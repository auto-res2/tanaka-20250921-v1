import hashlib, os, json, shutil, math, itertools, random, multiprocessing as mp
from pathlib import Path

import torch
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging
from tqdm import tqdm
logging.set_verbosity_error()

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()

# -------------------------------------------------------------
# generic per-dataset mapping into  (prompt, chosen, rejected)
# -------------------------------------------------------------
def _map_pair(sample):
    cols = sample.keys()
    if {"prompt", "chosen", "rejected"}.issubset(cols):
        return sample["prompt"], sample["chosen"], sample["rejected"]
    if {"question", "response_0", "response_1", "winner"}.issubset(cols):
        c, r = (sample["response_0"], sample["response_1"]) if sample["winner"] == 0 else \
               (sample["response_1"], sample["response_0"])
        return sample["question"], c, r
    if {"instruction", "output_1", "output_2", "label"}.issubset(cols):
        c, r = (sample["output_1"], sample["output_2"]) if sample["label"] == 0 else \
               (sample["output_2"], sample["output_1"])
        return sample["instruction"], c, r
    if {"question", "answer", "options"}.issubset(cols) and sample.get("answer_idx"):
        # For MedQA
        prompt = sample['question'] + "\nOptions:\n" + "\n".join([f"{k}) {v}" for k,v in sample['options'].items()])
        chosen = sample['options'][sample['answer_idx']]
        # Create a plausible but incorrect rejected answer
        rejected_options = [v for k,v in sample['options'].items() if k != sample['answer_idx']]
        rejected = random.choice(rejected_options) if rejected_options else "No alternative available."
        return prompt, chosen, rejected
    raise ValueError(f"Unknown pair format: {cols}")

# -------------------------------------------------------------
# reward-model for entropy feature
# -------------------------------------------------------------
class RewardModelEntropy:
    _model_cache = {}
    def __init__(self, model_id="OpenAssistant/reward-model-deberta-v3-large-v2"):
        if model_id not in RewardModelEntropy._model_cache:
            if torch.cuda.is_available():
                RewardModelEntropy._model_cache[model_id] = AutoModelForSequenceClassification.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16).eval().to("cuda:0")
            else:
                print("Warning: CUDA not available. Reward model will not be used.")
                RewardModelEntropy._model_cache[model_id] = None

        self.model = RewardModelEntropy._model_cache[model_id]
        if self.model:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    @torch.no_grad()
    def entropy(self, txts):
        if not self.model:
            return torch.zeros(len(txts))
        toks = self.tokenizer(txts, padding=True, truncation=True, return_tensors="pt").to("cuda:0")
        logits = self.model(**toks).logits.float()
        probs = logits.softmax(-1)
        ent = (-probs * probs.log()).sum(-1)        # [B]
        return ent.cpu()

# -------------------------------------------------------------
def tokenise_pair(tokenizer, prompt, answer, max_len):
    text = prompt + tokenizer.eos_token + answer if prompt else answer
    tok = tokenizer(text,
                    truncation=True, max_length=max_len,
                    padding='max_length', return_tensors="pt")
    return {k: v.squeeze(0) for k, v in tok.items()}

def build_features(samples, tokenizer, ref_model=None, rm_entropy=None, max_len=512, dataset_id=0):
    out = {k: [] for k in
           ["input_ids_c", "attention_mask_c",
            "input_ids_r", "attention_mask_r",
            "feat_len", "feat_ll_margin", "feat_entropy", "feat_dsid"]}
    # batched reference log-likelihood
    with torch.no_grad():
        for prompt, ch, rej in tqdm(samples, desc="feature-build", leave=False):
            tok_c = tokenise_pair(tokenizer, prompt, ch, max_len)
            tok_r = tokenise_pair(tokenizer, prompt, rej, max_len)
            # log-likelihood margin under frozen reference model
            ll_margin = 0.0 # Placeholder, computed in trainer
            # reward model entropy (uncertainty)
            ent = rm_entropy.entropy([prompt + ch]).item() if rm_entropy and prompt else 0.0
            # length
            length = tok_c["input_ids"].ne(tokenizer.pad_token_id).sum().item()
            # store
            out["input_ids_c"].append(tok_c["input_ids"])
            out["attention_mask_c"].append(tok_c["attention_mask"])
            out["input_ids_r"].append(tok_r["input_ids"])
            out["attention_mask_r"].append(tok_r["attention_mask"])
            out["feat_len"].append(length)
            out["feat_ll_margin"].append(ll_margin)
            out["feat_entropy"].append(ent)
            out["feat_dsid"].append(dataset_id)
    # stack
    out = {k: torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v)
           for k, v in out.items()}
    return out


def prepare_dataset(cfg, tokenizer, ref_model=None, rm_entropy=None, split="train"):
    """
    Returns torch tensors already ready for a PyTorch DataLoader
    """
    name = cfg["name"]
    ds_key = f"{name.replace('/', '_')}_{split}"
    path = CACHE_DIR / f"{ds_key}.pt"
    if path.exists():
        print(f"Loading cached dataset from {path}")
        return torch.load(path)
    print(f"Downloading {name} [{split}] …")
    try:
        raw = load_dataset(name, split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset {name} from {name}. Error: {e}")

    #  ----------       deduplicate & map        ----------
    seen = set()
    pairs = []
    for s in tqdm(raw, desc=f"Processing {name}"):
        try:
            prompt_text, chosen_text, rejected_text = _map_pair(s)
            # Ensure prompt, chosen, and rejected are strings
            prompt = str(prompt_text) if prompt_text is not None else ""
            chosen = str(chosen_text) if chosen_text is not None else ""
            rejected = str(rejected_text) if rejected_text is not None else ""
            h = sha1(prompt + chosen + rejected)
            if h in seen:                    # dedup
                continue
            seen.add(h)
            pairs.append((prompt, chosen, rejected))
        except (ValueError, KeyError) as e:
            # print(f"Skipping sample due to error: {e}")
            continue

    random.shuffle(pairs)
    #   length filter after tokenisation
    proc_pairs = []
    for (p, c, r) in tqdm(pairs, desc="len-filter"):
        lc = len(tokenizer(p + tokenizer.eos_token + c if p else c)["input_ids"])
        lr = len(tokenizer(p + tokenizer.eos_token + r if p else r)["input_ids"])
        if 4 <= lc <= cfg["max_len"] and 4 <= lr <= cfg["max_len"]:
            proc_pairs.append((p, c, r))
    # tiny subset for smoke test
    if cfg.get("limit"):
        proc_pairs = proc_pairs[: cfg["limit"]]
    print(f"Pairs kept for {name}: {len(proc_pairs)}")
    tensors = build_features(proc_pairs, tokenizer, ref_model, rm_entropy,
                             max_len=cfg["max_len"], dataset_id=cfg["id"])
    torch.save(tensors, path)
    return tensors
