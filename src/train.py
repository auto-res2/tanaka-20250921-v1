import json, math, os, time, copy, itertools, random
from pathlib import Path
from collections import defaultdict

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm

from src.preprocess import prepare_dataset, RewardModelEntropy

# ─────────────────────── C3PO LOSS ──────────────────────────────
class C3POLoss(nn.Module):
    def __init__(self, feat_dim=4, hidden=16, kappa_init=0.0, eta_dual=0.05,
                 grow=1.5, beta_min=1e-3, beta_max=10.0):
        super().__init__()
        self.kappa_max = kappa_init
        self.eta = eta_dual
        self.grow = grow
        self.beta_min, self.beta_max = beta_min, beta_max
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Softplus()
        )

    def dual_update(self, mean_kl):
        # projected stochastic dual ascent
        lagrange = getattr(self, "_lagrange", 0.0)
        lagrange = max(0.0, lagrange + self.eta * (mean_kl - self.kappa_max))
        self._lagrange = lagrange
        return lagrange

    def forward(self, feats, ll_c, ll_r, ll_ref_c, ll_ref_r):
        """
        feats        : [B, F]  (no grad)
        ll_*         : scalar log-likelihood of the whole sequence
        """
        beta_i = self.mlp(feats).squeeze(-1).clamp(self.beta_min, self.beta_max)
        # ρ_i   (policy advantage vs reference)
        rho = (ll_c - ll_r) - (ll_ref_c - ll_ref_r)   # [B]
        loss = -F.logsigmoid(beta_i * rho).mean()

        # KL for the batch  (reference → policy, symmetric across pair)
        kl_batch = 0.5 * (ll_c - ll_ref_c + ll_r - ll_ref_r).mean()

        return loss, kl_batch, beta_i.mean()

# ─────────────────────── TRAINER ────────────────────────────────
class Trainer:
    def __init__(self, cfg, run_name):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], torch_dtype=dtype).to(self.device)
        # frozen reference
        self.ref_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.loss_fn = C3POLoss(feat_dim=4, hidden=16,
                                kappa_init=0.0, eta_dual=0.05, grow=1.5).to(self.device)

        # optimiser
        self.opt = AdamW(self.model.parameters(), lr=cfg["lr"],
                         betas=(0.9, 0.95), weight_decay=0.1)
        self.sched = get_cosine_schedule_with_warmup(
            self.opt, num_warmup_steps=cfg["warmup"],
            num_training_steps=cfg["max_steps"])
        self.batch, self.grad_accum = cfg["batch_size"], cfg["grad_accum"]
        self.run_name = run_name
        self.log = defaultdict(list)
        self.out_dir = Path(".research/iteration2") / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    @torch.no_grad()
    def _seq_log_prob(self, input_ids, attn_mask, model):
        logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
        shift = input_ids[:, 1:]
        ll = logits[:, :-1, :].log_softmax(-1).gather(-1,
                                                     shift.unsqueeze(-1)).squeeze(-1)
        ll = (ll * attn_mask[:, 1:]).sum(-1)          # length-norm NOT needed
        return ll                                     # [B]

    # -----------------------------------------
    def train_one_cfg(self, data):
        ds = TensorDataset(*data)
        loader = DataLoader(ds, sampler=RandomSampler(ds),
                            batch_size=self.batch, pin_memory=True)
        step, best_val_reward, plateau = 0, -1e9, 0

        pbar = tqdm(total=self.cfg["max_steps"])
        while step < self.cfg["max_steps"]:
            for batch in loader:
                (ids_c, mask_c, ids_r, mask_r,
                 feat_len, feat_ll_m, feat_entropy, feat_dsid) = \
                    [b.to(self.device) for b in batch]
                feats = torch.stack([feat_len.float(),
                                     feat_ll_m.float(),
                                     feat_entropy.float(),
                                     feat_dsid.float()], -1)

                # current policy log-probs
                ll_c = self._seq_log_prob(ids_c, mask_c, self.model)
                ll_r = self._seq_log_prob(ids_r, mask_r, self.model)
                # reference log-probs
                ll_ref_c = self._seq_log_prob(ids_c, mask_c, self.ref_model)
                ll_ref_r = self._seq_log_prob(ids_r, mask_r, self.ref_model)

                loss, kl, beta_mean = self.loss_fn(feats, ll_c, ll_r,
                                                   ll_ref_c, ll_ref_r)
                loss.backward()
                if (step + 1) % self.grad_accum == 0:
                    self.opt.step()
                    self.sched.step()
                    self.opt.zero_grad()

                # dual variable update
                lagr = self.loss_fn.dual_update(kl.item())

                # -------- logging every 10 steps -------------
                if step % 10 == 0:
                    self.log["train_loss"].append(loss.item())
                    self.log["kl"].append(kl.item())
                    self.log["beta"].append(beta_mean.item())
                    pbar.set_postfix(loss=f"{loss.item():.3f}",
                                     kl=f"{kl.item():.3f}",
                                     kappa=f"{self.loss_fn.kappa_max:.2f}",
                                     β=f"{beta_mean.item():.2f}")
                step += 1
                pbar.update(1)
                # ------------------ κmax growth ------------------
                if step % self.cfg["val_interval"] == 0:
                    val_reward, val_kl = self.validate(data[:512])  # cheap proxy
                    self.log["val_reward"].append(val_reward)
                    if val_reward < best_val_reward + 1e-4:
                        plateau += 1
                    else:
                        plateau = 0
                        best_val_reward = val_reward
                    if plateau >= 3 and val_kl < 0.8 * self.loss_fn.kappa_max:
                        self.loss_fn.kappa_max *= self.loss_fn.grow
                        plateau = 0
                if step >= self.cfg["max_steps"]:
                    break
        pbar.close()
        torch.save(self.model.state_dict(), self.out_dir / "final.pt")
        # ----------- dump JSON --------------
        json_path = self.out_dir / "train_log.json"
        with open(json_path, "w") as fp:
            json.dump(self.log, fp)
        return str(json_path)

    # -----------------------------------------
    @torch.no_grad()
    def validate(self, data):
        ids_c, mask_c, ids_r, mask_r, *_ = [d.to(self.device) for d in data]
        ll_c = self._seq_log_prob(ids_c, mask_c, self.model)
        ll_r = self._seq_log_prob(ids_r, mask_r, self.model)
        reward = (ll_c - ll_r).mean().item()
        ll_ref_c = self._seq_log_prob(ids_c, mask_c, self.ref_model)
        ll_ref_r = self._seq_log_prob(ids_r, mask_r, self.ref_model)
        kl = 0.5 * (ll_c - ll_ref_c + ll_r - ll_ref_r).mean().item()
        return reward, kl

# ─────────────────────── utility ───────────────────────────────
def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)

def run(cfg, run_name):
    set_seed(cfg["seed"])
    # ---------------- data ----------------
    try:
        rm_entropy = RewardModelEntropy()
    except Exception as e:
        print(f"Could not initialize RewardModelEntropy, features will be zero: {e}")
        rm_entropy = None

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token

    datasets = []
    for i, dcfg in enumerate(cfg["datasets"]):
        dcfg["id"] = i
        tensors = prepare_dataset(dcfg,
                                  tokenizer,
                                  ref_model=None, # Not passing ref model to save memory, will be created in Trainer
                                  rm_entropy=rm_entropy,
                                  split=dcfg["split"])
        datasets.append(tensors)

    all_keys = datasets[0].keys()
    merged_data = {k: torch.cat([d[k] for d in datasets], 0) for k in all_keys}

    trainer = Trainer(cfg, run_name)
    json_path = trainer.train_one_cfg(list(merged_data.values()))
    return json_path
