
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
# Dataset metadata
annotations_creators:
  - expert-generated
language_creators:
  - found
language:
  - en
license:
  - apache-2.0
multilinguality:
  - monolingual
pretty_name: MedQA-USMLE Preprocessed
homepage: "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options"
tags:
  - medical
  - usmle
  - multiple-choice
  - reasoning
task_categories:
  - question-answering
configs:
- config_name: default
  data_files:
  - split: solve
    path: "data/solve-*.parquet"
  - split: unsolve
    path: "data/unsolve-*.parquet"
---

# MedQA-USMLE Preprocessed Dataset

This dataset is a preprocessed version of [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

The data has been formatted into a `question` and `answer` structure suitable for training or evaluating instruction-following language models.

## Data Structure

- `question`: The original medical question combined with the four multiple-choice options.
- `answer`: The correct answer index, prefixed with `####`.

### Example

**Question:**
```
A 60-year-old woman comes to the emergency department because of a 3-day history of fever, chills, and a productive cough. She has a 40-pack-year history of smoking. Her temperature is 38.5°C (101.3°F), blood pressure is 130/80 mm Hg, pulse is 100/min, and respirations are 22/min. Physical examination shows crackles in the left lower lung field. A chest x-ray shows consolidation in the left lower lobe. Which of the following is the most likely causative organism?

A) Streptococcus pneumoniae
B) Mycoplasma pneumoniae
C) Legionella pneumophila
D) Klebsiella pneumoniae
```

**Answer:**
```
#### A
```

## Splits

The original `train`, `test`, and `validation` splits are preserved.

## How to Use

```python
from datasets import load_dataset

ds = load_dataset("daichira/MedQA-USMLE-4-options_preprocess", split="train")
print(ds[0])
```

## Original Dataset

For more information, please refer to the original dataset card at [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

Output:
{
    "extracted_code": "from datasets import load_dataset\n\nds = load_dataset(\"daichira/MedQA-USMLE-4-options_preprocess\", split=\"train\")\nprint(ds[0])"
}
