
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: answer
    dtype: string
  - name: options
    dtype: string
  - name: meta_info
    dtype: string
  - name: answer_idx
    dtype: string
  - name: metamap_phrases
    dtype: string
  - name: question
    dtype: string
  splits:
  - name: test
    num_bytes: 1870599
    num_examples: 1221
  - name: incomplete
    num_bytes: 70746
    num_examples: 52
  download_size: 1086096
  dataset_size: 1941345
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
  - split: incomplete
    path: data/incomplete-*
license: cc-by-4.0
task_categories:
- question-answering
language:
- en
tags:
- medical
---
# MedQA-USMLE-4-options-clean Dataset

## Overview
MedQA-USMLE-4-options-clean is an enhanced medical question-answering benchmark that builds upon the MedQA-USMLE dataset. Physicians analyzed the 1373 questions in the original dataset and moved 52 questions that were either malformed or incomplete to another split `incomplete`.

## Key Features
- Relabeled malformed/incorrect questions

## Dataset Details
- **Size**: 1373
- **Language**: English

## Data Source
- [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) dataset

## Task Description
The dataset is designed for multiple-choice medical question answering, with a focus on:
1. Clinical knowledge assessment

## Citation
If you use this dataset in your research, please cite:

```

```
Output:
{
    "extracted_code": ""
}
