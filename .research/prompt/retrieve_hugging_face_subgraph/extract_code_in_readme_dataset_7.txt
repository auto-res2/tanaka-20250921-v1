
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  - name: options
    struct:
    - name: A
      dtype: string
    - name: B
      dtype: string
    - name: C
      dtype: string
    - name: D
      dtype: string
  - name: meta_info
    dtype: string
  - name: answer_idx
    dtype: string
  - name: metamap_phrases
    sequence: string
  splits:
  - name: train
    num_bytes: 15175834
    num_examples: 10178
  - name: test
    num_bytes: 1946030
    num_examples: 1273
  download_size: 8869925
  dataset_size: 17121864
---
# Dataset Card for "medqa_usmle"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
Output:
{
    "extracted_code": ""
}
