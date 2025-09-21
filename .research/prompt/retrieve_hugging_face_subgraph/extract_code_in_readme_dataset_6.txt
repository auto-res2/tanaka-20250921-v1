
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: sent1
    dtype: string
  - name: sent2
    dtype: string
  - name: ending0
    dtype: string
  - name: ending1
    dtype: string
  - name: ending2
    dtype: string
  - name: ending3
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 8960967
    num_examples: 10178
  - name: validation
    num_bytes: 1118434
    num_examples: 1272
  - name: test
    num_bytes: 1146462
    num_examples: 1273
  download_size: 6433072
  dataset_size: 11225863
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

Output:
{
    "extracted_code": ""
}
