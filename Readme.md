# Text Summarization Using Large Language Models (LLMs)

## Overview
This project evaluates the performance of various fine-tuned Large Language Models (LLMs) for text summarization tasks. The models compared include:
- T5
- RoBERTa
- PEGASUS
- DistillBART

These models have been fine-tuned on two benchmark datasets:
- XSUM
- Gigaword

The objective is to identify the most efficient and accurate model for abstractive summarization.

## Dataset
### XSUM 
- A single-document summarization dataset.
- Contains over 200,000 BBC articles with one-sentence summaries.
- Designed for creating short, highly abstracted summaries.

### Gigaword
- A large-scale summarization dataset from newswire sources.
- Task involves generating short, headline-like summaries.

## Commands to Reproduce the Results

### 1. Create Conda Env 

```html
conda env create -f environment.yml
```

### 2. Preprocess the Data
We have preprocessed the datasets for XSUM and Gigaword, and the processed files are stored in the following directories:

#### Gigaword
Preprocessed files are located in the directory : `dataset/gigaword/test.csv`. 
Each row contains `document` and `summary` columns.

#### XSUM
Preprocessed files are located in the directory : `dataset/xsum/xsum.csv`. 
Each row contains `input` and `target` columns.

### 3. Commands to reproduce the evaluation results

```html
python train.py --dataset_name gigaword --model_type t5 --eval_ok True
```

```html
python train.py --dataset_name gigaword --model_type roberta --eval_ok True
```

```html
python train.py --dataset_name gigaword --model_type bart_large --eval_ok True
```

```html
python train.py --dataset_name gigaword --model_type distill_bart --eval_ok True
```

```html
python train.py --dataset_name gigaword --model_type pegasus --eval_ok True
```

For reproducing results for xsum replace `gigaword` to `xsum`

### 4. Results
All the results are save into the `result/[dataset_name]/[model_type]/output_metrics.json`

**XSUM** dataset 

| Model       | Parameters | Rouge-1 | Rouge-2 | Rouge-l | Inference (sec) | 
|-------------|------------|---------|---------|---------|-----------------|
| T5          | 7.37 M     | 0.354   | 0.145   | 0.302   | 6.739           | 
| RoBERTA     | 1.53 M     | 0.444   | 0.176   | 0.333   | 5.974           | 
| PEGASUS     | 5.69 M     | 0.533   | 0.428   | 0.5333  | 22.548          | 
| DistillBART | 2.21 M     | 0.378   | 0.176   | 0.322   | 1.949           | 
| BART        | 4.06 M     | 0.430   | 0.213   | 0.370   | 9.348           | 

**Gigaword** dataset

| Model       | Parameters | Rouge-1 | Rouge-2 | Rouge-l | Inference (sec) | 
|-------------|------------|---------|---------|---------|-----------------|
| T5          | 7.37 M     | 0.316   | 0       | 0.316   | 5.805           | 
| RoBERTA     | 1.53 M     | 0.355   | 0.175   | 0.334   | 2.126           | 
| PEGASUS     | 5.69 M     | 0.279   | 0.116   | 0.253   | 3.226           | 
| DistillBART | 2.21 M     | 0.108   | 0.019   | 0.099   | 0.528           | 
| BART        | 4.06 M     | 0.173   | 0.046   | 0.158   | 1.684           | 


