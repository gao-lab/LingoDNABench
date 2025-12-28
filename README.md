
---

# LingoDNABench

**Canonical pretraining paradigms constrain the capacity of genomic language models to decode regulatory code**

LingoDNABench was developed to address a fundamental yet underexplored question:

> **Does canonical Masked Language Model (MLM) pretraining truly equip genomic language models (gLMs) with the ability to decode complex regulatory codes?**

Rather than focusing exclusively on downstream leaderboard rankings, this benchmark scrutinizes the **alignment between pretraining dynamics and downstream regulatory task performance**.

---

## Benchmark Overview

### Models

The benchmark evaluates a diverse suite of models:

* **Genomic Language Models (gLMs):**  Please refer to the table in `benchmark/README.md` for detailed task information.
* **Random baselines (using the architecture defined in `pretrain_gLM/BERT`):** 
  * **RandomWeight:** Models initialized with completely random parameters.
  * **RandomSeq:** Models pretrained using MLM on purely random nucleotide sequences 


### Tasks

Please refer to the table in `benchmark/datasets/README.md` for detailed task information.

### Evaluation Protocol

* **Standardized Metrics:** Task-specific performance metrics (AUROC for classification, SpearmanR for regression applictions).
* **Unified Pipeline:** A consistent training and evaluation protocol across models to ensure fair comparison.
* **Robustness Measures:** Each adapter model is trained across **five random seeds** to ensure statistical significance.

---

## Pretraining Analysis

### Alignment: Pretraining vs. Downstream Performance

* Systematically analyze the correlation between pretraining loss trajectories and the resulting downstream task performance.(see `pretrain_analysis/pretrain_downstream_alignment`)

### A Mutual Information Perspective

* **Different mutual information structure in genome region:** gLM show significantly different performance in predicting mask token in different genome region. (see `pretrain_analysis/mask_token_prediction`)


## Repository Structure

```text
LingoDNABench/
├── benchmark/      # datasets and evaluation pipelines
├── pretrain_gLM/       # pretraining scripts or configs
└── pretrain_analysis/       # pretrain–downstream analysis
```
