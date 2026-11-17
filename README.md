
---

# LingoDNABench

**Conventional self-supervised pretraining paradigms constrain the capacity of genomic language models on regulatory decoding**

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
* **Architecture-matched random weights (match-RW):** Fixed random encoders for
  ten gLM architectures. Generate weights locally with seed 666 in
  `benchmark/matched-RW/<model-type>/`, then extract embeddings with
  `--random-weights`; see the [generation and benchmark workflow](benchmark/README.md#architecture-matched-random-weights-match-rw).


### Tasks

Please refer to the table in `benchmark/datasets/README.md` for detailed task information.

### Evaluation Protocol

* **Standardized Metrics:** Task-specific performance metrics (AUROC for classification, SpearmanR for regression applictions).
* **Unified Pipeline:** A consistent training and evaluation protocol across models to ensure fair comparison.
* **Robustness Measures:** Each adapter model is trained across **five random seeds** to ensure statistical significance.

---

## Pretraining Analysis

### Alignment: Pretraining vs. Downstream Performance

Compare checkpoint-level pretraining loss with downstream performance for the
human-genome (`model_H`) and multi-species (`model_M`) BERT-155M models, using
the same evaluation protocol across checkpoints. See the
[analysis workflow](pretrain_analysis/pretrain_application_alignment/README.md).

### A Mutual Information Perspective

* **Different mutual information structure in genome region:** gLM show significantly different performance in predicting mask token in different genome region. (see `pretrain_analysis/mask_token_prediction`)

## Repository Structure

```text
LingoDNABench/
├── benchmark/      # datasets and evaluation pipelines
├── pretrain_gLM/       # pretraining scripts or configs
└── pretrain_analysis/       # pretrain–downstream analysis
```
