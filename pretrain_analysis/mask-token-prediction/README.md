# Intra-species Mutual Information (MI) Probe via Masked-Token Recovery

This README documents a simple **masked-token recovery probe** to estimate how much **position-specific statistical dependency** (a proxy for mutual information) a genomic language model (gLM) has learned in different human genomic regions (e.g., exon vs Alu).

**Key idea:** for each nucleotide position, mask that position, ask the model to predict the masked base, and record the probability assigned to the *true* base. Higher probability suggests the model can better exploit surrounding context at that position.

---

## What this measures (intuitively)

For an input sequence `x = (x₁, …, xₙ)`, for each position `i`:

1. Construct `x^{(i→MASK)}` by masking `xᵢ`.
2. Compute the model’s distribution `p(· | x^{(i→MASK)})`.
3. Record `p(xᵢ | x^{(i→MASK)})`.

Aggregating these per-position probabilities across regions provides a **comparative signal** of how much contextual dependence the model captures in those regions.

> This is a *probe* (not a strict MI estimator). It is most meaningful for **comparisons** (exon vs Alu) under the same masking/evaluation protocol.

---

## Inputs & Outputs

### Inputs
- `exon.txt`, `alu.txt`: plain text files, **one DNA sequence per line**.
  - Sequences should match the model’s expected alphabet.

### Outputs
- `exon-<model_type>_prob.txt`, `alu-<model_type>_prob.txt`
  - One score per masked position (or per sequence-position pair, depending on script implementation).
  - The score corresponds to the model probability of the original nucleotide at the masked site.

---

## Run (BERT-Series scripts)

> The commands below are written for the **BERT-Series** implementation (MLM-style masking).  

```bash
# Random-weight baseline
model_type=RandomWeight
python mask-token-prediction.py exon.txt exon-${model_type}_prob.txt ${model_type}
python mask-token-prediction.py alu.txt  alu-${model_type}_prob.txt  ${model_type}

# Human-genome pretrained gLM
model_type=human-genome
python mask-token-prediction.py exon.txt exon-${model_type}_prob.txt ${model_type}
python mask-token-prediction.py alu.txt  alu-${model_type}_prob.txt  ${model_type}

# Multi-species pretrained gLM
model_type=multi-species
python mask-token-prediction.py exon.txt exon-${model_type}_prob.txt ${model_type}
python mask-token-prediction.py alu.txt  alu-${model_type}_prob.txt  ${model_type}
```

---

## Adapting to other gLMs

This repo’s current scripts are **BERT-Series**-oriented (MLM masking + masked-token head). For other gLM types, adapt the probe as follows.

### 1) Tokenization / input encoding
- **1-mer models:** easiest—mask one nucleotide directly.
- **k-mer tokenization (e.g., 3-mer, 6-mer non-overlapped):**
  - Masking “one nucleotide” may affect a k-mer token. Decide whether to:
    - mask the **token** containing the nucleotide, or
    - mask **all tokens** influenced by that nucleotide (overlapped k-mers), or
    - evaluate at the **token level** instead of nucleotide level.
  - Keep the protocol consistent across regions/models.

### 2) Objective type (MLM vs CLM)
- **MLM models (BERT-like):** direct masked-token prediction as above.
- **Causal LMs (CLM):**
  - Standard CLM cannot condition on *both sides* of a masked position without modification.
    - **Prefix-only scoring:** score `xᵢ` using only left context (positions `< i`).

