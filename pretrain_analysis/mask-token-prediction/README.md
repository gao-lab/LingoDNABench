# Intra-species Mutual Information (MI) Probe via Masked-Token Recovery

This README documents a simple **masked-token recovery probe** to estimate how much **position-specific statistical dependency** (a proxy for mutual information) a genomic language model (gLM) has learned in different human genomic regions (e.g., exon vs Alu).

**Key idea:** for each nucleotide position, mask that position, ask the model to predict the masked base, and record the probability assigned to the *true* base. Higher probability suggests the model can better exploit surrounding context at that position.

---

## What this measures (intuitively)

For an input sequence `x = (x₁, …, xₙ)`, for each position `i`:

1. Construct `x^{(i→MASK)}` by masking `xᵢ`.
2. Compute the model’s distribution `p(· | x^{(i→MASK)})`.
3. Record `p(xᵢ | x^{(i→MASK)})`.

Aggregating these per-position probabilities (or log-probabilities) across regions provides a **comparative signal** of how much contextual dependence the model captures in those regions.

> This is a *probe* (not a strict MI estimator). It is most meaningful for **comparisons** (exon vs Alu; multi-species vs human-genome; etc.) under the same masking/evaluation protocol.

---

## Inputs & Outputs

### Inputs
- `exon.txt`, `alu.txt`: plain text files, **one DNA sequence per line**.
  - Sequences should match the model’s expected alphabet (typically `A/C/G/T` plus possibly `N`).

### Outputs
- `exon-<model_type>_prob.txt`, `alu-<model_type>_prob.txt`
  - One score per masked position (or per sequence-position pair, depending on script implementation).
  - The score corresponds to the model probability of the original nucleotide at the masked site.

---

## Run (BERT-Series scripts)

> **Important:** The commands below are written for the **BERT-Series** implementation (MLM-style masking).  
> For other gLM families (e.g., causal LMs, state-space models, Hyena-based models), you should adapt masking, tokenization, and scoring as described in [Adapting to other gLMs](#adapting-to-other-glms).

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

### Recommended: store log-probabilities
If `mask-token-prediction.py` supports it, prefer saving **log-probabilities** (more stable for aggregation):
- per-position: `log p(xᵢ | context)`
- per-sequence: sum/mean over positions

If not supported, you can convert `p` to `log p` downstream.

---

## How to interpret results

Common summaries:
- **Mean probability** (or mean log-probability) across positions
- **Distribution plots** (histograms / violin) comparing exon vs Alu
- **Per-position curves** (to see whether specific motifs/contexts stand out)

Typical expectation (qualitative):
- More constrained / structured regions (e.g., exon) may show higher contextual predictability than repeats, depending on model and training data.
- Multi-species pretraining can strengthen conservation-driven signals, potentially increasing predictability in conserved regions.

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
  - Practical alternatives:
    - **Prefix-only scoring:** score `xᵢ` using only left context (positions `< i`).
    - **Pseudo-bidirectional scoring:** run two passes (left-to-right and right-to-left) if you have a reverse model.
    - **Span corruption / infilling models:** if the model supports infilling, use its native infill interface.
  - Do **not** compare MLM and CLM scores naively unless the conditioning context is matched.

### 3) What score to record
- Use **(log) probability of the true base** at the probed location.
- If the model outputs logits for a larger alphabet (e.g., includes `N`), make sure you map bases consistently.

### 4) Efficiency
Masking every position is expensive.
Options:
- Subsample positions (fixed random seed).
- Probe a fixed window per sequence.
- Cache tokenization and only change the masked position.

---

## Minimal checklist for fair comparisons

- Same input sequences and preprocessing
- Same masking protocol (what is masked, how it’s represented)
- Same scoring definition (prob vs log-prob)
- Same set of probed positions (or same sampling seed)
- Same batch size / precision settings (to avoid subtle numeric drift)

---

## Notes

- If you see consistently ~uniform probabilities (~0.25 for A/C/G/T), confirm:
  - the model is actually loaded,
  - the mask token is correct for your tokenizer,
  - your evaluation is not accidentally using random weights.
- For `N` or ambiguous bases, define whether you:
  - filter them out, or
  - keep them but skip scoring at those positions.

---

## Original snippet (for reference)

The optimized README is based on the original commands and description in `README.md`.
