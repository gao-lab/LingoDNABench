# Alu/exon token prediction

Final sequence inputs and **mean predicted probability** for each model.
No genome download, annotation processing, matching, plotting or statistical testing is needed here.
This is a within-model Alu/exon comparison, not a mutual-information estimator or cross-model accuracy ranking.

## Final data

- **data/paired_sequences.tsv**: the actual sequences, one Alu/exon pair per row.
  Columns: cohort, pair_id, alu_id, exon_id, alu_sequence, exon_sequence.
- **data/mean_predictions.tsv**: final, already-computed results for all 11 models/controls.
  Columns: cohort, pair_id, model, alu_mean_predicted, exon_mean_predicted.
- **data/provenance.json**: counts, source fingerprints and scoring definition.

There are 1,000 pairs in each cohort and 3,605 unique 200-bp windows in total.
IDs are chrom:start-end, using 0-based, half-open GRCh38 coordinates.
All sequences are reference-forward A/C/G/T strings.

The primary cohort matches chromosome, length and GC fraction (absolute difference ≤0.02).
The conservation_matched cohort additionally matches mean 100-way phyloP (difference ≤0.10),
retaining all 902 original pairs at the stricter 0.05 threshold and adding 98 pairs.
Alu annotations came from RepeatMasker and exons from GENCODE v44 protein-coding genes/transcripts.
Alu parents do not overlap annotated exons; exon parents do not overlap annotated repeats.
The cohorts share some windows and are not independent replication datasets.
The conservation extension was designed after earlier results, without reading model scores during matching.

## What is the mean predicted value?

For each observed target token, retain the model's **native DNA-sequence output IDs** (G):
complete k-mers/BPE pieces and N where present, excluding control, annotation,
protein/text and unused outputs. This set is fixed for a model, not target-specific.

~~~text
p_target = exp(logit_of_observed_token) / sum(exp(logits[G]))
mean_predicted = mean(p_target over the scorable tokens in one window)
~~~

Normalization is done **per token before averaging**. BPE tokens are not split into bases,
and candidates are not restricted to the target's length. Inputs contain no N,
but a model's native N output token is still included in G.
This probability is not top-1 accuracy. The uniform reference is 1/len(G), not always 0.25.

| Model name for --model | Prediction | Native tokens | DNA output IDs | Target scoring |
|---|---|---|---:|---|
| RandomWeight | MLM | 1-mer | 5 | CLS + DNA + EOS; mask each target |
| HyenaDNA | CLM | 1-mer | 5 | Left context; omit the first token |
| Caduceus | MLM | 1-mer | 5 | Native DNA + SEP; restore shared weights; mask each target |
| DNABERT3mer | MLM | Overlapping 3-mer | 64 | Mask every token overlapping the target's nucleotide span |
| DeepGENE | MLM | BPE | 4,091 | Native CLS, SEP prefix; mask each complete target |
| DNABERT2 | MLM | BPE | 4,091 | Native CLS + DNA + SEP; mask each complete target |
| OmniNA | CLM | BPE | 8,895 | Native BOS + DNA; predict every DNA token, including the first |
| LucaOne | MLM | Gene-alphabet 1-mer | 5 | Original gene-mask logits, not the protein output vocabulary |
| NT-2.5B-MS | MLM | Non-overlapping 6-mer | 4,101 | Native CLS prefix; score all 33 complete 6-mers |
| GENERator | CLM | Non-overlapping 6-mer | 4,096 | Left context; score 32 targets after the first context token |
| Evo2 | CLM | DNA byte/1-mer | 5 | Left context; omit the first token |

MLM: evaluate each target in a separately masked copy; an internal overlapping 3-mer target
requires masking up to five tokens to avoid leakage.
CLM: insert no MASK; logits at position j-1 predict token j.
The 6-mer models use the first 198 bp of each window and discard the trailing 2 bp.
No model receives flanking sequence outside the supplied window, no annotation prompt is added,
and no sequence is reverse-complement averaged.

RandomWeight uses the supplied model_init_666.pt, not newly generated weights.
It is one Lingo BERT-155M initialization, not an architecture-matched control for every gLM.
Only DNA **sequence** tokens are retained: NT includes its 4,096 six-mers plus A/T/C/G/N,
whereas GENERator annotation tokens such as <cds> are excluded.

## Read the final results without running a model

~~~python
import pandas as pd

scores = pd.read_csv("data/mean_predictions.tsv", sep="\t")
print(scores.groupby(["cohort", "model"])[["alu_mean_predicted", "exon_mean_predicted"]].mean())
~~~

Each result row refers to one pair; each of its two means is calculated over one window's targets.
Repeated windows across cohorts have the same score. Do not pool tokens across windows:
the experiment gives each window equal weight.

## Recompute from sequences

Run from this directory, in the model's validated environment, with its local checkpoint.
The script does not download checkpoints or train models.
See [model downloads/environments](../../benchmark/README.md) and the existing
[RandomWeight model implementation](../../pretrain_gLM/BERT/model.py).

~~~bash
python mask-token-prediction.py \
  --model DNABERT2 --checkpoint /path/to/DNABERT-2-117M \
  --input data/paired_sequences.tsv --output DNABERT2.mean_predictions.tsv
~~~

The same command works for Caduceus, HyenaDNA, DNABERT3mer, OmniNA,
NT-2.5B-MS and GENERator by changing the model name and checkpoint directory.

~~~bash
python mask-token-prediction.py --model RandomWeight \
  --checkpoint /path/to/RandomWeights/model_init_666.pt \
  --output RandomWeight.mean_predictions.tsv --batch-size 32

python mask-token-prediction.py --model DeepGENE \
  --checkpoint /path/to/DeepGene/model/pretrain_params_epoch_20 \
  --code-path /path/to/DeepGene/PanGeneGraphTrans \
  --tokenizer-path /path/to/DeepGene/PanGeneGraphTrans/tokenizer.json \
  --output DeepGENE.mean_predictions.tsv

python mask-token-prediction.py --model LucaOne \
  --checkpoint /path/to/LucaOne/checkpoint-directory \
  --code-path /path/to/LucaOneApp-master --output LucaOne.mean_predictions.tsv

# Use the separate, validated Evo2 A100-compatible environment.
python mask-token-prediction.py --model Evo2 \
  --checkpoint /path/to/evo2_7b.pt --output Evo2.mean_predictions.tsv
~~~

LucaOne's checkpoint directory must contain config.json, training_args.bin and pytorch.pth;
its code root must expose algorithms/. Use trusted checkpoints and original model implementations.
RandomWeight requires CUDA FlashAttention. Do not assume every model runs in one Python environment.

Use --cohort primary or --cohort conservation_matched for a single cohort.
Use --limit-pairs 1 for a small test; omit it for the final results.
The script scores each unique interval once, writes the two means per pair,
and saves a small .meta.json with checkpoint paths, input hash and retained DNA IDs.
Small hardware/precision differences are possible; the supplied table is the final experiment output.
The simplified script was GPU-smoke-tested on one Alu/exon pair per model for all 11 models;
the largest absolute difference from the supplied values was 0.000066. This was not a full rerun.

