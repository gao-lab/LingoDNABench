# Pretraining loss and downstream performance

Compare pretraining loss with downstream performance across checkpoints of
human-genome (`model_H`) and multi-species (`model_M`) BERT-155M models.
Use frozen last-layer representations for supervised tasks and zero-shot
variant effect prediction.

## Checkpoints and loss

Download [BERT-155M-Series](../../pretrain_gLM/README.md#cehckpoints) and place it
in this directory:

```text
pretrain_application_alignment/BERT-155M-Series/
├── model_H/
│   ├── model_0.pt ... model_18.pt
│   └── epoch_metrics.tsv
└── model_M/
    ├── model_0_79109.pt ... model_8_711981.pt
    └── loss.txt
```

| Family | Checkpoint index | Loss record |
|---|---|---|
| model_H | 0–18 | `epoch_metrics.tsv`: `epoch = index`, column `train_loss` |
| model_M | 0–8 | `loss.txt`: `epoch = index`, column `Loss` |

Both loss files use zero-based epochs matching checkpoint indices.
Index 0 is the first saved training checkpoint. H loss decreases from 1.2543 to
0.7467; M loss decreases from 0.9789 to 0.7721 across the released checkpoints.
Use the checkpoint index to match each loss record with downstream results.

## Extract representations

Run from this directory in the BERT-series environment described in the
[benchmark README](../../benchmark/README.md#31-bert-series-environment).
For the proximal TATA-promoter task:

```bash
data_dir=../../benchmark/datasets/promoter/prom_300_tata
family=model_M

for index in {0..8}; do
  for split in train dev test; do
    python get-embedding-pretrain_loss.py \
      "${data_dir}/${split}_data_0.txt" "${data_dir}/${family}-${index}" \
      300 -1 "${index}" --model-family "${family}" \
      --checkpoint-dir ./BERT-155M-Series
  done
done
```

To extract model_H representations, use `family=model_H` and indices `{0..18}`.
The five positional arguments remain input file, output directory, token length,
layer and checkpoint index. `--model-family` defaults to `model_M`.

## Evaluate downstream performance

### Supervised tasks

After extraction, use the downstream evaluation environment:

```bash
data_dir=../../benchmark/datasets/promoter/prom_300_tata
family=model_M

for index in {0..8}; do
  for seed in {42..46}; do
    python ../../benchmark/script/benchmark-default-applications.py \
      "${family}-${index}" "${data_dir}" -1 "${seed}" 1 False
  done
done
```

Other supervised tasks follow the [benchmark workflow](../../benchmark/README.md#5-downstream-evaluation)
with the corresponding `<family>-<index>` model name.

### Variant effect prediction (zero-shot)

Example using a single model_H checkpoint. Run extraction in the BERT-series
environment:

```bash
data_dir=../../benchmark/datasets/variant/var_disease_noncoding
family=model_H
index=0
model_name=${family}-${index}

for split in train dev test; do
  for allele in 0 1; do
    python get-embedding-pretrain_loss.py \
      "${data_dir}/${split}_data_${allele}.txt" "${data_dir}/${model_name}" \
      512 -1 "${index}" --model-family "${family}" \
      --checkpoint-dir ./BERT-155M-Series
  done
done
```

`data_0` contains reference sequences and `data_1` contains alternative sequences.
In the downstream evaluation environment, run:

```bash
python ../../benchmark/script/variant_effect_prediction-zero-shot.py \
  "${data_dir}" "${model_name}" -1
```

The script scores variants by the Euclidean distance between mean-pooled
reference and alternative representations and reports AUROC in
`<data_dir>/metrics-<family>-<index>--1.txt`.
Change `index` to evaluate another checkpoint and pair its AUROC with its loss.

## Relate loss to performance

For supervised tasks, results are written to
`<data_dir>/<family>-<index>/checkpoint/metrics--1.txt`.
Pair each checkpoint's pretraining loss with its downstream test metric:
AUROC for classification and Spearman R for regression. Summarize the five
adapter seeds as mean and standard deviation. For zero-shot variant effect
prediction, use the AUROC directly; no adapter training is involved.

Compare loss and performance across checkpoints within each family and task.
Plot pretraining loss against mean downstream performance, or report their
Spearman correlation. A negative correlation means lower pretraining loss is
associated with higher downstream performance; a weak or positive correlation
indicates that the loss reduction does not consistently track downstream gains.
