# GC and CpG prediction

Compare GC and CpG prediction from frozen pretrained and matched random-weight
encoders using Ridge regression. Models use mean last-layer representations;
Evo2 uses its benchmark layer, `blocks.28.mlp.l3`.

## Data

`data/windows.tsv` provides 5,000 GRCh38 windows of 300 bp.
Training uses chr1–16, validation chr17–19, and testing chr20–22.
GC is the fraction of G/C bases; CpG is the number of CG dinucleotides divided by 299.

To regenerate the data:

```bash
python prepare_data.py --reference-fasta data/GRCh38.primary_assembly.genome.fa
```

## Run

First [generate random weights with seed 666](../../benchmark/README.md#architecture-matched-random-weights-match-rw)
from the repository's `benchmark` directory. Weights are stored in
`benchmark/matched-RW/<model-type>/`.

Then run from this directory in the corresponding model environment:

```bash
python extract_features.py --model-type nt \
  --checkpoint ../../benchmark/models/nucleotide-transformer-2.5b-multi-species \
  --output results/nt/pretrained

python extract_features.py --model-type nt --random-weights \
  --random-weights-dir ../../benchmark/matched-RW \
  --output results/nt/random

python predict.py --features results/nt/random results/nt/pretrained \
  --output results/nt/prediction
```

Model types: `dnabert`, `dnabert2`, `nt`, `caduceus`, `hyenadna`, `omnina`,
`deepgene`, `lucaone`, `GEN`, `evo2`. For Evo2, `--checkpoint` points to the `.pt` file.

Here `--random-weights-dir ../../benchmark/matched-RW` selects the generated
weight directory. Use the same option if you store weights elsewhere.

## Results

Ridge regularization is selected on the validation set.
`metrics.tsv` reports test Spearman R, R-squared and MAE for pretrained and random
encoders. `predictions.tsv` contains test predictions, and `analysis.json` records
run information.
