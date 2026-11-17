# Token representation visualization

Compare pretrained and matched random-weight token representations using the
same 500 DNA sequences sampled from chromatin regions (510 bp each). Models use their final output;
Evo2 uses the benchmark layer `blocks.28.mlp.l3`.

Each token identity contributes one occurrence per sequence. One combined UMAP
figure shows random/pretrained representations colored by token identity in the
top two rows and token GC in the bottom two rows.

## Run

First [generate random weights with seed 666](../../benchmark/README.md#architecture-matched-random-weights-match-rw)
from the repository's `benchmark` directory. Weights are stored in
`benchmark/matched-RW/<model-type>/`.

Then run from this directory in the corresponding model environment:

```bash
python extract_tokens.py --model-type nt \
  --checkpoint ../../benchmark/models/nucleotide-transformer-2.5b-multi-species \
  --output results/nt/pretrained

python extract_tokens.py --model-type nt --random-weights \
  --random-weights-dir ../../benchmark/matched-RW \
  --output results/nt/random
```

Model types and checkpoint paths follow the [GC/CpG scripts](../gc-cpg-prediction/README.md).
Here `--random-weights-dir ../../benchmark/matched-RW` selects the generated
weight directory. Use the same option if you store weights elsewhere.

Plot with NumPy, Matplotlib, scikit-learn and umap-learn:

```bash
python visualize.py --features results/nt/random results/nt/pretrained \
  --output results/figures
```

Add more model directories to `--features` for a combined figure.
UMAP seed is 42. Token colors use up to eight
equally spaced identities in the lexically sorted native DNA vocabulary.
