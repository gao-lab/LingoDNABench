# Datasets & Preprocessing Guide (gLM Benchmark)


## Dataset Download

All benchmark datasets are hosted at:

```text
http://ftp.cbi.pku.edu.cn/pub/LingoDNABench/datasets/
```

### Download with `wget`

```bash
wget -r -np -nH --cut-dirs=3 -R "index.html*"  \
  "http://ftp.cbi.pku.edu.cn/pub/LingoDNABench/datasets/"


```


## Datasets Information

The table below summarizes the tasks and dataset statistics.

<details>
<summary><strong>Click to expand the full dataset table</strong></summary>

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Category</th>
      <th>Sequence length (bp)</th>
      <th>Number of data</th>
      <th>Output dim</th>
      <th>Description</th>
      <th>Task</th>
      <th>Reference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Chromatin profiling</td>
      <td>Histone modification</td>
      <td>510</td>
      <td>5,446,162</td>
      <td>3,090</td>
      <td>ENCODE ChIP-seq data</td>
      <td>Classification</td>
      <td>https://www.encodeproject.org/</td>
    </tr>
    <tr>
      <td>DNA accessibility</td>
      <td>510</td>
      <td>3,530,399</td>
      <td>1,770</td>
      <td>ENCODE DNase-seq and ATAC-seq data</td>
      <td>Classification</td>
      <td>https://www.encodeproject.org/</td>
    </tr>
    <tr>
      <td>DNA methylation</td>
      <td>41</td>
      <td>5mC: 4,688; 6mA: 36,670</td>
      <td>1</td>
      <td>Detection of 5mC and 6mA modifications</td>
      <td>Classification</td>
      <td>https://www.biorxiv.org/content/10.1101/2024.08.16.608288v1</td>
    </tr>
    <tr>
      <td rowspan="7">Transcription regulation</td>
      <td>TFBS</td>
      <td>510</td>
      <td>2,075,603</td>
      <td>3,572</td>
      <td>ENCODE ChIP-seq data</td>
      <td>Classification</td>
      <td>https://www.encodeproject.org/</td>
    </tr>
    <tr>
      <td>Promoter</td>
      <td>300 / 70</td>
      <td>TATA: 6,130; non-TATA: 53,066; all: 59,196</td>
      <td>1</td>
      <td>-249~+50bp / -34~+35bp around TSS; includes TATA and non-TATA promoters</td>
      <td>Classification</td>
      <td>https://arxiv.org/abs/2306.15006</td>
    </tr>
    <tr>
      <td>Enhancer</td>
      <td>200</td>
      <td>Human: 12,639; Mouse: 13,378</td>
      <td>1</td>
      <td>Vista Enhancer Browser (241106). Split to 200 bp bins; CD-HIT removes high similarity (&gt;0.8). Tissue/stage: fb_e11.5 (pos) vs e11.5 (neg).</td>
      <td>Classification</td>
      <td>https://enhancer.lbl.gov/vista/</td>
    </tr>
    <tr>
      <td>Silencer</td>
      <td>200</td>
      <td>4,000</td>
      <td>1</td>
      <td>Candidate silencer prediction</td>
      <td>Classification</td>
      <td>https://github.com/xy-chen16/DeepSilencer, https://academic.oup.com/nar/article/49/D1/D221/5921294</td>
    </tr>
    <tr>
      <td>CRE activity</td>
      <td>230</td>
      <td>HepG2: 245,852; K562: 393,328</td>
      <td>1</td>
      <td>lentiMPRA activity for candidate promoters/enhancers in HepG2 and K562</td>
      <td>Regression</td>
      <td>https://www.nature.com/articles/s41586-024-08430-9</td>
    </tr>
    <tr>
      <td>Promoter–promoter interaction</td>
      <td>1000 + 1000</td>
      <td>tB: 162,391; Mon: 136,164; FoeT: 143,347; tCD4: 155,821; nCD4: 166,806; tCD8: 154,059</td>
      <td>1</td>
      <td>Promoter–promoter interaction in 6 cell types (tB, Mon, FoeT, tCD4, nCD4, tCD8)</td>
      <td>Classification</td>
      <td>https://academic.oup.com/nar/article/47/10/e60/5380496</td>
    </tr>
    <tr>
      <td>Enhancer–promoter interaction</td>
      <td>2000 + 1000</td>
      <td>tB: 176,096; Mon: 162,104; FoeT: 133,263; tCD4: 161,364; nCD4: 174,269; tCD8: 162,990</td>
      <td>1</td>
      <td>Enhancer–promoter interaction in 6 cell types (tB, Mon, FoeT, tCD4, nCD4, tCD8)</td>
      <td>Classification</td>
      <td>https://academic.oup.com/nar/article/47/10/e60/5380496</td>
    </tr>
    <tr>
      <td rowspan="5">Post-transcription regulation</td>
      <td>Polyadenylation signals</td>
      <td>600</td>
      <td>AATAAA: 22,602; all: 41,864</td>
      <td>1</td>
      <td>PolyA signal variants (including canonical AATAAA)</td>
      <td>Classification</td>
      <td>https://academic.oup.com/bioinformatics/article/38/17/4053/6633307</td>
    </tr>
    <tr>
      <td>Translation initiation sites</td>
      <td>600</td>
      <td>56,488</td>
      <td>1</td>
      <td>Translation initiation site prediction</td>
      <td>Classification</td>
      <td>https://academic.oup.com/bioinformatics/article/38/17/4053/6633307</td>
    </tr>
    <tr>
      <td>Splice site</td>
      <td>400</td>
      <td>Acceptor: 22,154; Donor: 21,945</td>
      <td>1</td>
      <td>Balanced, error-free. Negatives from exon/intron regions and GT/AG dinucleotides.</td>
      <td>Classification</td>
      <td>https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04471-3</td>
    </tr>
    <tr>
      <td>Exon PSI</td>
      <td>400 + 400</td>
      <td>40,737</td>
      <td>56</td>
      <td>Percent spliced-in (PSI) across 56 tissues. Use the custom loss function in the dataset dictionary for training.</td>
      <td>Regression</td>
      <td>https://github.com/gao-lab/DNALingo-dev/blob/main/www.biorxiv.org/content/10.1101/2024.02.29.582810v2</td>
    </tr>
    <tr>
      <td>Intron retention</td>
      <td>600</td>
      <td>75,922</td>
      <td>1</td>
      <td>Alternative splicing: intron retention</td>
      <td>Classification</td>
      <td>https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012755</td>
    </tr>
    <tr>
      <td>Gene expression</td>
      <td>Bulk RNA-seq</td>
      <td>2000</td>
      <td>21,700</td>
      <td>53</td>
      <td>53 GTEx tissues; promoter sequence (TSS -1k~+1k) predicts mRNA/lncRNA expression</td>
      <td>Regression</td>
      <td>https://www.nature.com/articles/s41592-021-01252-x</td>
    </tr>
    <tr>
      <td rowspan="4">Variant effect prediction</td>
      <td>Disease-related (coding)</td>
      <td>1</td>
      <td>122,237</td>
      <td>1</td>
      <td>ClinVar (240307) + GENCODE v44, GRCh38; SNVs labeled Pathogenic vs Benign</td>
      <td>Classification</td>
      <td>https://www.ncbi.nlm.nih.gov/clinvar/</td>
    </tr>
    <tr>
      <td>Disease-related (noncoding)</td>
      <td>1</td>
      <td>111,202</td>
      <td>1</td>
      <td>ClinVar (240307) + GENCODE v44, GRCh38; SNVs labeled Pathogenic vs Benign</td>
      <td>Classification</td>
      <td>https://www.ncbi.nlm.nih.gov/clinvar/</td>
    </tr>
    <tr>
      <td>Transcription-related (eQTL)</td>
      <td>1</td>
      <td>108 ~ 5,480</td>
      <td>1</td>
      <td>eQTLs and negatives in 49 GTEx tissues; GRCh38</td>
      <td>Classification</td>
      <td>https://www.nature.com/articles/s41592-021-01252-x</td>
    </tr>
    <tr>
      <td>Transcription-related (MPRA)</td>
      <td>1</td>
      <td>21,000</td>
      <td>1</td>
      <td>K562 &amp; HepG2; sampled 210k (pos:neg = 1:20) from REVA; GRCh38</td>
      <td>Classification</td>
      <td>https://academic.oup.com/gpb/article/19/4/590/7230396</td>
    </tr>
  </tbody>
</table>

</details>

---

## Data Processing

### Default setting (all tasks unless stated otherwise)

- `train/dev/test_data_0.txt`: DNA sequences (one per line).
- `train/dev/test_label.txt`: labels aligned with `data_0` (one per line).

In the benchmark workflow:
1. **Convert sequences → gLM embeddings** (per model).
2. **Train a downstream adapter / predictor** using embeddings + labels.

### Expression level prediction (Gene expression)

- `train/dev/test_data_0.txt`: DNA sequences (one per line).
- `train/dev/test_target.npy`: regression targets (`N × 53`).

### Exon PSI prediction (dual-input regression)

- `train/dev/test_data_0.txt` and `train/dev/test_data_1.txt`: two aligned DNA sequences per sample.
- `train/dev/test_target.npy`: regression targets (`N × 56`).

> The PSI task typically requires a **custom loss**; use the loss function provided with the dataset metadata/dictionary.

### PPI / EPI (Promoter–promoter / Enhancer–promoter interaction)

- `train/dev/test_data_0.txt` and `train/dev/test_data_1.txt`: dual-input sequences.
- For large datasets, data may be **sharded** and loaded as a **streaming dataset** during training.

The script accepts `.h5` and can split into chunks (e.g., **5,000 records per shard**):
- `script/benchmark-PPI_EPI.py`

### TFBS / DNA accessibility / Histone modification (TFRecord pipeline)

Due to dataset size, embeddings are converted into **TFRecords** for efficient TensorFlow input pipelines.

Typical steps:
1. Prepare and split sequences into manageable chunks (e.g., **25k sequences per file**).
2. Extract embeddings and write TFRecords.
3. Train via TF pipeline.

Example commands (edit paths to match your local layout):

```bash
conda install bioconda::bedtools=2.26.0
wget -c https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/GRCh38.primary_assembly.genome.fa.gz
gunzip GRCh38.primary_assembly.genome.fa.gz
mv  GRCh38.primary_assembly.genome.fa ../data
# 1) Prepare Basset-style sequence shards
# sh script/01_process_basset_data.sh <feature_name> <input_dir> <output_dir> <blacklist_bed> <reference_fasta>
sh script/01_process_basset_data.sh TFBS ./TFBS ./TFBS ../data/GRCh38.primary_assembly.genome.fa

# 2) Split into chunks and write index files / metadata
# python script/02_process_basset_data.py <feature_name> <input_dir> <output_dir> <output_dim> <split_size>
python script/02_process_basset_data.py TFBS ./TFBS ./TFBS 3572 25000
```

Embedding extraction + TFRecord writing (fill in model variables):

```bash
model_name="nucleotide-transformer-2.5b-multi-species"
model_type="nt"          # e.g., nt / dnabert2 / ...
layer=-1
embedding_len=510        # bp length for TFBS/Accessibility/Histone
token_len=$((embedding_len / 6 + 1))  # for non-overlapped 6-mer models (adjust per tokenizer)

input_dir="./datasets/TFBS"
feature_name="TFBS"
mkdir -p "${input_dir}/${model_name}"
output_dir="${input_dir}/${model_name}"

# Train shards
for idx in {0..57}; do
  seq_file="${input_dir}/${feature_name}_train_${idx}.seq"
  python ../get-embeddings.py ../models "${model_type}" "${model_name}" "${seq_file}" "${output_dir}" "${token_len}" "${layer}"
  python 03_embed2tfr.py "${idx}" train "${feature_name}" "${model_name}" "${layer}" "${input_dir}"
done

# Valid shards
for idx in {0..11}; do
  seq_file="${input_dir}/${feature_name}_valid_${idx}.seq"
  python ../get-embeddings.py ../models "${model_type}" "${model_name}" "${seq_file}" "${output_dir}" "${token_len}" "${layer}"
  python 03_embed2tfr.py "${idx}" valid "${feature_name}" "${model_name}" "${layer}" "${input_dir}"
done

# Test shards
for idx in {0..14}; do
  seq_file="${input_dir}/${feature_name}_test_${idx}.seq"
  python ../get-embeddings.py ../models "${model_type}" "${model_name}" "${seq_file}" "${output_dir}" "${token_len}" "${layer}"
  python 03_embed2tfr.py "${idx}" test "${feature_name}" "${model_name}" "${layer}" "${input_dir}"
done
```

### Variant effect prediction

- `train/dev/test.vcf`: VCF sites.
- `train/dev/test_data_0.txt`: reference (pre-mutation) 512 bp sequence centered at the mutation site.
- `train/dev/test_data_1.txt`: alternate (post-mutation) 512 bp sequence centered at the mutation site.
- `train/dev/test_label.txt`: labels.

For **zero-shot** evaluation, you can merge train/dev/test splits.  
The provided split is retained to enable optional adapter tuning experiments.

