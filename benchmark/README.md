# Benchmark
## Genomic language model preparation
Download the genomic language model to benchmarking
```
sh models-download.sh
```
## Datasets
<!-- GitHub Markdown文件中的HTML表格，支持合并单元格 -->
<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Category</th>
      <th>Sequence length (bp)</th>
      <th>Number of data</th>
      <th>Description</th>
      <th>Classification/Regression</th>
      <th>Reference</th>
    </tr>
  </thead>
  <tbody>
    <!-- Chromatin profiling -->
    <tr>
      <td rowspan="3">Chromatin profiling</td>
      <td>Histone modification</td>
      <td>510</td>
      <td>5,446,162</td>
      <td>ENCODE ChIP-seq data</td>
      <td>Classification</td>
      <td>https://www.encodeproject.org/</td>
    </tr>
    <tr>
      <td>DNA accessibility</td>
      <td>510</td>
      <td>3,530,399</td>
      <td>ENCODE Dnase-seq and ATAC-seq data</td>
      <td>Classification</td>
      <td>https://www.encodeproject.org/</td>
    </tr>
    <tr>
      <td>DNA methylation</td>
      <td>41</td>
      <td>5mC: 4,688; 6mA: 36,670</td>
      <td>Detection of 5-methylcytosine (5mC) and N6-methyladenosine (6mA) modifications in DNA sequences.</td>
      <td>Classification</td>
      <td>https://www.biorxiv.org/content/10.1101/2024.08.16.608288v1</td>
    </tr>
    
    <!-- Transcription regulation -->
    <tr>
      <td rowspan="7">Transcription regulation</td>
      <td>TFBS</td>
      <td>510</td>
      <td>2,075,603</td>
      <td>ENCODE ChIP-seq data</td>
      <td>Classification</td>
      <td>https://www.encodeproject.org/</td>
    </tr>
    <tr>
      <td>Promoter</td>
      <td>300/70</td>
      <td>TATA: 6,130; non-TATA: 53,066; all: 59,196</td>
      <td>-249~+50bp / -34~+35bp of TSS, Including TATA promoter and non-TATA promoter</td>
      <td>Classification</td>
      <td>https://arxiv.org/abs/2306.15006</td>
    </tr>
    <tr>
      <td>Enhancer</td>
      <td>200</td>
      <td>Human: 12,639; Mouse: 13,378</td>
      <td>Data was downloaded from Vista Enhancer Browser. Positive and negative enhancers were split to 200 bp bins.</td>
      <td>Classification</td>
      <td>https://enhancer.lbl.gov/vista/</td>
    </tr>
    <tr>
      <td>Silencer</td>
      <td>200</td>
      <td>4,000</td>
      <td>Candidate silencer prediction</td>
      <td>Classification</td>
      <td>https://github.com/xy-chen16/DeepSilencer</td>
    </tr>
    <tr>
      <td>CRE activity</td>
      <td>230</td>
      <td>HepG2: 245,852; K562: 393,328</td>
      <td>CRE: cis regulatory element, here is potential promoter and enhancer. Regulatory activity measured by lentiMPRA in HepG2 and K562.</td>
      <td>Regression</td>
      <td>https://www.nature.com/articles/s41586-024-08430-9</td>
    </tr>
    <tr>
      <td>Promoter-promoter interaction</td>
      <td>1000+1000</td>
      <td>tB: 162,391; Mon: 136,164; FoeT: 143,347; tCD4: 155,821; nCD4: 166,806; tCD8: 154,059</td>
      <td>Promoter-promoter interaction. Six cell lines: total B cells (tB), monocytes (Mon), foetal thymus (FoeT), total CD4+ T cells (tCD4), naive CD4+ T cells (nCD4), and total CD8+ T cells (tCD8).</td>
      <td>Classification</td>
      <td>https://academic.oup.com/nar/article/47/10/e60/5380496?login=true</td>
    </tr>
    <tr>
      <td>Enhancer-promoter interaction</td>
      <td>2000+1000</td>
      <td>tB: 176,096; Mon: 162,104; FoeT: 133,263; tCD4: 161,364; nCD4: 174,269; tCD8: 162,990</td>
      <td>Enhancer-promoter interaction. Six cell lines: total B cells (tB), monocytes (Mon), foetal thymus (FoeT), total CD4+ T cells (tCD4), naive CD4+ T cells (nCD4), and total CD8+ T cells (tCD8).</td>
      <td>Classification</td>
      <td>https://academic.oup.com/nar/article/47/10/e60/5380496?login=true</td>
    </tr>
    
    <!-- Post-transcription regulation -->
    <tr>
      <td rowspan="6">Post-transcription regulation</td>
      <td>Polyadenylation signals</td>
      <td>600</td>
      <td>AATAAA: 22,602; all: 41,864</td>
      <td>Polyadenylation signals, with AATAAA variant and all variant</td>
      <td>Classification</td>
      <td>https://academic.oup.com/bioinformatics/article/38/17/4053/6633307?login=true</td>
    </tr>
    <tr>
      <td>Translation initiation sites</td>
      <td>600</td>
      <td>56,488</td>
      <td>Translation initiation sites</td>
      <td>Classification</td>
      <td>https://academic.oup.com/bioinformatics/article/38/17/4053/6633307?login=true</td>
    </tr>
    <tr>
      <td>Splice site</td>
      <td>400</td>
      <td>Acceptor: 22,154; Donor: 21,945</td>
      <td>Two datasets: acceptor and negative; donor and negative. Balanced dataset, error-free.</td>
      <td>Classification</td>
      <td>https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04471-3</td>
    </tr>
    <tr>
      <td>Exon PIS</td>
      <td>400+400</td>
      <td>40,737</td>
      <td>Percent spliced-in (PSI): skipping level of on exon in alternative splicing. Input sequence: 300 + 100 (acceptor) and 100 + 300 (donor); Number of target tissues: 56.</td>
      <td>Regression</td>
      <td>https://github.com/gao-lab/DNALingo-dev/blob/main/www.biorxiv.org/content/10.1101/2024.02.29.582810v2</td>
    </tr>
    <tr>
      <td>Intron retention</td>
      <td>600</td>
      <td>75,922</td>
      <td>Alternative splicing: intron retention</td>
      <td>Classification</td>
      <td>https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012755</td>
    </tr>
    
    <!-- Gene expression -->
    <tr>
      <td rowspan="1">Bulk RNA-seq</td>
      <td>2000</td>
      <td>21,700</td>
      <td>53 GTEx tissues, mRNA and lncRNA transcription prediction based on promoter sequence (TSS -1k~+1k).</td>
      <td>Regression</td>
      <td>https://www.nature.com/articles/s41592-021-01252-x</td>
    </tr>
    
    <!-- Variant effect prediction -->
    <tr>
      <td rowspan="4">Variant effect prediction</td>
      <td>Disease-related coding</td>
      <td>1</td>
      <td>122,237</td>
      <td>Based on clinvar 240307 and gencode v44, GRCh38. Only consider "Pathogenic" and "Benign" SNVs.</td>
      <td>Classification</td>
      <td>https://www.ncbi.nlm.nih.gov/clinvar/</td>
    </tr>
    <tr>
      <td>Disease-related noncoding</td>
      <td>1</td>
      <td>111,202</td>
      <td>Based on clinvar 240307 and gencode v44, GRCh38. Only consider "Pathogenic" and "Benign" SNVs.</td>
      <td>Classification</td>
      <td>https://www.ncbi.nlm.nih.gov/clinvar/</td>
    </tr>
    <tr>
      <td>Transcription-related (eQTL)</td>
      <td>1</td>
      <td>108~5,480</td>
      <td>eQTLs and negative variants in 49 GTEx tissues. Reference genome version: GRCh38.</td>
      <td>Classification</td>
      <td>https://www.nature.com/articles/s41592-021-01252-x</td>
    </tr>
    <tr>
      <td>Transcription-related (MPRA)</td>
      <td>1</td>
      <td>21,000</td>
      <td>K562 and HepG2 cell lines, random select 210k (pos: neg = 1:20) from REVA benchmark dataset. Reference genome version: GRCh38.</td>
      <td>Classification</td>
      <td>https://academic.oup.com/gpb/article/19/4/590/7230396</td>
    </tr>
  </tbody>
</table>





## Extracting embeddings
An example for extracting embedding in TATA-promoter (proximal) prediction dataset.
```
sh run-get-embedding.sh ./dataset/promoter/prom_300_tata 300
```

## Downstream applications
### default applications:
```
data_dir="./dataset/promoter/prom_300_tata"
layer=-1 
random_seed=42
model_name=DNABERT-2
output_dim=1
regression=False 
python benchmark-default-tasks.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### Exon PSI applications
```
data_dir="./dataset/Exon_PSI"
layer=-1 
model_name=DNABERT-2
random_seed=42
output_dim=56
regression=True 
python benchmark-Exon_PSI.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### Gene expression level prediction applications
```
data_dir="./dataset/exp"
model_name=DNABERT-2
layer=-1
random_seed=42
output_dim=53
regression=True
python benchmark-exp.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### Gene expression level prediction applications
```
data_dir="./dataset/exp"
layer=-1
random_seed=42
output_dim=53
regression=True
python benchmark-exp.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### Promoter-promoter/Enhancer-promoter interaction applications
```
data_dir="./dataset/PPI_EPI/tB/P-P"
layer=-1
random_seed=42
output_dim=1
regression=False
python benchmark-PPI-PEI.py  $model_name ${data_dir}/${model_name} $layer  $random_seed $output_dim $regression
```

### TFBS/DNA accessibility/Histone modification
```

model_name=DNABERT-2
layer=-1
embedding_len=510
model_dim=768
dataset_dir=./dataset/TFBS
checkpoint_dir=your/path/to/save/checkpoints
eval_dir=your/path/to/save/evaluation/results
python TFBS_510-embedding-all_model.py $model_name $layer 510 ${dataset_dir}/${model_name} $checkpoint_dir $eval_dir
```

### Variant effect prediction
```
data_dir=./variant/var_disease_noncoding
model_name=DNABERT-2
layer=-1
python variant_effect_prediction-zero-shot.py $data_dir $model_name $layer
```
