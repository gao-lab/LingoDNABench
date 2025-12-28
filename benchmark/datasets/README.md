## Datasets Information

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Category</th>
      <th>Sequence length (bp)</th>
      <th>Number of data</th>
      <th>Output dim</th>
      <th>Description</th>
      <th>Classification/Regression</th>
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
      <td>ENCODE Dnase-seq and ATAC-seq data</td>
      <td>Classification</td>
      <td>https://www.encodeproject.org/</td>
    </tr>
    <tr>
      <td>DNA methylation</td>
      <td>41</td>
      <td>5mC: 4,688; 6mA: 36,670</td>
      <td>1</td>
      <td>Detection of 5-methylcytosine (5mC) and N6-methyladenosine (6mA) modifications in DNA sequences</td>
      <td>Classification</td>
      <td>https://www.biorxiv.org/content/10.1101/2024.08.16.608288v1</td>
    </tr>
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
      <td>300/70</td>
      <td>TATA: 6,130; non-TATA: 53,066; all: 59,196</td>
      <td>1</td>
      <td>-249~+50bp / -34~+35bp of TSS, Including TATA promoter and non-TATA promoter</td>
      <td>Classification</td>
      <td>https://arxiv.org/abs/2306.15006</td>
    </tr>
    <tr>
      <td>Enhancer</td>
      <td>200</td>
      <td>Human: 12,639; Mouse: 13,378</td>
      <td>1</td>
      <td>Data was downloaded from Vista Enhancer Browser (241106). Positive and negative enhancers were split to 200 bp bins. CD-hit was used to remove seqs with high similarity (>0.8). Tissue and stage: fb_e11.5 (forebrain, positive) and e11.5 (negative)</td>
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
      <td>https://github.com/xy-chen16/DeepSilencer, https://academic.oup.com/nar/article/49/D1/D221/5921294?login=true#221747952</td>
    </tr>
    <tr>
      <td>CRE activity</td>
      <td>230</td>
      <td>HepG2: 245,852; K562: 393,328</td>
      <td>1</td>
      <td>CRE: cis regulatory element, here is potential promoter and enhancer. Regulatory activity measured by lentiMPRA in HepG2 and K562</td>
      <td>Regression</td>
      <td>https://www.nature.com/articles/s41586-024-08430-9</td>
    </tr>
    <tr>
      <td>Promoter-promoter interaction</td>
      <td>1000+1000</td>
      <td>tB: 162,391; Mon: 136,164; FoeT: 143,347; tCD4: 155,821; nCD4: 166,806; tCD8: 154,059</td>
      <td>1</td>
      <td>Promoter-promoter interaction. Six cell lines: total B cells (tB), monocytes (Mon), foetal thymus (FoeT), total CD4+ T cells (tCD4), naive CD4+ T cells (nCD4), and total CD8+ T cells (tCD8)</td>
      <td>Classification</td>
      <td>https://academic.oup.com/nar/article/47/10/e60/5380496?login=true</td>
    </tr>
    <tr>
      <td>Enhancer-promoter interaction</td>
      <td>2000+1000</td>
      <td>tB: 176,096; Mon: 162,104; FoeT: 133,263; tCD4: 161,364; nCD4: 174,269; tCD8: 162,990</td>
      <td>1</td>
      <td>Enhancer-promoter interaction. Six cell lines: total B cells (tB), monocytes (Mon), foetal thymus (FoeT), total CD4+ T cells (tCD4), naive CD4+ T cells (nCD4), and total CD8+ T cells (tCD8)</td>
      <td>Classification</td>
      <td>https://academic.oup.com/nar/article/47/10/e60/5380496?login=true</td>
    </tr>
    <tr>
      <td rowspan="5">Post-transcription regulation</td>
      <td>Polyadenylation signals</td>
      <td>600</td>
      <td>AATAAA: 22,602; all: 41,864</td>
      <td>1</td>
      <td>Polyadenylation signals, with AATAAA variant and all variant</td>
      <td>Classification</td>
      <td>https://academic.oup.com/bioinformatics/article/38/17/4053/6633307?login=true</td>
    </tr>
    <tr>
      <td>Translation initiation sites</td>
      <td>600</td>
      <td>56,488</td>
      <td>1</td>
      <td>Translation initiation sites</td>
      <td>Classification</td>
      <td>https://academic.oup.com/bioinformatics/article/38/17/4053/6633307?login=true</td>
    </tr>
    <tr>
      <td>Splice site</td>
      <td>400</td>
      <td>Acceptor: 22,154; Donor: 21,945</td>
      <td>1</td>
      <td>Two datasets: acceptor and negative; donor and negative. Balanced dataset, error-free. Negative seqs are from randomly selected exon regions, intron regions and GT or AG dinucleotides</td>
      <td>Classification</td>
      <td>https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-04471-3</td>
    </tr>
    <tr>
      <td>Exon PIS</td>
      <td>400+400</td>
      <td>40,737</td>
      <td>56</td>
      <td>Percent spliced-in (PSI): skipping level of on exon in alternative splicing. Input sequence: 300 + 100 (acceptor) and 100 + 300 (donor); Number of target tissues: 56. Please use custom loss function provided in the data dictionary for downstream model</td>
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
      <td>53 GTEx tissues, mRNA and lncRNA transcription prediction based on promoter sequence (TSS -1k~+1k)</td>
      <td>Regression</td>
      <td>https://www.nature.com/articles/s41592-021-01252-x</td>
    </tr>
    <tr>
      <td rowspan="4">Variant effect prediction</td>
      <td>Disease-related coding</td>
      <td>1</td>
      <td>122,237</td>
      <td>1</td>
      <td>Based on clinvar 240307 and gencode v44, GRCh38. Only consider "Pathogenic" and "Benign" SNVs</td>
      <td>Classification</td>
      <td>https://www.ncbi.nlm.nih.gov/clinvar/</td>
    </tr>
    <tr>
      <td>Disease-related noncoding</td>
      <td>1</td>
      <td>111,202</td>
      <td>1</td>
      <td>Based on clinvar 240307 and gencode v44, GRCh38. Only consider "Pathogenic" and "Benign" SNVs</td>
      <td>Classification</td>
      <td>https://www.ncbi.nlm.nih.gov/clinvar/</td>
    </tr>
    <tr>
      <td>Transcription-related (eQTL)</td>
      <td>1</td>
      <td>108~5,480</td>
      <td>1</td>
      <td>eQTLs and negative variants in 49 GTEx tissues. Reference genome version: GRCh38</td>
      <td>Classification</td>
      <td>https://www.nature.com/articles/s41592-021-01252-x</td>
    </tr>
    <tr>
      <td>Transcription-related (MPRA)</td>
      <td>1</td>
      <td>21,000</td>
      <td>1</td>
      <td>K562 and HepG2 cell lines, random select 210k (pos: neg = 1:20) from REVA benchmark dataset. Reference genome version: GRCh38</td>
      <td>Classification</td>
      <td>https://academic.oup.com/gpb/article/19/4/590/7230396</td>
    </tr>
  </tbody>
</table>


## Data Precessing
### Default setting
The file `train/dev/test_data_0.txt` contains DNA sequences. In the gLM benchmark, these need to be converted into gLM embeddings before performing adapter tuning. The file `train/dev/test_label.txt` contains the corresponding labels, which are used for model training.

### Expression level prediction
The file `train/dev/test_data_0.txt` contains DNA sequences. In the gLM benchmark, these need to be converted into gLM embeddings before performing adapter tuning. The file `train/dev/test_target.npy` contains the corresponding labels (each record corresponds to 53 continuous values), which are used for model training.


### Exon PSI prediction
The files  `train/dev/test_data_0.txt` and `train/dev/test_data_1.txt` contain DNA sequences. This is a dual-sequence input task. In the gLM benchmark, these need to be converted into gLM embeddings before performing adapter tuning. The file `train/dev/test_target.npy` contains the corresponding labels (each record corresponds to 56 continuous values), which are used for model training.


### PPI-EPI(Promoter-promoter/Enhancer-promoter interaction) prediction
The files `train/dev/test_data_0.txt` and `train/dev/test_data_1.txt` contain DNA sequences. However, due to the large volume of data, it can be split into multiple files and then read as a streaming dataset during training. (The `../scripts/benchmark-PPI_EPI.py` script accepts .h5 files and is configured to split the data into smaller files, each containing 5000 records.)

### TFBS/DNA accessibility/Histone modification
Given the large size of the dataset, the embeddings are converted into the TFRecord format for training. This process involves splitting the data into appropriately sized chunks(25k sequence per file), storing them as TFRecords, and subsequently training them through a TensorFlow pipeline.

```
# sh ./script/01_process_basset_data.sh feature_name input_dir output_dir blacklist_file reference_genome
sh ./script/01_process_basset_data.sh TFBS ./TFBS ./TFBS ../data/hg38-blacklist.v2.bed ../data/hg38-genome.fa
# python ./script/02_process_basset_data.py feature_name input_dir output_dir output_dim split_size
python ./script/02_process_basset_data.py TFBS ./TFBS ./TFBS 3572 25000

# Extract embedding and write it into tfrecords
model_name=nucleotide-transformer-2.5b-multi-species
model_type
layer=-1
input_dir=./datasets/TFBS
feature_name=TFBS
mkdir ${input_dir}/${model_name}
output_dir=${input_dir}/${model_name}
for idx in {0..57}
do
python ../get-embeddings.py ../models/$model_name $model_type $model_name ${input_dir}/${feature_name}_train_{idx}.seq $output_dir $((embedding_len / 6 + 1 )) $layer 
python 03_embed2tfr.py $idx train TFBS $model_name $layer $input_dir
done
for idx in {0..11}
do
python ../get-embeddings.py ../models/$model_name $model_type $model_name ${input_dir}/${feature_name}_train_{idx}.seq $output_dir $((embedding_len / 6 + 1 )) $layer 
python 03_embed2tfr.py $idx train TFBS $model_name $layer $input_dir
done
for idx in {0..14}
do
python ../get-embeddings.py ../models/$model_name $model_type $model_name ${input_dir}/${feature_name}_train_{idx}.seq $output_dir $((embedding_len / 6 + 1 )) $layer 
python 03_embed2tfr.py $idx train TFBS $model_name $layer $input_dir
done
```

### Variant effect prediction
The files `train/dev/test.vcf` are VCF files for mutation sites. The files `train/dev/test_data_0.txt` and `train/dev/test_data_1.txt` contain DNA sequences (both files contain sequences centered on the mutation site, totaling 512 bp. `data_0` contains the pre-mutation sequence, and `data_1` contains the post-mutation sequence). The file `train/dev/test_label.txt` contains the corresponding labels. For zero-shot evaluation, the training, development, and test datasets can be merged directly. The distinction provided here enables potential attempts at adapter tuning.

