## Pretrain gLMs with genome sequences and random sequences
### Prepare training data
Download multi-species reference genomes from GenBank.
```bash
sh download_genomes.sh
```

Process data and construct human and multi-species training dataset.
```bash
sh prepare_genome_dataset.sh
```

### Train gLMs
Train gLM with human and multi-species training dataset.
```bash
# get help
python BaselineBERT.py -h

# train human gLM with 1 GPU
torchrun --nnodes 1 --nproc_per_node 1 BaselineBERT.py \
--train_data ./data/dataset/human.4096.h5 \
--model_save_path ./model/human_gLM

# train multi-species gLM with 4 GPUs
torchrun --nnodes 1 --nproc_per_node 4 BaselineBERT.py \
--train_data ./data/dataset/ms.4096.h5 \
--model_save_path ./model/ms_gLM
```

### RandomWeight model
RandomWeight model uses random weights to initialize the model and save <b>without any training</b>.
```bash 
python RandomWeight.py \
--model_save_path ./model/RandomWeight
```
### Cehckpoints
checkpoints are available through
```
wget -r -np -nH --cut-dirs=3 -R "index.html*"  \
  "http://ftp.cbi.pku.edu.cn/pub/LingoDNABench/BERT-155M-Series/"
```

The download link above is currently experiencing some accessibility issues. As a temporary alternative, the dataset can be downloaded from our FTP server:
```
ftp://ftp.gao-lab.org
Username: lingodnabench_download
Password: lingodnabench_download
```

We recommend using lftp for command-line downloading:
```
lftp -u lingodnabench_download,lingodnabench_download ftp://ftp.gao-lab.org \
  -e "mirror --continue --parallel=8 LingoDNABench/BERT-155M-Series ./BERT-155M-Series; quit"
```
The `--continue` option allows interrupted downloads to be resumed, while `--parallel=8` enables parallel downloading.

Alternatively, the FTP server can be accessed using an FTP client such as FileZilla.

> Note: This FTP server is provided as a temporary download mirror. The primary download link above will remain the preferred source once its accessibility issue is resolved.

