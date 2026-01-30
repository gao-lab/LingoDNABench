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

Construct random sequence dataset.
```bash
sh prepare_random_dataset.sh
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

