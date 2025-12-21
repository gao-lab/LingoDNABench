import os
import sys
from numpy.lib.format import open_memmap
input_seq=sys.argv[1]
output_dir=sys.argv[2]
embedding_len=int(sys.argv[3])
layer=int(sys.argv[4])
epoch=int(sys.argv[5])

name=input_seq.split('/')[-1].split('.')[0]
output_file=output_dir+'/'+name+f"-embedding-layer_{layer}.npy"
model_dir="/lustre/grp/gglab/liangyx/data/benchmark"

if (os.path.exists(output_file) and os.path.getsize(output_file) != 0):
    sys.exit()


from transformers import AutoTokenizer,AutoModel,AutoModelForMaskedLM,AutoModelForSequenceClassification
import torch
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import json
import sys
import h5py

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device=torch.device('cuda')
import time

def find_files_with_prefix(directory, prefix):
    if not os.path.exists(directory):
        raise ValueError(f"dir {directory} not exists")
    return [f for f in os.listdir(directory) if f.startswith(prefix) and os.path.isfile(os.path.join(directory, f))]

batch_size=4
import os
import sys
sys.path.append("./BERT-155M-Series")
import torch
from torch.utils.data import DataLoader
import numpy as np
from dpb_bert.data import  DNADataset, DNATokenizer
from dpb_bert.model import DNALingo
model_dir="./BERT-155M-Series/MS7-4K-8-K1"
model_version=find_files_with_prefix(model_dir,f"model_{epoch}_")[0]
model_checkpoint = f"{model_dir}/{model_version}"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
'''model parameters'''
d_kv = 64 # dimension of K(=Q), V
n_heads = 16 # number of heads in Multi-Head Attention
n_layers = 12 # number of Encoder of Encoder Layer
max_vocab = 16
d_model = n_heads * d_kv
'''tokenization'''
kmer = 1
'''training config'''

def load_model():
    # init model
    print("Initializing model and loading checkpoint...")

    model = DNALingo(max_vocab, n_heads, d_kv, n_layers, eval_mode=True)
    model.to(device)
    
    # loading checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # remove unused keys
    model_state = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state}
    model_state.update(state_dict)
    model.load_state_dict(model_state)
    
    return model
model = load_model()
print(model)
seq_path=input_seq
out_path=output_file
layer_num = -1
mean_mode = False
# get seq number
input_seq = np.loadtxt(seq_path, dtype=str, delimiter="\t")
seq_num = input_seq.shape[0]
print(seq_num)
# load data
dna_tokenizer = DNATokenizer(kmer = kmer)
# default: no padding
seq_dataset = DNADataset(dna_tokenizer, seq_path, data_type="seq")
seq_data_loader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# placeholder for output embedding
if mean_mode:
    output_embedding = np.empty((seq_num, d_model))
else:
    output_embedding = np.empty((seq_num, embedding_len, d_model))
    
print("Extracting embedding...")
model.eval().cuda()
# flash attention only support fp16 and bf16
with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
    with torch.no_grad():
        for x, input_ids in enumerate(seq_data_loader):
            embedding = model(input_ids[0].cuda())
            if mean_mode:
                # cal mean embedding, exclude CLS and SEP
                embedding_mean = torch.mean(embedding[layer_num][:,1:1+embedding_len,:], dim=1)
                output_embedding[x*batch_size : x*batch_size + embedding_mean.shape[0]] = embedding_mean.cpu().detach().numpy().astype(np.float32)
            else:
                # exclude CLS and SEP
                output_embedding[x*batch_size : x*batch_size + embedding[layer_num].shape[0]] = embedding[layer_num][:,1:1+embedding_len,:].cpu().detach().numpy().astype(np.float32)


np.save(output_file,output_embedding.astype(np.float32))
