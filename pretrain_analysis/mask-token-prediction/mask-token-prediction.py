import warnings
import torch
from torch.utils.data import Dataset
import itertools
import random
from random import shuffle
import numpy as np
import h5py
import os
import sys
sys.path.append("../../pretrain_gLM")
from BERT import BaselineBERT
import copy
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
input_seq=sys.argv[1]
output_file=sys.argv[2]
model_type=sys.argv[3]

def enforce_determinism():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

enforce_determinism()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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


word2idx={}
idx2word={}
cls_token="[CLS]"
eos_token="[EOS]"
mask_token="[MASK]"
pad_token="[PAD]"
unk_token="[UNK]"
vocab = ['A', 'C', 'G', 'T', 'N']
special_tokens=[pad_token, cls_token, eos_token, mask_token, unk_token]
word2idx = {f'{name}': idx for idx, name in enumerate(special_tokens)}
kmer=1
k_mers=1
kmer_tokens = [''.join(p) for p in itertools.product(vocab, repeat=kmer)]
num_special_tokens=len(special_tokens)
word2idx.update({kmer: idx + num_special_tokens for idx, kmer in enumerate(kmer_tokens)})


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
if model_type=="human-genome":
    model_checkpoint = "../../pretrain_gLM/checkpoints/H-4K-13-K1/model_13_610919.pt"
if model_type=="RandomWeight":
    model_checkpoint = "../../pretrain_gLM/checkpoints/RandomWeight/model_init_666.pt"
if model_type=="multi-species":
    model_checkpoint = "../../pretrain_gLM/checkpoints/MS7-4K-8-K1/model_8_711981.pt"
if model_type=="RandomSeq":
    model_checkpoint = "../../pretrain_gLM/checkpoints/RandomSeq/model_0_43637.pt"

def load_model():
    # init model
    print("Initializing model and loading checkpoint...")
    model = BaselineBERT(max_vocab, n_heads, d_kv, n_layers, eval_mode=False)
    print("trans loading")
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

model=load_model()

'''
Dataset for DNA sequence
'''
class DNADataset(Dataset):
    def __init__(
        self, 
        sequence
        ):
        super().__init__()
        self.sequence=sequence
        self.max_length= 4096
        self.seq_len = len(sequence)
        self.ref_token = [word2idx[self.sequence[i:i+k_mers]] for i in range(0, len(self.sequence) - k_mers + 1)]
        self.ref_token = [word2idx['[CLS]']] + self.ref_token + [word2idx['[EOS]']]
        if len(self.ref_token)<self.max_length:
            self.ref_token=self.ref_token+[word2idx['[PAD]']]*(self.max_length-len(self.ref_token))
        
    def __len__(self):
        return self.seq_len
    def __getitem__(self, idx):
        # data type
        masked_pos=[idx+1]
        mask_seq=copy.copy(self.ref_token)
        ref_token_mask=[self.ref_token[idx+1]]
        mask_seq[idx+1]=word2idx['[MASK]']
        # to tensor
        return torch.tensor(mask_seq), torch.tensor(masked_pos), torch.tensor(ref_token_mask)

all_seq=np.loadtxt(input_seq,dtype=str)
all_result=[]
for seq_idx in range(len(all_seq)):
    dataset=DNADataset(all_seq[seq_idx])
    dataloader=DataLoader(dataset,batch_size=64,shuffle=False)
    current_result=[]
    for x in dataloader:
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            with torch.no_grad():
                logits=model(x[0].to(device),x[1].to(device))
                probabilities = torch.softmax(logits[:,0,:], dim=-1).to(torch.device('cpu'))
                probabilities=torch.gather(probabilities,1,x[2]).squeeze(1).numpy().tolist()
                current_result=current_result+probabilities
    all_result.append(current_result)


with open(output_file,'w') as f:
    for line in all_result:
        temp_str=list(map(str,line))
        temp_str='\t'.join(temp_str)
        f.write(f"{temp_str}\n")
