import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from dpb_bert.data import  DNADataset, DNATokenizer
from dpb_bert.model import DNALingo



model_checkpoint = "./MS7-4K-8-K1/model_8_711981.pt"
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
batch_size = 128

target_file_path = "./test.txt"
output_file_path = "./test.npy"
seq_length = 200


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

def extract_embedding(model, seq_path, out_path, layer_num = -1, mean_mode = False):
    # get seq number
    input_seq = np.loadtxt(seq_path, dtype=str, delimiter="\t")
    seq_num = input_seq.shape[0]
    
    # load data
    dna_tokenizer = DNATokenizer(kmer = kmer)
    # default: no padding
    seq_dataset = DNADataset(dna_tokenizer, seq_path, data_type="seq", eval_mode=True)
    seq_data_loader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # placeholder for output embedding
    if mean_mode:
        output_embedding = np.empty((seq_num, d_model))
    else:
        output_embedding = np.empty((seq_num, seq_length, d_model))
    # predict
    print("Extracting embedding...")
    model.eval().cuda()
    # flash attention only support fp16 and bf16
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
        with torch.no_grad():
            for x, input_ids in enumerate(seq_data_loader):
                embedding = model(input_ids[0].cuda())
                if mean_mode:
                    # cal mean embedding, exclude CLS and SEP
                    embedding_mean = torch.mean(embedding[layer_num][:,1:-1,:], dim=1)
                    output_embedding[x*batch_size : x*batch_size + embedding_mean.shape[0]] = embedding_mean.cpu().detach().numpy()
                else:
                    # exclude CLS and SEP
                    output_embedding[x*batch_size : x*batch_size + embedding[layer_num].shape[0]] = embedding[layer_num][:,1:-1,:].cpu().detach().numpy()
    # save
    np.save(out_path, output_embedding)
    print("Saved embedding to: " + out_path)

if __name__ == "__main__":
    model = load_model()
    extract_embedding(model, target_file_path, output_file_path)