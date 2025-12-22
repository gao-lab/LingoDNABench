# This script extracts embeddings from a pre-trained BERT model for DNA sequences.
import argparse
import os
import sys
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
import numpy as np
from BERT.data import  DNADataset, DNATokenizer
from BERT.model import BaselineBERT
from BERT.utils import load_config


parser = argparse.ArgumentParser(description='Extract embeddings from a pre-trained BERT-like gLM model.')
parser.add_argument('--config_file', type = str, default = "./config/Baseline_gLM.config.json")
parser.add_argument('--model_checkpoint', type = str, default = None, help = 'Path to the pre-trained model checkpoint.')
parser.add_argument('--batch_size', type = int, default = 128, help = 'Batch size for data loading.')
parser.add_argument('--target_file', type = str, default = "./test/test.txt", help = 'Path to the input file containing sequences.')
parser.add_argument('--output_file', type = str, default = "./test/test.npy", help = 'Path to save the extracted embeddings.')
parser.add_argument('--seq_length', type = int, default = None, help = 'Length of the input sequences.')
parser.add_argument('--layer_num', type = int, default = -1, help = 'Layer number from which to extract embeddings. Default is the last layer.')
parser.add_argument('--mean_mode', type = str, default = "false", help = 'Whether to take the mean of the embeddings across all layers. Default is False.') 
args = parser.parse_args()

# device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

config = load_config(args.config_file)

'''model parameters'''
model_config = config["model_config"]
d_kv = model_config['d_kv']
n_heads = model_config['n_heads']
n_layers = model_config['n_layers']
max_vocab = model_config['max_vocab']
d_model = n_heads * d_kv

'''tokenization'''
kmer = model_config['kmer']

'''training config'''
batch_size = args.batch_size

'''Input file and output file'''
target_file_path = args.target_file
output_file_path = args.output_file
if args.seq_length is not None:
    seq_length = args.seq_length
else:
    # load first line to get seq length
    with open(target_file_path, "r") as f:
        first_line = f.readline()
        seq_length = len(first_line) - 1
print("Sequence length:", seq_length)


def load_model():
    # init model
    print("Initializing model and loading checkpoint...")

    model = BaselineBERT(max_vocab, n_heads, d_kv, n_layers, eval_mode=True)
    model.to(device)
    
    # loading checkpoint
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # remove unused keys
    model_state = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_state}
    model_state.update(state_dict)
    model.load_state_dict(model_state)
    
    return model

def extract_embedding(model, seq_path, out_path, layer_num, mean_mode):
    # get seq number
    input_seq = np.loadtxt(seq_path, dtype=str, delimiter="\t")
    seq_num = input_seq.shape[0]
    
    # load data
    dna_tokenizer = DNATokenizer(kmer = kmer)
    # default: no padding
    seq_dataset = DNADataset(dna_tokenizer, seq_path, data_type="seq", eval_mode=True)
    seq_data_loader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    # placeholder for output embedding
    if mean_mode == "true":
        output_embedding = np.empty((seq_num, d_model))
    elif mean_mode == "false":
        output_embedding = np.empty((seq_num, seq_length, d_model))
    else:
        raise ValueError("Invalid mean_mode argument. Please choose 'true' or 'false'.")
    
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
    extract_embedding(model, args.target_file, args.output_file, args.layer_num, args.mean_mode)