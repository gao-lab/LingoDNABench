# this code is used to train a dna model with bert and random weight.
import argparse
import sys
sys.path.append('..')
import os
import torch
from torch import nn
from BERT.model import BaselineBERT
from BERT.utils import get_param_num, load_config


parser = argparse.ArgumentParser(description='Generate random weight for BERT-like gLM model.')
parser.add_argument('--config_file', type = str, default = "./config/Baseline_gLM.config.json")
parser.add_argument('--model_save_path', type = str, default = "./model/RandomWeight", help = 'Path to save the model with random weight.')
parser.add_argument('--random_seed', type = int, default = 666, help = 'Random seed, default is 666.')
args = parser.parse_args()


'''path'''
os.makedirs(args.model_save_path, exist_ok=True)
config = load_config(args.config_file)

'''model parameters'''
model_config = config["model_config"]
d_kv = model_config['d_kv']
n_heads = model_config['n_heads']
n_layers = model_config['n_layers']
max_vocab = model_config['max_vocab']

'''random seed for torch'''
seed = args.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# init model
print("Init model...")
device = torch.device("cuda")
model = BaselineBERT(max_vocab, n_heads, d_kv, n_layers)
print("Model parameters:", get_param_num(model))
model.to(device)

# random weight and save model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0,std=0.02)
    elif isinstance(m, nn.LayerNorm):
        pass
    elif hasattr(nn, 'RMSNorm') and isinstance(m, nn.RMSNorm):
        pass
    elif isinstance(m, nn.Conv1d):
        nn.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    else:
        pass

print("Random weight...")
model.apply(init_weights)

print("Save model...")
# save model
torch.save(
    {'model_state_dict': model.state_dict(),},
    os.path.join(args.model_save_path, "model_init.pt")
)

print("Model saved to", os.path.join(args.model_save_path, "model_init.pt"))