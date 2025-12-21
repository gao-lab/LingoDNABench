# this code is used to train a dna model with bert and random weight.
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from torch import nn
from dpb_bert.model import DNALingo
from dpb_bert.utils import get_param_num


'''path'''
model_save_path = "./model_test2_bert_4k_random_weight_k1"
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

'''model parameters'''
d_kv = 64 # dimension of K(=Q), V
n_heads = 16 # number of heads in Multi-Head Attention
n_layers = 12 # number of Encoder of Encoder Layer

max_vocab = 16
kmer = 1


'''random seed for torch'''
seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# init model
print("Init model...")
device = torch.device("cuda")
model = DNALingo(max_vocab, n_heads, d_kv, n_layers)
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

# check if model weight is random
'''
for name, param in model.named_parameters():
    if "weight" in name:
        print(name, param.data)
'''
print("Save model...")
# save model
torch.save(
    {'model_state_dict': model.state_dict(),},
    model_save_path + "/model_init.pt"
)

print("Model saved to", model_save_path + "/model_init.pt")