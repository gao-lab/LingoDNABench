import torch
from torch import nn
from BERT.attention import MHA_FlashAttention
import math


# Embedding
class Embedding(nn.Module):
    def __init__(self, max_vocab, d_model, dropout_rate=0.0):
        super(Embedding, self).__init__()
        #token embdding
        self.token_embedding = nn.Embedding(max_vocab, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        '''
        x: [batch_size, seq_length]
        '''
        # token embedding
        token_embedding = self.token_embedding(x)      
        return self.dropout(self.norm(token_embedding))

# FNN with swiGLU
class FeedForwardNetwork_swiGLU(nn.Module):
    def __init__(self, d_model, ibas=False, dropout_rate=0.1, bias=False):
        super(FeedForwardNetwork_swiGLU, self).__init__()
        self.dff = int((d_model * 4 * (2/3) + 256 - 1) // 256 * 256) # intermidiate size for FFN, in normal case, dff = d_model * 4
        self.gate = nn.Linear(d_model, self.dff, bias=bias)
        self.down = nn.Linear(self.dff, d_model, bias=bias)
        self.up = nn.Linear(d_model, self.dff, bias=bias)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        '''
        x: [batch_size, seq_length, d_model]
        '''
        x = self.down(self.silu(self.gate(x)) * self.up(x))
        
        return self.dropout(x)

# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_kv, d_model, layer_num, dropout_rate=0.0, bias=False):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MHA_FlashAttention(n_heads, d_kv, d_model)
        self.feed_forward_network = FeedForwardNetwork_swiGLU(d_model)
        
        # for deep norm, for encoder only, depreciated
        self.layer_num = layer_num

        # RMSNorm
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

           
    def forward(self, x, attn_mask = None):
        '''
        x: [batch_size, seq_length, d_model]
        attn_mask: [batch_size, seq_length, seq_length]
        pre-norm
        '''
        # multi-head attention
        residual = x
        #norm
        x = self.norm1(x)
        # mha
        x = self.multi_head_attention(x, x, x)
        # add
        x = x + residual
        
        # FFN
        residual = x
        # norm
        x = self.norm2(x)
        # ffn
        x = self.feed_forward_network(x)
        # add
        x = x + residual

        return x