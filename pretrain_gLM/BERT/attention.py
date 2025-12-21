import torch
from torch import nn
import numpy as np
from flash_attn import flash_attn_qkvpacked_func
from BERT.rope import RotaryEmbedding


#Flash attention
class MHA_FlashAttention(nn.Module):
    def __init__(self, n_heads, d_kv, d_model, dropout_rate=0.1, bias=False):
        super(MHA_FlashAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_kv
        self.d_v = d_kv
        # W_q, W_k, W_v, W_o
        self.W_Q = nn.Linear(d_model, n_heads*self.d_k, bias=bias)
        self.W_K = nn.Linear(d_model, n_heads*self.d_k, bias=bias)
        self.W_V = nn.Linear(d_model, n_heads*self.d_v, bias=bias)
        self.W_O = nn.Linear(n_heads*self.d_v, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    @staticmethod
    def RoPE_pos_emb(d_k, Qs, Ks):
        # Qs, Ks: [batch_size, seq_length, n_heads, d_k]
        rotary_emb = RotaryEmbedding(emb_dim=d_k, device='cuda', use_FA_triton_kernel=True)
        Qs, Ks, *_ = rotary_emb(Qs, Ks, start_pos=0)
        return Qs, Ks

    def forward(self, Q, K, V):
        '''
        Q, K, V: [batch_size, seq_length, d_model]
        '''
        batch_size = Q.size(0)

        '''
        split Q, K, V into n_heads: [batch_size, seq_length, n_heads, d_k] 
        '''
        Q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k) # [batch_size, seq_length, n_heads, d_k]
        K_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k) # [batch_size, seq_length, n_heads, d_k]
        V_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v) # [batch_size, seq_length, n_heads, d_v]
        
        '''RoPE position embedding''' # fixed in the forward pass, convert freqs to t.device
        Q_s, K_s = Q_s.to(torch.float32), K_s.to(torch.float32) # convert to fp32 for FA kernel
        Q_s, K_s = self.RoPE_pos_emb(self.d_k, Q_s, K_s)
        Q_s, K_s = Q_s.to(torch.bfloat16), K_s.to(torch.bfloat16) # convert to bf16
        
        '''packed Q, K, V'''
        QKV_packed = torch.stack([Q_s, K_s, V_s], dim=2) # [batch_size, seq_length, 3, n_heads, d_k/d_v]
        # Flash attention context [batch, seq_length, n_head, d_k/d_v]
        context = flash_attn_qkvpacked_func(QKV_packed, dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1))
        context = context.view(batch_size, -1, self.n_heads*self.d_v) # [batch_size, seq_length, n_heads*d_v]
        #output
        output = self.W_O(context) # [batch_size, seq_length, d_model]
        output = self.dropout(output)
        
        return output
