import torch
from torch import nn
from BERT.modules import Embedding, EncoderLayer

# model
class BaselineBERT(nn.Module):
    def __init__(self, max_vocab, n_heads, d_kv, 
                n_layers, dropout_rate=0.0, bias=False, eval_mode=False
                ):
        super().__init__()
        # cal d_model
        self.d_model = int(n_heads * d_kv)
        # embedding
        self.embedding = Embedding(max_vocab, self.d_model)

        # encoder
        self.encoders = nn.ModuleList([EncoderLayer(n_heads, d_kv, self.d_model, n_layers, dropout_rate, bias) for _ in range(n_layers)])

        
        # activation
        self.gelu = nn.GELU()
        '''shared weights between linear layer'''
        self.linear = nn.Linear(self.d_model, self.d_model)
        '''shared weights between token embedding layer'''
        shared_embedding_weight = self.embedding.token_embedding.weight
        self.word_classifier = nn.Linear(self.d_model, max_vocab, bias=False)
        self.word_classifier.weight = shared_embedding_weight
        self.word_classifier_bias = nn.Parameter(torch.zeros(max_vocab))
        # normalization
        self.norm = nn.LayerNorm(self.d_model)
        # eval mode
        self.eval_mode = eval_mode
    
    def forward(self, tokens, masked_pos=None):
        '''
        tokens: [batch_size, seq_length]
        masked_pos: [batch_size, n_mask] 
        '''
        # check input
        if self.eval_mode == False and masked_pos is None:
            raise ValueError("masked_pos is required in training mode")
        # embedding
        embedding = self.embedding(tokens)
        # encoder; eval mode
        if self.eval_mode:
            embedding_out = []
            for encoder in self.encoders:
                embedding = encoder(embedding)
                embedding_out.append(embedding)
            return embedding_out
        # encoder; training mode
        else:
            for encoder in self.encoders:
                embedding = encoder(embedding)
            # output: [batch_size, seq_length, d_model]
            # masked language model task
            masked_pos = masked_pos.unsqueeze(-1).expand(-1, -1, embedding.size(-1))
            # get masked position from final output of transformer
            h_masked = torch.gather(embedding, 1, masked_pos)
            # linear
            h_masked = self.norm(self.gelu(self.linear(h_masked))) # [batch_size, n_pred, d_model]
            # output
            logits_lm = self.word_classifier(h_masked) + self.word_classifier_bias # [batch_size, n_pred, max_vocab]
            return logits_lm
