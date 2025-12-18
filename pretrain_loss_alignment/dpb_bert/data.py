import warnings
import torch
from torch.utils.data import Dataset
import itertools
import random
from random import shuffle
import numpy as np
import h5py
import os
from Bio.Seq import Seq
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

"""
Tokenizer DNA sequence into k-mers
"""
class DNATokenizer:
    DNA_vocab = ['A', 'C', 'G', 'T', 'N']
    def __init__(
        self, 
        vocab = None, 
        kmer: int = 3, 
        overlap_token: bool = True,
        cls_token: str = "CLS", 
        eos_token: str = "EOS", 
        mask_token: str = "MASK", 
        pad_token: str = "PAD", 
        unk_token: str = "UNK",
        ):
        super().__init__()
        if vocab is None:
            vocab = self.DNA_vocab
        self.vocab = vocab
        self.special_tokens = [pad_token, cls_token, eos_token, mask_token, unk_token]
        self.kmer = kmer
        self.overlap_token = overlap_token

        self.token2id = {f'[{name}]': idx for idx, name in enumerate(self.special_tokens)}
        self.num_special_tokens = len(self.special_tokens)
        
        kmer_tokens = [''.join(p) for p in itertools.product(vocab, repeat=kmer)]
        self.token2id.update({kmer: idx + self.num_special_tokens for idx, kmer in enumerate(kmer_tokens)})
        self.vocab_size = len(self.token2id)

        self.id2token = {idx: token for token, idx in self.token2id.items()}
    
    def reverse_complement(self, seq: str):
        return str(Seq(seq).reverse_complement()).upper()

    def __call__(self, seq: str):
        tokens = [] 
        if self.overlap_token:
            for i in range(0, len(seq) - self.kmer + 1):
                if seq[i:i+self.kmer] not in self.token2id:
                    tokens.append(self.token2id['[UNK]'])
                else:
                    tokens.append(self.token2id[seq[i:i+self.kmer]])
        else:
            for i in range(0, len(seq), self.kmer):
                if len(seq[i:i+self.kmer]) < self.kmer:
                    break
                if seq[i:i+self.kmer] not in self.token2id:
                    tokens.append(self.token2id['[UNK]'])
                else:
                    tokens.append(self.token2id[seq[i:i+self.kmer]])
        tokens = [self.token2id['[CLS]']] + tokens + [self.token2id['[EOS]']]
        return tokens

'''
Dataset for DNA sequence
'''
class DNADataset(Dataset):
    def __init__(
        self, 
        tokenizer: DNATokenizer,
        data_path, 
        data_type: str = "h5",
        data_key_h5: str = "seq", 
        padding: bool = False, 
        max_length: int = 4096,
        eval_mode: bool = False
        ):
        super().__init__()

        self.tokenizer = tokenizer

        self.data_path = data_path
        self.data_type = data_type
        self.data_key_h5 = data_key_h5
        
        if self.data_type == "h5":         
            with h5py.File(data_path, "r") as f:
                self.length = len(f[self.data_key_h5])
        else:
            self.input_seqs = np.loadtxt(data_path, dtype=str, delimiter="\t")
            self.length = self.input_seqs.shape[0]
        
        self.padding = padding
        self.max_length = max_length
        self.eval_mode = eval_mode

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # data type
        if self.data_type == "h5":
            with h5py.File(self.data_path, "r") as f:
                input_seq = f[self.data_key_h5][idx].decode()
        else:
            input_seq = self.input_seqs[idx]
        
        # 50% ratio to get reverse complementary sequence
        if not self.eval_mode:
            if random.random() < 0.5:
                input_seq = self.tokenizer.reverse_complement(input_seq)

        # check input length
        available_seq_max_length = self.max_length - 2 + (self.tokenizer.kmer - 1) # including [CLS] and [EOS]
        if len(input_seq) > available_seq_max_length:
            input_seq = input_seq[:available_seq_max_length]
        #print(input_seq)
        
        # tokenize
        input_ids = self.tokenizer(input_seq)

        # padding
        if self.padding:
            input_ids = input_ids + [self.tokenizer.token2id['[PAD]']] * (self.max_length - len(input_ids))
        
        input_length = len(input_ids) # including [CLS] and [EOS]
        
        # to tensor
        return torch.tensor(input_ids), input_length
            
'''
Customized DataCollator function for DNADataset
'''
class DataCollatorForDNA:
    def __init__(
        self, 
        tokenizer: DNATokenizer,
        p_mask: float = 0.8, 
        p_replace: float = 0.1, 
        mlm_prob: float = 0.15,
        continue_mask: bool = True,
        max_length: int = None, 
        dynamic_length: bool = False,
        dynamic_length_prob: float = 0.1,
        dynamic_min_length: int = 50,
        ):
        super().__init__()
        self.tokenizer = tokenizer
        self.p_mask = p_mask
        self.p_replace = p_replace
        self.mlm_prob = mlm_prob
        self.continue_mask = continue_mask
        self.max_length = max_length
        self.dynamic_length = dynamic_length
        self.dynamic_length_prob = dynamic_length_prob
        self.dynamic_min_length = dynamic_min_length


    def mask_fn(self, cand_pos, input_ids):
        masked_pos, masked_tokens = [], []
        if self.continue_mask:
            for pos in cand_pos:
                for i in range(self.tokenizer.kmer):
                    pos_tmp = pos + i
                    '''
                    # avoid mask [EOS]
                    if input_ids[pos_tmp] == self.tokenizer.token2id['[EOS]']:
                        continue
                    # avoid out of range
                    if pos_tmp > len(input_ids):
                        break
                    '''
                    masked_pos.append(pos_tmp)
                    masked_tokens.append(input_ids[pos_tmp])
                    # maks
                    if random.random() < self.p_mask:
                        input_ids[pos_tmp] = self.tokenizer.token2id['[MASK]']
                    elif random.random() > self.p_mask + self.p_replace:
                        input_ids[pos_tmp] = random.randint(4, self.tokenizer.vocab_size-1)
                    else:
                        pass # do noting with 1-p_mask-p_replace probability
        else:
            for pos in cand_pos:
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos])
                if random.random() < self.p_mask:
                    input_ids[pos] = self.tokenizer.token2id['[MASK]']
                elif random.random() > self.p_mask + self.p_replace:
                    input_ids[pos] = random.randint(4, self.tokenizer.vocab_size-1)
                else:
                    pass # do noting with 1-p_mask-p_replace probability
        return input_ids, masked_pos, masked_tokens

    def mask_tokens(self, padded_input_ids: torch.Tensor):
        #random select 15%/mask_len(k-mers) tokens keeping with a min distance of mask_len(k-mers)
        if self.continue_mask:
            mlm_mask_num = int((self.mlm_prob * len(padded_input_ids))/self.tokenizer.kmer)
            cand_mask_pos = [self.tokenizer.kmer*idx + x for idx, x in enumerate(sorted(random.sample(range(1, len(padded_input_ids)- (self.tokenizer.kmer - 1) - self.tokenizer.kmer * mlm_mask_num), mlm_mask_num)))]
            return self.mask_fn(cand_mask_pos, padded_input_ids)
        else:
            cand_pos = [i for i, t in enumerate(padded_input_ids) if t != self.tokenizer.token2id['[CLS]'] and t != self.tokenizer.token2id['[EOS]']]
            shuffle(cand_pos)
            mlm_mask_num = int(self.mlm_prob * len(padded_input_ids))
        return self.mask_fn(cand_pos[0:mlm_mask_num], padded_input_ids)
    
    def __call__(self, batch_data: torch.Tensor):
        # sbatch_data: List
        seqs = []
        lengths = []
        for sample in batch_data:
            seqs.append(sample[0].tolist())
            lengths.append(sample[1])
        
        if self.max_length is None:
            max_length = max(lengths)
        else:
            max_length = self.max_length

        if self.dynamic_length:
            if random.random() < self.dynamic_length_prob:
                crop_length = random.randint(self.dynamic_min_length, max_length)
            else:
                crop_length = max_length
        else:
            crop_length = max_length
        #print("crop_length:", crop_length)
        # crop, padding and mask
        batch_input_ids = []
        batch_masked_pos = []
        batch_masked_tokens = []

        for i, seq in enumerate(seqs):
            # crop and padding
            '''Following codes should be modified if input length is less than crop_length'''          
            if len(seq) > crop_length:
                # ranodm crop from the middle
                start_pos = random.randint(0, len(seq) - crop_length) + 1 # avoid [CLS]
                seq = [self.tokenizer.token2id['[CLS]']] + seq[start_pos:start_pos+crop_length-2] + [self.tokenizer.token2id['[EOS]']]
            elif len(seq) < crop_length:
                seq = seq + [self.tokenizer.token2id['[PAD]']] * (crop_length - len(seq))
            else:
                pass
            #print("#", seq_onehot.size())
            # mask
            input_ids, masked_pos, masked_tokens = self.mask_tokens(seq)        

            batch_input_ids.append(torch.tensor(input_ids))
            batch_masked_pos.append(torch.tensor(masked_pos))
            batch_masked_tokens.append(torch.tensor(masked_tokens))
        return torch.stack(batch_input_ids), torch.stack(batch_masked_pos), torch.stack(batch_masked_tokens)
