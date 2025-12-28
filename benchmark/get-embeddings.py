

# ===============================
# Global imports & config
# ===============================

import os
import sys
import json
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)

# ===============================
# CLI arguments
# ===============================
model_dir = sys.argv[1]
model_type = sys.argv[2]
model_name = sys.argv[3]
input_seq = sys.argv[4]
output_dir = sys.argv[5]
embedding_len = int(sys.argv[6])
layer = int(sys.argv[7])

name = os.path.basename(input_seq).split(".")[0]
output_file = os.path.join(
    output_dir, f"{name}-embedding-layer_{layer}.npy"
)


batch_size = 4

if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    sys.exit(0)

# ===============================
# Dataset definitions
# ===============================

class Caduceus_GFM_Dataset(Dataset):
    def __init__(self, sequence_path, model_name, embedding_len):
        from tokenization_caduceus import CaduceusTokenizer
        self.seqs = np.loadtxt(sequence_path, dtype=str, delimiter="\t")
        self.embedding_len = embedding_len
        self.tokenizer = CaduceusTokenizer.from_pretrained(
            f"{model_dir}/{model_name}",
            padding="max_length",
            max_length=embedding_len + 1,
        )

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        out = self.tokenizer(
            self.seqs[idx],
            return_tensors="pt",
            padding="max_length",
            max_length=self.embedding_len + 1,
        )
        return out["input_ids"].squeeze(0)


class HyenaDNA_GFM_Dataset(Dataset):
    def __init__(self, sequence_path, model_name, embedding_len):
        from tokenization_hyena import HyenaDNATokenizer
        self.seqs = np.loadtxt(sequence_path, dtype=str, delimiter="\t")
        self.embedding_len = embedding_len
        self.tokenizer = HyenaDNATokenizer.from_pretrained(
            f"{model_dir}/{model_name}",
            padding="max_length",
            max_length=embedding_len + 1,
        )

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        out = self.tokenizer(
            self.seqs[idx],
            return_tensors="pt",
            padding="max_length",
            max_length=self.embedding_len + 1,
        )
        return {k: v.squeeze(0) for k, v in out.items()}


class OmniNA_GFM_Dataset(Dataset):
    def __init__(self, sequence_path, model_name, embedding_len):
        self.seqs = np.loadtxt(sequence_path, dtype=str, delimiter="\t")
        self.embedding_len = embedding_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            f"{model_dir}/{model_name}"
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        out = self.tokenizer(
            self.seqs[idx],
            return_tensors="pt",
            padding="max_length",
            max_length=self.embedding_len,
        )
        return {k: v.squeeze(0)[-self.embedding_len :] for k, v in out.items()}


class NT_GFM_Dataset(Dataset):
    def __init__(self, sequence_path, tokenizer, embedding_len):
        self.seqs = np.loadtxt(sequence_path, dtype=str, delimiter="\t")
        self.embedding_len = embedding_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        out = self.tokenizer(
            self.seqs[idx],
            return_tensors="pt",
            padding="max_length",
            max_length=self.embedding_len + 1,
        )
        return {
            "input_ids": out["input_ids"].squeeze(0)[: self.embedding_len + 1],
            "attention_mask": out["attention_mask"].squeeze(0)[
                : self.embedding_len + 1
            ],
        }


# ===============================
# Common embedding runner
# ===============================

def run_dataloader_embedding(dataloader, extract_fn):
    total = len(dataloader.dataset)
    test_flag = True
    cnt = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
            else:
                batch = batch.to(device)

            temp = extract_fn(batch).cpu().detach().numpy()
            num = temp.shape[0]

            if test_flag:
                result = np.zeros(
                    (total, temp.shape[1], temp.shape[2]), dtype=np.float32
                )
                test_flag = False

            result[cnt : cnt + num] = temp
            cnt += num

    return result


# ===============================
# Model runners
# ===============================

def run_caduceus():
    sys.path.append(f"{model_dir}/{model_name}")
    model = AutoModelForMaskedLM.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    ).to(device).eval()

    dataset = Caduceus_GFM_Dataset(input_seq, model_name, embedding_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    def extract(batch):
        hs = model(batch, output_hidden_states=True)["hidden_states"]
        return hs[layer][:, :-1, :]

    return run_dataloader_embedding(loader, extract)


def run_hyenadna():
    model = AutoModelForSequenceClassification.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    ).to(device).eval()

    dataset = HyenaDNA_GFM_Dataset(input_seq, model_name, embedding_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    def extract(batch):
        hs = model(**batch, output_hidden_states=True)["hidden_states"]
        return hs[layer][:, :-1, :]

    return run_dataloader_embedding(loader, extract)


def run_nt():
    tokenizer = AutoTokenizer.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    )
    model = AutoModelForMaskedLM.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    ).to(device).eval()

    dataset = NT_GFM_Dataset(input_seq, tokenizer, embedding_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    def extract(batch):
        hs = model(**batch, output_hidden_states=True)["hidden_states"]
        return hs[layer][:, 1 : 1 + embedding_len, :]

    return run_dataloader_embedding(loader, extract)


def run_omnina():
    model = AutoModel.from_pretrained(
        f"{model_dir}/{model_name}", trust_remote_code=True
    ).to(device).eval()

    dataset = OmniNA_GFM_Dataset(input_seq, model_name, embedding_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    def extract(batch):
        hs = model(**batch, output_hidden_states=True)["hidden_states"]
        return hs[layer]

    return run_dataloader_embedding(loader, extract)


def run_bert_series():
    sys.path.append('../pretrain_gLM')
    from BERT.data import  DNADataset, DNATokenizer
    from BERT.model import BaselineBERT
    from BERT.utils import load_config
    config = load_config("../pretrain_gLM/config/Baseline_gLM.config.json")
    if model_name == "RandomWeight":
        model_checkpoint = "../pretrain_gLM/checkpoints/RandomWeight/model_init_666.pt"
    elif model_name == "RandomSeq":
        model_checkpoint = "../pretrain_gLM/checkpoints/RandomSeq/model_0_43637.pt"
    elif model_name == "MS7-4K-8-K1":
        model_checkpoint = "../pretrain_gLM/checkpoints/MS7-4K-8-K1/model_8_711981.pt"
    elif model_name == "H-4K-13-K1":
        model_checkpoint = "../pretrain_gLM/checkpoints/H-4K-13-K1/model_13_610919.pt"
    else:
        raise ValueError(model_name)
    model_config = config["model_config"]
    d_kv = model_config['d_kv']
    n_heads = model_config['n_heads']
    n_layers = model_config['n_layers']
    max_vocab = model_config['max_vocab']
    d_model = n_heads * d_kv
    kmer = model_config['kmer']

    def load_model():
        # init model
        print("Initializing model and loading checkpoint...")

        model = BaselineBERT(max_vocab, n_heads, d_kv, n_layers, eval_mode=True)
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
    
    

    dna_tokenizer = DNATokenizer(kmer = kmer)
    # default: no padding
    seqs = np.loadtxt(input_seq, dtype=str)
    seq_dataset = DNADataset(dna_tokenizer, input_seq, data_type="seq", eval_mode=True)
    seq_data_loader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    model = load_model()
    model.eval().cuda()
    
    result = np.zeros((len(seqs), embedding_len, d_model), dtype=np.float32)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            for i, batch in enumerate(seq_data_loader):
                emb = model(batch[0].cuda())[layer][:, 1:1 + embedding_len, :]
                result[
                    i * batch_size : i * batch_size + emb.shape[0]
                ] = emb.cpu().numpy()

    return result

def run_dnabert():
    tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/{model_name}")
    model = AutoModel.from_pretrained(f"{model_dir}/{model_name}")
    model.to(device).eval()

    k_mers = int(model_name.split("_")[-1])
    seqs = np.loadtxt(input_seq, dtype=str)

    result = np.zeros((len(seqs), embedding_len, model.config.hidden_size))

    def tokenize(seq):
        kmers = " ".join(
            seq[i:i + k_mers] for i in range(len(seq) - k_mers + 1)
        )
        return tokenizer(
            kmers,
            return_tensors="pt",
            padding="max_length",
            max_length=embedding_len,
        )

    with torch.no_grad():
        for i, seq in enumerate(seqs):
            x = tokenize(seq)
            x = {k: v.to(device) for k, v in x.items()}
            hs = model(**x, output_hidden_states=True)["hidden_states"]
            result[i] = hs[layer][0, 1:-1, :].cpu().numpy()

    return result

def run_deepgene():
    sys.path.append(f"{model_dir}/DeepGene-main/PanGeneGraphTrans")
    from modeling_roformer import RoFormerForMaskedLM
    import tokenizers

    param_file = f"{model_dir}/DeepGene-main/model/pretrain_params_epoch_20"
    model = RoFormerForMaskedLM.from_pretrained(param_file).to(device).eval()

    # 原始 graph dataset 构造逻辑
    from numpy.lib.format import open_memmap

    def load_dataset():
        from tokenizers import Tokenizer
        graphs = []
        tokenizer = Tokenizer.from_file(
            f"{model_dir}/DeepGene-main/data/vocab/tokenizer.json"
        )
        with open(input_seq) as f:
            for line in f:
                if line.startswith("sequence"):
                    continue
                ids = tokenizer.encode(line.strip().split(",")[0]).ids
                ids = torch.tensor(ids[: embedding_len], dtype=torch.long)
                graphs.append({
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                })
        return graphs

    graphs = load_dataset()
    result = np.zeros((len(graphs), embedding_len, model.config.hidden_size))

    with torch.no_grad():
        for i, g in enumerate(graphs):
            g = {k: v.unsqueeze(0).to(device) for k, v in g.items()}
            hs = model(**g, output_hidden_states=True)["hidden_states"]
            result[i] = hs[layer][0, :embedding_len, :].cpu().numpy()

    return result

def run_dnabert2():
    from transformers import BertConfig

    model_path = f"{model_dir}/DNABERT-2-117M"

    config = BertConfig.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        model_max_length=embedding_len,
    )

    seqs = np.loadtxt(input_seq, dtype=str)
    hidden_size = model.config.hidden_size

    
    result = np.zeros(
        (len(seqs), embedding_len, hidden_size),
        dtype=np.float32,
    )

    with torch.no_grad():
        for i, seq in enumerate(seqs):
            x = tokenizer(seq, return_tensors="pt")
            x = {k: v.to(device) for k, v in x.items()}

            encoded_layers = model(
                **x,
                output_all_encoded_layers=True
            )

            # encoded_layers[layer]: [seq_len+2, hidden]
            seq_len = x["input_ids"].shape[1] - 2

            temp = (
                encoded_layers[layer][1:-1, :]
                .detach()
                .cpu()
                .numpy()
            )

            result[i, 0:seq_len, :] = temp

    return result

def run_generator():
    from transformers import AutoTokenizer, AutoModel

    model_path = f"/lustre/grp/gglab/liangyx/data/benchmark/{model_name}"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
    ).to(device).eval()

    seqs = np.loadtxt(input_seq, dtype=str)
    hidden_size = model.config.hidden_size
    token_len = embedding_len

    result = np.zeros(
        (len(seqs), token_len, hidden_size),
        dtype=np.float32,
    )

    tokenizer.padding_side = "right"

    with torch.inference_mode():
        for i, seq in enumerate(seqs):
            inputs = tokenizer(
                [seq],
                add_special_tokens=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.config.max_position_embeddings,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            
            result[i, :, :] = (
                hidden_states[layer][0, 1:1 + token_len, :]
                .detach()
                .cpu()
                .numpy()
            )

    return result


def run_evo2():
    from evo2 import Evo2
    model_path = f"{model_dir}/{model_name}"
    model = Evo2('evo2_7b',local_path=f"{model_path}.pt")
    layer_name=layer# 7b
    model=model.cuda()
    model=model.eval()
    seqs = np.loadtxt(input_seq, dtype=str)
    hidden_size = 4096
    token_len = embedding_len

    result = np.zeros(
        (len(seqs), token_len, hidden_size),
        dtype=np.float32,
    )

    with torch.no_grad():
        for i, seq in enumerate(seqs):
            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq),
                dtype=torch.int,
            ).unsqueeze(0).to('cuda:0')

            outputs,embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
            
            result[i, :, :] = (
                embeddings[layer_name][0, 1:1 + token_len, :]
                .detach()
                .cpu()
                .numpy()
            )

    return result


def run_lucaone_hf():
    import numpy as np
    import torch
    from lucagplm import LucaGPLMModel, LucaGPLMTokenizer

    model_name_hf = f"{model_dir}/LucaOne-default-step36M"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model & tokenizer (HF official)
    model = LucaGPLMModel.from_pretrained(
        model_name_hf,
    ).to(device).eval()

    tokenizer = LucaGPLMTokenizer.from_pretrained(
        model_name_hf
    )

    # Load sequences
    seqs = np.loadtxt(input_seq, dtype=str)
    hidden_size = model.config.hidden_size

    # Fixed-length output (benchmark aligned)
    result = np.zeros(
        (len(seqs), embedding_len, hidden_size),
        dtype=np.float32,
    )

    with torch.no_grad():
        for i, seq in enumerate(seqs):
            inputs = tokenizer(
                seq,
                seq_type="gene",
                return_tensors="pt",
                truncation=True,
                max_length=embedding_len + 2,  # CLS + EOS
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            # last_hidden_state: [1, L+2, H]
            token_emb = outputs.last_hidden_state[:, 1:-1, :]

            L = min(token_emb.shape[1], embedding_len)
            result[i, :L, :] = token_emb[:, :L, :].cpu().numpy()

    return result

# ===============================
# Dispatcher
# ===============================

RUNNERS = {
    "bert-series": run_bert_series,
    "dnabert": run_dnabert,
    "dnabert2": run_dnabert2,
    "caduceus": run_caduceus,
    "hyenadna": run_hyenadna,
    "nt": run_nt,
    "omnina": run_omnina,
    "deepgene": run_deepgene,
    "lucaone": run_lucaone_hf,
    "GEN": run_generator,
    "evo2": run_evo2,
}

if model_type not in RUNNERS:
    raise ValueError(f"Unsupported model_type: {model_type}")

result = RUNNERS[model_type]()
np.save(output_file, result.astype(np.float32))
