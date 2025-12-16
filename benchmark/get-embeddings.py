
from transformers import AutoTokenizer,AutoModel,AutoModelForMaskedLM,AutoModelForSequenceClassification
import torch
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import json
import sys
import h5py
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device=torch.device('cuda')
import time
from numpy.lib.format import open_memmap
import sys
torch.manual_seed(42)
np.random.seed(42)

model_type=sys.argv[1]
model_name=sys.argv[2]
input_seq=sys.argv[3]
output_dir=sys.argv[4]
embedding_len=int(sys.argv[5])
layer=int(sys.argv[6])

name=input_seq.split('/')[-1].split('.')[0]
output_file=output_dir+'/'+name+f"-embedding-layer_{layer}.npy"
model_dir="./models"
batch_size=4
if os.path.exists(output_file) and  os.path.getsize(output_file) != 0:
    sys.exit()

class Caduceus_GFM_Dataset(Dataset):
    def __init__(self, sequence_path,model_name,embedding_len):
        super(Caduceus_GFM_Dataset, self).__init__()
        self.DNA_input_seq = np.loadtxt(sequence_path, dtype=str, delimiter="\t")
        # self.DNA_input_seq = np.load(DNA_data_path,allow_pickle=True)
        self.model_name=model_name
        self.embedding_len=embedding_len
        self.num_DNA=self.DNA_input_seq.shape[0]
        self.pre_tokenizer=CaduceusTokenizer.from_pretrained(f"{model_dir}/{model_name}",padding="max_length",max_length=self.embedding_len+1)
    def tokenize(self, seq):
        token_result=self.pre_tokenizer(seq, return_tensors = 'pt',padding="max_length",max_length=self.embedding_len+1)
        input_ids=token_result["input_ids"].squeeze(0)
        return input_ids
    def __len__(self):
        return int(self.num_DNA)
    def __getitem__(self, idx):
        DNA_input_ids = self.tokenize(self.DNA_input_seq[idx])
        return DNA_input_ids

class HyenaDNA_GFM_Dataset(Dataset):
    def __init__(self, sequence_path,model_name,embedding_len):
        super(HyenaDNA_GFM_Dataset, self).__init__()
        self.DNA_input_seq = np.loadtxt(sequence_path, dtype=str, delimiter="\t")
        # self.DNA_input_seq = np.load(DNA_data_path,allow_pickle=True)
        self.model_name=model_name
        self.embedding_len=embedding_len
        self.num_DNA=self.DNA_input_seq.shape[0]
        self.pre_tokenizer=HyenaDNATokenizer.from_pretrained(f"{model_dir}/{model_name}",padding='max_length',max_length=self.embedding_len+1)
    def tokenize(self, seq):
        token_result=self.pre_tokenizer(seq, return_tensors = 'pt',padding='max_length',max_length=self.embedding_len+1)
        new_dict={key:token_result[key].squeeze(0) for key in token_result}
        # input_ids=token_result["input_ids"].squeeze(0)
        return new_dict
    def __len__(self):
        return int(self.num_DNA)
    def __getitem__(self, idx):
        DNA_input_ids = self.tokenize(self.DNA_input_seq[idx])
        return DNA_input_ids

class OmniNA_GFM_Dataset(Dataset):
    def __init__(self, sequence_path,model_name,embedding_len):
        super(OmniNA_GFM_Dataset, self).__init__()
        self.DNA_input_seq = np.loadtxt(sequence_path, dtype=str, delimiter="\t")
        self.model_name=model_name
        self.embedding_len=embedding_len
        self.num_DNA=self.DNA_input_seq.shape[0]
        self.pre_tokenizer=AutoTokenizer.from_pretrained(f"{model_dir}/{model_name}")
        self.pre_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    def tokenize(self, seq):
        token_result=self.pre_tokenizer(seq, return_tensors = 'pt',padding="max_length",max_length=self.embedding_len)
        new_dict={key:token_result[key].squeeze(0)[-embedding_len:] for key in token_result}
        return new_dict
    def __len__(self):
        return int(self.num_DNA)
    def __getitem__(self, idx):
        DNA_input_ids = self.tokenize(self.DNA_input_seq[idx])
        return DNA_input_ids

class GROVER_GFM_Dataset(Dataset):
    def __init__(self, sequence_path,model_name,embedding_len):
        super(GROVER_GFM_Dataset, self).__init__()
        self.DNA_input_seq = np.loadtxt(sequence_path, dtype=str, delimiter="\t")
        # self.DNA_input_seq = np.load(DNA_data_path,allow_pickle=True)
        self.model_name=model_name
        self.embedding_len=embedding_len
        self.num_DNA=self.DNA_input_seq.shape[0]
        self.pre_tokenizer=AutoTokenizer.from_pretrained(f"{model_dir}/{model_name}")
    def tokenize(self, seq):
        token_result=self.pre_tokenizer(seq, return_tensors = 'pt',padding="max_length",max_length=self.embedding_len+1)
        new_dict={key:token_result[key].squeeze(0) for key in token_result}
        # input_ids=token_result["input_ids"].squeeze(0)
        return new_dict
    def __len__(self):
        return int(self.num_DNA)
    def __getitem__(self, idx):
        DNA_input_ids = self.tokenize(self.DNA_input_seq[idx])
        return DNA_input_ids

class GraphDataset(Dataset):
    def __init__(self):
        """
            graph = {'input_ids': input_ids,
                     'attention_mask':attention_mask
                     'pos_ids':pos_ids}
        """
        self.graphs = []
        # self.labels = []
    def __getitem__(self, index):
        # return self.graphs[index], self.labels[index]
        return self.graphs[index]
    def __len__(self):
        return len(self.graphs)

def load_finetune_dataset(file_dir,embedding_len):
    dataset = GraphDataset()
    tokenizer = tokenizers.Tokenizer.from_file("/lustre/grp/gglab/liangyx/tools/DeepGene-main/data/vocab/tokenizer.json")
    seqs = []
    seq_max_len = embedding_len
    max_position_embeddings = 5120
    with open(file_dir) as f:
        for line in f:
            line = line.split(',')
            if line[0] == 'sequence':
                continue
            seq = torch.tensor(tokenizer.encode(line[0], add_special_tokens=False).ids, dtype=torch.long)
            if seq.shape[0] + 2 >= max_position_embeddings:
                seq = seq[0:max_position_embeddings-2]
            seqs.append(seq)
            seq_max_len = max(seq_max_len, seq.shape[0])
            # dataset.labels.append(torch.tensor(int(line[1]), dtype=torch.long))
    seq_max_len += 2
    for x in seqs:
        num_nodes = x.shape[0]
        cls_id = 1
        sep_id = 2
        pad_id = 3
        input_ids = pad_id * torch.ones(seq_max_len, dtype=torch.long)
        input_ids[0] = cls_id
        input_ids[1] = sep_id
        input_ids[2:2 + num_nodes] = x
        attention_mask = torch.zeros(seq_max_len, dtype=torch.bool)
        attention_mask[:2 + num_nodes] = True
        pos_ids = (seq_max_len - 1) * torch.ones(seq_max_len, dtype=torch.long)  # dep[[PAD]] = seq_max_len-1
        pos_ids[0] = 0  # dep[[CLS]] = 0
        pos_ids[1] = 1  # dep[[SEP]] = 1
        for i in range(2, 2 + num_nodes):
            pos_ids[i] = i
        graph = {'input_ids': input_ids.cpu(),
                 'attention_mask': attention_mask.cpu(),
                 'pos_ids': pos_ids.cpu()}
        dataset.graphs.append(graph)
    # print(dataset.graphs)
    return dataset



def get_model(model_name):
    special_tokens = (['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] +
                      ["+", '-', '*', '/', '=', "&", "|", "!"] +
                      ['M', 'B'] + ['P'] + ['R', 'I', 'K', 'L', 'O', 'Q', 'S', 'U', 'V'] + ['W', 'Y', 'X', 'Z'])
    if model_name in ('dna_gpt0.1b_h',):
        print("non-dynamic")
        tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=False)
    else:
        tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=True)
    vocab_size = len(tokenizer)
    model = DNAGPT.from_name(model_name, vocab_size)
    return model, tokenizer




def load_model(model, weight_path, device=None, dtype=None):
    state = torch.load(weight_path, map_location="cpu")
    if 'model' in state.keys():
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    print(f"loading model weights from {weight_path}")
    model.to(device=device, dtype=dtype)
    model = model.eval()
    return model

def load_model_lucaone(log_filepath,model_dirpath):
    with open(log_filepath, "r") as rfp:
        for line_idx, line in enumerate(rfp):
            if line_idx == 0:
                try:
                    args_info = json.loads(line.strip(), encoding="UTF-8")
                except Exception as e:
                    args_info = json.loads(line.strip())
                break

    print("Model dirpath: %s" % model_dirpath)
    # create tokenizer
    tokenizer_dir = os.path.join(model_dirpath, "tokenizer")
    tokenizer = AlphabetV2_0.from_predefined("gene_prot")
    config_class, model_class = LucaGPLMConfigV2_0, LucaGPLMV2_0
    model_config: PretrainedConfig = config_class.from_json_file(os.path.join(model_dirpath, "config.json"))
    # load the pretrained model or create the model
    print("Load pretrained model: %s" % model_dirpath)
    embedding_inference=True
    args = Args()
    args.pretrain_tasks = args_info["pretrain_tasks"]
    args.ignore_index = args_info["ignore_index"]
    args.label_size = args_info["label_size"]
    args.loss_type = args_info["loss_type"]
    args.output_mode = args_info["output_mode"]
    args.max_length = args_info["max_length"]
    args.classifier_size = args_info["classifier_size"]
    args.pretrained_model_name = None
    args.embedding_inference = embedding_inference

    model = model_class(model_config, args=args)
    pretrained_net_dict = torch.load(os.path.join(model_dirpath, "pytorch.pth"),
                                        map_location=torch.device("cpu"))
    model_state_dict_keys = set()
    for key in model.state_dict():
        model_state_dict_keys.add(key)
    new_state_dict = OrderedDict()
    for k, v in pretrained_net_dict.items():
        if k.startswith("module."):
            name = k[7:]
        else:
            name = k
        if name in model_state_dict_keys:
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model,tokenizer,args,args_info,model_config


def encoder(args_info, model_config, seq, seq_type, tokenizer):
    if args_info["tokenization"]:
        # seq to seq ids
        encoding = tokenizer.encode_plus(
            text=seq,
            text_pair=None,
            add_special_tokens=args_info["add_special_tokens"],
            padding="max_length",
            max_length=model_config.max_position_embeddings,
            return_attention_mask=True,
            return_token_type_ids=not model_config.no_token_type_embeddings,
            return_length=False,
            truncation=True
        )
        processed_seq_len = sum(encoding["attention_mask"])
    elif args_info["model_type"] in ["lucaone_gplm", "lucaone", "lucagplm"]:
        seqs = [seq]
        seq_types = [seq_type]
        seq_encoded_list = [tokenizer.encode(seq)]
        if "max_length" in args_info and args_info["max_length"] and args_info["max_length"] > 0:
            seq_encoded_list = [encoded[:args_info["max_length"]] for encoded in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        processed_seq_len = max_len + int(tokenizer.prepend_bos) + int(tokenizer.append_eos)
        input_ids = torch.empty(
            (
                1,
                processed_seq_len,
            ),
            dtype=torch.int64,
        )
        input_ids.fill_(tokenizer.padding_idx)
        position_ids = None
        if not model_config.no_position_embeddings:
            position_ids = torch.empty(
                (
                    1,
                    processed_seq_len,
                ),
                dtype=torch.int64,
            )
            position_ids.fill_(tokenizer.padding_idx)
        token_type_ids = None
        if not model_config.no_token_type_embeddings:
            token_type_ids = torch.empty(
                (
                    1,
                    processed_seq_len,
                ),
                dtype=torch.int64,
            )
            token_type_ids.fill_(tokenizer.padding_idx)
        for i, (seq_type, seq_str, seq_encoded) in enumerate(
                zip(seq_types, seqs, seq_encoded_list)
        ):
            if tokenizer.prepend_bos:
                input_ids[i, 0] = tokenizer.cls_idx
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            input_ids[i, int(tokenizer.prepend_bos): len(seq_encoded) + int(tokenizer.prepend_bos)] = seq
            if tokenizer.append_eos:
                input_ids[i, len(seq_encoded) + int(tokenizer.prepend_bos)] = tokenizer.eos_idx
            if not model_config.no_position_embeddings:
                cur_len = int(tokenizer.prepend_bos) + len(seq_encoded) + int(tokenizer.append_eos)
                for idx in range(0, cur_len):
                    position_ids[i, idx] = idx
            if not model_config.no_token_type_embeddings:
                if seq_type == "gene":
                    type_value = 0
                else:
                    type_value = 1
                cur_len = int(tokenizer.prepend_bos) + len(seq_encoded) + int(tokenizer.append_eos)
                for idx in range(0, cur_len):
                    token_type_ids[i, idx] = type_value
        encoding = {"input_ids": input_ids, "token_type_ids": token_type_ids, "position_ids": position_ids}
    else:
        max_length = model_config.max_position_embeddings
        if args_info["add_special_tokens"]:
            max_length = max_length - 2
        if len(seq) > max_length:
            if args_info["truncation"] == "right":
                seq = seq[:max_length]
            elif args_info["truncation"] == "left":
                seq = seq[-max_length:]
            processed_seq_len = max_length + 2 if args_info["add_special_tokens"] else max_length
        else:
            processed_seq_len = len(seq) + 2 if args_info["add_special_tokens"] else len(seq)
        seq = " ".join(list(seq))
        encoding = tokenizer.encode_plus(
            text=seq,
            text_pair=None,
            add_special_tokens=args_info["add_special_tokens"],
            padding="max_length",
            max_length=model_config.max_position_embeddings,
            return_attention_mask=True,
            return_token_type_ids=not model_config.no_token_type_embeddings,
            return_length=False,
            truncation=True
        )
    if seq_type == "prot":
        new_encoding = {}
        for item in encoding.items():
            new_encoding[item[0] + "_b"] = item[1]
        encoding = new_encoding
    return encoding, processed_seq_len


class DNABERT_GFM_Dataset(Dataset):
    def __init__(self, DNA_data_path, k_mers,pre_tokenizer,embedding_len):
        super(DNABERT_GFM_Dataset, self).__init__()
        self.DNA_input_seq = np.loadtxt(DNA_data_path, dtype=str, delimiter="\t")
        self.embedding_len=embedding_len
        self.num_DNA=self.DNA_input_seq.shape[0]
        self.k_mers=k_mers
        self.pre_tokenizer=pre_tokenizer
    def tokenize(self, seq):
        token = [seq[i:i+self.k_mers] for i in range(0, len(seq) - self.k_mers + 1)]
        token=" ".join(token)
        token_result=self.pre_tokenizer(token, return_tensors = 'pt',padding='max_length',max_length=self.embedding_len)
        input_ids=token_result["input_ids"].squeeze(0)
        token_type_ids=token_result['token_type_ids'].squeeze(0)
        attn_mask=token_result["attention_mask"].squeeze(0)
        return {"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attn_mask}
    def __len__(self):
        return int(self.num_DNA)
    def __getitem__(self, idx):
        tkn=self.tokenize(self.DNA_input_seq[idx])
        return tkn

class DNABERT2_GFM_Dataset(Dataset):
    def __init__(self, DNA_data_path,pre_tokenizer):
        super(DNABERT2_GFM_Dataset, self).__init__()
        self.DNA_input_seq = np.loadtxt(DNA_data_path, dtype=str, delimiter="\t")
        # self.DNA_input_seq = np.load(DNA_data_path,allow_pickle=True)
        
        self.num_DNA=self.DNA_input_seq.shape[0]
        self.pre_tokenizer=pre_tokenizer
    def tokenize(self, seq):
        
        token_result=self.pre_tokenizer(seq,padding="max_length", return_tensors = 'pt')
        input_ids=token_result["input_ids"].squeeze(0)
        token_type_ids=token_result['token_type_ids'].squeeze(0)
        attn_mask=token_result["attention_mask"].squeeze(0)
        return {"input_ids":input_ids,"token_type_ids":token_type_ids,"attention_mask":attn_mask}
    def __len__(self):
        return int(self.num_DNA)
    def __getitem__(self, idx):
        tkn=self.tokenize(self.DNA_input_seq[idx])
        return tkn


class NT_GFM_Dataset(Dataset):
    def __init__(self, DNA_data_path, pre_tokenizer,embedding_len):
        super(NT_GFM_Dataset, self).__init__()
        self.DNA_input_seq = np.loadtxt(DNA_data_path, dtype=str, delimiter="\t")
        self.num_DNA=self.DNA_input_seq.shape[0]
        self.pre_tokenizer=pre_tokenizer
        self.embedding_len=embedding_len
    def tokenize(self, seq):
        token_result=self.pre_tokenizer(seq, return_tensors = 'pt',padding='max_length',max_length=self.embedding_len+1)
        input_ids=token_result["input_ids"].squeeze(0)[0:self.embedding_len+1]
        attn_mask=token_result["attention_mask"].squeeze(0)[0:self.embedding_len+1]
        return {"input_ids":input_ids,"attention_mask":attn_mask}
    def __len__(self):
        return int(self.num_DNA)
    def __getitem__(self, idx):
        DNA_input_ids = self.tokenize(self.DNA_input_seq[idx])
        return DNA_input_ids




if model_type=="dnabert2":
    from transformers import AutoModel, BertConfig
    config = BertConfig.from_pretrained(f"{model_dir}/DNABERT-2-117M")
    model=AutoModel.from_pretrained(f"{model_dir}/DNABERT-2-117M",config=config, trust_remote_code=True)
    model=model.to(device)
    model=model.eval()
    tokenizer=AutoTokenizer.from_pretrained(f"{model_dir}/DNABERT-2-117M", trust_remote_code=True,model_max_length=embedding_len)
    dnabert2_dataset=DNABERT2_GFM_Dataset(DNA_data_path=input_seq,pre_tokenizer=tokenizer)
    dnabert2_dataloader = DataLoader(dnabert2_dataset, batch_size=batch_size)
    def get_embedding(sequence_set,tokenizer,model,embedding_len,layer=-1):
        result=np.zeros((len(sequence_set),embedding_len,768))
        with torch.no_grad():
            for i,seq in enumerate(sequence_set):
                x = tokenizer(seq,return_tensors='pt')
                x={key:value.to(device) for key,value in x.items()}
                seq_len=x['input_ids'].shape[1]-2
                temp=model(**x,output_all_encoded_layers=True)[layer][1:-1,:].cpu().detach().numpy()
                result[i,0:seq_len,:] = temp
        return result
    result=get_embedding(np.loadtxt(input_seq,dtype=str),tokenizer,model,embedding_len,layer)
if model_type=="nt":
    model=AutoModelForMaskedLM.from_pretrained(f"{model_dir}/{model_name}",trust_remote_code=True)
    model=model.to(device)
    model=model.eval()
    tokenizer=AutoTokenizer.from_pretrained(f"{model_dir}/{model_name}",trust_remote_code=True)
    nt_dataset=NT_GFM_Dataset(DNA_data_path=input_seq,pre_tokenizer=tokenizer,embedding_len=embedding_len)
    nt_dataloader = DataLoader(nt_dataset, batch_size=batch_size)
    def get_embedding(dataloader,model,embedding_len,layer=-1):
        test_flag=True
        cnt=0
        with torch.no_grad():
            for batch in dataloader:
                batch={key:value.to(device) for key,value in batch.items()}
                if test_flag:
                    temp=model(**batch,output_hidden_states=True)['hidden_states'][layer][:,1:1+embedding_len,:]
                    temp=temp.cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result=np.zeros((len(dataloader.dataset),temp.shape[1],temp.shape[2]))
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
                    test_flag=False
                else:
                    temp=model(**batch,output_hidden_states=True)['hidden_states'][layer][:,1:1+embedding_len,:]
                    temp=temp.cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num

        return embedding_result
    result=get_embedding(nt_dataloader,model,embedding_len,layer)
 
if model_type=="caduceus":
    def has_nan(arr):
        return np.isnan(arr).any()
    sys.path.append(f"{model_dir}/{model_name}")
    from tokenization_caduceus import CaduceusTokenizer
    model = AutoModelForMaskedLM.from_pretrained(f"{model_dir}/{model_name}",trust_remote_code=True)
    model=model.to(device)
    model=model.eval()
    caduceus_dataset=Caduceus_GFM_Dataset(input_seq,model_name,embedding_len=embedding_len)
    caduceus_dataloader=DataLoader(caduceus_dataset,batch_size=batch_size)
    def get_embedding(dataloader,model,layer=-1):
        test_flag=True
        cnt=0
        with torch.no_grad():
            for batch in dataloader:
                if test_flag:
                    temp=model(batch.to(device),output_hidden_states=True)['hidden_states']
                    print(f"layer_num={len(temp)}")
                    temp=temp[layer][:,0:-1,:]
                    temp=temp.cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result=np.zeros((len(dataloader.dataset),temp.shape[1],temp.shape[2]))
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
                    test_flag=False
                    print(embedding_result[0])
                    if has_nan(embedding_result[0]):
                        sys.exit()
                    
                else:
                    temp=model(batch.to(device),output_hidden_states=True)['hidden_states'][layer][:,0:-1,:]
                    temp=temp.cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
        return embedding_result
    result=get_embedding(caduceus_dataloader,model,layer)

if model_type=="hyenadna":
    sys.path.append(model_dir+'/'+model_name)
    from tokenization_hyena import HyenaDNATokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir+'/'+model_name,trust_remote_code=True)
    model.to(device)
    model=model.eval()
    hyenaDNA_dataset=HyenaDNA_GFM_Dataset(input_seq,model_name,embedding_len)
    hyenaDNA_dataloader=DataLoader(hyenaDNA_dataset,batch_size=batch_size)
    def get_embedding(dataloader,model,layer=-1):
        test_flag=True
        cnt=0
        with torch.no_grad():
            for batch in dataloader:
                batch={key:value.to(device) for key,value in batch.items()}
                if test_flag:
                    start=time.time()
                    embedding_result=model(**batch,output_hidden_states=True)['hidden_states'][layer][:,0:-1,:]
                    temp=embedding_result.cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result=np.zeros((len(dataloader.dataset),temp.shape[1],temp.shape[2]))
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
                    end=time.time()
                    print(end-start)
                    test_flag=False
                else:
                    start=time.time()
                    temp=model(**batch,output_hidden_states=True)['hidden_states'][layer][:,0:-1,:].cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
                    end=time.time()
                    print(end-start)
        return embedding_result
    result=get_embedding(hyenaDNA_dataloader,model,layer)
if model_type=='omnina':
    model = AutoModel.from_pretrained(model_dir+'/'+model_name,trust_remote_code=True)
    model.to(device)
    model=model.eval()
    omnina_dataset=OmniNA_GFM_Dataset(input_seq,model_name,embedding_len)
    omnina_dataloader=DataLoader(omnina_dataset,batch_size=batch_size)
    def get_embedding(dataloader,model,layer=-1):
        test_flag=True
        cnt=0
        with torch.no_grad():
            for batch in dataloader:
                batch={key:value.to(device) for key,value in batch.items()}
                if test_flag:
                    start=time.time()
                    embedding_result=model(**batch,output_hidden_states=True)['hidden_states'][layer][:,:,:]
                    temp=embedding_result.cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result=np.zeros((len(dataloader.dataset),temp.shape[1],temp.shape[2]))
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
                    end=time.time()
                    print(end-start)
                    test_flag=False
                else:
                    start=time.time()
                    temp=model(**batch,output_hidden_states=True)['hidden_states'][layer][:,:,:].cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
                    end=time.time()
                    print(end-start)
        return embedding_result
    result=get_embedding(omnina_dataloader,model,layer)
if model_type=='grover':
    model = AutoModelForMaskedLM.from_pretrained(model_dir+'/'+model_name,trust_remote_code=True)
    model.to(device)
    model=model.eval()
    grover_dataset=GROVER_GFM_Dataset(input_seq,model_name,embedding_len)
    grover_dataloader=DataLoader(grover_dataset,batch_size=batch_size)
    def get_embedding(dataloader,model,layer=-1):
        test_flag=True
        cnt=0
        with torch.no_grad():
            for batch in dataloader:
                batch={key:value.to(device) for key,value in batch.items()}
                if test_flag:
                    start=time.time()
                    embedding_result=model(**batch,output_hidden_states=True)['hidden_states'][layer][:,1:,:]
                    temp=embedding_result.cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result=np.zeros((len(dataloader.dataset),temp.shape[1],temp.shape[2]))
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
                    end=time.time()
                    print(end-start)
                    test_flag=False
                else:
                    start=time.time()
                    temp=model(**batch,output_hidden_states=True)['hidden_states'][layer][:,1:,:].cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
                    end=time.time()
                    print(end-start)
        return embedding_result
    result=get_embedding(grover_dataloader,model,layer)

if model_type=='deepgene':
    sys.path.append("/lustre/grp/gglab/liangyx/tools/DeepGene-main/PanGeneGraphTrans")
    from modeling_roformer import *
    import tokenizers
    param_file="/lustre/grp/gglab/liangyx/tools/DeepGene-main/model/pretrain_params_epoch_20"
    model = RoFormerForMaskedLM.from_pretrained(param_file)
    deepgene_dataset=load_finetune_dataset(input_seq,embedding_len=embedding_len)
    deepgene_dataloader = DataLoader(deepgene_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model=model.to(device)
    model.eval()
    def get_embedding(dataloader,model,layer=-1):
        test_flag=True
        cnt=0
        with torch.no_grad():
            for batch in dataloader:
                batch={key:value.to(device) for key,value in batch.items()}
                if test_flag:
                    # print(batch['input_ids'].shape)
                    start=time.time()
                    embedding_result=model(**batch,output_hidden_states=True)['hidden_states']
                    print(f"layer_num={len(embedding_result)}")
                    embedding_result=embedding_result[layer][:,2:,:]
                    temp=embedding_result.cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result=np.zeros((len(dataloader.dataset),temp.shape[1],temp.shape[2]))
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
                    end=time.time()
                    print(end-start)
                    test_flag=False
                else:
                    start=time.time()
                    temp=model(**batch,output_hidden_states=True)['hidden_states'][layer][:,2:,:].cpu().detach().numpy()
                    num=temp.shape[0]
                    embedding_result[cnt:cnt+num,:,:]=temp
                    cnt+=num
                    end=time.time()
                    # print(end-start)
        return embedding_result
    result=get_embedding(deepgene_dataloader,model,layer)

if model_type=='lucaone':
    from transformers import AutoTokenizer, PretrainedConfig, BertTokenizer
    from collections import OrderedDict
    sys.path.append("/lustre/grp/gglab/liangyx/tools/LucaOneApp-master/algorithms/llm")
    sys.path.append("/lustre/grp/gglab/liangyx/tools/LucaOneApp-master")
    sys.path.append("/lustre/grp/gglab/liangyx/tools/LucaOneApp-master/algorithms")
    sys.path.append("/lustre/grp/gglab/liangyx/tools/LucaOneApp-master/algorithms/llm/lucagplm")
    model_dirpath="/lustre/grp/gglab/liangyx/tools/LucaOne"
    log_filepath="/lustre/grp/gglab/liangyx/tools/LucaOneApp-master/logs.txt"
    from algorithms.args import Args
    from algorithms.file_operator import fasta_reader, csv_reader, tsv_reader
    from algorithms.utils import set_seed, to_device, get_labels, get_parameter_number, seq_type_is_match_seq, \
        gene_seq_replace, clean_seq_luca, available_gpu_id, download_trained_checkpoint_lucaone, calc_emb_filename_by_seq_id
    from algorithms.llm.lucagplm.v2_0.lucaone_gplm import LucaGPLM as LucaGPLMV2_0
    from algorithms.llm.lucagplm.v2_0.lucaone_gplm_config import LucaGPLMConfig as LucaGPLMConfigV2_0
    from algorithms.llm.lucagplm.v2_0.alphabet import Alphabet as AlphabetV2_0
    model,tokenizer,args,args_info,model_config=load_model_lucaone(log_filepath,model_dirpath)
    model.to(device)
    model=model.eval()
    def get_embedding(sequence_set,tokenizer,model,embedding_len,layer=-1):
        result=np.zeros((len(sequence_set),embedding_len,2560))
        with torch.no_grad():
            for i,seq in enumerate(sequence_set):
                seq_type="gene"
                seq = gene_seq_replace(seq)
                batch, processed_seq_len = encoder(args_info, model_config, seq, seq_type, tokenizer)
                new_batch = {}
                for item in batch.items():
                    if torch.is_tensor(item[1]):
                        new_batch[item[0]] = item[1].to(device)
                new_batch["return_contacts"] = True
                new_batch["return_dict"] = True
                new_batch["repr_layers"] = list(range(args_info["num_hidden_layers"] + 1))
                batch = new_batch
                output = model(**batch)['representation_matrix']
                print(f"layer_num={len(output)}")
                output=output[layer].cpu().detach().numpy()[:,1:1+embedding_len,:]
                result[i,:,:]=output
        return result
    result=get_embedding(np.loadtxt(input_seq,dtype=str),tokenizer,model,embedding_len=embedding_len,layer=layer)
if model_type=='GEN':
    import torch
    from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
    import sys
    import numpy as np 
    import os
    tokenizer = AutoTokenizer.from_pretrained(f"/lustre/grp/gglab/liangyx/data/benchmark/{model_name}",trust_remote_code=True)
    model = AutoModel.from_pretrained(f"/lustre/grp/gglab/liangyx/data/benchmark/{model_name}",trust_remote_code=True)

    # Get model configuration and maximum sequence length
    config = model.config
    max_length = config.max_position_embeddings
    device=torch.device('cuda')
    # Define input sequences

    model=model.to(device)

    # Tokenize the sequences
    # The add_special_tokens=True adds special tokens
    tokenizer.padding_side = "right"
    sequence_set=np.loadtxt(input_seq,dtype=str)
    token_len=embedding_len
    result=np.zeros((len(sequence_set),token_len,config.hidden_size))
    for i in range(len(sequence_set)):
        # print(i)
    # Perform a forward pass through the model to obtain the outputs, including hidden states
        inputs = tokenizer(
            [sequence_set[i]],
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs={key:value.to(device) for key,value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
        print(f"layer_num={len(outputs.hidden_states)}")
        result[i,:,:]=outputs.hidden_states[layer][0,1:1+token_len,:].detach().cpu().numpy()


np.save(output_file,result.astype(np.float32))