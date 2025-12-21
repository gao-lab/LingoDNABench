# LingoDNABench: https://github.com/gao-lab/LingoDNABench

import argparse
import os
import sys
sys.path.append("./")
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import functools
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    FullStateDictConfig,
    FullOptimStateDictConfig,
    StateDictType,
)
# BERT modules
from BERT.modules import EncoderLayer
from BERT.data import  DNADataset, DNATokenizer, DataCollatorForDNA
from BERT.model import BaselineBERT
from BERT.utils import get_param_num, load_config


parser = argparse.ArgumentParser(description='Train BERT-like gLM baseline model.')
parser.add_argument('--train_data', type = str, default = "./data/dataset/human.4096.h5")
parser.add_argument('--config_file', type = str, default = "./config/Baseline_gLM.config.json")
parser.add_argument('--model_save_path', type = str, default = "./model/human_gLM")
parser.add_argument('--random_seed', type = int, default = 666, help = 'Random seed, default is 666.')
parser.add_argument('--verbose_step', type = int, default = 100, help = 'Print training info every N steps, default is 100.')
parser.add_argument('--resume', type = str, default = "false", help = 'Continue training from checkpoint or start new training, default is false.')
parser.add_argument('--model_checkpoint', type = str, default = None, help = 'Model checkpoint path for resuming training.')
args = parser.parse_args()


# load model config and training config from json file
os.makedirs(args.model_save_path, exist_ok=True)
config = load_config(args.config_file)

'''model config'''
model_config = config["model_config"]
d_kv = model_config['d_kv']
n_heads = model_config['n_heads']
n_layers = model_config['n_layers']
max_vocab = model_config['max_vocab']
kmer = model_config['kmer']

'''training config'''
training_config = config['training_config']
batch_size = training_config['batch_size']
lr_init = training_config['lr_init']
lr_max = training_config['lr_max']
lr_min = training_config['lr_min']
epochs = training_config['epochs']
warmup_steps = training_config['warmup_steps']
train_steps = training_config['train_steps']

'''random seed for torch'''
seed = args.random_seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

'''precision'''
bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

def setup():
    # initialize the process group
    dist.init_process_group(backend="nccl")

def cleanup():
    dist.destroy_process_group()

def resume_checkpoint(model, optimizer, model_checkpoint, device):
    checkpoint = torch.load(model_checkpoint, map_location=device)
    state_dict = checkpoint['model_state_dict']
    opt_state = checkpoint['optimizer_state_dict']
    lr_scheduler_state = checkpoint['lr_schedule']
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    #resume model
    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=False),
        FullOptimStateDictConfig(rank0_only=False),
    )
    model.load_state_dict(state_dict)
    optim_state_dict = FSDP.optim_state_dict_to_load(
        model, optimizer, opt_state
    )
    optimizer.load_state_dict(optim_state_dict)
    return model, optimizer, epoch, step, lr_scheduler_state

def save_checkpoint(model, global_rank, epoch, step, scheduler, optimizer, model_save_path):
    with FSDP.state_dict_type(model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(rank0_only=True),
        ):
        #print("Get model state dict...")
        model_state_dict = model.state_dict()
        #print("Get optimizer state dict...")
        optimizer_state_dict = FSDP.optim_state_dict(
            model,
            optimizer
        )
    if global_rank == 0:
        print("Save model...")
        torch.save({'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer_state_dict,
                    'step': step,
                    'lr_schedule':scheduler.state_dict()}, 
                    os.path.join(model_save_path, f"/model_{epoch}.pt"))

def train(device, step, model, scheduler, global_rank, train_dataloader, optimizer, epoch, sampler):
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    fsdp_loss = torch.zeros(2).to(device)
    
    sampler.set_epoch(epoch)
    for one_batch in train_dataloader:
        if global_rank == 0:
            start_time = time.time()
        
        input_ids, masked_pos, masked_tokens = map(lambda x: x.to(device), one_batch)
        
        optimizer.zero_grad()
        logits_lm = model(input_ids, masked_pos)
        loss_lm = loss_fn(logits_lm.view(-1, max_vocab), masked_tokens.view(-1))
        
        loss_lm = (loss_lm.float()).mean()
        loss_lm.backward()
        #clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #update parameters
        optimizer.step()
        
        if global_rank == 0 and step % args.verbose_step == 0:
            print(f"epoch {epoch}, step {step}, train_loss {loss_lm}")
        
        fsdp_loss[0] += loss_lm.item()
        fsdp_loss[1] += 1
        #update lr schedule
        scheduler.step()
        step += 1

    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)
    fsdp_loss = fsdp_loss[0] / fsdp_loss[1]
    return fsdp_loss, step


def fsdp_main():
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    device = torch.device("cuda", local_rank)
    
    setup()
    
    '''train data'''
    
    dna_tokenizer = DNATokenizer(kmer = kmer)
    dna_collator = DataCollatorForDNA(dna_tokenizer, dynamic_length=True, dynamic_length_prob=0.05)
    train_dataset = DNADataset(dna_tokenizer, args.train_data)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, collate_fn=dna_collator, sampler=train_sampler, batch_size=batch_size, num_workers=8)

    #load model
    model = BaselineBERT(max_vocab, n_heads, d_kv, n_layers)

    #get model parms number
    if global_rank == 0:
        get_param_num(model)
    
    #wrap model
    model_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            EncoderLayer,
        },
    )

    torch.cuda.set_device(local_rank)

    model = FSDP(model,
        auto_wrap_policy=model_auto_wrap_policy,
        mixed_precision=bfSixteen,
        device_id=torch.cuda.current_device(),
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    )

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9,0.999), eps=1e-8, lr=lr_max, weight_decay=0.01)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=lr_init/lr_max, end_factor=1.0, total_iters=warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_min/lr_max, total_iters=train_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])
    
    if args.resume.lower() == "true" and args.model_checkpoint is not None:
        if global_rank == 0:
            print("Resume model...")
        model, optimizer, epoch_save, step_save, lr_scheduler_state = resume_checkpoint(model, optimizer, args.model_checkpoint, device)
        scheduler.load_state_dict(lr_scheduler_state)
        epoch_start = epoch_save + 1
        step_start = step_save + 1
    else:
        epoch_start = 0
        step_start = 0
    
    if global_rank == 0:
        print("Start training...")
    
    step = step_start
    for epoch in range(epoch_start, epochs):
        train_epoch_loss, step = train(device, step, model, scheduler, global_rank, train_dataloader, optimizer, epoch, train_sampler)
        if global_rank == 0:
            print(f"epoch {epoch}, train_epoch_loss {train_epoch_loss}")
        save_checkpoint(model, global_rank, epoch, step, scheduler, optimizer, args.model_save_path)
    
    dist.barrier()
    cleanup()

if __name__ == "__main__":
    fsdp_main()