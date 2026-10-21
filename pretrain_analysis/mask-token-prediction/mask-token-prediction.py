"""Mean observed-token probability over each model's native DNA output tokens."""

import argparse
import csv
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path


# Objective, tokenization, number of native DNA output IDs (including native N).
MODELS = {
    'RandomWeight': ('MLM', '1mer', 5),
    'HyenaDNA': ('CLM', '1mer', 5),
    'Caduceus': ('MLM', '1mer', 5),
    'DNABERT3mer': ('MLM', '3mer', 64),
    'DeepGENE': ('MLM', 'BPE', 4091),
    'DNABERT2': ('MLM', 'BPE', 4091),
    'OmniNA': ('CLM', 'BPE', 8895),
    'LucaOne': ('MLM', '1mer', 5),
    'NT-2.5B-MS': ('MLM', '6mer', 4101),
    'GENERator': ('CLM', '6mer', 4096),
    'Evo2': ('CLM', '1mer', 5),
}


def load_model(args):
    import torch

    checkpoint = args.checkpoint.resolve()
    if args.code_path:
        sys.path.insert(0, str(args.code_path.resolve()))
    if args.model == 'RandomWeight':
        if not args.device.startswith('cuda'):
            raise ValueError('RandomWeight uses the original CUDA FlashAttention implementation')
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pretrain_gLM'))
        from BERT.model import BaselineBERT
        from BERT.data import DNATokenizer
        model = BaselineBERT(16, 16, 64, 12, eval_mode=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
        if not torch.equal(state['embedding.token_embedding.weight'], state['word_classifier.weight']):
            raise ValueError('Inconsistent shared embedding/output weights')
        model.load_state_dict(state, strict=True)
        tokenizer = DNATokenizer(kmer=1)
    elif args.model == 'LucaOne':
        from algorithms.llm.lucagplm.v2_0.lucaone_gplm import LucaGPLM
        from algorithms.llm.lucagplm.v2_0.lucaone_gplm_config import LucaGPLMConfig
        training_args = torch.load(checkpoint / 'training_args.bin', map_location='cpu', weights_only=False)
        training_args.embedding_inference = False
        training_args.pretrained_model_name = None
        config = LucaGPLMConfig.from_json_file(checkpoint / 'config.json')
        model = LucaGPLM(config, args=training_args)
        state = torch.load(checkpoint / 'pytorch.pth', map_location='cpu', weights_only=False)
        state = state.get('state_dict', state)
        model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=True)
        tokenizer = model.alphabet
    elif args.model == 'DeepGENE':
        from modeling_roformer import RoFormerForMaskedLM
        from tokenizers import Tokenizer
        if args.tokenizer_path is None:
            raise ValueError('DeepGENE requires --tokenizer-path tokenizer.json')
        tokenizer = Tokenizer.from_file(str(args.tokenizer_path.resolve()))
        model = RoFormerForMaskedLM.from_pretrained(str(checkpoint), local_files_only=True)
    elif args.model == 'Evo2':
        import evo2
        from evo2 import Evo2
        if not args.device.startswith('cuda'):
            raise ValueError('Evo2 requires its validated CUDA environment')
        original_directory = Path.cwd()
        try:
            os.chdir(Path(evo2.__file__).resolve().parent)
            model = Evo2('evo2_7b', local_path=str(checkpoint))
        finally:
            os.chdir(original_directory)
        model.model.eval()
        return model, model.tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
        factory = AutoModelForCausalLM if MODELS[args.model][0] == 'CLM' else AutoModelForMaskedLM
        model = factory.from_pretrained(str(checkpoint), trust_remote_code=True, local_files_only=True)
        if args.model == 'Caduceus':
            if not model.config.bidirectional_weight_tie:
                raise ValueError('Expected the released weight-tied Caduceus checkpoint')
            for block in model.caduceus.backbone.layers:
                mixer = block.mixer.submodule if model.config.rcps else block.mixer
                for name in ('in_proj', 'out_proj'):
                    forward = getattr(mixer.mamba_fwd, name)
                    reverse = getattr(mixer.mamba_rev, name)
                    reverse.weight, reverse.bias = forward.weight, forward.bias
            model.tie_weights()
            if model.get_output_embeddings().weight is not model.get_input_embeddings().weight:
                raise ValueError('Caduceus embedding/output weights are not tied')
    model.to(args.device).eval()
    return model, tokenizer


def encode_sequence(sequence, tokenizer, name):
    """Return content IDs, base spans, full input IDs, target positions and MASK ID."""
    objective, tokenization, _ = MODELS[name]
    if tokenization == '6mer':
        sequence = sequence[:len(sequence) // 6 * 6]  # 200 bp -> 198 bp.
    if name == 'RandomWeight':
        native = tokenizer(sequence)
        ids = native[1:-1]
    elif name == 'LucaOne':
        from algorithms.utils import gene_seq_replace
        ids = tokenizer.encode(gene_seq_replace(sequence))
    elif name == 'Evo2':
        ids = tokenizer.tokenize(sequence)
    elif name == 'DeepGENE':
        encoded = tokenizer.encode(sequence, add_special_tokens=False)
        ids, offsets = encoded.ids, encoded.offsets
    elif tokenization == '3mer':
        kmers = [sequence[i:i+3] for i in range(len(sequence)-2)]
        ids = tokenizer(' '.join(kmers), add_special_tokens=False)['input_ids']
    else:
        encoded = tokenizer(sequence, add_special_tokens=False, return_offsets_mapping=tokenization == 'BPE')
        ids = encoded['input_ids']
        if tokenization == 'BPE':
            offsets = encoded['offset_mapping']
    ids = [int(value) for value in ids]
    if tokenization != 'BPE':
        size, step = (3, 1) if tokenization == '3mer' else ((6, 6) if tokenization == '6mer' else (1, 1))
        offsets = [(i*step, i*step+size) for i in range(len(ids))]
        if len(ids) != (len(sequence)-size)//step+1:
            raise ValueError('Unexpected native token count')
    if len(ids) != len(offsets) or not ids or any(not 0 <= a < b <= len(sequence) for a, b in offsets):
        raise ValueError('Invalid native token spans')
    if objective == 'CLM':
        full = [tokenizer.bos_token_id, *ids] if name == 'OmniNA' else ids
        positions = list(range(1 if name == 'OmniNA' else 0, len(full)))
        if name == 'OmniNA' and full != tokenizer(sequence, add_special_tokens=True)['input_ids']:
            raise ValueError('OmniNA native BOS layout mismatch')
        return ids, offsets, full, positions, None
    if name == 'RandomWeight':
        full, positions, mask = native, list(range(1, len(ids)+1)), tokenizer.token2id['[MASK]']
    elif name == 'LucaOne':
        full = [tokenizer.cls_idx, *ids, tokenizer.eos_idx]
        positions, mask = list(range(1, len(ids)+1)), tokenizer.mask_idx
    elif name == 'DeepGENE':
        full = [tokenizer.token_to_id('[CLS]'), tokenizer.token_to_id('[SEP]'), *ids]
        positions, mask = list(range(2, len(ids)+2)), tokenizer.token_to_id('[MASK]')
    elif name == 'DNABERT2':
        full = [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]
        positions, mask = list(range(1, len(ids)+1)), tokenizer.mask_token_id
        if full != tokenizer(sequence, add_special_tokens=True)['input_ids']:
            raise ValueError('DNABERT2 native CLS/SEP layout mismatch')
    else:
        full = tokenizer.build_inputs_with_special_tokens(ids)
        special = tokenizer.get_special_tokens_mask(full, already_has_special_tokens=True)
        positions = [i for i, is_special in enumerate(special) if not is_special]
        mask = tokenizer.mask_token_id
    if len(positions) != len(ids) or mask is None:
        raise ValueError('Native MASK or target positions are missing')
    return ids, offsets, full, positions, mask


def forward_logits(model, input_ids, name):
    import torch

    if name == 'RandomWeight':
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model(input_ids, masked_pos=positions)
    if name == 'LucaOne':
        from contextlib import nullcontext
        context = torch.autocast('cuda', dtype=torch.bfloat16) if input_ids.is_cuda else nullcontext()
        with context:
            _, logits, _, _ = model.__forword__(input_ids=input_ids,
                token_type_ids=torch.zeros_like(input_ids), output_keys={'token_level':['gene_mask']},
                repr_layers=[0], need_head_weights=False, return_contacts=False)
        return logits['token_level']['gene_mask']
    if name == 'DeepGENE':
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        return model(input_ids=input_ids, pos_ids=positions).logits
    if name == 'Evo2':
        return model.model(input_ids)[0]
    kwargs = {'input_ids':input_ids}
    signature = inspect.signature(model.forward).parameters
    if 'attention_mask' in signature:
        kwargs['attention_mask'] = torch.ones_like(input_ids)
    if 'use_cache' in signature:
        kwargs['use_cache'] = False
    return model(**kwargs).logits


def mean_prediction(sequence, model, tokenizer, name, dna_ids, batch_size, device):
    import torch

    ids, offsets, full, positions, mask = encode_sequence(sequence, tokenizer, name)
    if not set(ids) <= set(dna_ids):
        raise ValueError('Observed target is outside the native DNA output vocabulary')
    support = torch.tensor(dna_ids, device=device)
    probabilities = []
    if MODELS[name][0] == 'CLM':
        targets = list(range(len(ids))) if name == 'OmniNA' else list(range(1, len(ids)))
        inputs = torch.tensor([full], device=device)
        logits = forward_logits(model, inputs, name)[0]
        selected = logits[torch.tensor([positions[t]-1 for t in targets], device=device)].float()
        true_ids = torch.tensor([ids[t] for t in targets], device=device)
        log_p = selected.gather(1, true_ids[:, None]).squeeze(1) - torch.logsumexp(selected[:, support], dim=-1)
        probabilities.extend(log_p.exp().cpu().tolist())
    else:
        for begin in range(0, len(ids), batch_size):
            targets = list(range(begin, min(begin+batch_size, len(ids))))
            inputs = torch.tensor(full, device=device).repeat(len(targets), 1)
            for row, target in enumerate(targets):
                start, end = offsets[target]
                overlapping = [positions[k] for k, (a, b) in enumerate(offsets) if a < end and b > start]
                inputs[row, overlapping] = mask
            logits = forward_logits(model, inputs, name)
            selected = logits[torch.arange(len(targets), device=device),
                torch.tensor([positions[t] for t in targets], device=device)].float()
            true_ids = torch.tensor([ids[t] for t in targets], device=device)
            log_p = selected.gather(1, true_ids[:, None]).squeeze(1) - torch.logsumexp(selected[:, support], dim=-1)
            probabilities.extend(log_p.exp().cpu().tolist())
    if not probabilities or not all(0 <= value <= 1.00001 for value in probabilities):
        raise ValueError('Empty or non-finite predicted probabilities')
    return sum(probabilities) / len(probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--input', type=Path, default=Path(__file__).resolve().parent / 'data/paired_sequences.tsv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cohort', choices=['all','primary','conservation_matched'], default='all')
    parser.add_argument('--code-path', type=Path, help='DeepGENE PanGeneGraphTrans or LucaOneApp root')
    parser.add_argument('--tokenizer-path', type=Path, help='DeepGENE tokenizer.json')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit-pairs', type=int, help='Optional small inference test; omitted for final results')
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit_pairs is not None and args.limit_pairs < 1):
        parser.error('Batch size and pair limit must be positive')
    with args.input.open() as handle:
        pairs = list(csv.DictReader(handle, delimiter='\t'))
    if args.cohort != 'all':
        pairs = [row for row in pairs if row['cohort'] == args.cohort]
    if args.limit_pairs is not None:
        pairs = pairs[:args.limit_pairs]
    if not pairs:
        raise ValueError('No input pairs')
    sequences = {}
    pair_ids = set()
    for row in pairs:
        key = (row['cohort'], row['pair_id'])
        if key in pair_ids:
            raise ValueError('Duplicate pair ID within a cohort')
        pair_ids.add(key)
        for region in ('alu','exon'):
            sequence_id, sequence = row[f'{region}_id'], row[f'{region}_sequence']
            if len(sequence) != 200 or set(sequence) - set('ACGT'):
                raise ValueError('Final inputs must be 200-bp canonical reference-forward sequences')
            if sequence_id in sequences and sequences[sequence_id] != sequence:
                raise ValueError('An interval ID has conflicting sequences')
            sequences[sequence_id] = sequence
    import torch
    from tqdm import tqdm
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, tokenizer = load_model(args)
    if args.model == 'RandomWeight':
        dna_ids = [tokenizer.token2id[base] for base in 'ACGTN']
    elif args.model == 'LucaOne':
        from algorithms.utils import gene_seq_replace
        dna_ids = tokenizer.encode(gene_seq_replace('ACGTN'))
    elif args.model == 'Evo2':
        dna_ids = tokenizer.tokenize('ACGTN')
    else:
        special = set(getattr(tokenizer, 'all_special_ids', []))
        dna_ids = []
        for token, token_id in tokenizer.get_vocab().items():
            dna = token.removeprefix('▁').removeprefix('Ġ')
            if dna and set(dna) <= set('ACGTN') and token_id not in special:
                dna_ids.append(token_id)
    dna_ids = sorted(set(int(value) for value in dna_ids))
    if len(dna_ids) != MODELS[args.model][2]:
        raise ValueError(f'Unexpected DNA vocabulary size: {len(dna_ids)}; check checkpoint/tokenizer version')
    means = {}
    with torch.inference_mode():
        for sequence_id, sequence in tqdm(sequences.items(), desc=args.model):
            means[sequence_id] = mean_prediction(sequence, model, tokenizer, args.model,
                dna_ids, args.batch_size, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.partial')
    with temporary.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['cohort','pair_id','model','alu_mean_predicted','exon_mean_predicted'],
            delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in pairs:
            writer.writerow(dict(cohort=row['cohort'], pair_id=row['pair_id'], model=args.model,
                alu_mean_predicted=means[row['alu_id']], exon_mean_predicted=means[row['exon_id']]))
    temporary.replace(args.output)
    metadata = dict(model=args.model, checkpoint=str(args.checkpoint.resolve()),
        code_path=str(args.code_path.resolve()) if args.code_path else None,
        tokenizer_path=str(args.tokenizer_path.resolve()) if args.tokenizer_path else None,
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), cohort=args.cohort,
        pair_count=len(pairs), unique_windows=len(sequences), limit_pairs=args.limit_pairs,
        dna_token_ids=dna_ids, uniform_reference=1/len(dna_ids), torch_version=torch.__version__,
        definition='mean observed-token probability conditional on native DNA-sequence output IDs, including N')
    Path(str(args.output) + '.meta.json').write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
