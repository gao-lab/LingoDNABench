import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_model(kind, path, device):
    if kind != 'evo2':
        from transformers import AutoConfig, AutoModel, AutoTokenizer, BertModel

    torch.manual_seed(666)
    if kind == 'dnabert':
        model = BertModel.from_pretrained(path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    elif kind == 'deepgene':
        sys.path.insert(0, str(path / 'code'))
        from configuration_roformer import RoFormerConfig
        from modeling_roformer import RoFormerForMaskedLM
        from tokenizers import Tokenizer

        config = RoFormerConfig.from_pretrained(path, local_files_only=True)
        model = RoFormerForMaskedLM(config)
        tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
    elif kind == 'lucaone':
        sys.path.insert(0, str(path / 'code'))
        from algorithms.llm.lucagplm.v2_0.lucaone_gplm import LucaGPLM
        from algorithms.llm.lucagplm.v2_0.lucaone_gplm_config import LucaGPLMConfig

        config = LucaGPLMConfig.from_pretrained(path, local_files_only=True)
        model = LucaGPLM(config, args=SimpleNamespace(
            pretrained_model_name=None, embedding_inference=True))
        tokenizer = model.alphabet
    elif kind == 'evo2':
        import yaml
        from evo2.models import StripedHyena, dotdict
        from vortex.model.tokenizer import CharLevelTokenizer

        config = dotdict(yaml.safe_load((path / 'config.yml').read_text()))
        config.use_fp8_input_projections = False
        torch.cuda.manual_seed_all(666)
        model = StripedHyena(config)
        model.to_bfloat16_except_pr_lc()
        tokenizer = CharLevelTokenizer(512)
    else:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        model = AutoModel.from_config(config, trust_remote_code=True)
        if kind in ('caduceus', 'hyenadna'):
            sys.path.insert(0, str(path))
            if kind == 'caduceus':
                from tokenization_caduceus import CaduceusTokenizer
                tokenizer = CaduceusTokenizer.from_pretrained(path, local_files_only=True)
            else:
                from tokenization_hyena import HyenaDNATokenizer
                tokenizer = HyenaDNATokenizer.from_pretrained(path, local_files_only=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)

    if kind != 'dnabert':
        state = torch.load(path / 'random-weights.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        del state
    return model.to(device).eval(), tokenizer


def tokenize_sequence(kind, tokenizer, sequence):
    if kind == 'dnabert':
        sequence = ' '.join(sequence[index:index + 3] for index in range(len(sequence) - 2))
    if kind == 'lucaone':
        sequence = sequence.translate(str.maketrans('ATCGN', '12345'))
        prefix = [tokenizer.cls_idx] if tokenizer.prepend_bos else []
        suffix = [tokenizer.eos_idx] if tokenizer.append_eos else []
        content = tokenizer.encode(sequence)
    elif kind == 'deepgene':
        prefix, content, suffix = [1, 2], tokenizer.encode(sequence, add_special_tokens=False).ids, []
    elif kind == 'evo2':
        prefix, content, suffix = [], tokenizer.tokenize(sequence), []
    else:
        encoded = tokenizer(sequence, truncation=False, padding=False)
        ids = encoded['input_ids']
        mask = tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
        positions = [index for index, special in enumerate(mask) if not special]
        if not positions:
            raise ValueError('Sequence has no DNA tokens')
        first, last = positions[0], positions[-1] + 1
        prefix, content, suffix = ids[:first], ids[first:last], ids[last:]
    if not content:
        raise ValueError('Sequence has no DNA tokens')
    return prefix, content, suffix


def extract_sequence(kind, model, tokenizer, sequence, layer, device):
    prefix, content, suffix = tokenize_sequence(kind, tokenizer, sequence)
    config = model.config
    if kind == 'evo2':
        limit = int(config.get('max_seqlen', 131072))
    else:
        limit = int(getattr(config, 'max_seq_len', getattr(config, 'max_position_embeddings', 131072)))
    if kind == 'nt':
        limit = min(limit - int(config.pad_token_id) - 1, int(tokenizer.model_max_length))
    capacity = limit - len(prefix) - len(suffix)
    if capacity < 1:
        raise ValueError('Model window has no space for DNA tokens')

    parts = []
    with torch.inference_mode():
        for start in range(0, len(content), capacity):
            chunk = content[start:start + capacity]
            ids = torch.tensor([prefix + chunk + suffix], dtype=torch.long, device=device)
            batch = {'input_ids': ids}
            if kind not in ('hyenadna', 'caduceus', 'evo2'):
                batch['attention_mask'] = torch.ones_like(ids)
            if kind == 'lucaone':
                depth = config.num_hidden_layers
                selected = depth if layer == -1 else layer
                batch.update(token_type_ids=torch.zeros_like(ids),
                             position_ids=torch.arange(ids.shape[1], device=device).unsqueeze(0),
                             repr_layers=list(range(depth + 1)), need_head_weights=False)
                hidden = model(**batch)['representation_matrix'][selected]
            elif kind == 'dnabert2':
                if layer == 0:
                    raise ValueError('DNABERT-2 layers start at 1')
                selected = layer - 1 if layer > 0 else layer
                hidden = model(**batch, output_all_encoded_layers=True)[selected]
            elif kind == 'evo2':
                selected = 28 if layer == -1 else layer
                captured = []

                def capture(module, inputs, output):
                    captured.append(output[0] if isinstance(output, tuple) else output)

                handle = model.get_submodule(f'blocks.{selected}.mlp.l3').register_forward_hook(capture)
                try:
                    model(ids)
                finally:
                    handle.remove()
                hidden = captured[0]
            else:
                if kind == 'deepgene':
                    batch['pos_ids'] = torch.arange(ids.shape[1], device=device).unsqueeze(0)
                output = model(**batch, output_hidden_states=True, return_dict=True)
                if kind in ('caduceus', 'hyenadna') and layer in (-1, config.n_layer):
                    hidden = output.last_hidden_state
                else:
                    hidden = output.hidden_states[layer]
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(0)
            if hidden.shape[1] != ids.shape[1] and hidden.shape[2] == ids.shape[1]:
                hidden = hidden.transpose(1, 2)
            if hidden.shape[1] != ids.shape[1]:
                raise ValueError(f'Unexpected embedding shape: {tuple(hidden.shape)}')
            parts.append(hidden[0, len(prefix):len(prefix) + len(chunk)].float().cpu().numpy())
    values = np.concatenate(parts, axis=0)
    if not np.isfinite(values).all():
        raise ValueError('Embedding contains non-finite values')
    return values


def run_matched_rw(args, sequences, total):
    kind = args.model_type
    path = Path(args.random_weights_dir) / kind
    if kind not in ('dnabert', 'dnabert2', 'nt', 'caduceus', 'hyenadna',
                    'omnina', 'deepgene', 'lucaone', 'GEN', 'evo2'):
        raise ValueError(f'No matched-RW model for {kind}')
    if args.embedding_len < 1 or total < 1:
        raise ValueError('Input and embedding length must be nonempty')
    initialization = json.loads((path / 'initialization.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if kind == 'evo2' and device.type != 'cuda':
        raise ValueError('Evo2 requires CUDA')

    output_dir = Path(args.output_dir)
    if not output_dir.name.endswith('-RW'):
        output_dir = output_dir.with_name(output_dir.name + '-RW')
    extension = '.npy' if args.save_format == 'npy' else '.h5'
    name = Path(args.input_seq).name.split('.')[0]
    output = Path(args.output_file) if args.output_file else output_dir / (
        f'{name}-embedding-layer_{args.layer}{extension}')
    metadata_path = Path(str(output) + '.json')
    signature = dict(model_type=kind, weights=str(path.resolve()),
                     input=str(Path(args.input_seq).resolve()),
                     input_size=Path(args.input_seq).stat().st_size,
                     input_mtime_ns=Path(args.input_seq).stat().st_mtime_ns,
                     layer=args.layer, embedding_len=args.embedding_len, format=args.save_format)
    if output.exists():
        if metadata_path.exists() and json.loads(metadata_path.read_text())['signature'] == signature:
            print(f'[OK] saved: {output}')
            return
        raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(kind, path, device)
    temporary = output.with_name(output.name + f'.{os.getpid()}.partial')
    handle = None
    values = None
    try:
        for row, sequence in enumerate(sequences):
            embedding = extract_sequence(kind, model, tokenizer, sequence.upper(), args.layer, device)
            if embedding.shape[0] > args.embedding_len:
                embedding = embedding[:args.embedding_len]
            if values is None:
                shape = (total, args.embedding_len, embedding.shape[1])
                if args.save_format == 'npy':
                    values = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=shape)
                else:
                    import h5py
                    handle = h5py.File(temporary, 'w')
                    values = handle.create_dataset('embeddings', shape=shape, dtype='float32')
            values[row] = 0
            values[row, :len(embedding)] = embedding
        if row + 1 != total:
            raise ValueError('Input sequence count changed during extraction')
        if handle is not None:
            handle.close()
            handle = None
        else:
            values.flush()
        del values
        values = None
        os.replace(temporary, output)
        metadata_path.write_text(json.dumps(dict(
            signature=signature, shape=shape, dtype='float32', seed=initialization['seed'],
            random_weights=True), indent=2) + '\n')
    finally:
        if handle is not None:
            handle.close()
        if temporary.exists():
            temporary.unlink()
    print(f'[OK] saved: {output}')
