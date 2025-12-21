# Thanks for Chen ZY's implementation.
import os
from abc import abstractmethod
from typing import Optional, Literal, List, Union
import math

import torch
from torch import nn
import torch.nn.functional as F

from flash_attn.layers.rotary import apply_rotary_emb

class RelativePositionEmbedding(nn.Module):
    def __init__(self, apply_on: Literal["qk", "attn_mtx"]):
        super().__init__()
        self.apply_on = apply_on

    @abstractmethod
    def forward(self, position_ids: Optional[torch.Tensor] = None, **kwargs):
        pass


class RotaryEmbedding(RelativePositionEmbedding):
    def __init__(self, emb_dim: int, max_seqlen: int = 4096, base: int = 10000,
                 scaling_factor: float = 1.0, device: Optional[Union[torch.device, str]] = None,
                 use_FA_triton_kernel: bool = False,
                 num_posids: Optional[int] = None, learnable: bool = False):
        r"""
        For general purpose RoPE, num_posids is 1 or None, for GLM like models with 2 position
        indices, num_posids = 2

        Parameters
        --------------
        use_FA_triton_kernel: bool
            Whether to use the FA Triton kernel (fused) for the operation, only applicable for start_pos
            or cu_seqlens.
        """
        super().__init__(apply_on="qk")
        
        if num_posids is None:
            num_posids = 1
        self.num_posids = num_posids

        self.dim = emb_dim // self.num_posids
        self.max_seqlen = max_seqlen
        self.base = base
        self.scaling_factor = scaling_factor

        # Different from original paper, uses a different permutation in order to obtain the same calculation
        # check implementation in GPT-J and GPT-Neox for comparison

        # Note: use float32 to prevent models from using bfloat16 (which will round 1995.0 to 2000.0)
        # dim: [emb_dim // 2]
        wave_num = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))

        t = torch.arange(self.max_seqlen, device=device, dtype=torch.float32)
        t = t / self.scaling_factor
        # dim: [max_seqlen, emb_dim // 2]
        rotation_angles = torch.outer(t, wave_num)

        self.use_FA_triton_kernel = use_FA_triton_kernel
        if use_FA_triton_kernel:
            self.register_buffer("cached_cos", rotation_angles.cos())
            self.register_buffer("cached_sin", rotation_angles.sin())
        else:
            emb = torch.cat((rotation_angles, rotation_angles), dim=-1)
            # dim: [max_seqlen, emb_dim]
            self.register_buffer("cached_cos", emb.cos())
            self.register_buffer("cached_sin", emb.sin())

    @staticmethod
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin):
        """Applies Rotary Position Embedding to the query and key tensors.
        """
        # sin/cos_cache: [batch_size, seq_len, emb_dim] or [seq_len, emb_dim]
        # q, k: [batch_size, seq_len, num_heads, emb_dim] or [seq_len, num_heads, emb_dim]
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        q_embed = (q * cos) + (RotaryEmbedding.rotate_half(q) * sin)
        k_embed = (k * cos) + (RotaryEmbedding.rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(
        self, q, k,
        position_ids: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        precomputed_cos: Optional[torch.Tensor] = None,
        precomputed_sin: Optional[torch.Tensor] = None,
    ):
        r"""
        Position ids are used to index the precomputed sin/cos vectors, which are then used to rotate
        the query and key tensors. Should be provided in one of the following ways:
        - position_ids: a tensor of shape [batch_size, seq_len] or [seq_len,] containing the position ids.
        - start_pos: assume the position ids are **contiguous** starting from start_pos in the batch.
        - cu_seqlens: Tensor of shape [batch_size + 1,], used for flash attention

        Parameters
        --------------
        q, k: torch.Tensor
            The query and key tensors. Shape [batch_size, seq_len, num_heads, head_dim] if not using
            flash attention, otherwise [total_seqlen, num_heads, head_dim]
        precomputed_cos, precomputed_sin: torch.Tensor
            The precomputed cosine and sine vectors.

        Returns
        --------------
        q_embed, k_embed: torch.Tensor
            The rotated query and key tensors. Shape [batch_size, seq_len, num_heads, head_dim]
        precomputed_cos, precomputed_sin: torch.Tensor
            The precomputed cos and sin vectors.
        """

        # FA triton kernel
        if self.use_FA_triton_kernel:
            assert start_pos is not None or cu_seqlens is not None
            q_embed = apply_rotary_emb(q, self.cached_cos, self.cached_sin, inplace=True,
                                       seqlen_offsets=start_pos if start_pos is not None else 0,
                                       cu_seqlens=cu_seqlens)
            k_embed = apply_rotary_emb(k, self.cached_cos, self.cached_sin, inplace=True,
                                       seqlen_offsets=start_pos if start_pos is not None else 0,
                                       cu_seqlens=cu_seqlens)
        else:
            INPUT_ERROR_MSG = "Pass exactly one of position_ids, start_pos, or cu_seqlens"
            if precomputed_cos is None or precomputed_sin is None:
                if position_ids is not None:
                    assert start_pos is None and cu_seqlens is None, INPUT_ERROR_MSG
                    # position_ids: [batch_size, seq_len] or [seq_len], sin/cos_cache: [max_seqlen, emb_dim]
                    precomputed_cos = F.embedding(position_ids, self.cached_cos)
                    precomputed_sin = F.embedding(position_ids, self.cached_sin)
                    # sin/cos_cache: [batch_size, seq_len, emb_dim] or [seq_len, emb_dim]
                elif start_pos is not None:
                    assert position_ids is None and cu_seqlens is None, INPUT_ERROR_MSG
                    precomputed_cos = self.cached_cos[start_pos:start_pos + q.size(1)]
                    precomputed_sin = self.cached_sin[start_pos:start_pos + q.size(1)]
                elif cu_seqlens is not None:
                    assert position_ids is None and start_pos is None, INPUT_ERROR_MSG
                    position_ids = torch.cat(
                        [torch.arange(s, e, device=q.device) for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:])],
                        dim=0
                    )  # [total_seqlen, ]
                    precomputed_cos = F.embedding(position_ids, self.cached_cos)
                    precomputed_sin = F.embedding(position_ids, self.cached_sin)
                    # sin/cos_cache: [total_seqlen, emb_dim]
                else:
                    raise ValueError("Either position_ids or start_pos should be provided")

            q_embed, k_embed = self.apply_rotary_pos_emb(q, k, precomputed_cos, precomputed_sin)

        return q_embed, k_embed, precomputed_cos, precomputed_sin