# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# coding=utf-8

import importlib.util
import torch
from torch import einsum, nn
from deepspeed.accelerator import get_accelerator

try:
    from einops import rearrange
except ImportError:
    rearrange = None

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb', 'apply_rotary_emb_unpad']

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        if rearrange is None:
             raise RuntimeError("einops is required for Rotary Embedding")
             
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq.to(get_accelerator().current_device_name())
        self.theta = theta

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        
        # emb [seq_length, .., dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # [seq, 1, 1, dim]
        base = rearrange(emb, 'n d -> n 1 1 d')
        
        # Always return [cos, sin]
        rope = [base.cos(), base.sin()]
        return rope


def _rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    rot_dim = freqs[0].shape[-1]
    t_pass = None
    if t.shape[-1] != rot_dim:
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    cos, sin = freqs
    cos = cos[:t.shape[0]].to(t.dtype)
    sin = sin[:t.shape[0]].to(t.dtype)

    t = (t * cos) + (_rotate_half(t) * sin)
    
    if t_pass is None:
        return t
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_emb_unpad(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    ) -> torch.Tensor:
    """
    Unpadded (Jagged) RoPE application for ModernBERT (FlashAttention VarLen).
    """
    rot_dim = cos.shape[-1]
    t_pass = None
    
    if t.shape[-1] != rot_dim:
        t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    
    total_nnz = t.shape[0]
    seq_idx = torch.arange(total_nnz, device=t.device, dtype=torch.int32)
    
    # Find which sample each token belongs to
    batch_idx = torch.searchsorted(cu_seqlens, seq_idx, right=True) - 1
    # Calculate position ID within the sequence
    pos_ids = seq_idx - cu_seqlens[batch_idx]
    
    # Gather cos/sin based on position
    cos_gathered = cos[pos_ids]
    sin_gathered = sin[pos_ids]
    
    # Adjust dimensions [total, 1, 1, d] -> [total, 1, d]
    if cos_gathered.dim() > 3:
        cos_gathered = cos_gathered.squeeze(1)
        sin_gathered = sin_gathered.squeeze(1)
    
    cos_gathered = cos_gathered.to(t.dtype)
    sin_gathered = sin_gathered.to(t.dtype)

    t_rotated = (t * cos_gathered) + (_rotate_half(t) * sin_gathered)
    
    if t_pass is not None:
        t_rotated = torch.cat((t_rotated, t_pass), dim=-1)
        
    return t_rotated