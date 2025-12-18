# models/attention.py
"""注意力机制"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class CrossAttention(nn.Module):
    """交叉注意力"""
    
    def __init__(
        self,
        dim: int,
        context_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=bias)
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = context.shape
        
        # Q from x, KV from context
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x
