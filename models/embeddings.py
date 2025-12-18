# models/embeddings.py
"""嵌入层"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEmbedding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()
        self.dim = dim
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, dim)
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)


class TimestepEmbedding(nn.Module):
    """时间步嵌入 (用于Flow-Matching)"""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch,) 时间步 [0, 1]
        # 使用正弦编码
        half_dim = self.mlp[0].in_features // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return self.mlp(emb)


class ConditionEmbedding(nn.Module):
    """条件嵌入"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LayerEmbedding(nn.Module):
    """Layer嵌入 (0或1)"""
    
    def __init__(self, num_layers: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_layers, embed_dim)
    
    def forward(self, layer: torch.Tensor) -> torch.Tensor:
        # layer: (batch,) 整数
        return self.embedding(layer)