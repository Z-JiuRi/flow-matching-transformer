# models/normalization.py
"""归一化层"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        return x_normed * self.weight


class PreNorm(nn.Module):
    """Pre-Normalization wrapper"""
    
    def __init__(self, dim: int, fn: nn.Module, norm_type: str = "rms"):
        super().__init__()
        if norm_type == "rms":
            self.norm = RMSNorm(dim)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        self.fn = fn
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(self.norm(x), **kwargs)
