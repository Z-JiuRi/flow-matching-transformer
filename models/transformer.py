# models/transformer.py
"""Transformer模型"""

import torch
import torch.nn as nn
from typing import Optional

from .normalization import RMSNorm, PreNorm
from .attention import MultiHeadAttention, CrossAttention
from .embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    ConditionEmbedding,
    LayerEmbedding
)


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization (用于条件注入)"""
    
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(cond_dim, dim * 2)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim)
        # cond: (B, cond_dim)
        x = self.norm(x)
        
        # 生成scale和shift
        scale_shift = self.proj(cond)  # (B, dim * 2)
        scale, shift = scale_shift.chunk(2, dim=-1)  # (B, dim) each
        
        # 应用条件
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer块 with AdaLN"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Self-attention with AdaLN
        self.adaln1 = AdaLN(dim, cond_dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        
        # FFN with AdaLN
        self.adaln2 = AdaLN(dim, cond_dim)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout)
        
        # 用于最终输出的scale
        self.adaln_final = nn.Linear(cond_dim, dim)
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.attn(self.adaln1(x, cond), mask)
        
        # FFN
        x = x + self.ffn(self.adaln2(x, cond))
        
        return x


class FlowMatchingTransformer(nn.Module):
    """Flow-Matching Transformer模型"""
    
    def __init__(
        self,
        H: int,
        W: int,
        condition_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        time_embed_dim: int = 256,
        cond_embed_dim: int = 256,
        layer_embed_dim: int = 64,
        num_layer_types: int = 2
    ):
        super().__init__()
        
        self.H = H
        self.W = W
        self.hidden_dim = hidden_dim
        
        # 输入投影: (H, W) -> (H*W, hidden_dim)
        self.input_proj = nn.Linear(1, hidden_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, H * W, hidden_dim) * 0.02)
        
        # 时间嵌入
        self.time_embed = TimestepEmbedding(time_embed_dim, hidden_dim)
        
        # 条件嵌入
        self.cond_embed = ConditionEmbedding(condition_dim, cond_embed_dim)
        
        # Layer嵌入
        self.layer_embed = LayerEmbedding(num_layer_types, layer_embed_dim)
        
        # 合并条件
        total_cond_dim = time_embed_dim + cond_embed_dim + layer_embed_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(total_cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                cond_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.final_norm = RMSNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        # 初始化权重
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.apply(_basic_init)
        
        # 输出层零初始化
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        layer: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, H, W) 噪声数据
            t: (B,) 时间步 [0, 1]
            condition: (B, condition_dim) 条件向量
            layer: (B,) layer索引 (0或1)
        
        Returns:
            velocity: (B, H, W) 预测的速度场
        """
        B = x.shape[0]
        
        # 展平并投影: (B, H, W) -> (B, H*W, 1) -> (B, H*W, hidden_dim)
        x = x.view(B, -1, 1)  # (B, H*W, 1)
        x = self.input_proj(x)  # (B, H*W, hidden_dim)
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 构建条件嵌入
        t_emb = self.time_embed(t)  # (B, time_embed_dim)
        cond_emb = self.cond_embed(condition)  # (B, cond_embed_dim)
        layer_emb = self.layer_embed(layer)  # (B, layer_embed_dim)
        
        # 合并条件
        cond = torch.cat([t_emb, cond_emb, layer_emb], dim=-1)  # (B, total_cond_dim)
        cond = self.cond_proj(cond)  # (B, hidden_dim)
        
        # Transformer块
        for block in self.blocks:
            x = block(x, cond)
        
        # 输出
        x = self.final_norm(x)
        x = self.output_proj(x)  # (B, H*W, 1)
        
        # 重塑为原始形状
        x = x.view(B, self.H, self.W)
        
        return x
