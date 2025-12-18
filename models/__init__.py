# models/__init__.py
"""模型模块"""

from .normalization import RMSNorm, PreNorm
from .embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    ConditionEmbedding,
    LayerEmbedding
)
from .attention import MultiHeadAttention, CrossAttention
from .transformer import FlowMatchingTransformer, TransformerBlock, FeedForward, AdaLN
from .flow_matching import FlowMatching


def build_model(config):
    """构建模型"""
    model = FlowMatchingTransformer(
        H=config.data.H,
        W=config.data.W,
        condition_dim=config.data.condition_dim,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_layers=config.model.num_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        time_embed_dim=config.model.time_embed_dim,
        cond_embed_dim=config.model.cond_embed_dim,
        layer_embed_dim=config.model.layer_embed_dim,
        num_layer_types=config.data.num_layers
    )
    
    flow_matching = FlowMatching(
        model=model,
        sigma_min=config.model.flow.sigma_min
    )
    
    return flow_matching


__all__ = [
    'RMSNorm',
    'PreNorm',
    'SinusoidalPositionalEmbedding',
    'TimestepEmbedding',
    'ConditionEmbedding',
    'LayerEmbedding',
    'MultiHeadAttention',
    'CrossAttention',
    'FlowMatchingTransformer',
    'TransformerBlock',
    'FeedForward',
    'AdaLN',
    'FlowMatching',
    'build_model'
]
