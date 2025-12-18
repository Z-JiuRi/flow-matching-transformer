# utils/helpers.py
"""通用工具函数"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

def setup_logging(log_dir):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format=('[%(asctime)s|%(name)s]- %(message)s'),
        datefmt='%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_dir / "output.log"),
            logging.StreamHandler()
        ]
    )


def mkdir(path: str):
    """创建目录，可以直接输入 dir1/dir2/dir3"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    loss: float,
    path: str,
    ema_model: Optional[nn.Module] = None
):
    """保存检查点"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if ema_model is not None:
        checkpoint['ema_model_state_dict'] = ema_model.state_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    ema_model: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """加载检查点"""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if ema_model is not None and 'ema_model_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'step': checkpoint.get('step', 0),
        'loss': checkpoint.get('loss', float('inf'))
    }


class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMA:
    """指数移动平均"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict
