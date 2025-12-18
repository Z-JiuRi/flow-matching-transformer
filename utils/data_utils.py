# utils/data_utils.py
"""数据处理工具"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import re
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

class ParamsDataset(Dataset):
    """Flow-Matching数据集"""
    
    def __init__(
        self,
        data_dir,
        cond_dir,
        H: int,
        W: int,
    ):
        self.linear1_dir = Path(data_dir) / "linear1"
        self.linear2_dir = Path(data_dir) / "linear2"
        self.cond_dir = Path(cond_dir)
        self.H = H
        self.W = W
        
        # 构建样本列表: (data_id, layer)
        self.samples = []
        # 获取所有文件ID
        linear1_files = list(self.linear1_dir.glob("*.pt"))
        
        for linear1_file in linear1_files:
            # 提取文件ID (如 "01105")
            data_id = linear1_file.stem
            
            # 对应的linear2文件
            linear2_file = self.linear2_dir / f"{data_id}.pt"
            
            # 对应的条件文件
            cond_file = self.cond_dir / f"{data_id}.pth"
            
            # linear1 对应 layer_id=0, linear2 对应 layer_id=1
            self.samples.append({
                'layer_id': 0,
                'data_path': str(linear1_file),
                'cond_file': str(cond_file)
            })
            
            self.samples.append({
                'layer_id': 1,
                'data_path': str(linear2_file),
                'cond_file': str(cond_file)
            })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        layer_id = sample['layer_id']
        data_path = sample['data_path']
        cond_file = sample['cond_file']
        
        # 加载目标tensor (H, W)
        tensor = torch.load(data_path, map_location='cpu')
        
        # 确保形状正确
        if tensor.shape != (self.H, self.W):
            tensor = tensor.view(self.H, self.W)
        
        # 加载条件向量
        condition = torch.load(cond_file, map_location='cpu')
        
        return {
            'tensor': tensor.float(),  # (H, W)
            'condition': condition.float(),  # (cond_dim,)
            'layer_id': torch.tensor(layer_id, dtype=torch.long),  # scalar
        }

def create_dataloaders(cfg, val_ratio: Optional[float] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
    """创建数据加载器"""
   
    # 创建数据集
    all_dataset = ParamsDataset(cfg.data.data_dir, cfg.data.cond_dir, cfg.data.H, cfg.data.W)
    logger.info(f"Loaded from {cfg.data.data_dir}, Total samples: {len(all_dataset)}, val_ratio: {val_ratio}")
    
    if val_ratio is not None and val_ratio > 0:
        tot_len = len(all_dataset)
        val_len = int(tot_len * val_ratio)
        train_len = tot_len - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            all_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(cfg.data.seed)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )
        return train_loader, val_loader
    else:
        test_loader = DataLoader(
            all_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )
        return test_loader
