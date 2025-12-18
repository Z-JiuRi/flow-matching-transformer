# core/tester.py
"""测试器"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import logging
import numpy as np

from models import build_model
from utils.data_utils import create_dataloaders
from utils.helpers import AverageMeter, load_checkpoint
from utils.helpers import set_seed, setup_logging


class Tester:
    """测试器"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: Any,
        device: torch.device,
        logger: logging.Logger
    ):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.logger = logger

        exp_dir = Path(self.config.train.exp_dir)
        self.output_dir = exp_dir / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        load_checkpoint(
            path=path,
            model=self.model,
            device=self.device
        )
        self.logger.info(f"Model loaded from {path}")

    @classmethod
    def from_config(cls, config: Any, logger: logging.Logger) -> "Tester":
        set_seed(int(config.data.seed))
        device = torch.device(config.data.device if torch.cuda.is_available() else "cpu")

        _, _, test_loader = create_dataloaders(config, val_ratio=config.data.val_ratio)

        logger.info("Building model...")
        model = build_model(config).to(device)

        return cls(
            model=model,
            test_loader=test_loader,
            config=config,
            device=device,
            logger=logger,
        )

    @classmethod
    def run(
        cls,
        config: Any,
        checkpoint_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> Dict[str, Any]:
        exp_dir = Path(config.train.exp_dir)
        log_dir = exp_dir / "logs"
        setup_logging(log_dir)

        logger = logger or logging.getLogger(__name__)
        logger.info("=" * 50)
        logger.info("Testing")
        logger.info("=" * 50)

        checkpoint_path = checkpoint_path or str(exp_dir / "ckpts" / "best.pt")

        tester = cls.from_config(config=config, logger=logger)
        tester.load_checkpoint(checkpoint_path)

        metrics = tester.test()
        per_layer_metrics = tester.evaluate_per_layer()
        return {"metrics": metrics, "per_layer_metrics": per_layer_metrics}
    
    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """测试"""
        self.model.eval()
        self.logger.info("Starting testing...")
        
        # 指标
        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        mae_meter = AverageMeter()
        
        all_results = []
        
        for batch in tqdm(self.test_loader, desc="Testing"):
            target = batch['target'].to(self.device)
            condition = batch['condition'].to(self.device)
            layer = batch['layer'].to(self.device)
            data_ids = batch['data_id']
            
            # 计算损失
            loss = self.model(target, condition, layer)
            loss_meter.update(loss.item(), target.size(0))
            
            # 生成样本
            samples = self.model.sample(
                condition=condition,
                layer=layer,
                H=self.config.data.H,
                W=self.config.data.W,
                num_steps=self.config.inference.num_steps,
                solver=self.config.model.flow.solver,
                guidance_scale=self.config.inference.guidance_scale
            )
            
            # 计算指标
            mse = torch.mean((samples - target) ** 2).item()
            mae = torch.mean(torch.abs(samples - target)).item()
            
            mse_meter.update(mse, target.size(0))
            mae_meter.update(mae, target.size(0))
            
            # 保存结果
            for i, data_id in enumerate(data_ids):
                all_results.append({
                    'data_id': data_id,
                    'layer': layer[i].item(),
                    'target': target[i].cpu().numpy(),
                    'generated': samples[i].cpu().numpy(),
                    'mse': ((samples[i] - target[i]) ** 2).mean().item(),
                    'mae': torch.abs(samples[i] - target[i]).mean().item()
                })
        
        # 汇总结果
        metrics = {
            'loss': loss_meter.avg,
            'mse': mse_meter.avg,
            'mae': mae_meter.avg,
            'rmse': np.sqrt(mse_meter.avg)
        }
        
        self.logger.info("Test Results:")
        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value:.6f}")
        
        # 保存详细结果
        self._save_results(all_results, metrics)
        
        return metrics
    
    def _save_results(self, results: List[Dict], metrics: Dict[str, float]):
        """保存测试结果"""
        output_dir = self.output_dir / "test_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存指标
        metrics_path = output_dir / "metrics.pt"
        torch.save(metrics, str(metrics_path))
        
        # 保存详细结果
        results_path = output_dir / "results.pt"
        torch.save(results, str(results_path))
        
        self.logger.info(f"Results saved to {output_dir}")
    
    @torch.no_grad()
    def evaluate_per_layer(self) -> Dict[int, Dict[str, float]]:
        """按layer分别评估"""
        self.model.eval()
        
        # 每个layer的指标
        layer_metrics = {0: {'mse': [], 'mae': []}, 1: {'mse': [], 'mae': []}}
        
        for batch in tqdm(self.test_loader, desc="Evaluating per layer"):
            target = batch['target'].to(self.device)
            condition = batch['condition'].to(self.device)
            layer = batch['layer'].to(self.device)
            
            # 生成样本
            samples = self.model.sample(
                condition=condition,
                layer=layer,
                H=self.config.data.H,
                W=self.config.data.W,
                num_steps=self.config.inference.num_steps,
                solver=self.config.model.flow.solver,
                guidance_scale=self.config.inference.guidance_scale
            )
            
            # 按layer分组计算指标
            for i in range(target.size(0)):
                l = layer[i].item()
                mse = ((samples[i] - target[i]) ** 2).mean().item()
                mae = torch.abs(samples[i] - target[i]).mean().item()
                layer_metrics[l]['mse'].append(mse)
                layer_metrics[l]['mae'].append(mae)
        
        # 计算平均值
        results = {}
        for l in [0, 1]:
            if layer_metrics[l]['mse']:
                results[l] = {
                    'mse': np.mean(layer_metrics[l]['mse']),
                    'mae': np.mean(layer_metrics[l]['mae']),
                    'rmse': np.sqrt(np.mean(layer_metrics[l]['mse'])),
                    'count': len(layer_metrics[l]['mse'])
                }
        
        self.logger.info("Per-layer Results:")
        for l, m in results.items():
            self.logger.info(f"  Layer {l}: MSE={m['mse']:.6f}, MAE={m['mae']:.6f}, RMSE={m['rmse']:.6f}, Count={m['count']}")
        
        return results
