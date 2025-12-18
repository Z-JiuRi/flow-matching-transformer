# core/trainer.py
"""训练器"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

PYTORCH2 = int(torch.__version__.split('.')[0]) >= 2

if PYTORCH2:
    from torch.amp import autocast, GradScaler
else:
    from torch.cuda.amp import autocast, GradScaler

from models import build_model
from utils.data_utils import create_dataloaders
from utils.helpers import AverageMeter, EMA, save_checkpoint, load_checkpoint, mkdir, set_seed, setup_logging
from utils.scheduler import get_lr_scheduler
from utils.visualize import plot_heatmap, plot_histogram, plot_gaussian
import logging

class Trainer:
    """训练器"""
    def __init__(
        self,
        config: Any,
        resume: Optional[str] = None,
    ):
        self.config = config
        self.exp_dir = mkdir(self.config.train.exp_dir)
        self.log_dir = mkdir(self.exp_dir / "logs")
        self.ckpt_dir = mkdir(self.exp_dir / "ckpts")
        self.heatmap_dir = mkdir(self.exp_dir / "results" / "heatmaps")
        self.histogram_dir = mkdir(self.exp_dir / "results" / "histograms")
        self.gaussian_dir = mkdir(self.exp_dir / "results" / "gaussians")
        
        setup_logging(self.log_dir)
        self.logger = logging.getLogger("Trainer")
        self.logger.info(f"Config:\n{OmegaConf.to_yaml(self.config)}")

        self.tb_writer = None
        if SummaryWriter is not None:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            self.tb_writer.add_text("config", OmegaConf.to_yaml(self.config), global_step=0)
            self.logger.info(f"TensorBoard log_dir: {self.log_dir}")
        else:
            self.logger.info("TensorBoard not available; skip SummaryWriter.")

        set_seed(int(self.config.data.seed))
        self.device = torch.device(self.config.data.device if torch.cuda.is_available() else "cpu")

        self.train_loader, self.val_loader = create_dataloaders(self.config, val_ratio=self.config.data.val_ratio)
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val   samples: {len(self.val_loader.dataset)}")
        
        self.logger.info("Building model...")
        self.model = build_model(self.config).to(self.device)
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=float(self.config.train.lr), 
                betas=tuple(self.config.train.betas), 
                weight_decay=float(self.config.train.weight_decay)
            )
        
        if self.config.train.scheduler_type == "cosine_warmup":
            self.scheduler = get_lr_scheduler(
                self.optimizer,
                scheduler_type=self.config.train.scheduler_type,
                warmup_epochs=max(1, int(0.1 * int(self.config.train.num_epochs))),
                max_epochs=int(self.config.train.num_epochs),
                warmup_start_lr=1.0e-8,
                eta_min=float(self.config.train.min_lr),
            )
        elif self.config.train.scheduler_type == "cosine":
            self.scheduler = get_lr_scheduler(
                self.optimizer,
                scheduler_type=self.config.train.scheduler_type,
                T_max=int(self.config.train.num_epochs),
                eta_min=float(self.config.train.min_lr),
            )
        
        # EMA
        self.ema = None
        if self.config.train.ema_decay and float(self.config.train.ema_decay) > 0:
            ema_target = self.model.model if hasattr(self.model, "model") else self.model
            self.ema = EMA(ema_target, float(self.config.train.ema_decay))
        
        # 混合精度
        self.scaler = None
        if self.config.train.mixed_precision:
            self.scaler = GradScaler()
        
        # 状态
        self.epoch = 1
        self.global_step = 0
        self.best_val_loss = float('inf')
        if resume:
            self.load_checkpoint(resume)

    def train(self):
        """训练主循环"""
        self.logger.info("Starting training...")

        for epoch in range(self.epoch, self.config.train.num_epochs + 1):
            self.epoch = epoch
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("epoch/train_loss", train_loss, epoch)
            
            # 验证
            val_loss = None
            if self.val_loader is not None and epoch % self.config.train.val_every == 0:
                val_loss = self.validate()
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("epoch/val_loss", val_loss, epoch)
                
                self.logger.info(f"[V] Epoch {epoch}/{self.config.train.num_epochs} - Val Loss: {val_loss:.4e}")
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
                    self.logger.info(f"New best model saved with val loss: {val_loss:.4e}")
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar("epoch/best_val_loss", self.best_val_loss, epoch)
            else:
                self.logger.info(
                    f"[T] Epoch {epoch}/{self.config.train.num_epochs} - Train Loss: {train_loss:.4e}"
                )

            if self.config.train.scheduler_type == "reduce_on_plateau":
                self.scheduler.step(val_loss if val_loss is not None else train_loss)
            else:
                self.scheduler.step()
            
            # 定期保存
            if epoch % self.config.train.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")
        
        # 保存最终模型
        self.save_checkpoint("final.pt")
        self.logger.info("Training completed!")
        if self.tb_writer is not None:
            self.tb_writer.close()
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        loss_meter = AverageMeter()
        cond_drop_prob = float(getattr(self.config.train, "cond_drop_prob", 0.0))
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            target = batch['tensor'].to(self.device)
            condition = batch['condition'].to(self.device)
            layer = batch['layer_id'].to(self.device)
            dropped = 0.0
            if cond_drop_prob > 0:
                drop_mask = torch.rand(condition.size(0), device=condition.device) < cond_drop_prob
                if torch.any(drop_mask):
                    condition = condition.clone()
                    condition[drop_mask] = 0
                dropped = float(drop_mask.float().mean().item())

            self.optimizer.zero_grad()

            if self.scaler is not None:
                if PYTORCH2:
                    # PyTorch 2.0+ 语法
                    with autocast(device_type=self.device.type):
                        loss, pred_velocity, target_velocity = self.model(target, condition, layer)
                else:
                    # PyTorch 1.x 语法 - autocast只能用于CUDA
                    with autocast(enabled=(self.device.type == 'cuda')):
                        loss, pred_velocity, target_velocity = self.model(target, condition, layer)
                
                self.scaler.scale(loss).backward()

                if self.config.train.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.train.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss, pred_velocity, target_velocity = self.model(target, condition, layer)
                loss.backward()

                if self.config.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.train.grad_clip
                    )

                self.optimizer.step()

            if self.ema is not None:
                self.ema.update()

            self.global_step += 1

            loss_meter.update(loss.item(), target.size(0))
            if self.tb_writer is not None:
                self.tb_writer.add_scalar("train/loss", float(loss.item()), self.global_step)
                self.tb_writer.add_scalar("train/lr", float(self.scheduler.get_last_lr()[0]), self.global_step)
                velocity_mse = torch.mean((pred_velocity - target_velocity) ** 2).item()
                self.tb_writer.add_scalar("train/velocity_mse", float(velocity_mse), self.global_step)
                self.tb_writer.add_scalar("train/cond_drop_rate", float(dropped), self.global_step)

            if self.epoch % self.config.train.plot_every == 0:
                plot_heatmap(pred_velocity, target_velocity, filename=self.heatmap_dir / f"[T] Epoch {self.epoch} Heatmap")
                plot_histogram(pred_velocity, pred_velocity, filename=self.histogram_dir / f"[T] Epoch {self.epoch} Histogram")
                plot_gaussian(pred_velocity - target_velocity, filename=self.gaussian_dir / f"[T] Epoch {self.epoch} Diff Gaussian")
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4e}', 'lr': f'{self.scheduler.get_last_lr()[0]:.4e}'})
        
        return loss_meter.avg
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        loss_meter = AverageMeter()
        velocity_mse_meter = AverageMeter()
        
        # 使用EMA模型验证
        if self.ema is not None:
            self.ema.apply_shadow()
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            target = batch['tensor'].to(self.device)
            condition = batch['condition'].to(self.device)
            layer = batch['layer_id'].to(self.device)
            
            loss, pred_velocity, target_velocity = self.model(target, condition, layer)
            loss_meter.update(loss.item(), target.size(0))
            velocity_mse = torch.mean((pred_velocity - target_velocity) ** 2).item()
            velocity_mse_meter.update(velocity_mse, target.size(0))

            if self.epoch % self.config.train.plot_every == 0:
                plot_heatmap(pred_velocity, target_velocity, filename=self.heatmap_dir / f"[V] Epoch {self.epoch} Heatmap")
                plot_histogram(pred_velocity, pred_velocity, filename=self.histogram_dir / f"[V] Epoch {self.epoch} Histogram")
                plot_gaussian(pred_velocity - target_velocity, filename=self.gaussian_dir / f"[V] Epoch {self.epoch} Diff Gaussian")
        
        # 恢复原始模型
        if self.ema is not None:
            self.ema.restore()
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("val/velocity_mse", float(velocity_mse_meter.avg), self.epoch)
        
        return loss_meter.avg
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = str(self.ckpt_dir / filename)
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            loss=self.best_val_loss,
            path=path,
            ema_model=self.ema
        )
        
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        info = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema_model=self.ema,
            device=self.device
        )
        
        self.epoch = info['epoch']
        self.global_step = info['step']
        self.best_val_loss = info['loss']
        
        self.logger.info(f"Checkpoint loaded from {path}")
        self.logger.info(f"Resuming from epoch {self.epoch}, step {self.global_step}")
