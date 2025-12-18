# core/inferencer.py
"""推理器"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict
import logging

from models import build_model
from utils.helpers import load_checkpoint
from utils.helpers import set_seed, setup_logging


class Inferencer:
    """推理器"""
    
    def __init__(
        self,
        model: nn.Module,
        config,
        device: torch.device,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        self.model.to(device)
        self.model.eval()
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        load_checkpoint(
            path=path,
            model=self.model,
            device=self.device
        )
        self.logger.info(f"Model loaded from {path}")

    @classmethod
    def from_config(cls, config, logger: Optional[logging.Logger] = None) -> "Inferencer":
        set_seed(int(config.data.seed))
        device = torch.device(config.data.device if torch.cuda.is_available() else "cpu")
        model = build_model(config).to(device)
        return cls(model=model, config=config, device=device, logger=logger)

    @classmethod
    def run(
        cls,
        config,
        checkpoint_path: Optional[str],
        condition_path: str,
        layer: int,
        output_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> torch.Tensor:
        exp_dir = Path(str(config.train.exp_dir))
        setup_logging(exp_dir / "logs")

        logger = logger or logging.getLogger(__name__)
        logger.info("=" * 50)
        logger.info("Inference")
        logger.info("=" * 50)

        checkpoint_path = checkpoint_path or str(exp_dir / "ckpts" / "best.pt")
        output_path = output_path or str(exp_dir / "results" / f"generated_linear{layer + 1}.pth")

        inferencer = cls.from_config(config=config, logger=logger)
        inferencer.load_checkpoint(checkpoint_path)
        return inferencer.generate_from_file(
            condition_path=condition_path,
            layer=layer,
            output_path=output_path,
        )
    
    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        layer: Union[int, torch.Tensor],
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
        guidance_scale: Optional[float] = None,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        生成样本
        
        Args:
            condition: (N, condition_dim) 或 (condition_dim,) 条件向量
            layer: 整数或 (N,) tensor，layer索引
            num_steps: 采样步数
            solver: ODE求解器
            guidance_scale: CFG引导强度
            batch_size: 批处理大小（用于大量生成）
        
        Returns:
            samples: (N, H, W) 生成的样本
        """
        # 默认参数
        num_steps = num_steps or self.config.inference.num_steps
        solver = solver or self.config.model.flow.solver
        guidance_scale = guidance_scale or self.config.inference.guidance_scale
        batch_size = batch_size or self.config.inference.batch_size
        
        # 处理输入
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)
        
        condition = condition.to(self.device)
        N = condition.size(0)
        
        # 处理layer
        if isinstance(layer, int):
            layer = torch.full((N,), layer, dtype=torch.long, device=self.device)
        else:
            layer = layer.to(self.device)
        
        # 分批生成
        all_samples = []
        
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch_cond = condition[i:end_idx]
            batch_layer = layer[i:end_idx]
            
            samples = self.model.sample(
                condition=batch_cond,
                layer=batch_layer,
                H=self.config.data.H,
                W=self.config.data.W,
                num_steps=num_steps,
                solver=solver,
                guidance_scale=guidance_scale
            )
            
            all_samples.append(samples)
        
        return torch.cat(all_samples, dim=0)
    
    @torch.no_grad()
    def generate_from_file(
        self,
        condition_path: str,
        layer: int,
        output_path: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        从文件加载条件并生成
        
        Args:
            condition_path: 条件向量文件路径 (.pt)
            layer: layer索引 (0或1)
            output_path: 输出路径（可选）
            **kwargs: 传递给generate的其他参数
        
        Returns:
            sample: (H, W) 生成的样本
        """
        # 加载条件
        condition = torch.load(condition_path, map_location='cpu')
        condition = condition.flatten().float()
        
        # 生成
        sample = self.generate(
            condition=condition,
            layer=layer,
            **kwargs
        )
        
        sample = sample.squeeze(0)  # (H, W)
        
        # 保存
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            torch.save(sample.cpu(), output_path)
            self.logger.info(f"Generated sample saved to {output_path}")
        
        return sample
    
    @torch.no_grad()
    def generate_batch_from_files(
        self,
        condition_paths: List[str],
        layers: List[int],
        output_dir: Optional[str] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        批量从文件生成
        
        Args:
            condition_paths: 条件向量文件路径列表
            layers: layer索引列表
            output_dir: 输出目录（可选）
            **kwargs: 传递给generate的其他参数
        
        Returns:
            samples: 生成的样本列表
        """
        assert len(condition_paths) == len(layers), "条件数量和layer数量必须相同"
        
        # 加载所有条件
        conditions = []
        for path in condition_paths:
            cond = torch.load(path, map_location='cpu')
            conditions.append(cond.flatten().float())
        
        conditions = torch.stack(conditions)
        layers_tensor = torch.tensor(layers, dtype=torch.long)
        
        # 生成
        samples = self.generate(
            condition=conditions,
            layer=layers_tensor,
            **kwargs
        )
        
        # 保存
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            for i, (path, layer) in enumerate(zip(condition_paths, layers)):
                basename = os.path.basename(path).replace('.pt', '')
                output_path = os.path.join(output_dir, f"{basename}_linear{layer + 1}_gen.pth")
                torch.save(samples[i].cpu(), output_path)
            self.logger.info(f"Generated samples saved to {output_dir}")
        
        return [samples[i] for i in range(len(samples))]
    
    @torch.no_grad()
    def interpolate(
        self,
        condition1: torch.Tensor,
        condition2: torch.Tensor,
        layer: int,
        num_interpolations: int = 10,
        **kwargs
    ) -> torch.Tensor:
        """
        条件插值生成
        
        Args:
            condition1: 起始条件
            condition2: 结束条件
            layer: layer索引
            num_interpolations: 插值数量
            **kwargs: 传递给generate的其他参数
        
        Returns:
            samples: (num_interpolations, H, W) 插值生成的样本
        """
        condition1 = condition1.flatten().float().to(self.device)
        condition2 = condition2.flatten().float().to(self.device)
        
        # 线性插值
        alphas = torch.linspace(0, 1, num_interpolations, device=self.device)
        conditions = []
        for alpha in alphas:
            cond = (1 - alpha) * condition1 + alpha * condition2
            conditions.append(cond)
        
        conditions = torch.stack(conditions)
        layers = torch.full((num_interpolations,), layer, dtype=torch.long, device=self.device)
        
        # 生成
        samples = self.generate(
            condition=conditions,
            layer=layers,
            **kwargs
        )
        
        return samples
    
    @torch.no_grad()
    def generate_variations(
        self,
        condition: torch.Tensor,
        layer: int,
        num_variations: int = 5,
        noise_scale: float = 0.1,
        **kwargs
    ) -> torch.Tensor:
        """
        生成变体（通过添加噪声到条件）
        
        Args:
            condition: 基础条件
            layer: layer索引
            num_variations: 变体数量
            noise_scale: 噪声强度
            **kwargs: 传递给generate的其他参数
        
        Returns:
            samples: (num_variations, H, W) 生成的变体
        """
        condition = condition.flatten().float().to(self.device)
        
        # 添加噪声生成变体条件
        conditions = []
        for _ in range(num_variations):
            noise = torch.randn_like(condition) * noise_scale
            noisy_cond = condition + noise
            conditions.append(noisy_cond)
        
        conditions = torch.stack(conditions)
        layers = torch.full((num_variations,), layer, dtype=torch.long, device=self.device)
        
        # 生成
        samples = self.generate(
            condition=conditions,
            layer=layers,
            **kwargs
        )
        
        return samples
    
    @torch.no_grad()
    def compare_solvers(
        self,
        condition: torch.Tensor,
        layer: int,
        solvers: List[str] = ["euler", "heun", "rk4"],
        num_steps_list: List[int] = [10, 25, 50, 100],
        **kwargs
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        比较不同求解器和步数的结果
        
        Args:
            condition: 条件向量
            layer: layer索引
            solvers: 求解器列表
            num_steps_list: 步数列表
            **kwargs: 其他参数
        
        Returns:
            results: {solver: {num_steps: sample}}
        """
        condition = condition.flatten().float()
        results = {}
        
        for solver in solvers:
            results[solver] = {}
            for num_steps in num_steps_list:
                sample = self.generate(
                    condition=condition,
                    layer=layer,
                    solver=solver,
                    num_steps=num_steps,
                    **kwargs
                )
                results[solver][num_steps] = sample.squeeze(0)
                self.logger.info(f"Generated with solver={solver}, steps={num_steps}")
        
        return results
