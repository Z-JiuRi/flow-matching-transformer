# models/flow_matching.py
"""Flow-Matching核心"""

import torch
import torch.nn as nn
from typing import Optional, Callable
import math


class FlowMatching(nn.Module):
    """Flow-Matching训练和采样"""
    
    def __init__(
        self,
        model: nn.Module,
        sigma_min: float = 0.001
    ):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
    
    def forward(
        self,
        x1: torch.Tensor,
        condition: torch.Tensor,
        layer: torch.Tensor
    ) -> torch.Tensor:
        """
        训练前向传播
        
        Args:
            x1: (B, H, W) 目标数据
            condition: (B, condition_dim) 条件向量
            layer: (B,) layer索引
        
        Returns:
            loss: 标量损失
        """
        B = x1.shape[0]
        device = x1.device
        
        # 采样时间 t ~ U(0, 1)
        t = torch.rand(B, device=device)
        
        # 采样噪声 x0 ~ N(0, I)
        x0 = torch.randn_like(x1)
        
        # 线性插值: x_t = (1 - t) * x0 + t * x1
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        
        # 目标速度: v = x1 - x0
        target_velocity = x1 - x0
        
        # 预测速度
        pred_velocity = self.model(x_t, t, condition, layer)
        
        # MSE损失
        loss = torch.mean((pred_velocity - target_velocity) ** 2)
        
        return loss, pred_velocity, target_velocity
    
    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        layer: torch.Tensor,
        H: int,
        W: int,
        num_steps: int = 50,
        solver: str = "euler",
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        采样生成
        
        Args:
            condition: (B, condition_dim) 条件向量
            layer: (B,) layer索引
            H, W: 输出尺寸
            num_steps: 采样步数
            solver: ODE求解器类型
            guidance_scale: CFG引导强度
        
        Returns:
            samples: (B, H, W) 生成的样本
        """
        B = condition.shape[0]
        device = condition.device
        
        # 初始噪声
        x = torch.randn(B, H, W, device=device)
        
        # 时间步
        dt = 1.0 / num_steps
        
        if solver == "euler":
            x = self._euler_solve(x, condition, layer, num_steps, dt, guidance_scale)
        elif solver == "heun":
            x = self._heun_solve(x, condition, layer, num_steps, dt, guidance_scale)
        elif solver == "rk4":
            x = self._rk4_solve(x, condition, layer, num_steps, dt, guidance_scale)
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        return x
    
    def _get_velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        layer: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """获取速度场 (支持CFG)"""
        if guidance_scale == 1.0:
            return self.model(x, t, condition, layer)
        
        # Classifier-Free Guidance
        # 需要无条件预测 (这里简化处理，使用零条件)
        B = x.shape[0]
        
        # 有条件预测
        v_cond = self.model(x, t, condition, layer)
        
        # 无条件预测 (使用零向量作为条件)
        zero_cond = torch.zeros_like(condition)
        v_uncond = self.model(x, t, zero_cond, layer)
        
        # CFG
        v = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        return v
    
    def _euler_solve(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        layer: torch.Tensor,
        num_steps: int,
        dt: float,
        guidance_scale: float
    ) -> torch.Tensor:
        """Euler求解器"""
        B = x.shape[0]
        device = x.device
        
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self._get_velocity(x, t, condition, layer, guidance_scale)
            x = x + v * dt
        
        return x
    
    def _heun_solve(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        layer: torch.Tensor,
        num_steps: int,
        dt: float,
        guidance_scale: float
    ) -> torch.Tensor:
        """Heun求解器 (改进的Euler)"""
        B = x.shape[0]
        device = x.device
        
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=device)
            t_next = torch.full((B,), (i + 1) * dt, device=device)
            
            # 预测步
            v1 = self._get_velocity(x, t, condition, layer, guidance_scale)
            x_pred = x + v1 * dt
            
            # 校正步
            v2 = self._get_velocity(x_pred, t_next, condition, layer, guidance_scale)
            x = x + 0.5 * (v1 + v2) * dt
        
        return x
    
    def _rk4_solve(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        layer: torch.Tensor,
        num_steps: int,
        dt: float,
        guidance_scale: float
    ) -> torch.Tensor:
        """RK4求解器"""
        B = x.shape[0]
        device = x.device
        
        for i in range(num_steps):
            t = i * dt
            
            t1 = torch.full((B,), t, device=device)
            t2 = torch.full((B,), t + 0.5 * dt, device=device)
            t3 = torch.full((B,), t + dt, device=device)
            
            k1 = self._get_velocity(x, t1, condition, layer, guidance_scale)
            k2 = self._get_velocity(x + 0.5 * dt * k1, t2, condition, layer, guidance_scale)
            k3 = self._get_velocity(x + 0.5 * dt * k2, t2, condition, layer, guidance_scale)
            k4 = self._get_velocity(x + dt * k3, t3, condition, layer, guidance_scale)
            
            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return x
