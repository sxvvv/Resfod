# models/degradation_parser.py
# 退化解析器：从输入图像估计全局权重w和空间强度图m_i

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


def sinusoidal_time_embedding(timesteps, dim):
    """正弦时间嵌入"""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DegradationParser(nn.Module):
    """
    退化解析器：估计全局权重 w ∈ R^4 和空间强度图 m_i ∈ R^{H×W}。
    
    支持：
    1. Time-dependent composition：w 依赖于时间 t
    2. 分类监督：输出 logits 用于多标签 BCE 损失
    
    输出：
        - w: (B, 4) 全局权重 [w_low, w_haze, w_rain, w_snow]
        - m: (B, 4, H, W) 空间强度图 [m_low, m_haze, m_rain, m_snow]
        - logits: (B, 4) 分类 logits（用于监督）
    """
    
    def __init__(
        self,
        in_ch=3,
        base_ch=32,
        emb_dim=128,
        time_emb_dim=64,
        use_time_dependent=True,
    ):
        super().__init__()
        self.use_time_dependent = use_time_dependent
        
        # 共享特征提取器
        self.conv1 = nn.Conv2d(in_ch, base_ch, 7, padding=3)
        self.conv2 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        
        # 全局权重预测（全局池化 + MLP）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 时间嵌入（如果使用 time-dependent）
        if use_time_dependent:
            self.time_emb_dim = time_emb_dim
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, emb_dim // 2),
                nn.ReLU(),
                nn.Linear(emb_dim // 2, emb_dim // 2),
            )
            # 每个原子退化学一对参数 (a_i, b_i)
            self.time_params = nn.Linear(emb_dim + emb_dim // 2, 4 * 2)  # 4个原子，每个2个参数
            self.global_mlp = nn.Sequential(
                nn.Linear(base_ch * 4, emb_dim),
                nn.ReLU(),
            )
        else:
            self.global_mlp = nn.Sequential(
                nn.Linear(base_ch * 4, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, 4),
            )
        
        # 分类 logits（用于监督）
        self.cls_mlp = nn.Sequential(
            nn.Linear(base_ch * 4, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 4),
        )
        
        # 空间强度图预测（上采样 + 卷积）
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_ch, base_ch, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(base_ch, 4, 3, padding=1),
        )
    
    def forward(
        self, 
        y: torch.Tensor, 
        t: Optional[torch.Tensor] = None,
        present: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            y: (B, 3, H, W) 退化输入图像
            t: (B,) 时间 [0, 1]（可选，用于 time-dependent composition）
            present: (B, 4) 因子存在标签（可选，用于mask m和w）
        
        Returns:
            w: (B, 4) 全局权重（sigmoid输出，支持多因子叠加）
            m: (B, 4, H, W) 空间强度图（sigmoid输出，支持多因子叠加）
            logits: (B, 4) 分类 logits（用于监督）
        """
        # 特征提取
        h = F.relu(self.conv1(y))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))  # (B, base_ch*4, H/4, W/4)
        
        # 全局特征
        h_global = self.global_pool(h)  # (B, base_ch*4, 1, 1)
        h_global = h_global.view(h_global.shape[0], -1)  # (B, base_ch*4)
        
        # 分类 logits（用于监督）
        logits = self.cls_mlp(h_global)  # (B, 4)
        
        # 全局权重（time-dependent 或 static）
        if self.use_time_dependent and t is not None:
            # Time-dependent composition
            h_feat = self.global_mlp(h_global)  # (B, emb_dim)
            
            # 时间嵌入
            t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)
            t_feat = self.time_mlp(t_emb)  # (B, emb_dim//2)
            
            # 拼接特征和时间
            h_concat = torch.cat([h_feat, t_feat], dim=1)  # (B, emb_dim + emb_dim//2)
            
            # 预测每个原子的参数 (a_i, b_i)
            params = self.time_params(h_concat)  # (B, 8)
            params = params.view(-1, 4, 2)  # (B, 4, 2)
            a = params[:, :, 0]  # (B, 4)
            b = params[:, :, 1]  # (B, 4)
            
            # 时间变换：τ(t) = t（也可以尝试其他变换）
            tau = t.view(-1, 1)  # (B, 1)
            
            # w_i(t) = σ(a_i + b_i * τ(t))，每个因子独立强度
            w_logits = a + b * tau  # (B, 4)
            w = torch.sigmoid(w_logits)  
        else:
            # Static composition（只看 y，不看 t）
            w_logits = self.global_mlp(h_global)  # (B, 4)
            w = torch.sigmoid(w_logits)  
        
        # 空间强度图
        m_logits = self.spatial_conv(h)  # (B, 4, H, W)
        m = torch.sigmoid(m_logits)  
        
        if present is not None:
            # w: 只保留存在的因子
            w = w * present
            # m: 用present mask，每个通道独立
            m = m * present[:, :, None, None]
            # 可选：温和归一化（避免幅度爆炸）
            # m = m / (m.mean((2, 3), keepdim=True) + 1e-8)
        
        return w, m, logits

