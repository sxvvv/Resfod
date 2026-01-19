# models/traj_cfm_net.py
# Traj-CFM: 轨迹感知的组合流匹配网络
# 
# 核心：组合速度场 v_θ(x,t,c) = v_share(x,t) + Σ α_i(t,w) * v_i(x,t,m_i)
# 其中 v_i 是4个专家速度场（low/haze/rain/snow）

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# 定义时间嵌入函数
def sinusoidal_time_embedding(timesteps, dim):
    """
    正弦时间嵌入。
    
    Args:
        timesteps: (B,) 时间步
        dim: 嵌入维度
    
    Returns:
        (B, dim) 时间嵌入
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:  # zero pad
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# 定义ResBlock
class ResBlock(nn.Module):
    """残差块，带时间嵌入"""
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.res_conv(x)


# 定义Downsample
class Downsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


# 定义Upsample
class Upsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


from models.degradation_parser import DegradationParser


class AdapterLayer(nn.Module):
    """
    轻量Adapter层，用于专家速度场（参数高效）。
    
    结构：down-proj -> activation -> up-proj
    """
    def __init__(self, in_dim, adapter_dim=64):
        super().__init__()
        self.down_proj = nn.Linear(in_dim, adapter_dim)
        self.activation = nn.SiLU()
        self.up_proj = nn.Linear(adapter_dim, in_dim)
    
    def forward(self, x):
        return self.up_proj(self.activation(self.down_proj(x)))


class ExpertVelocityField(nn.Module):
    """
    单个专家速度场 v_i(x, t, m_i)。
    
    使用轻量卷积层实现参数高效。
    """
    def __init__(
        self,
        base_ch=64,
        ch_mult=(1, 2, 4, 4),
        emb_dim=256,
        adapter_dim=64,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.base_ch = base_ch
        
        # 时间嵌入Adapter
        self.time_adapter = AdapterLayer(emb_dim, adapter_dim)
        
        # 使用1x1卷积作为轻量Adapter
        self.spatial_adapter = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch, 1),
        )
    
    def apply_adapters(self, h_final, temb):
        """
        应用Adapter到full-res特征。
        
        Args:
            h_final: (B, base_ch, H, W) full-res feature from SharedBackbone
            temb: (B, emb_dim) 时间嵌入
        
        Returns:
            adapted_feature: (B, base_ch, H, W) adapted特征
            adapted_temb: (B, emb_dim) adapted time embedding
        """
        adapted_temb = self.time_adapter(temb)
        adapted_feature = self.spatial_adapter(h_final)
        return adapted_feature, adapted_temb


class SharedBackbone(nn.Module):
    """
    共享骨干网络（U-Net结构）。
    """
    def __init__(
        self,
        in_ch=6,  # x_t(3) + y(3)
        out_ch=3,
        base_ch=64,
        ch_mult=(1, 2, 4, 4),
        emb_dim=256,
        use_depth=False,
        use_depth_anything=False,
        depth_anything_extractor=None,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        
        # 输入卷积
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        
        # 时间嵌入投影
        self.time_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
        )
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chs = [base_ch]
        for i, mult in enumerate(ch_mult):
            ch_out = base_ch * mult
            self.down_blocks.append(nn.ModuleList([
                ResBlock(chs[-1], ch_out, emb_dim),
                ResBlock(ch_out, ch_out, emb_dim),
            ]))
            if i < len(ch_mult) - 1:
                self.downsamples.append(Downsample(ch_out))
            chs.append(ch_out)
        
        # Middle blocks
        self.mid1 = ResBlock(chs[-1], chs[-1], emb_dim)
        self.mid2 = ResBlock(chs[-1], chs[-1], emb_dim)
        
        # Up blocks
        self.up_blocks = nn.ModuleList()
        # up blocks的通道数应该从middle开始，然后逐步减少
        up_ch = chs[-1]  # 从middle的通道数开始
        for i, mult in enumerate(reversed(ch_mult)):
            ch_out = base_ch * mult
            # skip connection的通道数（从对应的down block）
            skip_idx = len(ch_mult) - 1 - i  # 对应的down block索引
            # skip_idx是down block索引，down block的输出在chs[skip_idx+1]
            skip_ch = chs[skip_idx + 1] if skip_idx + 1 < len(chs) else chs[-1]
            concat_ch = up_ch + skip_ch
            self.up_blocks.append(nn.ModuleList([
                ResBlock(concat_ch, ch_out, emb_dim),
                ResBlock(ch_out, ch_out, emb_dim),
            ]))
            if i < len(ch_mult) - 1:
                self.up_blocks.append(Upsample(ch_out))
            up_ch = ch_out  # 更新为下一个up block的输入
        
        # 输出卷积
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, 3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        """
        Args:
            x: (B, 3, H, W) 当前状态 x_t
            y: (B, 3, H, W) 退化输入
            t: (B,) 时间
            temb: (B, emb_dim) 时间嵌入（可选，如果None则从t计算）
            depth: (B, 1, H, W) 深度图（可选，当前方案不使用）
            return_features: 是否返回中间特征（用于专家Adapter）
        
        Returns:
            u: (B, 3, H, W) 速度场
            features: (可选) List of intermediate features
        """
        # 时间嵌入
        if temb is None:
            temb = sinusoidal_time_embedding(t, self.emb_dim)
        temb = self.time_proj(temb)
        
        # 输入：拼接 x_t 和 y
        h = torch.cat([x, y], dim=1)  # (B, 6, H, W)
        h = self.in_conv(h)
        
        # 存储中间特征（用于专家Adapter）
        features = []
        
        # Down blocks
        skips = []
        for li, block in enumerate(self.down_blocks):
            h = block[0](h, temb)
            h = block[1](h, temb)
            features.append(h)  # 保存特征用于Adapter
            skips.append(h)
            if li != len(self.down_blocks) - 1:
                h = self.downsamples[li](h)
        
        # Middle
        h = self.mid1(h, temb)
        h = self.mid2(h, temb)
        
        # Up blocks
        skip_idx = len(skips) - 1  # 从最后一个skip开始
        for li, block in enumerate(self.up_blocks):
            if isinstance(block, Upsample):
                h = block(h)
            else:
                h = torch.cat([h, skips[skip_idx]], dim=1)
                h = block[0](h, temb)
                h = block[1](h, temb)
                skip_idx -= 1
        
        # Output
        h_final = h  # (B, base_ch, H, W) before out_conv - full-res feature
        u = self.out_conv(h_final)
        
        if return_features:
            return u, features, temb, h_final
        return u


class TrajCFMNet(nn.Module):
    """
    Traj-CFM 组合速度场网络。
    
    速度场：v_θ(x,t,c) = v_share(x,t) + Σ α_i(t,w) * v_i(x,t,m_i)
    
    当前方案：
    - use_prompt_pool=False（不使用Prompt Pool）
    - use_depth=False（不使用深度信息）
    - use_time_dependent=True（使用时间依赖的退化解析）
    """
    def __init__(
        self,
        in_ch=6,
        out_ch=3,
        base_ch=64,
        ch_mult=(1, 2, 4, 4),
        emb_dim=256,
        use_depth=False,
        use_parser=True,
        adapter_dim=64,
        use_prompt_pool=False,
        prompt_dim=256,
        use_depth_anything=False,
        depth_anything_extractor=None,
    ):
        super().__init__()
        self.use_parser = use_parser
        self.base_ch = base_ch
        self.ch_mult = ch_mult
        self.emb_dim = emb_dim
        
        # 退化解析器
        if use_parser:
            self.parser = DegradationParser(in_ch=3, base_ch=32, emb_dim=128)
        
        # 共享速度场 v_share
        self.shared_velocity = SharedBackbone(
            in_ch=in_ch,
            out_ch=out_ch,
            base_ch=base_ch,
            ch_mult=ch_mult,
            emb_dim=emb_dim,
            use_depth=False,  # 当前方案不使用depth
            use_depth_anything=False,
            depth_anything_extractor=None,
        )
        
        # 4个专家速度场（low/haze/rain/snow）
        self.experts = nn.ModuleList([
            ExpertVelocityField(base_ch=base_ch, ch_mult=ch_mult, emb_dim=emb_dim, adapter_dim=adapter_dim)
            for _ in range(4)
        ])
        
        # 使用base_ch（full-res feature）而不是final_ch
        self.expert_outputs = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, base_ch),
                nn.SiLU(),
                nn.Conv2d(base_ch, out_ch, 3, padding=1),
            ) for _ in range(4)
        ])
        
        # Time-conditioned gating with spatial stats
        # α = softmax(MLP([temb, w_norm, m_mean]))
        # 输入维度：emb_dim (temb) + 4 (w_norm) + 4 (m_mean) = emb_dim + 8
        alpha_input_dim = emb_dim + 8
        self.alpha_mlp = nn.Sequential(
            nn.Linear(alpha_input_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 4),
        )
    
    def forward(
        self,
        x_t: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        deg_name: Optional[str] = None,
        w: Optional[torch.Tensor] = None,
        m: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        return_alpha: bool = False,
        return_prompts: bool = False,
        return_experts: bool = False,
        return_aux: bool = False,
        weather_labels: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x_t: (B, 3, H, W) 当前状态
            y: (B, 3, H, W) 退化输入
            t: (B,) 时间 [0,1]
            deg_name: (str) 退化名称（可选，用于解析器）
            w: (B, 4) 全局权重（可选，如果None则从解析器获取）
            m: (B, 4, H, W) 空间强度图（可选，如果None则从解析器获取）
            depth: (B, 1, H, W) 深度图（可选）
            return_alpha: 是否返回门控权重α（用于轨迹监督/正则）
            return_experts: 是否返回专家输出（用于分析）
            return_aux: 是否返回辅助信息字典（包含 r_share, r_exp, alpha）
        
        Returns:
            v: (B, 3, H, W) 组合速度场
            或 (v, alpha): 如果return_alpha=True
            或 (v, v_share, v_experts, alpha): 如果return_experts=True
            或 (v, {"r_share": v_share, "r_exp": r_exp, "alpha": alpha}): 如果return_aux=True
        """
        B = x_t.shape[0]
        device = x_t.device
        
        # 获取退化信息（w, m）
        # 修复：支持time-dependent和2/3输出
        if self.use_parser and (w is None or m is None):
            # time-dependent parser优先尝试传t
            try:
                out = self.parser(y, t)
            except TypeError:
                out = self.parser(y)
            
            # 支持2个或3个输出
            if isinstance(out, (tuple, list)) and len(out) == 3:
                w_pred, m_pred, _ = out
            else:
                w_pred, m_pred = out
            
            w = w if w is not None else w_pred
            m = m if m is not None else m_pred
        
        # 确保 w 和 m 存在
        if w is None:
            # 默认：均匀权重
            w = torch.ones(B, 4, device=device) * 0.25
        if m is None:
            # 默认：均匀空间强度
            H, W = x_t.shape[2:]
            m = torch.ones(B, 4, H, W, device=device) * 0.25
        
        # 共享速度场和特征（只调用一次，节省显存）
        v_share, features, temb, h_final = self.shared_velocity(
            x_t, y, t, depth=None, return_features=True
        )
        # h_final: (B, base_ch, H, W) - full-res feature for experts
        
        # Time-conditioned gating with spatial stats
        # ✅修复：w现在用sigmoid，支持多因子叠加，需要归一化（只在present内）
        # 从空间强度图m提取全局统计：m_mean = m.mean([2,3]) -> (B,4)
        m_mean = m.mean([2, 3])  # (B, 4) 空间平均强度
        
        # w归一化（只在存在的因子内归一化，支持多因子叠加）
        # 如果w是sigmoid输出，可以按需归一化；这里直接用w（因为sigmoid已经限制在[0,1]）
        w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-8)  # 归一化用于gating输入
        
        # 构建 alpha MLP 输入
        alpha_input = torch.cat([temb, w_norm, m_mean], dim=1)
        alpha_logits = self.alpha_mlp(alpha_input)
        alpha = torch.sigmoid(alpha_logits)  # (B,4) in [0,1]
        # 用w>阈值作为存在mask
        w_mask = (w > 0.1).float()  # (B, 4) 存在mask
        alpha = alpha * w_mask  # mask掉不存在的因子
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        
        # 应用每个专家的Adapter并组合
        v_experts = []
        r_experts = []  # 用于 return_aux，存储原始专家 residual（未乘 alpha 和 m）
        
        for i, expert in enumerate(self.experts):
            adapted_feature, adapted_temb = expert.apply_adapters(h_final, temb)
            
            # 使用adapted特征生成速度场（这是 residual/residual field）
            r_i = self.expert_outputs[i](adapted_feature)  # (B, 3, H, W)
            
            # 数值稳定性：裁剪异常值
            r_i = torch.clamp(r_i, min=-10.0, max=10.0)
            
            # 保存原始 r_i（用于 return_aux）
            r_experts.append(r_i)
            
            # 应用空间强度图 m_i
            m_i = m[:, i:i+1, :, :]  # (B, 1, H_m, W_m)
            # 确保mask的空间维度与r_i匹配
            if m_i.shape[2:] != r_i.shape[2:]:
                m_i = F.interpolate(m_i, size=r_i.shape[2:], mode='bilinear', align_corners=False)
            v_i = r_i * m_i  # 先乘 m
            
            v_experts.append(v_i)
        
        # 组合：v = v_share + Σ α_i * v_i
        v = v_share.clone()
        # 数值稳定性：裁剪共享速度场
        v = torch.clamp(v, min=-10.0, max=10.0)
        
        for i, v_i in enumerate(v_experts):
            # 确保v_i的空间维度与v_share匹配
            if v_i.shape[2:] != v.shape[2:]:
                v_i = F.interpolate(v_i, size=v.shape[2:], mode='bilinear', align_corners=False)
            alpha_i = alpha[:, i].view(-1, 1, 1, 1)  # (B, 1, 1, 1)
            v = v + alpha_i * v_i
        
        # 最终数值稳定性检查
        v = torch.clamp(v, min=-10.0, max=10.0)
        
        # 检查是否有NaN/Inf
        if torch.isnan(v).any() or torch.isinf(v).any():
            # 如果出现NaN/Inf，返回零速度场
            v = torch.zeros_like(v_share)
        
        # 返回值
        if return_aux:
            # 返回统一格式：v 和辅助信息字典
            # r_exp: [B,4,3,H,W] 堆叠所有专家的 residual
            r_exp = torch.stack(r_experts, dim=1)  # [B,4,3,H,W]
            # 确保 r_share 和 r_exp 的空间尺寸匹配
            if r_exp.shape[3:] != v_share.shape[2:]:
                r_exp = F.interpolate(
                    r_exp.view(-1, *r_exp.shape[2:]), 
                    size=v_share.shape[2:],
                    mode='bilinear',
                    align_corners=False
                ).view(r_exp.shape[0], r_exp.shape[1], r_exp.shape[2], *v_share.shape[2:])
            return v, {"r_share": v_share, "r_exp": r_exp, "alpha": alpha}
        elif return_experts:
            # 返回组合速度场、共享速度场、专家速度场列表、alpha
            return v, v_share, v_experts, alpha
        elif return_prompts:
            # 当前方案不支持return_prompts（use_prompt_pool=False）
            # 返回None作为占位符
            return v, alpha, None, None
        elif return_alpha:
            return v, alpha
        return v

