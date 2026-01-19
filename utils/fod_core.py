# utils/fod_core.py

import math
import numpy as np
import torch
import enum


# ============================================================================
# Schedule 函数（支持多种schedule类型）
# ============================================================================

def get_jsd_schedule(num_diffusion_timesteps, scale=1.5):
    """JSD schedule"""
    betas = 1. / np.linspace(
        num_diffusion_timesteps, 1., num_diffusion_timesteps, dtype=np.float64
    )
    return betas ** scale

def get_linear_schedule(num_diffusion_timesteps):
    """Linear schedule"""
    scale = 1000 / num_diffusion_timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

def get_cosine_schedule(num_diffusion_timesteps, s=0.008):
    """
    Cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = num_diffusion_timesteps + 1
    t = np.linspace(0, num_diffusion_timesteps, steps) / num_diffusion_timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return betas_clipped

def get_named_schedule(schedule_name, timesteps):
    """获取指定名称的schedule"""
    if schedule_name == 'jsd':
        schedule = get_jsd_schedule(timesteps)
    elif schedule_name == 'linear':
        schedule = get_linear_schedule(timesteps)
    elif schedule_name == 'cosine':
        schedule = get_cosine_schedule(timesteps)
    elif schedule_name == 'const':
        schedule = np.ones(timesteps)
    elif schedule_name == 'none':
        schedule = np.zeros(timesteps)
    else:
        print(f'Warning: Unknown schedule "{schedule_name}", using linear')
        schedule = get_linear_schedule(timesteps)
    return schedule


# ============================================================================
# 辅助函数
# ============================================================================

def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)


# ============================================================================
# ModelType 枚举
# ============================================================================

class ModelType(enum.Enum):
    """Which type of output the model predicts."""
    FINAL_X = enum.auto()  # the model predicts x_T
    FLOW = enum.auto()  # the model predicts x_T - x_0
    SFLOW = enum.auto()  # the model predicts x_T - x_t


# ============================================================================
# FoDiffusion 类（核心实现）
# ============================================================================

class FoDiffusion:
    """
    Flow of Diffusion 核心类，实现严格的数学采样。
    
    支持SDE/ODE，多种schedule，以及EM/MC/NMC采样方式。
    """
    
    def __init__(self, thetas, sigma2s, model_type, sigmas_scale=1.):
        """
        Args:
            thetas: theta schedule (drift coefficient)
            sigma2s: sigma^2 schedule (diffusion coefficient)
            model_type: ModelType枚举，指定模型输出类型
            sigmas_scale: sigma的缩放因子
        """
        self.model_type = model_type

        thetas = np.array(thetas, dtype=np.float64)
        sigma2s = np.array(sigma2s, dtype=np.float64)
        self.thetas = np.append(0.0, thetas)
        if np.sum(sigma2s) > 0:
            sigma2s = sigmas_scale * sigma2s / np.sum(sigma2s)  # normalize sigma squares
        self.sigma2s = np.append(0.0, sigma2s)
        expo_mean = -(self.thetas + 0.5 * self.sigma2s)

        self.thetas_cumsum = np.cumsum(self.thetas)
        self.sigma2s_cumsum = np.cumsum(self.sigma2s)
        expo_mean_cumsum = -(self.thetas_cumsum + 0.5 * self.sigma2s_cumsum)

        self.dt = math.log(0.001) / expo_mean_cumsum[-1]

        #### sqrt terms  ####
        self.expo_mean = expo_mean * self.dt
        self.sqrt_expo_variance = np.sqrt(self.sigma2s * self.dt)
        self.expo_mean_cumsum = expo_mean_cumsum * self.dt
        self.sqrt_expo_variance_cumsum = np.sqrt(self.sigma2s_cumsum * self.dt)

        self.num_timesteps = int(thetas.shape[0])

    def expo_normal(self, t, noise=None):
        """Exponential normal for single step"""
        assert noise is not None
        return torch.exp(
            _extract_into_tensor(self.expo_mean, t, noise.shape) + 
            _extract_into_tensor(self.sqrt_expo_variance, t, noise.shape) * noise
        )

    def expo_normal_cumsum(self, t, noise=None):
        """Exponential normal cumulative sum (for NMC sampling)"""
        assert noise is not None
        return torch.exp(
            _extract_into_tensor(self.expo_mean_cumsum, t, noise.shape) + 
            _extract_into_tensor(self.sqrt_expo_variance_cumsum, t, noise.shape) * noise
        )

    def expo_normal_transition(self, s, t, noise=None):
        """Exponential normal transition from s to t"""
        assert noise is not None
        expo_mean_cumsum = _extract_into_tensor(self.expo_mean_cumsum, t, noise.shape) \
                            - _extract_into_tensor(self.expo_mean_cumsum, s, noise.shape)
        expo_variance_cumsum = _extract_into_tensor(self.sigma2s_cumsum * self.dt, t, noise.shape) \
                            - _extract_into_tensor(self.sigma2s_cumsum * self.dt, s, noise.shape)

        return torch.exp(expo_mean_cumsum + torch.sqrt(expo_variance_cumsum) * noise)

    def sde_step(self, x, x_final, t, noise):
        """SDE step (Euler-Maruyama)"""
        drift = _extract_into_tensor(self.thetas, t, x.shape) * (x_final - x)
        diffusion = _extract_into_tensor(np.sqrt(self.sigma2s), t, x.shape) * (x - x_final)
        return x + drift * self.dt + diffusion * math.sqrt(self.dt) * noise

    def forward_step(
        self, 
        model, 
        x, x_start, 
        t, t_next, 
        sample_type="EM", 
        clip_denoised=True, 
        model_kwargs=None):
        """
        Single forward step.
        
        Args:
            model: 模型函数，接受 (x, t, x_start, **model_kwargs)
            x: 当前状态
            x_start: 起始状态（通常是退化输入y）
            t: 当前时间（整数索引）
            t_next: 下一步时间（整数索引）
            sample_type: "EM", "MC", or "NMC"
            clip_denoised: 是否clip到[-1, 1]
            model_kwargs: 传递给模型的额外参数
        
        Returns:
            x_next: 下一步状态
        """
        if model_kwargs is None:
            model_kwargs = {}

        model_output = model(x, t, x_start, **model_kwargs)

        if self.model_type == ModelType.FINAL_X:
            x_final = model_output
        elif self.model_type == ModelType.FLOW:
            x_final = x_start + model_output
        elif self.model_type == ModelType.SFLOW:
            x_final = x + model_output

        if clip_denoised:
            x_final = x_final.clamp(-1, 1)
        
        noise = torch.randn_like(x)
        if sample_type == "EM":
            x = self.sde_step(x, x_final, t_next, noise)
        elif sample_type == "MC":
            x = (x - x_final) * self.expo_normal_transition(t, t_next, noise) + x_final
        elif sample_type == "NMC":
            x = (x - x_final) * self.expo_normal_cumsum(t_next, noise) + x_final
        return x

    def forward_loop(
        self, 
        model, 
        x_start, 
        num_steps=-1,
        sample_type="EM",
        clip_denoised=True, 
        model_kwargs=None, 
        device=None, 
        progress=False):
        """
        Forward process to get x(T) from x(0).
        
        Args:
            model: 模型函数
            x_start: 起始状态
            num_steps: 步数（-1表示使用self.num_timesteps）
            sample_type: "EM", "MC", or "NMC"
            clip_denoised: 是否clip
            model_kwargs: 模型额外参数
            device: 设备
            progress: 是否显示进度条
        
        Returns:
            x_T: 最终状态
        """
        assert x_start is not None
        if device is None:
            device = next(iter(model_kwargs.values())).device if model_kwargs else x_start.device

        if num_steps <= 0:
            num_steps = self.num_timesteps

        img = x_start
        indices = np.linspace(0, self.num_timesteps, num_steps + 1).astype(int)
        times = np.copy(indices)

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(enumerate(indices[:-1]))
        else:
            indices = enumerate(indices[:-1])

        for i, idx in indices:
            t = torch.tensor([idx] * x_start.shape[0], device=device)
            t_next = torch.tensor([times[i+1]] * x_start.shape[0], device=device)
            with torch.no_grad():
                img = self.forward_step(
                    model, img, x_start, t, t_next, sample_type, 
                    clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        return img

    def training_losses(self, model, x_start, x_final, t, model_kwargs=None, noise=None):
        """
        Compute training losses.
        
        Args:
            model: 模型函数
            x_start: 起始状态（通常是退化输入y）
            x_final: 目标状态（通常是清晰图x）
            t: 时间索引（整数）
            model_kwargs: 模型额外参数
            noise: 噪声（如果None则随机生成）
        
        Returns:
            terms: 包含loss的字典
        """
        if model_kwargs is None:
            model_kwargs = {}

        if noise is None:
            noise = torch.randn_like(x_final)

        # generate states using expo_normal_cumsum
        x_t = (x_start - x_final) * self.expo_normal_cumsum(t, noise) + x_final

        # model prediction
        model_output = model(x_t, t, x_start, **model_kwargs)

        target = {
            ModelType.FINAL_X: x_final,
            ModelType.FLOW: x_final - x_start,
            ModelType.SFLOW: x_final - x_t,
        }[self.model_type]

        terms = {}
        terms["loss"] = mean_flat(torch.abs(target - model_output))

        return terms


# ============================================================================
# 向后兼容的接口（保持原有API）
# ============================================================================

class FoDSchedule:
    """
    Flow of Diffusion Schedule
    """
    
    def __init__(self, T=100, delta=0.001, device="cuda", 
                 theta_schedule="cosine", sigma_schedule="linear",
                 diffusion_type="sde", prediction="sflow"):
        """
        Args:
            T: 离散时间步数
            delta: 最小噪声水平（用于向后兼容，实际使用schedule）
            device: 设备
            theta_schedule: theta的schedule类型（"cosine", "linear", "jsd", "const", "none"）
            sigma_schedule: sigma的schedule类型
            diffusion_type: "sde" or "ode"（ode时sigma_schedule会被设为"none"）
            prediction: "final", "flow", or "sflow"（对应FINAL_X, FLOW, SFLOW）
        """
        self.T = T
        self.delta = delta
        self.device = device
        
        # 构建时间调度
        self.timesteps = torch.linspace(0, 1, T + 1, device=device)
        
        # 创建FoDiffusion对象
        if diffusion_type == "ode":
            sigma_schedule = "none"
        
        thetas = get_named_schedule(theta_schedule, T)
        sigma2s = get_named_schedule(sigma_schedule, T)
        
        if prediction == "final":
            model_type = ModelType.FINAL_X
        elif prediction == "flow":
            model_type = ModelType.FLOW
        elif prediction == "sflow":
            model_type = ModelType.SFLOW
        else:
            print(f"Warning: Unknown prediction type '{prediction}', using 'sflow'")
            model_type = ModelType.SFLOW
        
        # 创建内部FoDiffusion对象
        self._fod = FoDiffusion(
            thetas=thetas,
            sigma2s=sigma2s,
            model_type=model_type,
        )
        
        # 向后兼容：保持sigma属性
        self.sigma = self.delta + (1.0 - self.delta) * self.timesteps
    
    def get_sigma(self, t):
        """获取时间 t 的噪声水平（向后兼容方法）"""
        if isinstance(t, torch.Tensor):
            t = t.clamp(0.0, 1.0)
        else:
            t = max(0.0, min(1.0, t))
        return self.delta + (1.0 - self.delta) * t


def fod_analytic_sample(x0, mu, t0, t1, schedule, epsilon):
    """
    FoD 解析采样：从 x0 到 x1 的路径（使用更严格的expo_normal_cumsum）。
    
    Args:
        x0: (B, C, H, W) 初始状态（通常是退化输入 y）
        mu: (B, C, H, W) 目标状态（通常是清晰图 x）
        t0: (B,) 初始时间（归一化[0,1]）
        t1: (B,) 结束时间（归一化[0,1]）
        schedule: FoDSchedule 对象
        epsilon: (B, C, H, W) 噪声样本
    
    Returns:
        xt: (B, C, H, W) 时间 t1 的状态
    """
    B = x0.shape[0]
    device = x0.device
    
    t0_idx = (t0 * schedule.T).long().clamp(0, schedule.T)
    t1_idx = (t1 * schedule.T).long().clamp(0, schedule.T)
    
    fod = schedule._fod
    expo = fod.expo_normal_cumsum(t1_idx, epsilon)
    
    xt = (x0 - mu) * expo + mu
    
    return xt


def fod_nmc_step(
    model,
    xt,
    y,
    t,
    dt,
    schedule,
    mu=None,
    w=None,
    m=None,
    deg_name=None,
    inject_noise=False,
    last_step_noise=0.0,
):
    """
    FoD NMC 单步更新（用于训练时的 unroll）。
    
    使用更严格的FoDiffusion.forward_step实现。
    
    Args:
        model: TrajCFMNet 模型
        xt: (B, 3, H, W) 当前状态
        y: (B, 3, H, W) 退化输入
        t: (B,) 当前时间（归一化[0,1]）
        dt: 时间步长（归一化）
        schedule: FoDSchedule 对象
        mu: (B, 3, H, W) 目标状态（可选，如果 None 则从模型预测）
        w: (B, 4) 全局权重（可选）
        m: (B, 4, H, W) 空间强度图（可选）
        deg_name: 退化名称（可选，可以是list或str）
        inject_noise: 是否注入噪声（已废弃，使用NMC采样方式）
        last_step_noise: 最后一步的噪声水平（已废弃）
    
    Returns:
        xt_next: (B, 3, H, W) 下一步状态
    """
    B = xt.shape[0]
    device = xt.device
    
    t_idx = (t * schedule.T).long().clamp(0, schedule.T)
    t_next_idx = ((t + dt) * schedule.T).long().clamp(0, schedule.T)
    
    def model_wrapper(x, t_int, x_start, **kwargs):
        # 将整数索引转换回归一化时间
        t_norm = t_int.float() / schedule.T
        # 确保t_norm是(B,)形状
        if t_norm.dim() == 0:
            t_norm = t_norm.unsqueeze(0).expand(B)
        elif t_norm.shape[0] != B:
            t_norm = t_norm[:B]
        # 调用原始模型（模型期望归一化时间）
        r_pred = model(x, x_start, t_norm, deg_name=deg_name, w=w, m=m)
        return r_pred  # 返回残差（FLOW类型）
    
    # 使用FoDiffusion的forward_step（NMC采样）
    fod = schedule._fod
    xt_next = fod.forward_step(
        model=model_wrapper,
        x=xt,
        x_start=y,
        t=t_idx,
        t_next=t_next_idx,
        sample_type="MC",  # ✅修复：使用MC而不是NMC，确保多步采样正确（transition而不是cumsum）
        clip_denoised=True,
        model_kwargs={},
    )
    
    return xt_next


def fod_nmc_sample(
    model,
    y,
    n_steps=10,
    schedule=None,
    w=None,
    m=None,
    deg_name=None,
    inject_noise=False,
):
    """
    FoD NMC 采样：从退化输入 y 采样到清晰图。
    
    使用更严格的FoDiffusion.forward_loop实现。
    
    Args:
        model: TrajCFMNet 模型
        y: (B, 3, H, W) 退化输入
        n_steps: 采样步数
        schedule: FoDSchedule 对象（如果 None 则创建默认的）
        w: (B, 4) 全局权重（可选）
        m: (B, 4, H, W) 空间强度图（可选）
        deg_name: 退化名称（可选，可以是list或str）
        inject_noise: 是否注入噪声（已废弃，NMC采样已包含噪声）
    
    Returns:
        x_hat: (B, 3, H, W) 恢复结果
    """
    if schedule is None:
        # ✅修复：显式指定prediction="sflow"
        schedule = FoDSchedule(T=100, delta=0.001, device=y.device, prediction="sflow")
    
    B = y.shape[0]
    device = y.device
    
    # 创建模型包装函数
    def model_wrapper(x, t_int, x_start, **kwargs):
        # 将整数索引转换回归一化时间
        t_norm = t_int.float() / schedule.T
        # 确保t_norm是(B,)形状
        if t_norm.dim() == 0:
            t_norm = t_norm.unsqueeze(0).expand(B)
        elif t_norm.shape[0] != B:
            t_norm = t_norm[:B]
        # 调用原始模型
        r_pred = model(x, x_start, t_norm, deg_name=deg_name, w=w, m=m)
        return r_pred
    
    # 使用FoDiffusion的forward_loop（NMC采样）
    fod = schedule._fod
    x_hat = fod.forward_loop(
        model=model_wrapper,
        x_start=y,
        num_steps=n_steps,
        sample_type="MC",  
        clip_denoised=True,
        model_kwargs={},
        device=device,
        progress=False,
    )
    
    return x_hat


# ============================================================================
# 简化版采样（正确的FoD逻辑，用于调试）
# ============================================================================

def fod_simple_sample(model, y, n_steps=50, schedule=None, w=None, m=None, deg_name=None):
    """
    核心更新公式：x_{t+1} = (x_t - x_final) * decay + x_final
    其中 x_final = x_t + r_pred (SFLOW)
    
    Args:
        model: 模型函数，接受 (x_t, y, t, w, m, deg_name)
        y: (B, 3, H, W) 退化输入
        n_steps: 采样步数
        schedule: FoDSchedule对象（可选）
        w: (B, 4) 全局权重（可选）
        m: (B, 4, H, W) 空间强度图（可选）
        deg_name: 退化名称（可选）
    
    Returns:
        x_hat: (B, 3, H, W) 恢复结果
    """
    if schedule is None:
        schedule = FoDSchedule(T=100, delta=0.001, device=y.device, prediction="sflow")
    
    B = y.shape[0]
    device = y.device
    
    xt = y.clone()
    
    for step in range(n_steps):
        t_norm = torch.full((B,), step / n_steps, device=device)
        
        with torch.no_grad():
            r_pred = model(xt, y, t_norm, deg_name=deg_name, w=w, m=m)
        
        x_final = xt + r_pred
        x_final = x_final.clamp(-1, 1)
        
        if step < n_steps - 1:
            noise = torch.randn_like(xt) * 0.01 * (1.0 - t_next)  # 噪声随t减小
        else:
            noise = 0
        
        xt = (xt - x_final) * decay + x_final + noise
        xt = xt.clamp(-1, 1)
    
    return xt


