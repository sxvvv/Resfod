import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from time import time
import argparse
import logging
import os
import random
import math

from models.traj_cfm_net import TrajCFMNet
from models.degradation_parser import DegradationParser
from utils.fod_core import FoDSchedule, fod_nmc_sample, fod_analytic_sample, fod_simple_sample
from utils.lmdb_dataset import LMDBAllWeatherDataset
from utils.misc import seed_everything, makedirs
from utils.factor_utils import parse_factors, factors_to_present
from utils.metrics import psnr_y_torch, ssim_torch_eval as ssim_torch
from utils.ema import EMA

try:
    import pyiqa
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: pyiqa not available, perceptual loss will be disabled")


def create_logger(experiment_dir):
    """创建logger"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if experiment_dir:
        log_file = os.path.join(experiment_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def cleanup():
    """清理分布式训练"""
    dist.destroy_process_group()


def save_checkpoint(model, ema, opt, scheduler, step, path, args):
    """保存checkpoint"""
    checkpoint = {
        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        "ema": ema.state_dict(),  
        "opt": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, ema, opt, scheduler, rank, path):
    """加载checkpoint"""
    checkpoint = torch.load(path, map_location=f"cuda:{rank}")
    
    model_state = checkpoint["model"]
    
    if hasattr(model, 'module'):
        first_key = list(model_state.keys())[0] if model_state else ""
        if not first_key.startswith("module."):
            model_state = {f"module.{k}": v for k, v in model_state.items()}
    
    model.load_state_dict(model_state, strict=False)
    
    ema.load_state_dict(checkpoint["ema"], device=f"cuda:{rank}")
    if "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])
    if scheduler and "scheduler" in checkpoint and checkpoint["scheduler"]:
        scheduler.load_state_dict(checkpoint["scheduler"])
    step = checkpoint.get("step", 0)
    return step


def deg_name_to_multi_hot(deg_names, device):
    """将退化名称转换为 multi-hot 标签"""
    from utils.factor_utils import FACTOR2IDX
    B = len(deg_names)
    labels = torch.zeros(B, 4, device=device, dtype=torch.float32)
    for i, name in enumerate(deg_names):
        if name is None:
            continue
        factors = parse_factors(name)
        for factor in factors:
            if factor in FACTOR2IDX:
                labels[i, FACTOR2IDX[factor]] = 1.0
    return labels


def gating_target_from_labels(labels, t, deg_names=None):
    B = labels.shape[0]
    device, dtype = labels.device, labels.dtype
    
    # 三段阈值方法（只对存在的因子生效）
    mod1 = torch.tensor([0.3, 0.3, 1.0, 1.0], device=device, dtype=dtype)  # early: rain/snow
    mod2 = torch.tensor([0.3, 1.0, 0.5, 0.5], device=device, dtype=dtype)  # mid: haze
    mod3 = torch.tensor([1.0, 0.5, 0.3, 0.3], device=device, dtype=dtype)  # late: low
    
    tt = t.view(B, 1)
    time_mod = torch.where(
        tt < 0.33, mod1.view(1, 4),
        torch.where(tt < 0.67, mod2.view(1, 4), mod3.view(1, 4))
    )
    
    # 只在active因子内分配权重
    alpha = labels * time_mod
    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
    return alpha


@torch.no_grad()
def evaluate_model(model, test_loader, device, schedule, nmc_steps=50, ema=None, max_samples=20, use_time_dependent=True):
    """评估模型性能"""
    model.eval()
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    psnr_sum, ssim_sum, nimg = 0.0, 0.0, 0
    
    # ========== 采样方式选择 ========
    USE_SIMPLE_SAMPLE = False  
    USE_DIRECT_PREDICTION = True 
    
    if ema is not None:
        ema_context = ema.average_parameters(model)
        ema_context.__enter__()
    else:
        ema_context = None
    
    try:
        for idx, batch in enumerate(test_loader):
            if max_samples is not None and idx >= max_samples:
                break
            
            y = batch["LQ"].to(device)
            x = batch["GT"].to(device)
            deg_name = batch.get("deg_name", None)
            
            if isinstance(deg_name, (list, tuple)) and len(deg_name) > 0:
                deg_one = deg_name[0]
            elif isinstance(deg_name, str):
                deg_one = deg_name
            else:
                deg_one = None
            
            B = y.shape[0]
            
            if USE_DIRECT_PREDICTION:
                net = model.module if hasattr(model, 'module') else model
                t_zero = torch.zeros(B, device=device)
                
                # 获取parser输出
                if hasattr(net, 'parser') and net.parser is not None:
                    if use_time_dependent:
                        w_init, m_init, _ = net.parser(y, t_zero)
                    else:
                        w_init, m_init, _ = net.parser(y)
                else:
                    w_init = torch.ones(B, 4, device=device) * 0.25
                    H, W = y.shape[2:]
                    m_init = torch.ones(B, 4, H, W, device=device) * 0.25
                
                # 直接预测：x_pred = y + r_pred
                r_pred = net(y, y, t_zero, w=w_init, m=m_init, deg_name=deg_one)
                x_hat = y + r_pred
                x_hat = x_hat.clamp(-1, 1)
            elif USE_SIMPLE_SAMPLE:
                net = model.module if hasattr(model, 'module') else model
                
                t_init = torch.zeros(B, device=device)
                if hasattr(net, 'parser') and net.parser is not None:
                    if use_time_dependent:
                        # Time-dependent: 在初始时间计算一次
                        w_init, m_init, _ = net.parser(y, t_init)
                    else:
                        # Static: 只计算一次
                        w_init, m_init, _ = net.parser(y)
                else:
                    w_init = torch.ones(B, 4, device=device) * 0.25
                    H, W = y.shape[2:]
                    m_init = torch.ones(B, 4, H, W, device=device) * 0.25
                
                x_hat = fod_simple_sample(
                    net, y, 
                    n_steps=nmc_steps, 
                    schedule=schedule,
                    w=w_init, 
                    m=m_init, 
                    deg_name=deg_one
                )
            else:
                t_init = torch.zeros(B, device=device)
                
                if hasattr(model, 'module') and hasattr(model.module, 'parser') and model.module.parser is not None:
                    if use_time_dependent:
                        w_init, m_init, _ = model.module.parser(y, t_init)
                    else:
                        w_init, m_init, _ = model.module.parser(y)
                else:
                    w_init = torch.ones(B, 4, device=device) * 0.25
                    H, W = y.shape[2:]
                    m_init = torch.ones(B, 4, H, W, device=device) * 0.25
                
                x_hat = fod_nmc_sample(
                    model, y,
                    n_steps=nmc_steps,
                    schedule=schedule,
                    w=w_init, m=m_init,
                    deg_name=deg_one,
                    inject_noise=False,
                )
            # ========================================
            
            psnr_val = psnr_y_torch(x_hat, x, data_range=2.0).item()
            ssim_val = ssim_torch(x_hat, x, data_range=2.0, size_average=True).item()
            psnr_sum += psnr_val
            ssim_sum += ssim_val
            nimg += 1
            
            if (idx + 1) % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 释放中间变量
            del y, x, x_hat
            if 'r_pred' in locals():
                del r_pred
    finally:
        if ema_context is not None:
            ema_context.__exit__(None, None, None)
        
        # 最终清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    model.train()
    return psnr_sum / max(1, nimg), ssim_sum / max(1, nimg)


def main():
    parser = argparse.ArgumentParser(description="FoD Image Restoration Training")
    
    # ==================== 数据集参数 ====================
    parser.add_argument("--lmdb-path", type=str, required=True,
                        help="训练集LMDB路径")
    parser.add_argument("--test-lmdb-path", type=str, default=None,
                        help="测试集LMDB路径（可选）")
    parser.add_argument("--train-on-test-set", action="store_true", default=False,
                        help="在测试集上继续训练（fine-tuning模式）")
    parser.add_argument("--patch-size", type=int, default=512,
                        help="训练时的patch大小")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="数据加载线程数")
    
    # ==================== 训练参数 ====================
    parser.add_argument("--global-batch-size", type=int, default=16,
                        help="全局batch size（会按GPU数量自动分配）")
    parser.add_argument("--niter", type=int, default=300_000,
                        help="总训练步数")
    parser.add_argument("--lr", type=float, default=1.5e-4,
                        help="初始学习率")
    parser.add_argument("--global-seed", type=int, default=0,
                        help="随机种子")
    parser.add_argument("--log-every", type=int, default=100,
                        help="每N步打印一次日志")
    parser.add_argument("--val-every", type=int, default=2500,
                        help="每N步验证一次")
    parser.add_argument("--val-max-samples", type=int, default=20,
                        help="验证时最多使用的样本数（减少内存使用）")
    parser.add_argument("--ckpt-every", type=int, default=20_000,
                        help="每N步保存一次checkpoint")
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的checkpoint路径")
    
    # ==================== 模型参数 ====================
    parser.add_argument("--run-name", type=str, default="fod_ir",
                        help="实验名称")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="结果保存目录")
    parser.add_argument("--base-ch", type=int, default=64,
                        help="基础通道数")
    parser.add_argument("--ch-mult", type=int, nargs='+', default=[1, 2, 4, 4],
                        help="通道倍数列表")
    parser.add_argument("--emb-dim", type=int, default=256,
                        help="嵌入维度")
    parser.add_argument("--use-time-dependent", action="store_true", default=True,
                        help="使用时间依赖的退化解析")
    parser.add_argument("--use-depth", action="store_true", default=False,
                        help="使用深度信息（需要数据集提供）")
    
    # ==================== FoD参数 ====================
    parser.add_argument("--T", type=int, default=100,
                        help="FoD时间步数")
    parser.add_argument("--delta", type=float, default=0.001,
                        help="FoD delta参数")
    parser.add_argument("--nmc-steps-train", type=int, default=20,
                        help="训练时NMC unroll步数（已实现渐进式：早期20步→中期50步→后期100步）")
    parser.add_argument("--nmc-steps-eval", type=int, default=100,
                        help="评估时NMC步数（建议100步以获得最佳PSNR，推理时可减少到50步加速）")
    
    # ==================== 损失权重 ====================
    parser.add_argument("--lambda-sfm", type=float, default=1.0,
                        help="SFM损失权重")
    parser.add_argument("--lambda-traj", type=float, default=1.0,
                        help="Trajectory损失权重")
    parser.add_argument("--lambda-mid", type=float, default=0.05,
                        help="中间态监督权重")
    parser.add_argument("--lambda-cls", type=float, default=0.1,
                        help="分类损失权重")
    parser.add_argument("--lambda-alpha", type=float, default=0.02,
                        help="Alpha门控监督权重")
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="EMA衰减率")
    
    # ==================== 困难样本加权 ====================
    parser.add_argument("--use-hard-sample-weighting", action="store_true", default=False,
                        help="启用困难样本加权（对PSNR低的样本给予更高权重）")
    parser.add_argument("--hard-sample-psnr-threshold", type=float, default=28.5,
                        help="困难样本PSNR阈值（低于此值的样本会被加权）")
    parser.add_argument("--hard-sample-weight-factor", type=float, default=2.0,
                        help="困难样本权重因子（困难样本权重 = 基础权重 * factor）")
    
    args = parser.parse_args()
    DEBUG_SIMPLE_MODE = False 
    
    if DEBUG_SIMPLE_MODE:
        print("!!! DEBUG MODE: Disabling complex components !!!")
        args.lambda_cls = 0.0      # 禁用分类损失
        args.lambda_alpha = 0.0    # 禁用alpha监督
        args.lambda_mid = 0.0      # 禁用中间态监督
        args.lambda_traj = 0.0     # 只使用SFM
        args.use_time_dependent = False  # 禁用时间依赖解析
    # ============================================
    
    # ==================== 初始化分布式训练 ====================
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        # 单GPU模式
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1 or "RANK" in os.environ:
        dist.init_process_group("nccl")
    else:
        if rank == 0:
            print("Warning: Single GPU mode without torchrun. Please use torchrun for proper initialization.")
            print("Example: torchrun --nnodes=1 --nproc_per_node=1 --master_port=34567 train_IR.py ...")
    assert args.global_batch_size % world_size == 0, \
        f"global_batch_size ({args.global_batch_size}) must be divisible by world_size ({world_size})"
    
    device = local_rank
    seed = seed_everything(args.global_seed + rank)
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Training Configuration:")
        print(f"  World Size: {world_size}")
        print(f"  Global Batch Size: {args.global_batch_size}")
        print(f"  Batch Size per GPU: {args.global_batch_size // world_size}")
        print(f"  Device: {device}")
        print(f"  Seed: {seed}")
    
    # ==================== 创建实验目录 ====================
    experiment_dir = f"{args.results_dir}/{args.run_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    training_state_path = f"{checkpoint_dir}/training_state.pt"
    
    if rank == 0:
        makedirs(args.results_dir)
        makedirs(checkpoint_dir)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory: {experiment_dir}")
        if DEBUG_SIMPLE_MODE:
            logger.info("!!! DEBUG MODE: Disabling complex components !!!")
        logger.info(f"Training Configuration:")
        for key, value in vars(args).items():
            logger.info(f"  {key}: {value}")
    else:
        logger = create_logger(None)
    
    # ==================== 创建模型 ====================
    model = TrajCFMNet(
        in_ch=6,
        out_ch=3,
        base_ch=args.base_ch,
        ch_mult=tuple(args.ch_mult),
        emb_dim=args.emb_dim,
        use_depth=args.use_depth,
        use_parser=True,
        adapter_dim=64,
        use_prompt_pool=False,
        prompt_dim=256,
        use_depth_anything=False,
        depth_anything_extractor=None,
    ).to(device)
    
    # 修改 parser 支持 time-dependent
    if args.use_time_dependent:
        model.parser = DegradationParser(
            in_ch=3,
            base_ch=32,
            emb_dim=128,
            use_time_dependent=True,
        ).to(device)
    
    ema = EMA(model, decay=args.ema_decay)
    
    # DDP
    model = DDP(model, device_ids=[device], find_unused_parameters=False)
    
    # FoD Schedule
    schedule = FoDSchedule(
        T=args.T, 
        delta=args.delta, 
        device=device,
        prediction="sflow",  
    )
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
        logger.info(f"Using compositional velocity field: Share + 4 Experts")
    
    # ==================== 优化器 ====================
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
        betas=(0.9, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.niter, eta_min=1e-6
    )
    
    # ==================== 数据集 ====================
    # 如果启用测试集训练，使用测试集作为训练数据
    if args.train_on_test_set:
        if not args.test_lmdb_path or not os.path.exists(args.test_lmdb_path):
            raise ValueError("--train-on-test-set requires --test-lmdb-path to be specified and exist")
        train_lmdb_path = args.test_lmdb_path
        if rank == 0:
            logger.info("⚠️  Training on TEST SET (fine-tuning mode)")
    else:
        train_lmdb_path = args.lmdb_path
    
    dataset = LMDBAllWeatherDataset(
        lmdb_path=train_lmdb_path,
        patch_size=args.patch_size,
        is_train=True, 
        use_precomputed_depth=args.use_depth,
        use_counterfactual_supervision=False,
        depth_extractor=None,
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size // world_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    if rank == 0:
        dataset_type = "TEST SET" if args.train_on_test_set else "TRAIN SET"
        logger.info(f"Dataset ({dataset_type}): {len(dataset):,} samples from {train_lmdb_path}")
        if args.use_hard_sample_weighting:
            logger.info(f"Hard sample weighting enabled (threshold={args.hard_sample_psnr_threshold:.2f}dB, factor={args.hard_sample_weight_factor:.2f})")
    
    # ==================== 准备模型 ====================
    model.train()
    ema.update(model)  # 初始化EMA
    
    # ==================== 评估设置 ====================
    if rank == 0:
        lpips_fn = pyiqa.create_metric('lpips', device=device) if LPIPS_AVAILABLE else None
        
        # 测试集
        if args.test_lmdb_path and os.path.exists(args.test_lmdb_path):
            test_dataset = LMDBAllWeatherDataset(
                lmdb_path=args.test_lmdb_path,
                patch_size=None,
                is_train=False,
                use_precomputed_depth=False,
            )
            test_loader = DataLoader(
                test_dataset, batch_size=1, shuffle=False, num_workers=4
            )
            logger.info(f"Test set: {len(test_dataset):,} samples")
        else:
            test_loader = None
            logger.info("Test set not provided, skipping evaluation")
    else:
        lpips_fn = None
        test_loader = None
    
    # ==================== 训练变量 ====================
    train_steps = 0
    log_steps = 0
    running_loss = 0
    epoch = 0
    start_time = time()
    
    # ==================== 恢复训练 ====================
    if args.resume and os.path.exists(args.resume):
        if rank == 0:
            logger.info(f"Loading checkpoint from {args.resume}")
        train_steps = load_checkpoint(model, ema, opt, scheduler, rank, path=args.resume)
        if rank == 0:
            logger.info(f"Resumed from step {train_steps}")
    
    if rank == 0:
        logger.info(f"Starting training for {args.niter} iterations...")
    
    # ==================== 训练循环 ====================
    while train_steps < args.niter:
        sampler.set_epoch(epoch)
        if epoch % 5 == 0 and rank == 0:
            logger.info(f"Beginning epoch {epoch}...")
        
        for data in loader:
            # 统一batch格式
            if "x" in data and "y" in data:
                x = data["x"].to(device, non_blocking=True)
                y = data["y"].to(device, non_blocking=True)
            else:
                y = data["LQ"].to(device, non_blocking=True)
                x = data["GT"].to(device, non_blocking=True)
            
            deg_name = data.get("deg_name", None)
            
            # 获取因子信息
            B = y.shape[0]
            present = torch.zeros(B, 4, device=device)
            w = torch.zeros(B, 4, device=device)
            m = torch.zeros(B, 4, y.shape[2], y.shape[3], device=device)
            
            if deg_name is not None:
                if isinstance(deg_name, list):
                    for b in range(B):
                        deg_name_b = deg_name[b] if b < len(deg_name) else None
                        if deg_name_b:
                            factors = parse_factors(deg_name_b)
                            present[b] = factors_to_present(factors).to(device)
                            w[b] = present[b].clone()
                            for i in range(4):
                                if present[b, i] > 0:
                                    m[b, i].fill_(1.0)
                else:
                    factors = parse_factors(deg_name) if deg_name else []
                    present_all = factors_to_present(factors).to(device)
                    present = present_all.unsqueeze(0).expand(B, -1)
                    w = present.clone()
                    for i in range(4):
                        if present[0, i] > 0:
                            m[:, i].fill_(1.0)
            
            # 获取multi-hot标签
            if deg_name is not None:
                if isinstance(deg_name, (list, tuple)):
                    labels = deg_name_to_multi_hot(list(deg_name), device)
                else:
                    labels = deg_name_to_multi_hot([deg_name] * B, device)
            else:
                labels = None
            
            
            use_traj = False
            
            teacher_forcing_ratio = max(0.0, 1.0 - 2.0 * train_steps / args.niter)  # 从1.0线性衰减到0.0
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            
            # 构造GT的w和m（如果使用teacher-forcing）
            if use_teacher_forcing and deg_name is not None:
                # 用GT present构造w_gt和m_gt
                w_gt = present.clone().float()  # (B, 4) 直接使用present作为权重
                # m_gt: 对存在的因子，设置为均匀强度图
                H, W = y.shape[2:]
                m_gt = torch.zeros(B, 4, H, W, device=device)
                for i in range(4):
                    mask = present[:, i] > 0  # (B,)
                    if mask.any():
                        # 对存在的因子，设置为均匀强度（或可以用更复杂的策略）
                        m_gt[mask, i] = 1.0  # 或者可以用0.5等
            else:
                w_gt, m_gt = None, None
            
            if not use_traj:
                u = torch.rand(B, device=device)
                t = u * u  # 平方，t会大量集中在0附近
            else:
                t = torch.zeros(B, device=device)
            
            # 训练损失
            if use_traj:
                # Trajectory Matching
                from utils.fod_core import fod_nmc_step
                x0 = y.clone()
                mu = x
                xt = y.clone()
                
                loss_traj = torch.tensor(0.0, device=device)
                
                # 使用命令行参数（固定值，减少内存使用）
                train_unroll_steps = args.nmc_steps_train
                
                k = schedule.T // train_unroll_steps
                dt = k / schedule.T
                
                # 预先获取parser输出（用于分类损失，但不一定用于主网）
                if not args.use_time_dependent:
                    _, _, logits = model.module.parser(y, present=present if use_teacher_forcing else None)
                else:
                    logits = None
                
                for n in range(train_unroll_steps):
                    t_n = torch.full((B,), n * dt, device=device)
                    
                    # 决定使用GT还是parser输出
                    if use_teacher_forcing and w_gt is not None and m_gt is not None:
                        # Teacher-forcing: 使用GT的w和m
                        w_use, m_use = w_gt, m_gt
                        # parser仍然需要更新（用于分类损失），但不传present避免影响
                        if args.use_time_dependent:
                            _, _, _ = model.module.parser(y, t_n)
                    else:
                        # 使用parser输出
                        if args.use_time_dependent:
                            w_pred, m_pred, _ = model.module.parser(y, t_n, present=None)
                        else:
                            w_pred, m_pred, _ = model.module.parser(y, present=None)
                        w_use, m_use = w_pred, m_pred
                    
                    # 模型预测
                    r_pred, alpha_pred = model.module(
                        xt, y, t_n, w=w_use, m=m_use, deg_name=deg_name, return_alpha=True
                    )
                    mu_hat = xt + r_pred
                    
                    # 中间态监督
                    if args.lambda_mid > 0:
                        loss_traj = loss_traj + args.lambda_mid * torch.nn.functional.l1_loss(mu_hat, mu)
                    
                    # Alpha时间先验
                    if args.lambda_alpha > 0 and labels is not None:
                        alpha_tgt = gating_target_from_labels(labels, t_n, deg_name)
                        loss_traj = loss_traj + args.lambda_alpha * torch.nn.functional.mse_loss(alpha_pred, alpha_tgt)
                    
                    # NMC更新
                    if n < train_unroll_steps - 1:
                        xt_next = fod_nmc_step(
                            model.module, xt, y, t_n, dt,
                            schedule=schedule,
                            mu=None,
                            w=w_use, m=m_use,  
                            deg_name=deg_name,
                            inject_noise=(n >= train_unroll_steps - 2),
                            last_step_noise=0.0,
                        )
                        xt = xt_next.detach().requires_grad_(True)
                    else:
                        xt = mu_hat
                
                # 最终损失
                loss_traj = loss_traj + torch.nn.functional.l1_loss(xt, mu)
                loss = args.lambda_traj * loss_traj
                
                # 分类损失
                if args.lambda_cls > 0 and labels is not None and logits is not None:
                    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                    loss = loss + args.lambda_cls * loss_cls
            else:
                # SFM训练
                epsilon = torch.randn_like(y)
                xt = fod_analytic_sample(y, x, torch.zeros(B, device=device), t, schedule, epsilon)
                
                if use_teacher_forcing and w_gt is not None and m_gt is not None:
                    # Teacher-forcing: 使用GT的w和m
                    w_use, m_use = w_gt, m_gt
                    if args.use_time_dependent:
                        _, _, logits = model.module.parser(y, t, present=present)
                    else:
                        _, _, logits = model.module.parser(y, present=present)
                else:
                    # 使用parser输出
                    if args.use_time_dependent:
                        w_pred, m_pred, logits = model.module.parser(y, t, present=None)
                    else:
                        w_pred, m_pred, logits = model.module.parser(y, present=None)
                    w_use, m_use = w_pred, m_pred
                
                r_pred, alpha_pred = model.module(
                    xt, y, t, w=w_use, m=m_use, deg_name=deg_name, return_alpha=True
                )
                r_target = x - xt
                
                with torch.no_grad():
                    q = (x - y).abs().mean(1, keepdim=True)  # (B,1,H,W)
                    q = q / (q.mean(dim=[2,3], keepdim=True) + 1e-6)  # normalize
                    w_map = (1.0 + 1.5 * q).clamp(1.0, 4.0)  # gamma=1.5 可调
                
                
                if args.use_hard_sample_weighting:
                    # 计算per-sample损失
                    loss_sfm_mse_per_sample = torch.nn.functional.mse_loss(r_pred, r_target, reduction='none')
                    loss_sfm_l1_per_sample = torch.nn.functional.l1_loss(r_pred, r_target, reduction='none')
                    # 对每个样本计算平均损失 (B, C, H, W) -> (B,)
                    loss_sfm_mse_per_sample = loss_sfm_mse_per_sample.view(B, -1).mean(dim=1)
                    loss_sfm_l1_per_sample = loss_sfm_l1_per_sample.view(B, -1).mean(dim=1)
                    loss_sfm_per_sample = 0.5 * loss_sfm_mse_per_sample + 0.5 * loss_sfm_l1_per_sample
                    
                    x_pred = (xt + r_pred).clamp(-1, 1)
                    with torch.no_grad():
                        psnr_per_sample = psnr_y_torch(x_pred.detach(), x, data_range=2.0, per_sample=True)  # (B,)
                        
                        # 计算困难样本权重
                        hard_mask = psnr_per_sample < args.hard_sample_psnr_threshold  # (B,)
                        sample_weights = torch.ones(B, device=device)
                        sample_weights[hard_mask] = args.hard_sample_weight_factor
                        sample_weights = sample_weights / sample_weights.mean()
                    
                    loss_sfm = (loss_sfm_per_sample * sample_weights.detach()).mean()
                else:
                    loss_sfm_l1 = ((r_pred - r_target).abs() * w_map).mean()
                    loss_sfm = loss_sfm_l1  # 使用加权L1损失
                
                loss = args.lambda_sfm * loss_sfm
                
                t0 = torch.zeros(B, device=device)
                if args.use_time_dependent:
                    w0, m0, _ = model.module.parser(y, t0, present=None)
                else:
                    w0, m0, _ = model.module.parser(y, present=None)
                r0 = model.module(y, y, t0, w=w0, m=m0, deg_name=deg_name, return_alpha=False)  # 只返回一个值
                loss_t0 = torch.nn.functional.l1_loss(r0, x - y)
                loss = loss + 0.5 * loss_t0  # 权重0.5（可在0.2~1.0之间调节）
                
                # 分类损失
                if args.lambda_cls > 0 and labels is not None and logits is not None:
                    loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                    loss = loss + args.lambda_cls * loss_cls
                
                # Alpha时间先验
                if args.lambda_alpha > 0 and labels is not None:
                    alpha_tgt = gating_target_from_labels(labels, t, deg_name)
                    loss = loss + args.lambda_alpha * torch.nn.functional.mse_loss(alpha_pred, alpha_tgt)
                
            
            # 反向传播
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            ema.update(model)
            
            # 记录
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            # 日志
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                
                if rank == 0:
                    # 显示当前训练步数（如果是Traj模式）
                    traj_steps_info = ""
                    if use_traj:
                        progress = train_steps / args.niter
                        if progress < 0.33:
                            current_traj_steps = 20
                        elif progress < 0.66:
                            current_traj_steps = 50
                        else:
                            current_traj_steps = 100
                        traj_steps_info = f", TrajSteps: {current_traj_steps}"
                    
                    logger.info(
                        f"step={train_steps:07d}) Loss: {avg_loss:.6f}, "
                        f"LR: {opt.param_groups[0]['lr']:.2e}, Steps/Sec: {steps_per_sec:.2f}, "
                        f"Mode: {'Traj' if use_traj else 'SFM'}{traj_steps_info}"
                    )
                
                running_loss = 0
                log_steps = 0
                start_time = time()
            
            
            # 保存checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    save_checkpoint(model, ema, opt, scheduler, train_steps, checkpoint_path, args)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
        
        epoch += 1
    
    # ==================== 训练完成 ====================
    if rank == 0:
        logger.info("Training completed!")
        # 保存最终checkpoint
        final_checkpoint_path = f"{checkpoint_dir}/final.pt"
        save_checkpoint(model, ema, opt, scheduler, train_steps, final_checkpoint_path, args)
        logger.info(f"Final checkpoint saved to {final_checkpoint_path}")
    
    cleanup()


if __name__ == "__main__":
    main()
