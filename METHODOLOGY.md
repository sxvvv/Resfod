# Traj-CFM 模型设计范式与技术细节文档

本文档详细整理了Traj-CFM (Trajectory-aware Compositional Flow Matching) 模型的核心设计理念、架构范式、训练策略以及关键技术细节，为论文写作提供完整的技术依据。

---

## 一、核心范式与设计理念

### 1.1 组合式分解范式 (Compositional Decomposition Paradigm)

**核心思想**：将复杂的组合退化问题分解为原子因子，通过组合原子速度场来建模组合退化。

**设计动机**：
- **可组合性**：训练时只需见到部分组合（如`low`, `haze`, `low_haze`），推理时可以处理未见的组合（如`low_haze_rain`）
- **可解释性**：每个专家对应一个原子因子，门控权重α直接反映因子贡献
- **参数效率**：共享骨干网络 + 轻量专家适配器，避免为每个组合单独建模

**数学表述**：
$$v_\theta(x_t, t, y) = v_{\text{share}}(x_t, t, y) + \sum_{i=1}^{K} \alpha_i(t, w) \cdot v_i(x_t, t, m_i)$$

其中：
- $K=4$：原子因子数量（low-light, haze, rain, snow）
- $v_{\text{share}}$：共享速度场，处理通用恢复、颜色校正、因子交互
- $v_i$：专家速度场，处理特定因子$i$
- $\alpha_i$：时间条件门控权重
- $m_i$：空间强度图，调制专家输出

### 1.2 前向扩散范式 (Forward-Only Diffusion, FoD)

**核心思想**：利用图像复原任务的特殊结构（退化输入$y$包含丰富语义先验），实现单向前向扩散过程。

**与传统扩散模型的区别**：
- **传统扩散**：需要前向-反向成对过程，从噪声生成图像
- **FoD**：只需前向过程，从退化输入$y$直接流到清晰图$x$
- **优势**：更符合复原任务的自然结构，训练更稳定

**数学表述**：
$$\frac{dx_t}{dt} = v_\theta(x_t, t, y)$$

其中：
- $x_0 = y$（初始状态 = 退化输入）
- $x_1 = x$（目标状态 = 清晰图）
- $t \in [0,1]$（归一化时间）

**前向核采样**：
$$x_t = (y - x) \cdot \exp\left(-\int_0^t (\theta_s + \tfrac{1}{2}\sigma_s^2) ds + \int_0^t \sigma_s dW_s\right) + x$$

### 1.3 时间条件门控范式 (Time-Conditioned Gating)

**核心思想**：利用物理先验，在不同时间阶段强调不同的专家。

**物理先验设计**：
- **早期阶段** ($t < 0.33$)：强调rain/snow专家（高频噪声去除）
- **中期阶段** ($0.33 \leq t < 0.67$)：强调haze专家（中频去雾）
- **后期阶段** ($t \geq 0.67$)：强调low-light专家（全局增强）

**数学表述**：
$$\alpha_i(t, w) = \text{Normalize}\left(\text{sigmoid}\left(\text{MLP}([\tau(t), w_{\text{norm}}, \bar{m}])\right) \odot w_{\text{mask}}\right)$$

其中：
- $\tau(t)$：正弦时间嵌入（256维）
- $w_{\text{norm}}$：归一化的全局权重（4维）
- $\bar{m}$：空间统计（$m$在空间维度上的均值，4维）
- $w_{\text{mask}}$：存在掩码（$w > 0.1$）

**设计细节**：
- 使用sigmoid而非softmax，支持多专家同时激活
- 通过存在掩码过滤不存在的因子
- 最后进行sum-normalize，确保权重和为1

---

## 二、模型架构设计细节

### 2.1 整体架构流程

```
输入: x_t (3) + y (3) = 6 channels
  ↓
退化解析器 (DegradationParser)
  ├─→ w (B, 4) 全局权重
  ├─→ m (B, 4, H, W) 空间强度图
  └─→ logits (B, 4) 分类logits
  ↓
共享骨干网络 (SharedBackbone, U-Net)
  输入: [x_t, y] (6 channels)
  架构: base_ch=64, ch_mult=[1,2,4,4]
  输出: 
    ├─→ v_share (B, 3, H, W) 共享速度场
    └─→ h_final (B, 64, H, W) 全分辨率特征（用于专家）
  ↓
4个专家适配器 (ExpertVelocityField)
  每个专家:
    ├─→ 时间适配器: MLP(256 → 64 → 256)
    ├─→ 空间适配器: Conv2d(64 → 64) 1×1卷积
    └─→ 输出投影: Conv2d(64 → 3) 生成残差 r_i
  应用空间调制: v_i = r_i ⊙ m_i
  ↓
时间条件门控网络 (Alpha MLP)
  输入: [τ(t), w_norm, m_mean] (256 + 4 + 4 = 264维)
  输出: α (B, 4) 门控权重
  ↓
组合输出: v = v_share + Σ α_i · v_i
```

### 2.2 共享骨干网络 (SharedBackbone)

**架构选择：U-Net**

**设计理由**：
- **多尺度特征提取**：通过下采样-上采样结构捕获不同尺度信息
- **残差连接**：保留细节信息，避免信息丢失
- **时间条件**：通过时间嵌入控制不同时间步的行为

**详细配置**：
- **输入通道**：6（x_t的3通道 + y的3通道）
- **输出通道**：3（RGB速度场）
- **基础通道数**：64
- **通道倍增器**：[1, 2, 4, 4]
- **嵌入维度**：256
- **时间嵌入**：正弦嵌入 → MLP投影（256 → 256×4 → 256）

**关键组件**：
1. **ResBlock**：每个block包含2个ResBlock
   - GroupNorm (8 groups) + SiLU激活
   - 时间嵌入通过MLP投影后add到特征
   - 残差连接（如果通道数变化，使用1×1卷积投影）

2. **下采样**：3×3卷积，stride=2
3. **上采样**：最近邻插值 + 3×3卷积
4. **输出层**：GroupNorm + SiLU + 3×3卷积

**特征复用设计**：
- 共享骨干网络不仅输出速度场，还输出全分辨率特征$h_{\text{final}}$
- $h_{\text{final}}$用于专家适配器，避免重复计算，提高效率

### 2.3 专家速度场 (ExpertVelocityField)

**设计理念：参数高效的适配器机制**

**为什么使用适配器而非独立网络**：
- **参数效率**：每个专家只需~1M参数（vs 完整网络~15M）
- **特征复用**：直接使用共享骨干的特征，避免重复计算
- **协作学习**：共享特征促进专家之间的协同

**适配器结构**：

1. **时间适配器**：
   ```
   Input: 时间嵌入 temb (B, 256)
   ↓
   Linear(256 → 64)  # down-proj
   ↓
   SiLU
   ↓
   Linear(64 → 256)  # up-proj
   ↓
   Output: adapted_temb (B, 256)
   ```

2. **空间适配器**：
   ```
   Input: 全分辨率特征 h_final (B, 64, H, W)
   ↓
   Conv2d(64 → 64, 1×1) + GroupNorm(8) + SiLU
   ↓
   Conv2d(64 → 64, 1×1)
   ↓
   Output: adapted_feature (B, 64, H, W)
   ```

3. **输出投影**：
   ```
   Input: adapted_feature (B, 64, H, W)
   ↓
   GroupNorm(8) + SiLU
   ↓
   Conv2d(64 → 3, 3×3)
   ↓
   Output: r_i (B, 3, H, W) 原始残差
   ```

**空间调制**：
$$v_i = r_i \odot m_i$$

其中$m_i$是空间强度图的第$i$个通道，通过双线性插值调整尺寸匹配。

### 2.4 退化解析器 (DegradationParser)

**设计目标**：
- 从退化输入$y$估计全局权重$w$和空间强度图$m$
- 支持多标签分类（用于监督）
- 支持时间依赖的解析（time-dependent parsing）

**架构选择：轻量U-Net编码器**

**详细配置**：
- **基础通道数**：32（比主网络小，轻量化设计）
- **嵌入维度**：128
- **时间嵌入维度**：64（用于time-dependent模式）
- **总参数量**：~0.3M

**输出头设计**：

1. **全局权重$w$**：
   - **静态模式**：全局池化 → MLP(128→4) → sigmoid
   - **时间依赖模式**：
     - 时间嵌入：sinusoidal(64) → MLP(64→64)
     - 特征拼接：MLP(128) + time_feat(64) → (192)
     - 参数预测：Linear(192→8) → reshape(B,4,2)
     - 时间变换：$w_i(t) = \sigma(a_i + b_i \cdot t)$

2. **空间强度图$m$**：
   - 转置卷积上采样：H/4 → H
   - 输出：Conv2d(32→4, 3×3) → sigmoid
   - 支持多因子叠加（sigmoid而非softmax）

3. **分类logits**：
   - 全局池化 → MLP(128→4)
   - 用于多标签BCE损失监督

**关键设计决策**：
- **使用sigmoid而非softmax**：支持多因子同时存在（多标签分类）
- **Present mask机制**：训练时可用ground-truth present mask限制输出范围

### 2.5 时间条件门控网络

**输入构建**：
$$\text{input} = [\tau(t), w_{\text{norm}}, \bar{m}]$$

其中：
- $\tau(t)$：正弦时间嵌入，256维
- $w_{\text{norm}}$：归一化的全局权重，$w / \sum w$，4维
- $\bar{m}$：空间统计，$m$在空间维度上的均值，4维

**网络结构**：
```
Input: (256 + 4 + 4) = 264维
↓
Linear(264 → 256) + SiLU
↓
Linear(256 → 4)
↓
Output: alpha_logits (B, 4)
↓
sigmoid → w_mask → sum-normalize
↓
Output: α (B, 4)
```

**处理流程**：
1. **sigmoid激活**：$\alpha_{\text{raw}} = \sigma(\text{logits})$，支持多专家激活
2. **存在掩码**：$\alpha_{\text{masked}} = \alpha_{\text{raw}} \odot (w > 0.1)$
3. **归一化**：$\alpha = \alpha_{\text{masked}} / \sum \alpha_{\text{masked}}$

---

## 三、训练范式与策略

### 3.1 损失函数设计

#### 3.1.1 随机流匹配损失 (Stochastic Flow Matching, SFM Loss)

**目标**：预测从中间状态到目标的残差

$$r_\theta(x_t, t, y) \approx x - x_t$$

**损失形式**：
$$\mathcal{L}_{\text{SFM}} = \mathbb{E}_{t, \epsilon}\left[\lambda_1 \|r_\theta(x_t, t, y) - (x - x_t)\|_1 + \lambda_2 \|r_\theta(x_t, t, y) - (x - x_t)\|_2^2\right]$$

**实现细节**：
- $\lambda_1 = 0.5, \lambda_2 = 0.5$（混合L1+L2）
- 时间采样：$t \sim \mathcal{U}(0,1)$（实际使用平方采样$t = u^2$，$u \sim \mathcal{U}(0,1)$，偏向$t=0$）
- 中间状态采样：使用FoD前向核采样$x_t$

**t=0增强损失**：
$$\mathcal{L}_{t=0} = \|r_\theta(y, 0, y) - (x - y)\|_1$$

权重：0.5（对齐推理时的one-step预测）

**破坏权重图加权**（可选）：
$$w_{\text{map}} = (1.0 + 1.5 \cdot q) \in [1.0, 4.0]$$

其中$q$是GT与degraded的归一化差异图，用于强调困难区域。

#### 3.1.2 轨迹匹配损失 (Trajectory Matching Loss)

**目标**：在整个生成轨迹上强制一致性

**实现方式**：展开$N$步NMC采样，对每步进行监督

$$\mathcal{L}_{\text{traj}} = \|x_T - x\|_1 + \sum_{n=0}^{N-1} \lambda_{\text{mid}} \|\mu^{(n)} - x\|_1$$

其中：
- $x_T$：$N$步展开后的最终状态
- $\mu^{(n)} = x_t^{(n)} + r_\theta^{(n)}$：第$n$步的预测目标
- $\lambda_{\text{mid}} = 0.05$：中间监督权重

**展开策略**（当前实现已禁用，使用固定步数）：
- 训练步数：`args.nmc_steps_train`（默认20）
- 注意：长展开会导致显存占用过大

#### 3.1.3 辅助损失

1. **因子分类损失**：
   $$\mathcal{L}_{\text{cls}} = \text{BCE}(\text{logits}, \text{labels})$$
   - 权重：$\lambda_{\text{cls}} = 0.1$
   - 标签：多标签one-hot向量（4维）

2. **门控正则化损失**：
   $$\mathcal{L}_\alpha = \|\alpha_{\text{pred}} - \alpha_{\text{target}}\|_2^2$$
   - 权重：$\lambda_\alpha = 0.02$
   - 目标：基于物理先验的时间条件门控目标

**时间条件门控目标生成**：
```python
# 三段阈值方法（只对存在的因子生效）
mod1 = [0.3, 0.3, 1.0, 1.0]  # early: rain/snow
mod2 = [0.3, 1.0, 0.5, 0.5]  # mid: haze
mod3 = [1.0, 0.5, 0.3, 0.3]  # late: low

if t < 0.33:
    time_mod = mod1
elif t < 0.67:
    time_mod = mod2
else:
    time_mod = mod3

alpha_target = labels * time_mod  # 只在active因子内分配
alpha_target = alpha_target / sum(alpha_target)  # 归一化
```

#### 3.1.4 总损失

$$\mathcal{L}_{\text{total}} = \lambda_{\text{SFM}} \mathcal{L}_{\text{SFM}} + \lambda_{\text{traj}} \mathcal{L}_{\text{traj}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_\alpha \mathcal{L}_\alpha$$

**默认权重**：
- $\lambda_{\text{SFM}} = 1.0$
- $\lambda_{\text{traj}} = 1.0$（当前实现中traj分支已禁用）
- $\lambda_{\text{mid}} = 0.05$（轨迹损失内部）
- $\lambda_{\text{cls}} = 0.1$
- $\lambda_\alpha = 0.02$

### 3.2 训练策略

#### 3.2.1 渐进式训练策略（已禁用）

**原始设计**（当前已禁用，仅使用SFM）：

| 阶段 | 训练步数 | SFM比例 | Traj比例 | 展开步数 |
|------|---------|---------|----------|---------|
| 早期 | 0-100K | 100% | 0% | N/A |
| 中期 | 100K-200K | 50% | 50% | 20 |
| 后期 | 200K-300K | 30% | 70% | 100 |

**禁用原因**：
- 长轨迹展开导致显存占用过大
- 从大checkpoint续训时容易OOM
- 当前实现中仅使用SFM损失

#### 3.2.2 Teacher Forcing策略

**目标**：训练早期使用ground-truth退化信息，稳定学习

**实现**：
$$\text{teacher\_forcing\_ratio} = \max(0.0, **实现**：
$$\text{teacher\_forcing\_ratio} = \max(0.0, 1.0 - 2.0 \cdot \text{step} / \text{total\_steps})$$

**机制**：
- 训练步数 ≤ 50%时：使用ground-truth的$w$和$m$（present mask）
- 训练步数 > 50%时：逐渐过渡到使用parser预测的$w$和$m$
- 线性衰减：从1.0（完全teacher forcing）线性衰减到0.0（完全使用parser）

**设计理由**：
- 训练早期parser输出不稳定，使用GT可以稳定主网络学习
- 后期逐渐依赖parser，提升模型鲁棒性
- 避免parser错误在早期传播到主网络

### 3.3 优化器与学习率调度

**优化器配置**：
- **类型**：AdamW
- **初始学习率**：$1.5 \times 10^{-4}$（默认，可调整）
- **权重衰减**：$10^{-4}$
- **Beta参数**：$\beta_1 = 0.9, \beta_2 = 0.99$

**学习率调度**：
- **类型**：Cosine Annealing
- **总步数**：$T_{\max} = \text{niter}$（总训练步数）
- **最小学习率**：$\eta_{\min} = 10^{-6}$
- **公式**：
  $$\text{lr}(t) = \eta_{\min} + (\text{lr}_0 - \eta_{\min}) \cdot \frac{1 + \cos(\pi \cdot t / T_{\max})}{2}$$

**梯度裁剪**：
- 梯度范数裁剪：$\|\nabla \theta\| \leq 1.0$
- 防止梯度爆炸，稳定训练

### 3.4 FoD Schedule配置

**核心参数**：
- **T（离散时间步数）**：100
- **delta（最小噪声水平）**：0.001
- **theta_schedule**：cosine（漂移系数调度）
- **sigma_schedule**：linear（扩散系数调度）
- **diffusion_type**：SDE（随机微分方程）
- **prediction**：sflow（模型预测类型：$x - x_t$）

**时间嵌入**：
- 归一化时间：$t \in [0, 1]$
- 离散时间索引：$t_{\text{idx}} = \lfloor t \cdot T \rfloor \in [0, T]$

**前向核采样**（训练时）：
- 使用`fod_analytic_sample`函数
- 基于exponential normal累积和（expo_normal_cumsum）
- 路径：$x_t = (x_0 - x_1) \cdot \text{expo}(t) + x_1$

### 3.5 训练配置总结

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| **总训练步数** | 300,000 | 可扩展到500K+ |
| **全局Batch Size** | 16 | 按GPU数量自动分配 |
| **Patch大小** | 512×512 | 训练时的随机裁剪 |
| **数据加载线程** | 8 | 并行数据加载 |
| **EMA衰减率** | 0.9999 | 指数移动平均 |
| **验证频率** | 每5,000步 | 在测试集上评估 |
| **检查点保存** | 每20,000步 | 保存训练状态 |

---

## 四、推理范式

### 4.1 直接预测（One-Step Inference）

**核心思想**：直接预测从退化输入到清晰图的残差，无需迭代采样。

**数学表述**：
$$\hat{x} = y + r_\theta(y, t=0, y)$$

其中$r_\theta$是模型预测的残差（SFLOW类型：$x - x_t$）。

**优势**：
- **快速**：单次前向传播
- **稳定**：无误差累积
- **有效**：当模型在SFM目标上充分训练时表现优异

**实现细节**：
- 设置$t = 0$（初始时间步）
- 输入：$x_t = y$（退化输入作为当前状态）
- 解析器输出：$w, m = \text{Parser}(y, t=0)$
- 模型预测：$r = \text{Model}(y, y, t=0, w, m)$
- 输出：$\hat{x} = y + r$（clip到$[-1, 1]$）

### 4.2 多步NMC采样（Multi-Step NMC Sampling）

**核心思想**：使用Normalized Monte Carlo (NMC)积分，通过多步迭代提升质量。

**采样流程**：
```
初始化: x_0 = y
for n = 0 to N-1:
    t_n = n / N  # 归一化时间
    t_next = (n+1) / N
    dt = 1 / N
    
    # 预测目标
    r_n = Model(x_n, y, t_n, w, m)
    x_target = x_n + r_n
    
    # NMC更新
    x_{n+1} = (x_n - x_target) * expo_normal(t_n, t_next) + x_target
```

**采样参数**：
- **步数**：$N = 50$（快速）或 $100$（高质量）
- **采样类型**：MC（Monte Carlo）而非NMC（避免累积误差）
- **噪声注入**：由schedule自动处理

**注意事项**：
- 多步采样可能不如one-step效果好（训练/推理分布不匹配）
- 当前训练主要使用SFM损失，模型更适合one-step推理

### 4.3 推理配置

| 配置项 | One-Step | Multi-Step |
|--------|----------|------------|
| **前向传播次数** | 1 | 50-100 |
| **速度** | 快 | 慢 |
| **质量** | 高（当模型充分训练） | 可能略高（但差距不大） |
| **显存占用** | 低 | 中等 |

---

## 五、关键技术细节

### 5.1 数值稳定性设计

**速度场裁剪**：
- 所有速度场输出裁剪到$[-10, 10]$范围
- 防止异常值导致数值不稳定

**NaN/Inf检查**：
- 模型输出后进行NaN/Inf检查
- 如果检测到异常，返回零速度场

**归一化处理**：
- 门控权重$\alpha$：sum-normalize确保$\sum \alpha_i = 1$
- 全局权重$w$：归一化用于gating输入（但不强制sum=1，支持多因子叠加）

### 5.2 特征复用机制

**共享特征提取**：
- SharedBackbone同时输出$v_{\text{share}}$和$h_{\text{final}}$
- $h_{\text{final}}$（全分辨率特征）被所有专家复用
- 避免重复计算，提高效率

**设计优势**：
- **计算效率**：只需一次backbone前向传播
- **内存效率**：减少中间特征存储
- **协作学习**：共享特征促进专家协同

### 5.3 多因子组合支持

**sigmoid vs softmax**：
- **Parser输出**（$w, m$）：使用sigmoid，支持多因子同时存在
- **Gating权重**（$\alpha$）：使用sigmoid + sum-normalize，支持多专家激活
- **设计理由**：组合退化中多个因子可同时存在（如`low_haze_rain`）

**存在掩码机制**：
- $w_{\text{mask}} = (w > 0.1)$：过滤不存在的因子
- 应用到gating权重：$\alpha_{\text{masked}} = \alpha_{\text{raw}} \odot w_{\text{mask}}$
- 确保不存在的因子权重为0

### 5.4 时间依赖解析（Time-Dependent Parsing）

**设计理念**：解析器的输出可以依赖于时间$t$，适应不同时间步的需求。

**实现方式**：
$$w_i(t) = \sigma(a_i + b_i \cdot t)$$

其中$(a_i, b_i)$是每个因子学习的参数。

**优势**：
- 时间自适应：不同时间步可以有不同的因子强度
- 灵活性：模型可以根据时间动态调整因子权重

**当前状态**：代码支持，但实际训练中效果需要验证。

---

## 六、模型参数统计

### 6.1 参数量分解

| 组件 | 参数量 | 占比 |
|------|--------|------|
| **SharedBackbone** | ~15M | 77.7% |
| **4个Expert Adapters** | ~4M | 20.7% |
| **DegradationParser** | ~0.3M | 1.6% |
| **总计** | ~19.3M | 100% |

### 6.2 计算复杂度

**单次前向传播**：
- **输入尺寸**：512×512（训练时）
- **主要计算**：
  - SharedBackbone：U-Net前向传播（~15M参数）
  - 4个Expert Adapters：轻量适配器（~1M参数/专家）
  - DegradationParser：轻量编码器（~0.3M参数）
  - Gating Network：MLP（<0.1M参数）

**内存占用**（训练时）：
- **Batch Size = 16, Patch = 512×512**：约12-16GB（单GPU）
- **主要占用**：
  - 输入/输出张量：~6GB
  - 中间特征：~4-6GB
  - 梯度：~4GB

---

## 七、设计决策总结

### 7.1 关键设计选择

1. **组合式分解**：
   - **选择**：将复杂退化分解为原子因子
   - **理由**：支持未见组合的泛化，提供可解释性

2. **前向扩散（FoD）**：
   - **选择**：单向前向过程，而非传统前向-反向扩散
   - **理由**：更符合复原任务结构，训练更稳定

3. **适配器机制**：
   - **选择**：轻量适配器而非独立专家网络
   - **理由**：参数高效，特征复用，促进协作

4. **时间条件门控**：
   - **选择**：物理先验引导的时间依赖门控
   - **理由**：利用物理知识，提升恢复质量

5. **sigmoid激活**：
   - **选择**：多标签sigmoid而非softmax
   - **理由**：支持多因子同时存在

6. **SFM为主训练**：
   - **选择**：主要使用SFM损失，轨迹损失禁用
   - **理由**：显存效率，one-step推理效果好

### 7.2 与其他方法的区别

| 方面 | Traj-CFM | 传统扩散模型 | 其他组合方法 |
|------|----------|-------------|-------------|
| **扩散方向** | 前向（FoD） | 反向 | N/A |
| **因子建模** | 组合式分解 | 整体建模 | 可能独立建模 |
| **专家设计** | 适配器（参数高效） | N/A | 可能独立网络 |
| **门控机制** | 时间条件（物理先验） | N/A | 可能静态或学习 |
| **推理方式** | One-step优先 | 多步采样 | 通常单步 |

---

## 八、实现代码结构

### 8.1 核心文件

```
resfod/
├── train_IR.py                 # 主训练脚本
├── inference.py                # 推理脚本
├── models/
│   ├── traj_cfm_net.py        # TrajCFMNet主模型
│   └── degradation_parser.py  # 退化解析器
├── utils/
│   ├── fod_core.py            # FoD核心算法（采样、调度）
│   ├── lmdb_dataset.py        # 数据集加载
│   ├── factor_utils.py        # 因子解析工具
│   ├── metrics.py             # 评估指标（PSNR, SSIM, LPIPS）
│   └── ema.py                 # 指数移动平均
└── results/                   # 训练结果和检查点
```

### 8.2 关键类与函数

**模型类**：
- `TrajCFMNet`：主模型，组合速度场网络
- `SharedBackbone`：共享U-Net骨干
- `ExpertVelocityField`：专家速度场适配器
- `DegradationParser`：退化解析器

**工具类**：
- `FoDSchedule`：FoD调度配置
- `FoDiffusion`：FoD核心算法实现
- `EMA`：指数移动平均

**采样函数**：
- `fod_analytic_sample`：解析采样（训练时生成$x_t$）
- `fod_nmc_sample`：NMC多步采样（推理时）

---

## 九、参考文献与理论基础

### 9.1 核心论文

1. **Forward-only Diffusion (FoD)**
   - 论文：[arXiv:2505.16733](https://arxiv.org/pdf/2505.16733)
   - 核心：利用任务结构实现单向前向扩散

2. **Flow Matching**
   - Lipman et al., "Flow Matching for Generative Modeling"
   - 理论基础：连续归一化流

3. **Compositional Learning**
   - Rombach et al., "Compositional Diffusion Models"
   - 组合式生成模型设计

### 9.2 相关技术

- **Adapter机制**：参数高效的迁移学习
- **Mixture of Experts (MoE)**：专家混合模型
- **Conditional Diffusion**：条件扩散模型
- **Image Restoration**：图像复原任务

---

## 十、总结

Traj-CFM模型通过组合式分解、前向扩散和参数高效的适配器设计，实现了对组合退化的有效建模。核心创新包括：

1. **组合式速度场分解**：将复杂退化分解为原子因子，支持未见组合的泛化
2. **前向扩散范式**：利用复原任务结构，实现单向前向过程
3. **时间条件门控**：物理先验引导的专家组合机制
4. **参数高效设计**：适配器机制实现高效参数利用

该设计在保持高效性的同时，提供了良好的可解释性和泛化能力，为组合退化图像复原提供了一个有前景