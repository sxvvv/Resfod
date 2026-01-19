# ResFoD: Compositional Flow Matching for Image Restoration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**ResFoD** is a novel compositional flow matching framework for image restoration that decomposes complex combined degradations into atomic factors. This repository contains the core implementation of the Traj-CFM (Trajectory-aware Compositional Flow Matching) model.

## 🎯 Key Features

- **Compositional Decomposition**: Models combined degradations as compositions of atomic factors (low-light, haze, rain, snow)
- **Forward-Only Diffusion**: Single-forward diffusion process optimized for restoration tasks
- **Parameter-Efficient Design**: Shared backbone network with lightweight expert adapters
- **Time-Conditioned Gating**: Physics-informed gating mechanism that emphasizes different experts at different time stages
- **One-Step Inference**: Direct prediction mode for fast and efficient inference

## 📋 Table of Contents

- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Core Components](#core-components)
- [Usage](#usage)
- [Methodology](#methodology)
- [Citation](#citation)
- [License](#license)

## 🔧 Installation

### Requirements

```bash
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
Pillow>=8.0.0
lmdb>=1.0.0
tqdm>=4.60.0
```

### Setup

```bash
git clone https://github.com/yourusername/resfod.git
cd resfod
pip install -r requirements.txt
```

## 🚀 Usage

### Model Initialization

```python
from models.traj_cfm_net import TrajCFMNet
from models.degradation_parser import DegradationParser

# Create model
model = TrajCFMNet(
    in_ch=6,              # [x_t, y] concatenated
    out_ch=3,             # RGB output
    base_ch=64,           # Base channels
    ch_mult=(1, 2, 4, 4), # Channel multipliers
    emb_dim=256,          # Time embedding dimension
    use_parser=True,      # Enable degradation parser
)

# Time-dependent degradation parser
parser = DegradationParser(
    in_ch=3,
    base_ch=32,
    emb_dim=128,
    use_time_dependent=True,
)
```

### Forward Pass

```python
import torch

# Inputs
x_t = torch.randn(B, 3, H, W)  # Current state
y = torch.randn(B, 3, H, W)    # Degraded input
t = torch.rand(B)               # Time [0, 1]

# Get parser outputs
w, m, logits = parser(y, t)

# Model forward
v_pred = model(x_t, y, t, w=w, m=m, deg_name=None)
```

### FoD Sampling

```python
from utils.fod_core import FoDSchedule, fod_nmc_sample

# Create schedule
schedule = FoDSchedule(
    T=100,
    delta=0.001,
    device=device,
    prediction="sflow",
)

# NMC sampling
x_hat = fod_nmc_sample(
    model, y,
    n_steps=50,
    schedule=schedule,
    w=w, m=m,
    deg_name=None,
)
```

## 📖 Methodology

### Forward-Only Diffusion (FoD)

Unlike traditional diffusion models that require forward-backward processes, FoD uses a single-forward process optimized for restoration:

$$\frac{dx_t}{dt} = v_\theta(x_t, t, y)$$

where:
- $x_0 = y$ (initial state = degraded input)
- $x_1 = x$ (target state = clean image)
- $t \in [0,1]$ (normalized time)

### Time-Conditioned Gating

The gating mechanism uses physics-informed priors:

- **Early stage** ($t < 0.33$): Emphasize rain/snow experts (high-frequency noise removal)
- **Mid stage** ($0.33 \leq t < 0.67$): Emphasize haze expert (mid-frequency dehazing)
- **Late stage** ($t \geq 0.67$): Emphasize low-light expert (global enhancement)

### Training Strategy

The model is trained using:
- **SFM (Straight Flow Matching)**: Main training objective
- **Classification Loss**: Supervises degradation parser
- **Alpha Supervision**: Guides gating mechanism with physics priors

## 🔬 Technical Details

### Compositional Decomposition

- **Atomic Factors**: 4 factors (low-light, haze, rain, snow)
- **Compositional Property**: Can handle unseen combinations by composing atomic experts
- **Parameter Efficiency**: ~18.8M parameters with shared backbone + lightweight adapters

### Expert Design

- **Shared Backbone**: Full U-Net (base_ch=64, ch_mult=[1,2,4,4])
- **Expert Adapters**: Lightweight (time MLP + 1×1 conv + output projection)
- **Spatial Modulation**: Expert outputs modulated by spatial intensity maps $m_i$

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions and issues, please open an issue on GitHub.

---

**Note**: This repository contains only the core model implementation. Training and inference scripts are not included to keep the repository focused on the core methodology.
