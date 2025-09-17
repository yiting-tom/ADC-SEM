# TinyViT SSL Training for 15-Channel Images

This project implements self-supervised learning (SSL) training for TinyViT models with 15-channel input images at 460x460 resolution using the lightly-ssl framework.

## Features

- **Multi-channel Support**: Modified TinyViT architecture to handle 15-channel input
- **SSL Training**: SimCLR with geometric augmentations; additional algorithms below
- **Algorithm Support**: MoCo v3 (`train_moco_v3.py`), DINOv2 (`train_dino_v2.py`), DINOv3 (`train_dino_v3.py`)
- **Flexible Data Loading**: Support for both directory of .npy files and HDF5 datasets
- **Intelligent Weight Initialization**: Averages RGB pretrained weights across new channels
- **Production Ready**: PyTorch Lightning integration with proper logging and checkpointing

## Installation

```bash
pip install -r requirements.txt
# or
make install
```

## Quick Start

### Training with Directory of Image Files

**For NumPy files:**
```bash
python train_ssl.py \
    --data_path /path/to/your/15_channel_images/ \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-3
```

**For TIFF files:**
```bash
python train_ssl.py \
    --data_path /path/to/your/tiff_images/ \
    --file_extension tiff \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-3
```

### Training with HDF5 Dataset

```bash
python train_ssl.py \
    --data_path /path/to/your/dataset.h5 \
    --use_h5 \
    --h5_dataset_key images \
    --batch_size 32 \
    --epochs 100
```

### Smoke Test (All Methods)
Run a tiny forward/backward step for SimCLR, MoCo v3, DINOv2, and DINOv3 without downloading weights:
```bash
python smoke_test.py --data_path ./example_data --file_extension npy
```
Use `--methods` to select subsets, e.g. `--methods simclr moco`.

Or via Makefile:
```bash
make smoke                           # default: example_data, npy, all methods
make smoke FILE_EXT=tiff             # test TIFF loader
make smoke METHODS="simclr moco"      # run a subset
make smoke DATA_PATH=./data          # custom data directory
```

### Train via Makefile
Convenience targets for common runs (set DATA_PATH/FILE_EXT/CHANNELS/EPOCHS/BS/LR/PRETRAINED_PATH/NO_TIMM):
```bash
# SimCLR
make train-simclr DATA_PATH=./data FILE_EXT=npy CHANNELS=15 EPOCHS=50 PRETRAINED_PATH=/path/to/tinyvit.pth NO_TIMM=1

# MoCo v3
make train-moco DATA_PATH=./tiff_data FILE_EXT=tiff CHANNELS=15 EPOCHS=50 MOCO_LR=6e-2 PRETRAINED_PATH=/path/to/tinyvit.pth NO_TIMM=1

# DINOv2
make train-dino2 DATA_PATH=./data FILE_EXT=npy CHANNELS=15 EPOCHS=50 PRETRAINED_PATH=/path/to/tinyvit.pth NO_TIMM=1

# DINOv3
make train-dino3 DATA_PATH=./data FILE_EXT=npy CHANNELS=15 EPOCHS=50 PRETRAINED_PATH=/path/to/tinyvit.pth NO_TIMM=1
```

## Data Format Requirements

### Option 1: Directory of Image Files
**NumPy Files (.npy)**
- Each image should be a `.npy` file with shape `(460, 460, 15)` or `(15, 460, 460)`
- Values should be normalized to [0, 1] or [0, 255] range

**TIFF Files (.tiff/.tif)**
- Multi-channel TIFF files with shape `(15, 460, 460)`
- Values in any numeric range (will be normalized automatically)
- Requires `tifffile` package for proper multi-channel support

Directory structure:
```
data/
├── image_001.npy  (or .tiff)
├── image_002.npy  (or .tiff)
└── ...
```

### Option 2: HDF5 Dataset
- Single `.h5` file with dataset containing all images
- Dataset shape should be `(N, 15, 460, 460)` where N is number of images
- Example creation:
```python
import h5py
import numpy as np

# Create HDF5 file
with h5py.File('dataset.h5', 'w') as f:
    # images: shape (N, 15, 460, 460)
    f.create_dataset('images', data=your_image_array)
```

## Model Variants

Available TinyViT models:
- `tiny_vit_5m_224.dist_in22k_ft_in1k` (5M parameters, 224px)
- `tiny_vit_11m_224.dist_in22k_ft_in1k` (11M parameters, 224px)
- `tiny_vit_21m_224.dist_in22k_ft_in1k` (21M parameters, 224px)
- `tiny_vit_21m_512.dist_in22k_ft_in1k` (21M parameters, 512px, default)

## Usage Examples

### Basic Training
```bash
python train_ssl.py --data_path ./data --batch_size 64 --epochs 200
```

### Using KAN Projection Head
Enable KAN (Kolmogorov–Arnold Network) projector instead of the default MLP:
```bash
python train_ssl.py \
  --data_path ./data \
  --projection_head kan \
  --kan_num_knots 16 \
  --kan_x_min -3.0 \
  --kan_x_max 3.0
```
Notes:
- KAN projector is a two-layer KAN with piecewise-linear 1D splines per feature and an optional linear skip path.
- Adjust `--kan_num_knots` and range if your feature scales differ.

### Advanced Configuration
```bash
python train_ssl.py \
    --data_path ./data \
    --model_name tiny_vit_21m_512.dist_in22k_ft_in1k \
    --batch_size 64 \
    --epochs 200 \
    --lr 5e-4 \
    --num_workers 16 \
    --accelerator gpu \
    --devices 1 \
    --output_dir ./experiments/exp_001
```

### Hyperparameter Optimization with Optuna

**For NumPy files:**
```bash
python optimize_hyperparams.py \
    --data_path ./data \
    --n_trials 100 \
    --max_epochs 50 \
    --study_name my_ssl_optimization \
    --save_plots
```

**For TIFF files:**
```bash
python optimize_hyperparams.py \
    --data_path ./tiff_data \
    --file_extension tiff \
    --n_trials 100 \
    --max_epochs 50 \
    --study_name my_ssl_optimization \
    --save_plots
```

Then train with optimized parameters:

**For NumPy files:**
```bash
python train_optimized.py \
    --config_path ./optimization_results/best_config.json \
    --data_path ./data \
    --epochs 200
```

**For TIFF files:**
```bash
python train_optimized.py \
    --config_path ./optimization_results/best_config.json \
    --data_path ./tiff_data \
    --file_extension tiff \
    --epochs 200
```

### Training with Custom Channels
```bash
python train_ssl.py \
    --data_path ./data \
    --num_channels 10 \
    --batch_size 32 \
    --epochs 100
```

### MoCo v3 SSL Training
```bash
# NPY directory
python train_moco_v3.py --data_path ./data --file_extension npy --num_channels 15 --batch_size 32 --epochs 50 \
  --pretrained_path /path/to/local_tinyvit_weights.pth --no_timm_pretrained

# TIFF directory
python train_moco_v3.py --data_path ./tiff_data --file_extension tiff --num_channels 15

# HDF5 dataset
python train_moco_v3.py --data_path ./dataset.h5 --use_h5 --h5_dataset_key images --num_channels 15 \
  --pretrained_path /path/to/local_tinyvit_weights.pth --no_timm_pretrained

# Quick sanity run
python train_moco_v3.py --data_path ./data --batch_size 4 --epochs 1 --num_workers 0
```
Notes:
- Uses NTXent with memory bank and momentum encoder updates via cosine schedule.
- Saves backbone to `outputs/tinyvit_moco_v3_backbone.pth` and projection head to `outputs/tinyvit_moco_v3_proj_head.pth`.

### Fine-tuning for Classification (TIFF)
File naming: `<label_id>_<...rest...>.tiff`. Example: `26_35_U2523.20_1368.tiff` → label id `26`.

- Binary classification: label id `1` is Normal (class 1); all others are Abnormal (class 0).
  ```bash
  python train_finetune.py --data_path ./tiff_data --task binary \
    --num_channels 15 --epochs 50 --batch_size 32 \
    --pretrained_path /path/to/tinyvit.pth --no_timm_pretrained
  ```

- Multiclass classification: classes discovered from dataset (unique first tokens).
  ```bash
  python train_finetune.py --data_path ./tiff_data --task multiclass \
    --num_channels 15 --epochs 50 --batch_size 32 \
    --pretrained_path /path/to/tinyvit.pth --no_timm_pretrained
  ```

Makefile shortcuts:
```bash
make train-finetune-binary DATA_PATH=./tiff_data CHANNELS=15 PRETRAINED_PATH=/path/to/tinyvit.pth NO_TIMM=1
make train-finetune-multi  DATA_PATH=./tiff_data CHANNELS=15 PRETRAINED_PATH=/path/to/tinyvit.pth NO_TIMM=1

# Auto-pick latest SSL backbone from outputs/ and fine-tune
make train-finetune-auto DATA_PATH=./tiff_data TASK=binary NO_TIMM=1
make train-finetune-auto DATA_PATH=./tiff_data TASK=multiclass NO_TIMM=1
```

Outputs:
- Model weights: `outputs/tinyvit_finetune_<task>_<timestamp>.pth`
- Confusion matrix (CSV): `outputs/tinyvit_finetune_<task>_<timestamp>_confusion_matrix.csv`
- Metrics (JSON): `outputs/tinyvit_finetune_<task>_<timestamp>_metrics.json` including per-class precision/recall/F1 and train/val loss curves.

## Output Files

After training, you'll get:
- `tinyvit_ssl_model.ckpt`: Full PyTorch Lightning checkpoint
- `tinyvit_ssl_model_backbone.pth`: Just the backbone weights for downstream tasks
 - MoCo v3: `outputs/tinyvit_moco_v3_backbone.pth` and projection head `..._proj_head.pth`
 - DINOv2/DINOv3: student backbone and head saved in `outputs/`

## Using Pretrained Backbone

```python
import torch
from model import TinyViTBackbone

# Load pretrained SSL backbone
backbone = TinyViTBackbone(num_channels=15)
backbone.load_state_dict(torch.load('tinyvit_ssl_model_backbone.pth'))

# Use for downstream tasks
features = backbone(your_15_channel_input)  # Shape: (batch_size, feature_dim)
```

## Local Pretrained Weights

All training CLIs support local TinyViT weights and disabling online downloads:
- Pass `--pretrained_path /path/to/tinyvit_weights.pth` and `--no_timm_pretrained`.
- Examples:
  - SimCLR: `python train_ssl.py --data_path ./data --pretrained_path /path/to/tinyvit.pth --no_timm_pretrained`
  - MoCo v3: `python train_moco_v3.py --data_path ./data --pretrained_path /path/to/tinyvit.pth --no_timm_pretrained`
  - DINOv2: `python train_dino_v2.py --data_path ./data --pretrained_path /path/to/tinyvit.pth --no_timm_pretrained`
  - DINOv3: `python train_dino_v3.py --data_path ./data --pretrained_path /path/to/tinyvit.pth --no_timm_pretrained`

## Key Files

- `dataset.py`: Custom dataset classes for multi-channel data
- `model.py`: TinyViT architecture modifications and utilities
- `ssl_training.py`: SSL training pipeline with lightly integration
- `train_ssl.py`: Main training script with CLI interface
- `moco_training.py`: MoCo model, collate, and trainer for multi-channel TinyViT
- `train_moco_v3.py`: CLI entrypoint for MoCo v3-style SSL training
 
### DINOv2 SSL Training
```bash
# NPY directory
python train_dino_v2.py --data_path ./data --file_extension npy --num_channels 15 --batch_size 32 --epochs 50 \
  --pretrained_path /path/to/local_tinyvit_weights.pth --no_timm_pretrained

# TIFF directory
python train_dino_v2.py --data_path ./tiff_data --file_extension tiff --num_channels 15

# HDF5 dataset
python train_dino_v2.py --data_path ./dataset.h5 --use_h5 --h5_dataset_key images --num_channels 15 \
  --pretrained_path /path/to/local_tinyvit_weights.pth --no_timm_pretrained

# Quick sanity run
python train_dino_v2.py --data_path ./data --batch_size 4 --epochs 1 --num_workers 0
```
Notes:
- Student/teacher with EMA and multi-crop views; geometric-only transforms.
- Uses DINOLoss with temperature warmup and centering.

### DINOv3 SSL Training
```bash
# NPY directory
python train_dino_v3.py --data_path ./data --file_extension npy --num_channels 15 --batch_size 32 --epochs 50 \
  --pretrained_path /path/to/local_tinyvit_weights.pth --no_timm_pretrained

# TIFF directory
python train_dino_v3.py --data_path ./tiff_data --file_extension tiff --num_channels 15

# HDF5 dataset
python train_dino_v3.py --data_path ./dataset.h5 --use_h5 --h5_dataset_key images --num_channels 15 \
  --pretrained_path /path/to/local_tinyvit_weights.pth --no_timm_pretrained

# Quick sanity run
python train_dino_v3.py --data_path ./data --batch_size 4 --epochs 1 --num_workers 0
```
Notes:
- DINOv3-style defaults: 2 global + 8 local crops, proj_dim=384, cosine EMA and LR.
- `optuna_optimization.py`: Hyperparameter optimization with Optuna
- `optimize_hyperparams.py`: CLI script for running optimization
- `train_optimized.py`: Train with optimized hyperparameters
- `dino_training.py`: DINOv2-style trainer with multi-crop and EMA teacher
- `train_dino_v2.py`: CLI entrypoint for DINOv2-style SSL training
- `dino_v3_training.py`: DINOv3-style trainer with expanded local crops and larger head
- `train_dino_v3.py`: CLI entrypoint for DINOv3-style SSL training

## Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | Required | Path to data directory or .h5 file |
| `--use_h5` | False | Use HDF5 dataset format |
| `--h5_dataset_key` | 'images' | Key for dataset in HDF5 file |
| `--file_extension` | 'npy' | Image file format (npy, tiff, tif) |
| `--num_channels` | 15 | Number of input channels |
| `--model_name` | tiny_vit_21m_512... | TinyViT model variant |
| `--projection_head` | mlp | Projection head: `mlp` or `kan` |
| `--kan_num_knots` | 16 | Spline knots per input (KAN) |
| `--kan_x_min` | -3.0 | Input range minimum (KAN) |
| `--kan_x_max` | 3.0 | Input range maximum (KAN) |
| `--kan_no_skip` | False | Disable KAN linear skip connection |
| `--batch_size` | 32 | Training batch size |
| `--epochs` | 100 | Number of training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--num_workers` | 8 | Data loader workers |
| `--accelerator` | auto | Hardware accelerator |
| `--devices` | 1 | Number of devices |
| `--save_path` | tinyvit_ssl_model.ckpt | Model save path |
| `--output_dir` | ./outputs | Output directory |

## Hyperparameter Optimization

The project includes Optuna-based automatic hyperparameter optimization to find the best training configuration.

### Optimization Process

1. **Run Optimization**: The optimizer searches through hyperparameter space including:
   - Learning rate (1e-5 to 1e-1, log scale)
   - Batch size (8, 16, 32, 64, 128)
   - Temperature for contrastive loss (0.05 to 0.5)
   - Projection dimensions (64, 128, 256, 512)
   - Weight decay (1e-8 to 1e-3, log scale)
   - Augmentation parameters (flip probabilities, blur, scaling)
   - Model architecture variants

2. **Early Stopping**: Uses median pruning to stop unpromising trials early

3. **Visualization**: Generates interactive plots for optimization analysis

### Optimization Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_trials` | 100 | Number of optimization trials |
| `--max_epochs` | 50 | Max epochs per trial |
| `--study_name` | ssl_optimization | Study name for tracking |
| `--storage` | None | Database URL for persistence |
| `--sampler` | tpe | Sampling strategy (tpe/random/cmaes) |
| `--pruner` | median | Pruning strategy for early stopping |
| `--save_plots` | False | Generate visualization plots |

### Example Optimization Workflow

```bash
# 1. Run hyperparameter search
python optimize_hyperparams.py \
    --data_path ./data \
    --n_trials 100 \
    --max_epochs 50 \
    --study_name ssl_opt_15ch \
    --save_plots \
    --output_dir ./opt_results

# 2. Train with best parameters
python train_optimized.py \
    --config_path ./opt_results/best_config.json \
    --data_path ./data \
    --epochs 200

# 3. View optimization plots (saved in opt_results/plots/)
```

## Performance Notes

- Use mixed precision training (automatic with PyTorch Lightning on GPU)
- Adjust batch size based on your GPU memory (32 works for most 16GB GPUs)
- Use multiple workers for data loading to avoid bottlenecks
- The model automatically applies geometric augmentations suitable for multi-channel data
- **Recommended**: Use hyperparameter optimization for best results - can improve performance by 20-40%

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient checkpointing
2. **Slow Training**: Increase num_workers or use faster storage
3. **Poor Convergence**: Try different learning rates or model variants

### Debug Mode

Add these flags for debugging:
```bash
python train_ssl.py --data_path ./data --batch_size 4 --epochs 1 --num_workers 0
```
