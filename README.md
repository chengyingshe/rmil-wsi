# RMIL: Recurrent Multiple Instance Learning for Whole Slide Image Analysis

PyTorch implementation of Recurrent Multiple Instance Learning (RMIL) models for whole slide image classification and analysis. This project extends the original DSMIL framework with various recurrent architectures including GRU, LSTM, Mamba, and TTT (Token-to-Token Transformer).

## Overview

This repository implements multiple recurrent MIL architectures for WSI analysis:

- **RMIL-GRU**: GRU-based recurrent MIL with intelligent patch selection
- **RMIL-LSTM**: LSTM-based recurrent MIL with attention mechanisms  
- **RMIL-Mamba**: State space model-based MIL for long sequence modeling
- **RMIL-TTT**: Token-to-Token Transformer MIL with linear/MLP variants

## Key Features

- Multiple recurrent MIL architectures for WSI analysis
- Intelligent patch selection with hybrid strategies
- Cross-validation and train/valid/test evaluation schemes
- Support for multiple datasets (TCGA, Camelyon16, custom datasets)
- Comprehensive visualization tools for WSI patch analysis
- WandB integration for experiment tracking
- Flexible model parameter configuration via JSON

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.8+
- CUDA (for GPU training)

### Setup
```bash
# Clone repository
git clone https://github.com/XXXXXX/rmil-wsi.git
cd rmil-wsi

# Install mamba
pip install -e requirements/causal_conv1d
pip install -e requirements/mamba-1p1p1

# Install other dependencies
pip install -r requirements/requirements.txt
```

## Dataset Preparation

### Dataset Download

It'll come soon.

### Dataset Structure
```
datasets/
├── mydatasets/
│   ├── TCGA-lung-uni2/
│   │   ├── pt_files/          # Feature vectors (.pt files)
│   │   └── TCGA.csv           # Dataset labels
│   └── CAMELYON16-uni2/
│       ├── pt_files/          # Feature vectors (.pt files)
│       └── Camelyon16.csv     # Dataset labels
```

## Training

### Basic Training Commands

**Train on TCGA Lung dataset:**
```bash
python train.py --model rmil_gru --dataset_dir datasets/mydatasets/TCGA-lung-uni2 --label_file TCGA.csv
```

**Train on Camelyon16 dataset:**
```bash
python train.py --model rmil_gru --dataset_dir datasets/mydatasets/CAMELYON16-uni2 --label_file Camelyon16.csv --num_classes 1
```

**Train with custom parameters:**
```bash
python train.py \
    --model rmil_gru \
    --dataset_dir datasets/mydatasets/TCGA-lung-uni2 \
    --label_file TCGA.csv \
    --num_epochs 100 \
    --lr 0.0002 \
    --eval_scheme 5-fold-cv \
    --use_wandb
```

## Visualization

Generate patch-level score visualizations for WSI analysis:

```bash
python visualization.py \
    --model rmil_gru \
    --model_weights weights/best_model.pth \
    --wsi_id TCGA-55-8091-01Z-00-DX1 \
    --wsi_dir ./sample_wsi
```

The visualization script supports:
- Red color mapping (deep red for positive, light red for negative)
- Custom patch sizes and transparency
- Multiple WSI formats
- Batch processing capabilities

## Citation

If you use this code in your research, please cite:

```bibtex

```
