# Experiment Strategy and Configuration Guide

This document describes the experiment configurations and training strategy for the CLIP model on COCO dataset.

## Configuration Files

### 1. `clip_coco_tiny.yaml`
**Purpose**: Ultra-small subset for debugging and quick testing
- **Training data**: 1% of COCO train set (~500 samples)
- **Validation data**: 5% of COCO val set (~250 samples)
- **Batch size**: 16
- **Epochs**: 5
- **Expected training time**: ~5-10 minutes on GPU
- **Use case**: Quick sanity checks, debugging data pipeline

### 2. `clip_coco_small.yaml`
**Purpose**: Small subset for development and experimentation
- **Training data**: 10% of COCO train set (~5,000 samples)
- **Validation data**: 20% of COCO val set (~1,000 samples)
- **Batch size**: 32
- **Epochs**: 10
- **Expected training time**: ~30-60 minutes on GPU
- **Use case**: Testing training loop, hyperparameter tuning

### 3. `clip_coco_medium.yaml`
**Purpose**: Medium-scale training for realistic results
- **Training data**: 50% of COCO train set (~27,000 samples)
- **Validation data**: Full COCO val set (~5,000 samples)
- **Batch size**: 64
- **Epochs**: 20
- **Expected training time**: ~4-6 hours on GPU
- **Use case**: Production-ready model training

### 4. `clip_coco_full.yaml`
**Purpose**: Full dataset training for best performance
- **Training data**: Full COCO train set (~118,000 samples)
- **Validation data**: Full COCO val set (~5,000 samples)
- **Batch size**: 128
- **Epochs**: 50
- **Expected training time**: ~2-3 days on GPU
- **Use case**: Final model training, research experiments

## Training Strategy

### Learning Rate Schedule
- **Warmup**: Linear warmup for first N steps (configurable)
- **Main training**: Cosine annealing decay
- **Initial LR**: 1e-4 (adjustable per config)

### Optimization
- **Optimizer**: AdamW
- **Weight decay**: 0.01
- **Gradient clipping**: 1.0
- **Mixed precision**: Enabled (AMP) for faster training

### Model Architecture
- **Vision encoder**: ResNet-50 backbone with projection head
- **Text encoder**: Transformer-based with configurable layers
- **Projection dimension**: 256 (shared embedding space)
- **Temperature**: 0.07 (for contrastive loss)

### Expected Results

#### Tiny Config
- Recall@1: ~5-10% (baseline)
- Training time: < 10 minutes

#### Small Config
- Recall@1: ~15-25%
- Training time: ~1 hour

#### Medium Config
- Recall@1: ~30-40%
- Training time: ~6 hours

#### Full Config
- Recall@1: ~40-50% (target)
- Training time: ~2-3 days

## Running Experiments

### Quick Start (Tiny)
```bash
python scripts/train.py --config configs/clip_coco_tiny.yaml
```

### Development (Small)
```bash
python scripts/train.py --config configs/clip_coco_small.yaml
```

### Production (Medium)
```bash
python scripts/train.py --config configs/clip_coco_medium.yaml
```

### Full Training
```bash
python scripts/train.py --config configs/clip_coco_full.yaml
```

### Resume Training
```bash
python scripts/train.py --config configs/clip_coco_medium.yaml --resume checkpoints/checkpoint_epoch_10.pt
```

## Evaluation

After training, evaluate with:
```bash
python scripts/eval.py --checkpoint checkpoints/best_model.pt --config configs/clip_coco_small.yaml
```

## Notes

- All configs use the same model architecture base
- Adjust batch size based on available GPU memory
- Larger datasets benefit from more training epochs
- Monitor validation loss to prevent overfitting
- Use tensorboard or similar for training visualization

