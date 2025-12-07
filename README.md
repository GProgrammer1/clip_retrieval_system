# CLIP Model for COCO Dataset

A PyTorch implementation of a CLIP-style model trained on the COCO 2017 dataset. This project implements a contrastive learning approach to learn joint embeddings for images and text captions.

## Overview

This project builds a vision-language model that can:
- Given an image, find the matching caption
- Given a caption, find the matching image

The model consists of:
- **Vision Encoder**: ResNet-50 or ViT backbone with projection head
- **Text Encoder**: Transformer-based encoder for caption embeddings
- **Contrastive Training**: InfoNCE loss (similar to CLIP)
- **Retrieval System**: Image ↔ text search using learned embeddings

## Features

- Training on COCO 2017 dataset (train/val splits)
- Contrastive learning with InfoNCE loss
- Image-to-text and text-to-image retrieval
- Evaluation metrics (Recall@1, Recall@5, Recall@10)
- CLI retrieval interface
- Inference API for real-time queries

## Project Structure

```
CLIP_model/
├── src/
│   ├── data/          # Dataset loaders
│   ├── models/        # Model architectures
│   ├── training/      # Training loop
│   ├── eval/          # Evaluation scripts
│   └── utils/         # Utilities (tokenization, etc.)
├── configs/           # Training configurations
├── scripts/           # Entry point scripts
└── results/           # Evaluation results

```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train.py --config configs/clip_coco_small.yaml
```

### Evaluation

```bash
python scripts/eval.py --checkpoint checkpoints/best_model.pt --config configs/clip_coco_small.yaml
```

### Retrieval

```bash
# Text to image
python scripts/retrieve.py --query "red sports car drifting on wet road" --checkpoint checkpoints/best_model.pt

# Image to text
python scripts/retrieve.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pt
```

### Inference API

```bash
python scripts/api.py --checkpoint checkpoints/best_model.pt --port 8000
```

## Dataset

This project uses the COCO 2017 dataset:
- Training images: `images/train2017/`
- Validation images: `images/val2017/`
- Annotations: `images/annotations_trainval2017/annotations/`

## License

MIT License - see LICENSE file for details.

