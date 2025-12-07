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
├── notebooks/         # Colab-compatible Jupyter notebooks
├── tests/             # Unit tests
└── results/           # Evaluation results

```

## Installation

```bash
pip install -r requirements.txt
```

## Usage (Google Colab)

**All functionality is available as Colab-compatible notebooks!** This project is designed to run entirely in Google Colab.

### Quick Start in Colab

**Start here:** Open `notebooks/00_GETTING_STARTED.ipynb` first! It has the complete guide.

**Minimal Pipeline (4 notebooks):**
1. **`05_train_script.ipynb`** - Train the model
2. **`06_eval_script.ipynb`** - Evaluate performance
3. **`07_export_embeddings.ipynb`** - Precompute embeddings
4. **`08_retrieve_script.ipynb`** - Use the model for search

**All Notebooks:**
- **`00_GETTING_STARTED.ipynb`** - **READ THIS FIRST!** Complete setup guide
- **`01_dataset_exploration.ipynb`** - Explore dataset (optional)
- **`02_training.ipynb`** - Interactive training (alternative to 05)
- **`03_evaluation.ipynb`** - Interactive evaluation (alternative to 06)
- **`04_inference_retrieval.ipynb`** - Interactive retrieval (alternative to 08)
- **`05_train_script.ipynb`** - Train model (use this!)
- **`06_eval_script.ipynb`** - Evaluate model (use this!)
- **`07_export_embeddings.ipynb`** - Export embeddings (use this!)
- **`08_retrieve_script.ipynb`** - Retrieval script (use this!)

**Important:** You don't run Python files directly! The `src/` directory contains code that notebooks import. Just run the notebooks.

Each notebook is self-contained and includes:
- Dependency installation (`%pip install ...`)
- Google Drive mounting (optional)
- All necessary imports
- Step-by-step execution cells

### Notebook Workflow

1. **Dataset Exploration** (`01_dataset_exploration.ipynb`)
   - Load and explore COCO dataset
   - Test data loaders
   - Build and test tokenizer
   - Visualize sample images

2. **Training** (`02_training.ipynb`)
   - Build vocabulary from dataset
   - Create data loaders with tokenization
   - Initialize model, optimizer, scheduler
   - Train with progress bars and checkpointing
   - Save best model automatically

3. **Evaluation** (`03_evaluation.ipynb`)
   - Load trained model checkpoint
   - Compute embeddings for validation set
   - Calculate Recall@1, Recall@5, Recall@10 metrics
   - Save evaluation results

4. **Inference & Retrieval** (`04_inference_retrieval.ipynb`)
   - Load trained model
   - Precompute image embeddings for search
   - Text-to-image search: "red sports car drifting on wet road" → find matching images
   - Image-to-text search: upload image → find matching captions
   - Visualize results with matplotlib

### Local/Command Line Scripts (Optional)

#### Training

```bash
python scripts/train.py --config configs/clip_coco_small.yaml
```

#### Evaluation

```bash
python scripts/eval.py --checkpoint checkpoints/best_model.pt --config configs/clip_coco_small.yaml
```

#### Retrieval

```bash
# Text to image
python scripts/retrieve.py --query "red sports car drifting on wet road" --checkpoint checkpoints/best_model.pt --config configs/clip_coco_small.yaml

# Image to text
python scripts/retrieve.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pt --config configs/clip_coco_small.yaml
```

#### Export Embeddings

```bash
python scripts/export_embeddings.py --checkpoint checkpoints/best_model.pt --config configs/clip_coco_small.yaml
```

#### Inference API (Optional - for production deployment)

```bash
python scripts/api.py --checkpoint checkpoints/best_model.pt --config configs/clip_coco_small.yaml --port 8000
```

The API provides endpoints:
- `POST /text-to-image?query=...` - Search images by text
- `POST /image-to-text` - Find captions for uploaded image
- `POST /image-to-image` - Find similar images

**Note:** For Colab usage, the notebooks provide all the functionality you need. The API is optional for production deployments.

## Dataset

This project uses the COCO 2017 dataset:
- Training images: `images/train2017/`
- Validation images: `images/val2017/`
- Annotations: `images/annotations_trainval2017/annotations/`

## License

MIT License - see LICENSE file for details.

