"""Evaluation script for retrieval tasks."""

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.clip_model import CLIPModel


def compute_recall_at_k(
    similarity_matrix: torch.Tensor, k_values: List[int] = [1, 5, 10]
) -> tuple:
    """Compute Recall@K for image-to-text and text-to-image retrieval."""
    batch_size = similarity_matrix.shape[0]
    labels = torch.arange(batch_size, device=similarity_matrix.device)

    # Image-to-text: for each image, find matching text
    i2t_recalls = {}
    for k in k_values:
        _, top_k_indices = similarity_matrix.topk(k, dim=1)
        correct = (top_k_indices == labels.unsqueeze(1)).any(dim=1).float()
        i2t_recalls[k] = correct.mean().item()

    # Text-to-image: for each text, find matching image
    t2i_recalls = {}
    for k in k_values:
        _, top_k_indices = similarity_matrix.t().topk(k, dim=1)
        correct = (top_k_indices == labels.unsqueeze(1)).any(dim=1).float()
        t2i_recalls[k] = correct.mean().item()

    return i2t_recalls, t2i_recalls


def evaluate_retrieval(
    model: CLIPModel,
    loader: DataLoader,
    device: torch.device,
    k_values: List[int] = [1, 5, 10],
) -> Dict:
    """Evaluate retrieval performance."""
    model.eval()

    image_embeddings = []
    text_embeddings = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing embeddings"):
            images = batch["image"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_mask = batch["text_mask"].to(device)

            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(text_tokens, text_mask)

            image_embeddings.append(img_emb.cpu())
            text_embeddings.append(txt_emb.cpu())

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(image_embeddings, text_embeddings.t())

    # Compute metrics
    i2t_recalls, t2i_recalls = compute_recall_at_k(similarity_matrix, k_values)

    return {
        "image_to_text": i2t_recalls,
        "text_to_image": t2i_recalls,
        "similarity_matrix_shape": list(similarity_matrix.shape),
    }

