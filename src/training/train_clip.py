"""Training loop for CLIP model."""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.clip_model import CLIPModel, compute_clip_loss


def train_epoch(
    model: CLIPModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    config: dict,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        text_mask = batch["text_mask"].to(device)

        optimizer.zero_grad()

        if config["training"]["use_amp"] and scaler:
            with autocast():
                outputs = model(images, text_tokens, text_mask)
                loss = compute_clip_loss(outputs["logits"])

            scaler.scale(loss).backward()
            if config["training"]["gradient_clip_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["gradient_clip_norm"]
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, text_tokens, text_mask)
            loss = compute_clip_loss(outputs["logits"])
            loss.backward()
            if config["training"]["gradient_clip_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["gradient_clip_norm"]
                )
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % config["logging"]["log_every"] == 0:
            pbar.set_postfix({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

    return total_loss / num_batches


def get_lr_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """Create cosine annealing scheduler with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: CLIPModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    tokenizer_vocab_size: int,
    is_best: bool = False,
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "tokenizer_vocab_size": tokenizer_vocab_size,
    }

    if is_best:
        checkpoint_path = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path

