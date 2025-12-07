"""Training script entry point."""

import argparse
from pathlib import Path

import torch
import yaml

from src.data.coco_dataset import build_coco_dataloader
from src.models.clip_model import CLIPModel
from src.training.train_clip import get_lr_scheduler, save_checkpoint, train_epoch
from src.utils.tokenization import SimpleTokenizer


def collate_fn(batch, tokenizer, max_seq_length):
    """Custom collate function to tokenize captions."""
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]

    token_ids = [
        tokenizer.encode(cap, max_length=max_seq_length) for cap in captions
    ]
    token_tensor = torch.tensor(token_ids)

    mask = token_tensor == tokenizer.get_pad_token_id()

    return {
        "image": images,
        "text_tokens": token_tensor,
        "text_mask": mask,
        "caption": captions,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CLIP model on COCO dataset")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = Path(args.config).parent.parent
    checkpoint_dir = base_dir / config["training"]["save_dir"]
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    print("Building tokenizer vocabulary...")
    tokenizer = SimpleTokenizer(
        vocab_size=config["model"]["text"]["vocab_size"], min_freq=2
    )

    temp_loader = build_coco_dataloader(
        annotation_file=str(base_dir / config["data"]["train"]["annotation_file"]),
        image_dir=str(base_dir / config["data"]["train"]["image_dir"]),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        max_samples=5000,
    )

    all_captions = []
    for batch in temp_loader:
        all_captions.extend(batch["caption"])

    tokenizer.build_vocab(all_captions)
    print(f"Vocabulary size: {len(tokenizer)}")

    from torch.utils.data import DataLoader

    train_dataset = build_coco_dataloader(
        annotation_file=str(base_dir / config["data"]["train"]["annotation_file"]),
        image_dir=str(base_dir / config["data"]["train"]["image_dir"]),
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        subset_percentage=config["data"]["train"].get("subset_percentage"),
    ).dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(
            b, tokenizer, config["model"]["text"]["max_seq_length"]
        ),
    )

    print(f"Train batches: {len(train_loader)}")

    model = CLIPModel(
        vision_config=config["model"]["vision"],
        text_config=config["model"]["text"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    num_training_steps = len(train_loader) * config["training"]["num_epochs"]
    scheduler = get_lr_scheduler(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=num_training_steps,
    )

    scaler = (
        torch.cuda.amp.GradScaler() if config["training"]["use_amp"] else None
    )

    start_epoch = 0
    best_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["loss"]
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, config
        )
        print(f"Train loss: {train_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config["training"]["save_every"] == 0:
            checkpoint_path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                train_loss,
                checkpoint_dir,
                len(tokenizer),
            )
            print(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            best_path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                train_loss,
                checkpoint_dir,
                len(tokenizer),
                is_best=True,
            )
            print(f"Best model saved: {best_path}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()

