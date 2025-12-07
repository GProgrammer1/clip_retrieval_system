"""Export precomputed embeddings for images and captions."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.coco_dataset import build_coco_dataloader
from src.models.clip_model import CLIPModel
from src.utils.tokenization import SimpleTokenizer


def collate_fn(batch, tokenizer, max_seq_length):
    """Custom collate function."""
    images = torch.stack([item["image"] for item in batch])
    captions = [item["caption"] for item in batch]
    image_ids = [item["image_id"] for item in batch]

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
        "image_id": image_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Export embeddings")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="embeddings")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = Path(args.config).parent.parent

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CLIPModel(
        vision_config=config["model"]["vision"],
        text_config=config["model"]["text"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Build tokenizer
    tokenizer = SimpleTokenizer(
        vocab_size=config["model"]["text"]["vocab_size"], min_freq=2
    )

    temp_loader = build_coco_dataloader(
        annotation_file=str(base_dir / config["data"]["val"]["annotation_file"]),
        image_dir=str(base_dir / config["data"]["val"]["image_dir"]),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        max_samples=5000,
    )

    all_captions = []
    for batch in temp_loader:
        all_captions.extend(batch["caption"])
    tokenizer.build_vocab(all_captions)

    # Create data loader
    from torch.utils.data import DataLoader

    val_dataset = build_coco_dataloader(
        annotation_file=str(base_dir / config["data"]["val"]["annotation_file"]),
        image_dir=str(base_dir / config["data"]["val"]["image_dir"]),
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=2,
        subset_percentage=config["data"]["val"].get("subset_percentage"),
    ).dataset

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: collate_fn(
            b, tokenizer, config["model"]["text"]["max_seq_length"]
        ),
    )

    # Compute embeddings
    image_embeddings = []
    text_embeddings = []
    image_ids = []
    captions = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            text_mask = batch["text_mask"].to(device)

            img_emb = model.encode_image(images)
            txt_emb = model.encode_text(text_tokens, text_mask)

            image_embeddings.append(img_emb.cpu().numpy())
            text_embeddings.append(txt_emb.cpu().numpy())
            image_ids.extend(batch["image_id"])
            captions.extend(batch["caption"])

    image_embeddings = np.concatenate(image_embeddings, axis=0)
    text_embeddings = np.concatenate(text_embeddings, axis=0)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    np.save(output_dir / "image_embeddings.npy", image_embeddings)
    np.save(output_dir / "text_embeddings.npy", text_embeddings)

    metadata = {
        "image_ids": image_ids,
        "captions": captions,
        "num_images": len(image_ids),
        "embedding_dim": image_embeddings.shape[1],
    }

    import json

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Embeddings saved to {output_dir}")
    print(f"  Image embeddings: {image_embeddings.shape}")
    print(f"  Text embeddings: {text_embeddings.shape}")


if __name__ == "__main__":
    main()

