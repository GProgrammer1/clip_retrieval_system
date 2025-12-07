"""Evaluation script entry point."""

import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data.coco_dataset import build_coco_dataloader
from src.eval.eval_retrieval import evaluate_retrieval
from src.models.clip_model import CLIPModel
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
    parser = argparse.ArgumentParser(description="Evaluate CLIP model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CLIPModel(
        vision_config=config["model"]["vision"],
        text_config=config["model"]["text"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch']}")

    # Build tokenizer
    tokenizer = SimpleTokenizer(
        vocab_size=config["model"]["text"]["vocab_size"], min_freq=2
    )

    # Build vocab from validation set
    temp_loader = build_coco_dataloader(
        annotation_file=str(Path(args.config).parent.parent / config["data"]["val"]["annotation_file"]),
        image_dir=str(Path(args.config).parent.parent / config["data"]["val"]["image_dir"]),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        max_samples=5000,
    )

    all_captions = []
    for batch in temp_loader:
        all_captions.extend(batch["caption"])

    tokenizer.build_vocab(all_captions)
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # Create data loader
    from torch.utils.data import DataLoader

    base_dir = Path(args.config).parent.parent
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
        pin_memory=True,
        collate_fn=lambda b: collate_fn(
            b, tokenizer, config["model"]["text"]["max_seq_length"]
        ),
    )

    # Evaluate
    results = evaluate_retrieval(
        model, val_loader, device, k_values=config["eval"]["top_k"]
    )

    # Print results
    print("\n=== Retrieval Results ===")
    print("\nImage-to-Text Retrieval:")
    for k in config["eval"]["top_k"]:
        print(f"  Recall@{k}: {results['image_to_text'][k]:.4f}")

    print("\nText-to-Image Retrieval:")
    for k in config["eval"]["top_k"]:
        print(f"  Recall@{k}: {results['text_to_image'][k]:.4f}")

    # Save results
    results["checkpoint"] = str(args.checkpoint)
    results["epoch"] = checkpoint["epoch"]

    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    results_path = results_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

