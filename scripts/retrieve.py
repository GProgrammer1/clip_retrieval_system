"""CLI script for text-to-image and image-to-text retrieval."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.models.clip_model import CLIPModel
from src.utils.tokenization import SimpleTokenizer


def main():
    parser = argparse.ArgumentParser(description="Retrieve images or captions")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--query", type=str, help="Text query for image search")
    parser.add_argument("--image", type=str, help="Image path for caption search")
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = CLIPModel(
        vision_config=config["model"]["vision"],
        text_config=config["model"]["text"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load embeddings if available
    embeddings_dir = Path(args.embeddings_dir)
    if embeddings_dir.exists():
        image_embeddings = np.load(embeddings_dir / "image_embeddings.npy")
        with open(embeddings_dir / "metadata.json", "r") as f:
            import json

            metadata = json.load(f)
        print(f"Loaded {len(metadata['image_ids'])} precomputed embeddings")
    else:
        print("Embeddings not found. Run export_embeddings.py first.")
        return

    # Build tokenizer
    tokenizer = SimpleTokenizer(
        vocab_size=config["model"]["text"]["vocab_size"], min_freq=2
    )
    # Note: In practice, you'd load the vocab from training
    # For now, we'll use a simple approach
    tokenizer.build_vocab(metadata["captions"])

    if args.query:
        # Text-to-image retrieval
        print(f"\nQuery: '{args.query}'")
        token_ids = tokenizer.encode(
            args.query, max_length=config["model"]["text"]["max_seq_length"]
        )
        token_tensor = torch.tensor([token_ids]).to(device)
        mask = token_tensor == tokenizer.get_pad_token_id()

        with torch.no_grad():
            query_embedding = model.encode_text(token_tensor, mask).cpu().numpy()

        similarities = np.dot(query_embedding, image_embeddings.T).squeeze()
        top_k_indices = np.argsort(similarities)[-args.top_k :][::-1]

        print(f"\nTop {args.top_k} results:")
        for i, idx in enumerate(top_k_indices, 1):
            print(f"{i}. Image ID: {metadata['image_ids'][idx]}")
            print(f"   Score: {similarities[idx]:.4f}")
            print(f"   Caption: {metadata['captions'][idx]}")

    elif args.image:
        # Image-to-text retrieval
        print(f"\nQuery Image: {args.image}")
        img = Image.open(args.image).convert("RGB")
        img_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img_tensor = img_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embedding = model.encode_image(img_tensor).cpu().numpy()

        similarities = np.dot(image_embedding, image_embeddings.T).squeeze()
        top_k_indices = np.argsort(similarities)[-args.top_k :][::-1]

        print(f"\nTop {args.top_k} similar images:")
        for i, idx in enumerate(top_k_indices, 1):
            print(f"{i}. Image ID: {metadata['image_ids'][idx]}")
            print(f"   Score: {similarities[idx]:.4f}")
            print(f"   Caption: {metadata['captions'][idx]}")

    else:
        print("Please provide either --query or --image")


if __name__ == "__main__":
    main()

