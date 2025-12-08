"""FastAPI inference server for CLIP model."""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from src.models.clip_model import CLIPModel
from src.utils.tokenization import SimpleTokenizer

app = FastAPI(title="CLIP Model Inference API")

# Global variables
model = None
tokenizer = None
device = None
image_embeddings = None
metadata = None
image_transform = None


def load_model(checkpoint_path: str, config_path: str):
    """Load model and tokenizer."""
    global model, tokenizer, device, image_transform

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CLIPModel(
        vision_config=config["model"]["vision"],
        text_config=config["model"]["text"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Initialize tokenizer
    tokenizer = SimpleTokenizer(
        vocab_size=config["model"]["text"]["vocab_size"], min_freq=2
    )
    # Note: Vocabulary will be built from metadata when embeddings are loaded

    # Image transform
    image_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    print(f"Model loaded from {checkpoint_path}")


def load_embeddings(embeddings_dir: str):
    """Load precomputed embeddings."""
    global image_embeddings, metadata, tokenizer

    embeddings_path = Path(embeddings_dir)
    if not embeddings_path.exists():
        print(f"Embeddings directory not found: {embeddings_dir}")
        return

    image_embeddings = np.load(embeddings_path / "image_embeddings.npy")

    import json

    with open(embeddings_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Build tokenizer vocabulary from metadata captions
    if tokenizer is not None and hasattr(tokenizer, 'build_vocab'):
        if 'captions' in metadata:
            tokenizer.build_vocab(metadata['captions'])
            print(f"Built tokenizer vocabulary from {len(metadata['captions'])} captions")
        else:
            print("Warning: No captions in metadata to build vocabulary")

    print(f"Loaded {len(metadata['image_ids'])} precomputed embeddings")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    # This will be set by the main function
    pass


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "CLIP Model Inference API",
        "endpoints": [
            "/text-to-image",
            "/image-to-text",
            "/image-to-image",
            "/health",
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/text-to-image")
async def text_to_image(query: str, top_k: int = 5):
    """
    Search for images matching a text query.

    Args:
        query: Text query string
        top_k: Number of results to return

    Returns:
        List of matching images with scores
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if image_embeddings is None:
        raise HTTPException(
            status_code=503, detail="Image embeddings not loaded. Run export_embeddings.py first."
        )

    # Encode query
    token_ids = tokenizer.encode(
        query, max_length=77  
    )
    token_tensor = torch.tensor([token_ids]).to(device)
    mask = token_tensor == tokenizer.get_pad_token_id()

    with torch.no_grad():
        query_embedding = model.encode_text(token_tensor, mask).cpu().numpy()

    # Compute similarities
    similarities = np.dot(query_embedding, image_embeddings.T).squeeze()
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_k_indices:
        results.append(
            {
                "image_id": metadata["image_ids"][idx],
                "score": float(similarities[idx]),
                "caption": metadata["captions"][idx],
            }
        )

    return {"query": query, "results": results}


@app.post("/image-to-text")
async def image_to_text(file: UploadFile = File(...), top_k: int = 5):
    """
    Find captions matching an uploaded image.

    Args:
        file: Uploaded image file
        top_k: Number of results to return

    Returns:
        List of matching captions with scores
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Load and process image
    try:
        image = Image.open(file.file).convert("RGB")
        img_tensor = image_transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # Encode image
    with torch.no_grad():
        image_embedding = model.encode_image(img_tensor).cpu().numpy()

    # If we have precomputed text embeddings, use them
    # Otherwise, encode captions on the fly 
    if image_embeddings is not None and metadata is not None:
        # Use image-to-image similarity as proxy
        similarities = np.dot(image_embedding, image_embeddings.T).squeeze()
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_k_indices:
            results.append(
                {
                    "caption": metadata["captions"][idx],
                    "score": float(similarities[idx]),
                    "image_id": metadata["image_ids"][idx],
                }
            )
    else:
        # Fallback: return generic message
        results = [
            {
                "message": "Text embeddings not precomputed. Run export_embeddings.py first."
            }
        ]

    return {"results": results}


@app.post("/image-to-image")
async def image_to_image(file: UploadFile = File(...), top_k: int = 5):
    """
    Find similar images to an uploaded image.

    Args:
        file: Uploaded image file
        top_k: Number of results to return

    Returns:
        List of similar images with scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if image_embeddings is None:
        raise HTTPException(
            status_code=503, detail="Image embeddings not loaded"
        )

    # Load and process image
    try:
        image = Image.open(file.file).convert("RGB")
        img_tensor = image_transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # Encode image
    with torch.no_grad():
        query_embedding = model.encode_image(img_tensor).cpu().numpy()

    # Compute similarities
    similarities = np.dot(query_embedding, image_embeddings.T).squeeze()
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []
    for idx in top_k_indices:
        results.append(
            {
                "image_id": metadata["image_ids"][idx],
                "score": float(similarities[idx]),
                "caption": metadata["captions"][idx],
            }
        )

    return {"results": results}


def main():
    parser = argparse.ArgumentParser(description="Start CLIP inference API")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--embeddings_dir", type=str, default="embeddings")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    load_model(args.checkpoint, args.config)

    load_embeddings(args.embeddings_dir)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

