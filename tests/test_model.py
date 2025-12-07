"""Test script to verify model forward pass."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.clip_model import CLIPModel


def test_model_forward():
    """Test that model forward pass works correctly."""
    print("Testing CLIP model forward pass...")

    vision_config = {
        "backbone": "resnet50",
        "embed_dim": 512,
        "projection_dim": 256,
    }

    text_config = {
        "vocab_size": 1000,
        "embed_dim": 512,
        "num_layers": 4,
        "num_heads": 8,
        "max_seq_length": 77,
        "projection_dim": 256,
    }

    model = CLIPModel(vision_config=vision_config, text_config=text_config)
    model.eval()

    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    text_tokens = torch.randint(0, 1000, (batch_size, 20))

    print(f"Input shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Text tokens: {text_tokens.shape}")

    # Forward pass
    with torch.no_grad():
        outputs = model(images, text_tokens)

    print(f"\nOutput shapes:")
    print(f"  Image embeddings: {outputs['image_embeddings'].shape}")
    print(f"  Text embeddings: {outputs['text_embeddings'].shape}")
    print(f"  Logits: {outputs['logits'].shape}")

    # Check embeddings are normalized
    img_norm = torch.norm(outputs['image_embeddings'], dim=1)
    txt_norm = torch.norm(outputs['text_embeddings'], dim=1)
    print(f"\nEmbedding norms (should be ~1.0):")
    print(f"  Image embeddings: {img_norm.mean().item():.4f} ± {img_norm.std().item():.4f}")
    print(f"  Text embeddings: {txt_norm.mean().item():.4f} ± {txt_norm.std().item():.4f}")

    # Test individual encoders
    img_emb = model.encode_image(images)
    txt_emb = model.encode_text(text_tokens)
    print(f"\nIndividual encoder outputs:")
    print(f"  Image encoder: {img_emb.shape}")
    print(f"  Text encoder: {txt_emb.shape}")

    print("\nModel forward pass test passed!")


if __name__ == "__main__":
    test_model_forward()

