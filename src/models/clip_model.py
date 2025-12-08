"""CLIP model combining vision and text encoders."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder
from .vision_encoder import VisionEncoder, ViTVisionEncoder


class CLIPModel(nn.Module):
    """CLIP model with vision and text encoders."""

    def __init__(
        self,
        vision_config: dict,
        text_config: dict,
        temperature: float = 0.07,
    ):
        """
        Initialize CLIP model.

        Args:
            vision_config: Configuration for vision encoder
            text_config: Configuration for text encoder
            temperature: Temperature parameter for contrastive loss
        """
        super().__init__()
        self.temperature = temperature

        # Initialize encoders
        backbone = vision_config.get("backbone", "resnet50")
        if backbone == "resnet50":
            self.vision_encoder = VisionEncoder(
                embed_dim=vision_config.get("embed_dim", 512),
                projection_dim=vision_config.get("projection_dim", 256),
            )
        elif backbone == "vit" or backbone == "vit_b_16":
            self.vision_encoder = ViTVisionEncoder(
                embed_dim=vision_config.get("embed_dim", 512),
                projection_dim=vision_config.get("projection_dim", 256),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.text_encoder = TextEncoder(
            vocab_size=text_config.get("vocab_size", 10000),
            embed_dim=text_config.get("embed_dim", 512),
            num_layers=text_config.get("num_layers", 4),
            num_heads=text_config.get("num_heads", 8),
            max_seq_length=text_config.get("max_seq_length", 77),
            projection_dim=text_config.get("projection_dim", 256),
        )

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            images: Input images [batch_size, 3, 224, 224]
            text_tokens: Token IDs [batch_size, seq_length]
            text_mask: Attention mask for text [batch_size, seq_length]

        Returns:
            Dictionary with:
                - image_embeddings: [batch_size, projection_dim]
                - text_embeddings: [batch_size, projection_dim]
                - logits: [batch_size, batch_size] similarity matrix
        """
        # Encode images and text
        image_embeddings = self.vision_encoder(images)
        text_embeddings = self.text_encoder(text_tokens, text_mask)

        # Compute similarity matrix
        logits = torch.matmul(image_embeddings, text_embeddings.t()) / self.temperature

        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "logits": logits,
        }

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images only."""
        return self.vision_encoder(images)

    def encode_text(self, text_tokens: torch.Tensor, text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode text only."""
        return self.text_encoder(text_tokens, text_mask)


def compute_clip_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss.

    Args:
        logits: Similarity matrix [batch_size, batch_size]

    Returns:
        Loss value
    """
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=logits.device)

    # Image-to-text loss
    loss_i2t = F.cross_entropy(logits, labels)

    # Text-to-image loss
    loss_t2i = F.cross_entropy(logits.t(), labels)

    # Average
    loss = (loss_i2t + loss_t2i) / 2.0

    return loss

