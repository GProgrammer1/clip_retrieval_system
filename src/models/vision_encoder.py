"""Vision encoder with ResNet backbone and projection head."""

import torch
import torch.nn as nn
from torchvision import models


class VisionEncoder(nn.Module):
    """Vision encoder using ResNet-50 backbone with projection head."""

    def __init__(self, embed_dim: int = 512, projection_dim: int = 256):
        """
        Initialize vision encoder.

        Args:
            embed_dim: Dimension of backbone output features
            projection_dim: Dimension of projected embeddings
        """
        super().__init__()
        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(2048, embed_dim),  # ResNet-50 outputs 2048-dim features
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [batch_size, 3, 224, 224]

        Returns:
            Projected embeddings [batch_size, projection_dim]
        """
        # Extract features
        features = self.backbone(x)  # [batch_size, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [batch_size, 2048]
        
        # Project to embedding space
        embeddings = self.projection(features)  # [batch_size, projection_dim]
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class ViTVisionEncoder(nn.Module):
    """Vision encoder using Vision Transformer (ViT) backbone."""

    def __init__(self, embed_dim: int = 512, projection_dim: int = 256):
        """
        Initialize ViT vision encoder.

        Args:
            embed_dim: Dimension of backbone output features
            projection_dim: Dimension of projected embeddings
        """
        super().__init__()
        # Load pretrained ViT
        vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Use the encoder part
        self.backbone = vit.encoder
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(768, embed_dim),  # ViT-B/16 outputs 768-dim features
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [batch_size, 3, 224, 224]

        Returns:
            Projected embeddings [batch_size, projection_dim]
        """
        # ViT expects different preprocessing, but we'll use standard transforms
        # Extract CLS token embedding
        x = vit.process_input(x) if hasattr(vit, 'process_input') else x
        features = self.backbone(x)
        # Get CLS token (first token)
        cls_token = features[:, 0]  # [batch_size, 768]
        
        # Project to embedding space
        embeddings = self.projection(cls_token)  # [batch_size, projection_dim]
        
        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings

