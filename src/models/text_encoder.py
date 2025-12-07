"""Text encoder with Transformer-based architecture."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """Text encoder using Transformer architecture."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_length: int = 77,
        projection_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize text encoder.

        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_length: Maximum sequence length
            projection_dim: Dimension of projected embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, projection_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Token IDs [batch_size, seq_length]
            mask: Attention mask [batch_size, seq_length] (True for padding)

        Returns:
            Projected embeddings [batch_size, projection_dim]
        """
        batch_size, seq_length = x.shape

        # Create position indices
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeds = self.token_embedding(x)  # [batch_size, seq_length, embed_dim]
        pos_embeds = self.position_embedding(positions)  # [batch_size, seq_length, embed_dim]
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)

        # Create attention mask (invert padding mask for transformer)
        if mask is not None:
            # Transformer expects True for positions to attend to
            attn_mask = ~mask
        else:
            attn_mask = None

        # Transformer encoder
        # Note: TransformerEncoder expects mask where True = ignore
        if mask is not None:
            # Convert to format expected by transformer (True = mask out)
            src_key_padding_mask = mask
        else:
            src_key_padding_mask = None

        encoded = self.transformer(
            embeddings, src_key_padding_mask=src_key_padding_mask
        )  # [batch_size, seq_length, embed_dim]

        # Use mean pooling over sequence (or CLS token if we had one)
        # For simplicity, we'll use mean pooling
        pooled = encoded.mean(dim=1)  # [batch_size, embed_dim]

        # Project to embedding space
        embeddings = self.projection(pooled)  # [batch_size, projection_dim]

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

