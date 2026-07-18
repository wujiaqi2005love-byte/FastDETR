"""
Positional Encoding for FastDETR.

We use 2D sine-cosine positional encodings as in the original Transformer,
generalized to handle multi-scale feature maps. The encoding is added to
queries and keys in every attention layer.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionEmbeddingSine2D(nn.Module):
    """
    2D Sine-Cosine Positional Encoding.

    For each spatial position (y, x), we encode:
        PE(y, 2i)   = sin(y / 10000^(2i/d_model))
        PE(y, 2i+1) = cos(y / 10000^(2i/d_model))
        PE(x, 2i)   = sin(x / 10000^(2i/d_model))
        PE(x, 2i+1) = cos(x / 10000^(2i/d_model))

    The y and x encodings are concatenated, giving d_model channels.
    """
    def __init__(
        self,
        num_pos_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask: (B, H, W) binary mask, True=valid, False=padding

        Returns:
            pos: (B, num_pos_feats*2, H, W) positional encoding
        """
        assert mask is not None

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32,
                             device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        # Apply sin to even indices, cos to odd indices
        pos_x = torch.stack(
            [pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4
        ).flatten(3)
        pos_y = torch.stack(
            [pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4
        ).flatten(3)

        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Learned positional embeddings for object queries.

    These are the "object queries" in DETR terminology — learnable
    embeddings that the decoder uses to detect objects at specific
    positions/sizes. Different from spatial positional encodings.
    """
    def __init__(
        self,
        num_queries: int = 300,
        num_pos_feats: int = 256,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.row_embed = nn.Embedding(50, num_pos_feats // 2)
        self.col_embed = nn.Embedding(50, num_pos_feats // 2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self) -> torch.Tensor:
        """Return learnable query embeddings (num_queries, hidden_dim)."""
        # Not used directly in FastDETR; here for reference
        pass
