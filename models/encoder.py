"""
Multi-Scale Deformable Transformer Encoder.

Improvement #3: O(HW) Complexity
    - Replaces the original DETR encoder's O((HW)^2) global self-attention
      with deformable attention that samples K points per query
    - Complexity: O(HW · K · L · d) — LINEAR in spatial resolution
    - For 800×800 input at stride 32 (HW=625): 39× reduction
    - For 1600×1600 input: 156× reduction — makes high-res inference feasible

Architecture:
    Each encoder layer:
        1. Multi-Scale Deformable Self-Attention (each pixel attends to
           K learnable points across L feature levels)
        2. Feed-Forward Network (FFN)
        3. Layer Norm + Residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from .deformable_attention import MSDeformableAttention


class DeformableEncoderLayer(nn.Module):
    """
    Single encoder layer with deformable self-attention.
    """
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = 'relu',
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()

        # Deformable self-attention
        self.self_attn = MSDeformableAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_levels=n_levels,
            n_points=n_points,
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(
        tensor: torch.Tensor, pos: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return tensor + pos if pos is not None else tensor

    def forward(
        self,
        src: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        reference_points: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: Flattened multi-scale features (ΣHW, B, d_model)
            spatial_shapes: (n_levels, 2) feature map sizes (H, W)
            level_start_index: (n_levels,) start indices for each level
            reference_points: (ΣHW, B, n_levels, 2) reference points
            pos: Positional encoding (ΣHW, B, d_model) or None
            padding_mask: (B, ΣHW) attention mask

        Returns:
            src: Updated features (ΣHW, B, d_model)
        """
        # Self-attention
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            query=q,
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            attention_mask=padding_mask,
        )
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.ffn(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class DeformableEncoder(nn.Module):
    """
    Multi-Scale Deformable Encoder.

    Process multi-scale feature maps with deformable self-attention.
    Each pixel attends to K=4 learnable points at each of L=4 pyramid levels,
    vs. all (HW) pixels in the original DETR.

    Input: Multi-scale FPN features {P2, P3, P4, P5}
    Output: Enhanced multi-scale feature representations
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
        num_layers: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.num_layers = num_layers

        # Encoder layers
        self.layers = nn.ModuleList([
            DeformableEncoderLayer(
                d_model=d_model,
                d_ffn=d_ffn,
                dropout=dropout,
                n_levels=n_levels,
                n_heads=n_heads,
                n_points=n_points,
            )
            for _ in range(num_layers)
        ])

        # Input projection for each feature level (if channels != d_model)
        self.level_embed = nn.Embedding(n_levels, d_model)

        # Generate reference points from spatial shapes
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def get_reference_points(
        spatial_shapes: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate normalized [0,1] reference points for all spatial positions.

        Returns:
            reference_points: (Σ(H_l*W_l), n_levels, 2)
            Each position gets its own normalized coordinate repeated for all levels
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, device=device),
                torch.linspace(0.5, W - 0.5, W, device=device),
                indexing='ij',
            )
            # Normalize to [0, 1]
            ref_y = ref_y / H
            ref_x = ref_x / W
            ref = torch.stack([ref_x, ref_y], dim=-1).reshape(-1, 2)  # (H*W, 2)
            reference_points_list.append(ref)

        # Stack with level dimension
        reference_points = [
            ref.unsqueeze(1).repeat(1, len(spatial_shapes), 1)
            for ref in reference_points_list
        ]
        return torch.cat(reference_points, dim=0)

    def forward(
        self,
        srcs: List[torch.Tensor],
        masks: List[torch.Tensor],
        pos_embeds: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            srcs: List of feature maps [P2, P3, P4, P5],
                  each (B, d_model, H_l, W_l)
            masks: List of masks, each (B, H_l, W_l)
            pos_embeds: List of positional encodings, each (B, d_model, H_l, W_l)

        Returns:
            Dict with 'memory' (ΣHW, B, d_model), 'spatial_shapes',
            'level_start_index', 'reference_points'
        """
        # Prepare multi-scale inputs
        spatial_shapes_list = []
        src_flatten_list = []
        mask_flatten_list = []
        pos_flatten_list = []

        for lvl, (src, mask, pos) in enumerate(zip(srcs, masks, pos_embeds)):
            B, C, H, W = src.shape
            spatial_shapes_list.append([H, W])

            # Flatten spatial dimensions
            src_flat = src.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
            mask_flat = mask.flatten(1)  # (B, H*W)
            pos_flat = pos.flatten(2).permute(2, 0, 1)  # (H*W, B, C)

            # Add level embedding to provide scale information
            level_emb = self.level_embed.weight[lvl].view(1, 1, -1)
            src_flat = src_flat + level_emb
            pos_flat = pos_flat + level_emb

            src_flatten_list.append(src_flat)
            mask_flatten_list.append(mask_flat)
            pos_flatten_list.append(pos_flat)

        # Concatenate across all levels
        src = torch.cat(src_flatten_list, dim=0)  # (ΣHW, B, d_model)
        mask = torch.cat(mask_flatten_list, dim=1)  # (B, ΣHW)
        pos = torch.cat(pos_flatten_list, dim=0)  # (ΣHW, B, d_model)

        spatial_shapes = torch.tensor(
            spatial_shapes_list, device=src.device
        )  # (n_levels, 2)

        # Compute level start indices for efficient slicing
        level_lengths = torch.prod(spatial_shapes, dim=1)
        level_start_index = torch.cat([
            torch.zeros(1, device=src.device, dtype=torch.long),
            level_lengths[:-1].cumsum(0)
        ])

        # Generate reference points
        reference_points = self.get_reference_points(
            spatial_shapes, src.device
        )  # (ΣHW, n_levels, 2)
        reference_points = reference_points.unsqueeze(1).repeat(1, B, 1, 1)

        # Prepare attention mask (invert: 1=valid attended, 0=padding ignored)
        # We use mask where 1=keep, 0=ignore
        # Convert to float: ~mask means False→True (valid), True→False (padding)
        attention_mask = (~mask).float()

        # Pass through encoder layers
        outputs = []
        for layer in self.layers:
            src = layer(
                src=src,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points,
                pos=pos,
                padding_mask=attention_mask,
            )
            outputs.append(src)

        return {
            'memory': src,
            'encoder_outputs': outputs,  # All intermediate outputs
            'spatial_shapes': spatial_shapes,
            'level_start_index': level_start_index,
            'mask': mask,
        }
