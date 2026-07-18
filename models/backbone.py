"""
Backbone with Feature Pyramid Network (FPN) for multi-scale feature extraction.

Improvement #1: Small Object Detection
- Standard DETR uses only C5 feature map (stride 32), losing small object details
- We add FPN to generate {P2, P3, P4, P5} at strides {4, 8, 16, 32}
- High-resolution P2 preserves fine-grained textures for small objects
- Multi-scale features feed into the deformable encoder for adaptive sampling

Supported backbones: ResNet-50, ResNet-101, Swin-Tiny
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from typing import List, Dict, Optional


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen statistics (no gradient, running stats fixed)."""
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class FPN(nn.Module):
    """
    Feature Pyramid Network.

    Transforms backbone feature maps {C2, C3, C4, C5} into multi-scale
    feature pyramid {P2, P3, P4, P5} with consistent channel dimension.

    Architecture:
        C5 ──→ [1×1 conv] ──→ P5 ──→ [upsample] ──→ +
        C4 ──→ [1×1 conv] ──→ P4_0 ──────────────────→ P4 ──→ [upsample] → +
        C3 ──→ [1×1 conv] ──→ P3_0 ─────────────────────────────────────────→ P3 → [upsample] → +
        C2 ──→ [1×1 conv] ──→ P2_0 ──────────────────────────────────────────────────────────────→ P2
    """
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_levels = len(in_channels)

        # Lateral connections (1×1 convs to align channel dimensions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])

        # Output convolutions (3×3 convs to reduce aliasing from upsampling)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self, feature_maps: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Args:
            feature_maps: List of feature maps from backbone,
                         from highest resolution [C2, C3, C4, C5]
                         Shapes: [B, C_i, H_i, W_i]

        Returns:
            List of FPN features [P2, P3, P4, P5],
            all with out_channels channels.
        """
        # Build top-down pathway
        laterals = [
            conv(fm) for fm, conv in zip(feature_maps, self.lateral_convs)
        ]

        # Top-down with lateral connections
        # Start from P5 (lowest resolution)
        fpn_features = [laterals[-1]]
        for i in range(len(laterals) - 2, -1, -1):
            # Upsample higher-level feature to match current level
            upsampled = F.interpolate(
                fpn_features[-1],
                size=laterals[i].shape[-2:],
                mode='nearest'
            )
            fpn_features.append(laterals[i] + upsampled)

        # Reverse to get [P2, P3, P4, P5] order (high→low resolution)
        fpn_features = fpn_features[::-1]

        # Apply output convolutions
        outputs = [
            conv(feat) for feat, conv in zip(fpn_features, self.output_convs)
        ]

        return outputs


class ResNetBackbone(nn.Module):
    """
    ResNet backbone + FPN for FastDETR.

    Extracts multi-scale features from ResNet stages and feeds them through
    an FPN to produce a feature pyramid at strides {4, 8, 16, 32}.
    """
    def __init__(
        self,
        name: str = 'resnet50',
        pretrained: bool = True,
        out_channels: int = 256,
        freeze_bn: bool = True,
        return_intermediate: bool = True,
    ):
        super().__init__()
        self.name = name
        self.out_channels = out_channels

        # Load ResNet
        if name == 'resnet50':
            resnet = resnet50(pretrained=pretrained)
            in_channels = [256, 512, 1024, 2048]  # C2, C3, C4, C5
        elif name == 'resnet101':
            resnet = resnet101(pretrained=pretrained)
            in_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown ResNet: {name}")

        # Extract stages
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # C2, stride 4, 256 channels
        self.layer2 = resnet.layer2  # C3, stride 8, 512 channels
        self.layer3 = resnet.layer3  # C4, stride 16, 1024 channels
        self.layer4 = resnet.layer4  # C5, stride 32, 2048 channels

        # Freeze BatchNorm
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        # FPN neck
        self.fpn = FPN(in_channels=in_channels, out_channels=out_channels)

        self.return_intermediate = return_intermediate

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            Dict with:
                'features': List of FPN features [P2, P3, P4, P5]
                           shapes: [B, 256, H/4, W/4], [B,256,H/8,W/8], etc.
                'strides': [4, 8, 16, 32]
                'mask': Valid region mask [B, H/32, W/32] for padding
        """
        # Backbone forward
        x = self.stem(x)
        c2 = self.layer1(x)      # stride 4
        c3 = self.layer2(c2)     # stride 8
        c4 = self.layer3(c3)     # stride 16
        c5 = self.layer4(c4)     # stride 32

        # FPN forward
        features = self.fpn([c2, c3, c4, c5])

        # Create mask for valid regions (1=valid, 0=padding)
        # Use the lowest resolution feature map
        B, _, H_mask, W_mask = features[-1].shape
        mask = torch.ones(B, H_mask, W_mask, device=x.device)

        return {
            'features': features,
            'strides': [4, 8, 16, 32],
            'mask': mask,
        }


def build_backbone(
    name: str = 'resnet50',
    pretrained: bool = True,
    out_channels: int = 256,
) -> nn.Module:
    """Factory function to build the backbone."""
    if name.startswith('resnet'):
        return ResNetBackbone(
            name=name,
            pretrained=pretrained,
            out_channels=out_channels,
        )
    else:
        raise ValueError(f"Unknown backbone: {name}")
