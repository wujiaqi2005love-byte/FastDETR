"""
FastDETR: Accelerated Detection Transformer

Key improvements over original DETR:
1. Multi-Scale Deformable Attention for small objects
2. Contrastive Denoising Training for fast convergence
3. O(HW) encoder complexity via sparse sampling
4. Dynamic Query Allocation beyond fixed-100 limit
5. Mixed Query Selection + Look-Forward Twice for stable training
"""

from .fast_detr import FastDETR, build_fast_detr
from .backbone import build_backbone
from .deformable_attention import MSDeformableAttention
from .encoder import DeformableEncoder
from .decoder import DeformableDecoder
from .matcher import HungarianMatcher
from .criterion import SetCriterion

__all__ = [
    'FastDETR',
    'build_fast_detr',
    'build_backbone',
    'MSDeformableAttention',
    'DeformableEncoder',
    'DeformableDecoder',
    'HungarianMatcher',
    'SetCriterion',
]
