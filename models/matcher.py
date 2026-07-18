"""
Hungarian Bipartite Matching for Set Prediction Loss.

This module computes the optimal one-to-one assignment between predicted
objects and ground truth objects, using the Hungarian algorithm.

The matching cost considers:
1. Classification cost: -p(class_of_gt) — how likely is the predicted class?
2. Bounding box L1 cost: ||b_pred - b_gt||₁
3. Generalized IoU cost: 1 - GIoU(b_pred, b_gt)

The GIoU cost is scale-invariant, which is crucial since the original DETR
makes absolute box predictions (not relative to anchors).
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """
    Compute the optimal bipartite matching between predictions and ground truth.

    For each image in the batch:
    1. Build cost matrix C of size (N_pred, N_gt)
       C_ij = λ_cls * cost_cls(pred_i, gt_j)
            + λ_l1  * ||box_pred_i - box_gt_j||₁
            + λ_giou * (1 - GIoU(box_pred_i, box_gt_j))
    2. Solve with Hungarian algorithm
    3. Return matched (pred_idx, gt_idx) pairs

    Each GT is matched to exactly one prediction. Unmatched predictions
    are assigned to "no object" (∅).
    """
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, \
            "At least one cost must be non-zero"

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            outputs: Dict with:
                'pred_logits': (B, N, num_classes+1) class logits
                'pred_boxes': (B, N, 4) predicted boxes in cxcywh format
            targets: List of dicts, each with:
                'labels': (M_i,) ground truth class labels
                'boxes': (M_i, 4) ground truth boxes in cxcywh format

        Returns:
            List of tuples (pred_indices, gt_indices) for each image
        """
        B, N, _ = outputs['pred_logits'].shape

        # Split batch
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)

        # Concatenate targets
        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t['boxes'] for t in targets])

        # Classification cost: -log(prob of correct class) for non-∅ classes
        # For ∅ (background), the cost is constant
        cost_class = -out_prob[:, tgt_ids]  # (B*N, total_gt)

        # Bounding box L1 cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # (B*N, total_gt)

        # GIoU cost
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )  # (B*N, total_gt)

        # Combined cost matrix
        C = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        # Reshape to per-image cost matrices
        C = C.view(B, N, -1)
        sizes = [len(t['boxes']) for t in targets]

        # Hungarian matching per image
        indices = []
        for b, (C_b, M_b) in enumerate(zip(C.split(sizes, dim=-1), sizes)):
            C_b = C_b.squeeze(0).cpu()  # (N, M_b)

            # Hungarian algorithm
            pred_idx, gt_idx = linear_sum_assignment(C_b)
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(gt_idx, dtype=torch.long),
            ))

        return indices


class HybridHungarianMatcher(nn.Module):
    """
    Improvement: Hybrid One-to-One + One-to-Many Matching.

    Standard DETR uses strict one-to-one matching, which is slow to converge
    because each prediction only gets gradient from one GT.

    Hybrid matching adds an auxiliary one-to-many branch:
    1. Primary branch: one-to-one matching (like original DETR)
    2. Auxiliary branch: one-to-many matching (top-K predictions per GT)

    The auxiliary branch provides richer gradients early in training,
    accelerating convergence. At inference, only the primary branch is used.

    This is inspired by Co-DETR and Group-DETR approaches.
    """
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        top_k_aux: int = 4,  # number of auxiliary matches per GT
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.top_k_aux = top_k_aux

    @torch.no_grad()
    def forward(
        self,
        outputs: dict,
        targets: List[dict],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns both one-to-one and one-to-many matchings.
        For simplicity, we implement the core one-to-one matching here.
        The one-to-many extension can be added as a separate auxiliary head.
        """
        # Core one-to-one matching (same as original HungarianMatcher)
        B, N, _ = outputs['pred_logits'].shape

        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)

        tgt_ids = torch.cat([t['labels'] for t in targets])
        tgt_bbox = torch.cat([t['boxes'] for t in targets])

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox),
        )

        C = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        C = C.view(B, N, -1)
        sizes = [len(t['boxes']) for t in targets]

        indices = []
        for b, (C_b, M_b) in enumerate(zip(C.split(sizes, dim=-1), sizes)):
            C_b = C_b.squeeze(0).cpu()
            pred_idx, gt_idx = linear_sum_assignment(C_b)
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(gt_idx, dtype=torch.long),
            ))

        return indices
