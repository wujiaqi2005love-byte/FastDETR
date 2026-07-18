"""
Set Prediction Criterion (Loss Function).

Computes the training loss after bipartite matching:
1. Classification loss: Cross-entropy for matched queries
2. Bounding box loss: L1 + Generalized IoU (GIoU)
3. Denoising loss (auxiliary): Direct supervision for denoising queries

The loss is computed for all decoder layers (auxiliary decoding losses).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class SetCriterion(nn.Module):
    """
    Loss function for FastDETR.

    L_total = λ_cls * L_cls + λ_l1 * L_l1 + λ_giou * L_giou

    Where:
    - L_cls: Cross-entropy over class predictions
    - L_l1: L1 loss on bounding box coordinates
    - L_giou: Generalized IoU loss (scale-invariant)

    All losses are normalized by the number of objects in the batch.
    """
    def __init__(
        self,
        num_classes: int,
        matcher,
        weight_dict: Optional[Dict[str, float]] = None,
        losses: Optional[List[str]] = None,
        eos_coef: float = 0.1,
        focal_alpha: float = 0.25,  # For focal loss variant
        use_focal_loss: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.focal_alpha = focal_alpha
        self.use_focal_loss = use_focal_loss

        # Default loss weights
        self.weight_dict = weight_dict or {
            'loss_ce': 1.0,
            'loss_bbox': 5.0,
            'loss_giou': 2.0,
        }

        # Which losses to compute
        self.losses = losses or ['labels', 'boxes']

        # Empty weight for "no object" class (down-weight to handle imbalance)
        if num_classes > 0:
            empty_weight = torch.ones(num_classes + 1)
            empty_weight[-1] = eos_coef  # much lower weight for ∅
            self.register_buffer('empty_weight', empty_weight)
        else:
            self.register_buffer('empty_weight', None)

    def _get_src_permutation_idx(
        self, indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert per-image matching indices to batch-level indices
        for efficient loss computation.
        """
        # Permute predictions: (batch_idx, src_idx)
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
        log: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Classification loss (cross-entropy or focal loss).

        Matched queries: loss is CE with their assigned GT class
        Unmatched queries: loss is CE with "no object" (∅) class
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (B, N, num_classes+1)

        # Get matched indices
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.long, device=src_logits.device
        )
        target_classes_o = torch.cat([
            t['labels'][J] for t, (_, J) in zip(targets, indices)
        ])
        target_classes[idx] = target_classes_o

        if self.use_focal_loss:
            # Focal loss variant (better for class imbalance)
            loss_ce = self._focal_loss(
                src_logits.flatten(0, 1),
                target_classes.flatten(0),
                alpha=self.focal_alpha,
                gamma=2.0,
            )
        else:
            # Standard cross-entropy
            loss_ce = F.cross_entropy(
                src_logits.flatten(0, 1),
                target_classes.flatten(0),
                self.empty_weight,
            )

        losses = {'loss_ce': loss_ce}
        return losses

    def _focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> torch.Tensor:
        """Focal loss for addressing class imbalance."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
        return focal_loss.mean()

    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Bounding box loss: L1 + GIoU.

        Only computed for matched queries (those assigned to a real object,
        not ∅).
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # GIoU loss
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_denoising(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        dn_info: Dict,
        num_boxes: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Improvement #2: Denoising Loss.

        Direct supervision for denoising queries — the model knows exactly
        which GT each denoising query corresponds to. No Hungarian matching
        needed, providing clean gradients from day one.
        """
        losses = {}
        if dn_info is None:
            return losses

        num_learnable = dn_info['num_learnable']
        dn_labels = dn_info['dn_labels']  # (B, G, max_n)
        dn_boxes = dn_info['dn_boxes']  # (B, G, max_n) — original clean boxes
        dn_mask = dn_info['dn_padding_mask']  # (B, G*max_n)

        # Extract denoising predictions (after learnable queries)
        dn_logits = outputs['pred_logits'][:, num_learnable:, :]  # (B, dn_q, cls+1)
        dn_pred_boxes = outputs['pred_boxes'][:, num_learnable:, :]  # (B, dn_q, 4)

        B, G, max_n, _ = dn_boxes.shape
        num_dn = G * max_n

        # Reshape denoising targets
        dn_labels_flat = dn_labels.reshape(B, num_dn)  # (B, G*max_n)
        dn_boxes_flat = dn_boxes.reshape(B, num_dn, 4)  # (B, G*max_n, 4)

        # Only compute loss on valid (non-padding) denoising queries
        valid_mask = dn_mask  # (B, num_dn)
        num_valid = valid_mask.sum()

        if num_valid == 0:
            # Return dummy zero losses
            losses['loss_ce_dn'] = torch.tensor(0.0, device=dn_logits.device)
            losses['loss_bbox_dn'] = torch.tensor(0.0, device=dn_logits.device)
            losses['loss_giou_dn'] = torch.tensor(0.0, device=dn_logits.device)
            return losses

        # Classification loss for denoising
        dn_logits_valid = dn_logits[valid_mask]  # (num_valid, cls+1)
        dn_labels_valid = dn_labels_flat[valid_mask]  # (num_valid,)
        loss_ce_dn = F.cross_entropy(dn_logits_valid, dn_labels_valid)
        losses['loss_ce_dn'] = loss_ce_dn

        # Box losses for denoising
        dn_boxes_valid = dn_pred_boxes[valid_mask]  # (num_valid, 4)
        dn_boxes_target = dn_boxes_flat[valid_mask]  # (num_valid, 4)

        loss_bbox_dn = F.l1_loss(dn_boxes_valid, dn_boxes_target, reduction='sum') / max(num_valid, 1)
        losses['loss_bbox_dn'] = loss_bbox_dn

        loss_giou_dn = (
            1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(dn_boxes_valid),
                box_cxcywh_to_xyxy(dn_boxes_target),
            ))
        ).sum() / max(num_valid, 1)
        losses['loss_giou_dn'] = loss_giou_dn

        return losses

    def get_loss(
        self,
        loss: str,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: int,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Dispatch to the appropriate loss function."""
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f"Unknown loss: {loss}. Options: {list(loss_map.keys())}"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        dn_info: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total training loss.

        Args:
            outputs: Dict with 'pred_logits' and 'pred_boxes'
            targets: List of target dicts
            dn_info: Optional denoising info from decoder

        Returns:
            Dict of loss components
        """
        # Handle auxiliary outputs (per decoder layer)
        if 'aux_outputs' in outputs:
            outputs_without_aux = {k: v for k, v in outputs.items()
                                   if k != 'aux_outputs'}
            indices = self.matcher(outputs_without_aux, targets)

            # Compute loss for all output layers
            losses = {}
            for loss in self.losses:
                kwargs = {}
                losses.update(
                    self.get_loss(loss, outputs_without_aux, targets, indices,
                                  self._get_num_boxes(targets), **kwargs)
                )

            # Auxiliary losses from intermediate decoder layers
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices_aux = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices_aux,
                        self._get_num_boxes(targets), **kwargs
                    )
                    l_dict = {f'{k}_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

            # Denoising loss
            if dn_info is not None:
                dn_losses = self.loss_denoising(
                    outputs_without_aux, targets, dn_info,
                    self._get_num_boxes(targets)
                )
                losses.update(dn_losses)

            return losses
        else:
            # Single output (inference or simple training)
            indices = self.matcher(outputs, targets)
            num_boxes = self._get_num_boxes(targets)

            losses = {}
            for loss in self.losses:
                kwargs = {}
                losses.update(
                    self.get_loss(loss, outputs, targets, indices, num_boxes,
                                  **kwargs)
                )

            if dn_info is not None:
                dn_losses = self.loss_denoising(
                    outputs, targets, dn_info, num_boxes
                )
                losses.update(dn_losses)

            return losses

    @staticmethod
    def _get_num_boxes(targets: List[Dict]) -> int:
        """Count total number of ground truth objects."""
        num_boxes = sum(len(t['labels']) for t in targets)
        return max(num_boxes, 1)  # Avoid division by zero
