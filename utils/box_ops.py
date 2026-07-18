"""
Bounding box operations for FastDETR.

Includes:
- Box format conversions (cxcywh ↔ xyxy)
- Box area computation
- Generalized IoU (GIoU)
- Box clamping and validation
"""

import torch
from typing import Tuple


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format.

    Args:
        boxes: Tensor of shape (..., 4) in (cx, cy, w, h) format

    Returns:
        Tensor of shape (..., 4) in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    # Ensure boxes are on the same device as input
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format.

    Args:
        boxes: Tensor of shape (..., 4) in (x1, y1, x2, y2) format

    Returns:
        Tensor of shape (..., 4) in (cx, cy, w, h) format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute the area of boxes in (x1, y1, x2, y2) format.

    Args:
        boxes: (..., 4) in xyxy format

    Returns:
        (...,) area values
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    return (x2 - x1) * (y2 - y1)


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise Intersection over Union (IoU).

    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format

    Returns:
        iou: (N, M) IoU matrix
        union: (N, M) union areas
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # Compute union
    union = area1[:, None] + area2 - inter  # (N, M)

    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Generalized IoU (GIoU) between two sets of boxes.

    GIoU = IoU - |C \ (A ∪ B)| / |C|
    where C is the smallest enclosing box containing both A and B.

    GIoU is scale-invariant and ranges from -1 to 1, making it suitable
    for the Hungarian matching cost and bounding box loss.

    Reference:
        "Generalized Intersection over Union" by Rezatofighi et al., CVPR 2019

    Args:
        boxes1: (N, 4) in xyxy format
        boxes2: (M, 4) in xyxy format

    Returns:
        giou: (N, M) Generalized IoU matrix
    """
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4, \
        "Boxes must be in xyxy format with 4 coordinates"

    # Standard IoU
    iou, union = box_iou(boxes1, boxes2)

    # Smallest enclosing box
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    area_c = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    # GIoU
    giou = iou - (area_c - union) / area_c.clamp(min=1e-6)

    return giou


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Convert binary masks to bounding boxes.

    Args:
        masks: (N, H, W) binary masks

    Returns:
        boxes: (N, 4) in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    n = masks.shape[0]
    boxes = torch.zeros(n, 4, device=masks.device)

    for i in range(n):
        mask = masks[i]
        if mask.sum() == 0:
            continue
        y_coords, x_coords = torch.where(mask)
        boxes[i, 0] = x_coords.min()
        boxes[i, 1] = y_coords.min()
        boxes[i, 2] = x_coords.max()
        boxes[i, 3] = y_coords.max()

    return boxes


def clip_boxes_to_image(
    boxes: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    """
    Clip boxes to lie within image boundaries.

    Args:
        boxes: (..., 4) in xyxy format
        size: (height, width) of the image

    Returns:
        Clipped boxes
    """
    h, w = size
    boxes = boxes.clone()
    boxes[..., 0].clamp_(min=0, max=w)
    boxes[..., 1].clamp_(min=0, max=h)
    boxes[..., 2].clamp_(min=0, max=w)
    boxes[..., 3].clamp_(min=0, max=h)
    return boxes


def box_scale(
    boxes: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """
    Scale boxes by a factor.

    Args:
        boxes: (..., 4) in cxcywh or xyxy format
        scale_factor: Scaling factor

    Returns:
        Scaled boxes
    """
    return boxes * scale_factor
