"""
Miscellaneous utilities for FastDETR.

Includes:
- NestedTensor for handling images with padding masks
- Collate function for DataLoader
- Utility functions for distributed training
- Metric logging and checkpointing
"""

import torch
import torch.nn as nn
import os
import time
import datetime
from typing import List, Dict, Optional, Union
from collections import defaultdict


class NestedTensor:
    """
    Wraps a tensor with an associated mask indicating valid (non-padding) regions.

    Used to handle images of different sizes in a batch by padding to the
    largest dimensions.
    """
    def __init__(self, tensor: torch.Tensor, mask: torch.Tensor):
        self.tensors = tensor
        self.mask = mask

    def to(self, device):
        return NestedTensor(
            self.tensors.to(device),
            self.mask.to(device),
        )

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return f"NestedTensor(tensors={self.tensors.shape}, mask={self.mask.shape})"


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.

    Pads images to the same size and stacks targets.
    """
    images = [item['image'] for item in batch]
    targets = [item['target'] for item in batch]

    # Pad images to same size
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    masks = []
    for img in images:
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
        padded_images.append(padded)

        # Create mask: True = valid, False = padding
        mask = torch.ones(h, w, dtype=torch.bool)
        mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), value=False)
        masks.append(mask)

    return {
        'image': torch.stack(padded_images),
        'mask': torch.stack(masks),
        'target': targets,
    }


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_world_size():
    """Get the number of distributed processes."""
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    """Get the rank of the current process."""
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def reduce_dict(input_dict: Dict[str, torch.Tensor], average: bool = True) -> Dict[str, torch.Tensor]:
    """
    Reduce dict values across distributed processes.

    Args:
        input_dict: Dict of scalar tensors
        average: If True, average; if False, sum

    Returns:
        Reduced dict
    """
    if not is_dist_avail_and_initialized():
        return input_dict

    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    reduced_dict = {}
    with torch.no_grad():
        for k, v in sorted(input_dict.items()):
            # Aggregate across processes
            torch.distributed.all_reduce(v)
            if average:
                v /= world_size
            reduced_dict[k] = v
    return reduced_dict


class SmoothedMetric:
    """
    Track the smoothed value of a metric using exponential moving average.
    """
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.values = []
        self.count = 0
        self.ema = None
        self.alpha = 2 / (window_size + 1)  # EMA smoothing factor

    def update(self, value: float):
        self.values.append(value)
        self.count += 1
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema

        # Keep only recent values
        if len(self.values) > self.window_size * 2:
            self.values = self.values[-self.window_size:]

    @property
    def median(self):
        if not self.values:
            return 0
        return sorted(self.values)[len(self.values) // 2]

    @property
    def avg(self):
        if not self.values:
            return 0
        return sum(self.values[-self.window_size:]) / min(len(self.values), self.window_size)

    @property
    def global_avg(self):
        if self.count == 0:
            return 0
        return sum(self.values) / self.count


class MetricLogger:
    """
    Log training metrics with smoothing and formatting.
    """
    def __init__(self, delimiter: str = "  "):
        self.meters = defaultdict(lambda: SmoothedMetric(window_size=20))
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        entries = []
        for name, meter in self.meters.items():
            entries.append(f"{name}: {meter.avg:.4f} ({meter.global_avg:.4f})")
        return self.delimiter.join(entries)

    def add_meter(self, name: str, meter: SmoothedMetric):
        self.meters[name] = meter


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    extra_info: Optional[Dict] = None,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    if extra_info:
        checkpoint.update(extra_info)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup and cosine decay.

    This schedule is beneficial for transformer training:
    - Linear warmup prevents early instability
    - Cosine decay provides smooth LR reduction
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        base_lr: float = 2e-4,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self, epoch: int = None):
        if epoch is not None:
            self.current_epoch = epoch

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr


import math
