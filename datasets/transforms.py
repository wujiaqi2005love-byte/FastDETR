"""
Data augmentation and transformation pipeline for FastDETR.

Transformations applied:
1. Random resize (shortest side 480-800, longest ≤ 1333)
2. Random horizontal flip
3. Random crop (improves global reasoning → +1 AP)
4. Normalize with ImageNet mean/std
5. Convert to tensor
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Tuple, Optional, Dict, List
import random


class RandomResize:
    """
    Randomly resize the image such that:
    - Shortest side is between min_size and max_size
    - Longest side is at most max_longest
    """
    def __init__(
        self,
        min_size: int = 480,
        max_size: int = 800,
        max_longest: int = 1333,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.max_longest = max_longest

    def __call__(
        self,
        image: torch.Tensor,
        target: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        # Randomly select target size for shortest side
        target_size = random.randint(self.min_size, self.max_size)

        _, h, w = image.shape

        # Scale so that shortest side = target_size
        scale = target_size / min(h, w)

        # Ensure longest side ≤ max_longest
        if max(h, w) * scale > self.max_longest:
            scale = self.max_longest / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        image = F.resize(image, [new_h, new_w])

        # Update target boxes
        if 'boxes' in target and target['boxes'].numel() > 0:
            target['boxes'] = target['boxes'] * scale
        if 'area' in target and target['area'].numel() > 0:
            target['area'] = target['area'] * scale * scale

        target['size'] = torch.tensor([new_h, new_w])

        return image, target


class RandomHorizontalFlip:
    """
    Randomly flip the image horizontally with probability p.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
        target: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        if random.random() < self.p:
            image = F.hflip(image)
            _, _, w = image.shape

            if 'boxes' in target and target['boxes'].numel() > 0:
                # Flip x coordinates: cx → w - cx
                boxes = target['boxes']
                boxes[:, 0] = w - boxes[:, 0]
                target['boxes'] = boxes

        return image, target


class RandomCrop:
    """
    Random crop augmentation.

    Crops a random rectangular region and resizes back to training size.
    This helps the model learn global relationships (important for DETR
    variants where self-attention reasons over the full image).
    """
    def __init__(
        self,
        p: float = 0.5,
        crop_ratio: Tuple[float, float] = (0.5, 0.9),
    ):
        self.p = p
        self.crop_ratio = crop_ratio

    def __call__(
        self,
        image: torch.Tensor,
        target: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        if random.random() > self.p:
            return image, target

        _, h, w = image.shape

        # Random crop size
        crop_h = int(h * random.uniform(*self.crop_ratio))
        crop_w = int(w * random.uniform(*self.crop_ratio))

        # Random position
        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))

        # Crop image
        image_cropped = image[:, top:top + crop_h, left:left + crop_w]

        # Adjust boxes
        if 'boxes' in target and target['boxes'].numel() > 0:
            boxes = target['boxes'].clone()
            # Convert to xyxy for easier crop logic
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # Shift to crop coordinates
            x1 = x1 - left
            y1 = y1 - top
            x2 = x2 - left
            y2 = y2 - top

            # Clamp to crop region
            x1 = x1.clamp(0, crop_w)
            y1 = y1.clamp(0, crop_h)
            x2 = x2.clamp(0, crop_w)
            y2 = y2.clamp(0, crop_h)

            # Convert back to cxcywh
            new_w = x2 - x1
            new_h = y2 - y1

            # Keep only boxes that are still valid (area > threshold)
            valid = (new_w > 2) & (new_h > 2)
            boxes = boxes[valid]
            boxes[:, 0] = (x1[valid] + x2[valid]) / 2
            boxes[:, 1] = (y1[valid] + y2[valid]) / 2
            boxes[:, 2] = new_w[valid]
            boxes[:, 3] = new_h[valid]

            target['boxes'] = boxes
            target['labels'] = target['labels'][valid]
            if 'area' in target:
                target['area'] = target['area'][valid]
            if 'iscrowd' in target:
                target['iscrowd'] = target['iscrowd'][valid]

        # Update size
        target['size'] = torch.tensor([crop_h, crop_w])

        # Resize back to standard size range
        image_cropped, target = RandomResize()(image_cropped, target)

        return image_cropped, target


class ToTensor:
    """Convert PIL Image to tensor and normalize boxes."""
    def __call__(
        self,
        image,
        target: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        image = F.to_tensor(image)
        return image, target


class Normalize:
    """Normalize with ImageNet statistics."""
    def __init__(
        self,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def __call__(
        self,
        image: torch.Tensor,
        target: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Compose:
    """Compose multiple transforms."""
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(
        self,
        image,
        target: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def build_transforms(
    is_train: bool = True,
    img_min_size: int = 480,
    img_max_size: int = 800,
    img_max_longest: int = 1333,
) -> Compose:
    """
    Build the data transformation pipeline.

    Training: RandomResize → RandomCrop → RandomHorizontalFlip → Normalize
    Inference: Resize (fixed) → Normalize
    """
    if is_train:
        transforms = [
            RandomResize(img_min_size, img_max_size, img_max_longest),
            RandomCrop(p=0.5),
            RandomHorizontalFlip(p=0.5),
            Normalize(),
        ]
    else:
        transforms = [
            RandomResize(img_max_size, img_max_size, img_max_longest),
            Normalize(),
        ]

    return Compose(transforms)
