"""
COCO Dataset loader for FastDETR.

Provides COCODetection dataset class that loads images and annotations
in the format expected by FastDETR: boxes in (cx, cy, w, h) normalized
to [0, 1], and integer class labels.
"""

import torch
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from typing import List, Dict, Optional, Tuple
from .transforms import build_transforms


class COCODetection(Dataset):
    """
    COCO Object Detection Dataset.

    Each item returns:
        {
            'image': Tensor (3, H, W) — normalized image
            'target': {
                'boxes': Tensor (N, 4) — normalized cxcywh format
                'labels': Tensor (N,) — class labels (0-indexed)
                'image_id': int — COCO image ID
                'area': Tensor (N,) — box areas
                'iscrowd': Tensor (N,) — crowd flags
                'orig_size': (H, W) — original image size
                'size': (H, W) — resized image size
            }
        }
    """
    def __init__(
        self,
        root: str,
        split: str = 'train2017',
        transforms=None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            root: Path to COCO dataset (e.g., 'data/coco')
            split: Dataset split ('train2017' or 'val2017')
            transforms: Data augmentation pipeline
            max_samples: Limit number of samples (for debugging)
        """
        self.root = root
        self.split = split

        # Image directory
        self.img_dir = os.path.join(root, split)

        # Annotation file
        ann_file = os.path.join(
            root, 'annotations',
            f'instances_{split}.json'
        )
        assert os.path.exists(ann_file), \
            f"Annotation file not found: {ann_file}"

        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        # Build image index
        self.images = self.coco['images']
        if max_samples is not None:
            self.images = self.images[:max_samples]

        # Build annotation index (image_id → list of annotations)
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # Build category mapping (COCO cat_id → 0-indexed label)
        self.categories = {
            cat['id']: i for i, cat in enumerate(self.coco['categories'])
        }
        self.num_classes = len(self.categories)

        # Transforms
        self.transforms = transforms or build_transforms(is_train=('train' in split))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict:
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Get annotations for this image
        anns = self.annotations.get(img_id, [])

        # Build target dict
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # COCO bbox: [x, y, width, height] (xywh, absolute pixels)
            x, y, w, h = ann['bbox']

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            # Convert to cxcywh, normalized to [0, 1]
            cx = (x + w / 2) / orig_w
            cy = (y + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h

            boxes.append([cx, cy, nw, nh])
            labels.append(self.categories[ann['category_id']])
            areas.append(w * h)
            iscrowd.append(ann.get('iscrowd', 0))

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32)
                     if boxes else torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long)
                      if labels else torch.zeros((0,), dtype=torch.long),
            'image_id': img_id,
            'area': torch.tensor(areas, dtype=torch.float32)
                    if areas else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.tensor(iscrowd, dtype=torch.long)
                       if iscrowd else torch.zeros((0,), dtype=torch.long),
            'orig_size': torch.tensor([orig_h, orig_w]),
        }

        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return {
            'image': image,
            'target': target,
        }


def build_coco(
    root: str,
    split: str = 'train2017',
    **kwargs,
) -> Dataset:
    """
    Build COCO dataset with appropriate transforms.

    Args:
        root: COCO dataset root directory
        split: 'train2017' or 'val2017'
        **kwargs: Additional arguments for COCODetection

    Returns:
        Dataset instance
    """
    is_train = 'train' in split
    transforms = build_transforms(is_train=is_train)
    return COCODetection(
        root=root,
        split=split,
        transforms=transforms,
        **kwargs,
    )
