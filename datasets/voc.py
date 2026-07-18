"""
Pascal VOC Dataset loader for FastDETR.

VOC is much smaller than COCO (~16K images vs 118K), making it ideal
for quick training, debugging, and resource-limited environments.

VOC2007: ~5K train, ~5K val, 20 classes
VOC2012: ~5.7K train, ~5.8K val, 20 classes
Combined: ~16.5K train, ~10.8K val

20 classes: person, bird, cat, cow, dog, horse, sheep, aeroplane,
bicycle, boat, bus, car, motorbike, train, bottle, chair,
dining table, potted plant, sofa, tv/monitor

Training time (single GPU, VOC07+12, 50 epochs): ~30-60 minutes
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
import numpy as np
from .transforms import build_transforms


VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]

VOC_CLASS_TO_IDX = {name: i for i, name in enumerate(VOC_CLASSES)}


class VOCDetection(Dataset):
    """
    Pascal VOC Object Detection Dataset.

    Detects objects in images and returns boxes in (cx, cy, w, h) format,
    normalized to [0, 1], consistent with the FastDETR pipeline.

    Dataset structure expected:
        VOCdevkit/
          VOC2007/
            Annotations/  (xml files)
            JPEGImages/   (image files)
            ImageSets/Main/trainval.txt
          VOC2012/
            Annotations/
            JPEGImages/
            ImageSets/Main/trainval.txt
    """
    def __init__(
        self,
        root: str,
        year: str = '2007',
        split: str = 'trainval',  # 'train', 'val', 'trainval'
        transforms=None,
        max_samples: Optional[int] = None,
        use_difficult: bool = False,
    ):
        """
        Args:
            root: Path to VOCdevkit directory (e.g., 'data/VOCdevkit')
            year: Dataset year ('2007', '2012', or '0712' for both)
            split: 'trainval' or 'train' or 'val'
            transforms: Data augmentation pipeline
            max_samples: Limit number of samples (for quick testing)
            use_difficult: Include 'difficult' objects
        """
        self.root = root
        self.year = year
        self.split = split
        self.use_difficult = use_difficult

        # Collect image IDs from specified year(s)
        years = ['2007', '2012'] if year == '0712' else [year]
        self.image_paths = []
        self.annotation_paths = []
        self.image_ids = []

        for y in years:
            voc_year_dir = os.path.join(root, f'VOC{y}')
            if not os.path.exists(voc_year_dir):
                print(f"[WARN] VOC{y} not found at {voc_year_dir}")
                continue

            # Read image set
            img_set_file = os.path.join(
                voc_year_dir, 'ImageSets', 'Main', f'{split}.txt'
            )
            if not os.path.exists(img_set_file):
                print(f"[WARN] Image set not found: {img_set_file}")
                continue

            with open(img_set_file, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]

            for img_id in ids:
                img_path = os.path.join(
                    voc_year_dir, 'JPEGImages', f'{img_id}.jpg'
                )
                ann_path = os.path.join(
                    voc_year_dir, 'Annotations', f'{img_id}.xml'
                )
                if os.path.exists(img_path) and os.path.exists(ann_path):
                    self.image_paths.append(img_path)
                    self.annotation_paths.append(ann_path)
                    self.image_ids.append(f'{y}_{img_id}')

        # Limit samples
        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]
            self.annotation_paths = self.annotation_paths[:max_samples]
            self.image_ids = self.image_ids[:max_samples]

        self.num_classes = 20
        self.transforms = transforms or build_transforms(
            is_train=('train' in split)
        )

        print(f"[VOC] Loaded {len(self.image_paths)} images "
              f"(years={years}, split={split})")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _parse_annotation(self, ann_path: str) -> Tuple:
        """
        Parse VOC XML annotation.

        Returns:
            boxes: (N, 4) in cxcywh, normalized
            labels: (N,) 0-indexed class indices
            difficult: (N,) bool
        """
        tree = ET.parse(ann_path)
        root = tree.findall('size')[0]
        img_w = float(root.find('width').text)
        img_h = float(root.find('height').text)

        boxes = []
        labels = []
        difficults = []

        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            if name not in VOC_CLASS_TO_IDX:
                continue

            difficult = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
            if difficult and not self.use_difficult:
                continue

            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            # Convert xyxy → cxcywh, normalize
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            boxes.append([cx, cy, w, h])
            labels.append(VOC_CLASS_TO_IDX[name])
            difficults.append(difficult)

        if len(boxes) == 0:
            return (
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.bool),
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(difficults, dtype=torch.bool),
        )

    def __getitem__(self, idx: int) -> Dict:
        img_path = self.image_paths[idx]
        ann_path = self.annotation_paths[idx]
        img_id = self.image_ids[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Parse annotations
        boxes, labels, difficults = self._parse_annotation(ann_path)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'area': boxes[:, 2] * boxes[:, 3] * orig_w * orig_h if len(boxes) > 0
                    else torch.zeros((0,), dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.long),
            'orig_size': torch.tensor([orig_h, orig_w]),
            'difficult': difficults,
        }

        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return {
            'image': image,
            'target': target,
        }


def build_voc(
    root: str = 'data/VOCdevkit',
    year: str = '0712',
    split: str = 'trainval',
    max_samples: Optional[int] = None,
    **kwargs,
) -> VOCDetection:
    """Build VOC dataset."""
    return VOCDetection(
        root=root,
        year=year,
        split=split,
        max_samples=max_samples,
        **kwargs,
    )
