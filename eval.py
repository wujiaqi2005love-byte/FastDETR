"""
Evaluation script for FastDETR on COCO.

Computes standard COCO metrics:
- AP, AP50, AP75 (average precision)
- AP_S, AP_M, AP_L (AP by object size)
- AR (average recall)

Usage:
    python eval.py \
        --config configs/fast_detr_r50.yaml \
        --resume outputs/fast_detr_r50_50ep/checkpoint.pth \
        --coco_path data/coco

    # With test-time augmentation
    python eval.py \
        --config configs/fast_detr_r50.yaml \
        --resume outputs/fast_detr_r50_50ep/checkpoint.pth \
        --coco_path data/coco \
        --tta
"""

import torch
import torch.nn.functional as F
import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fast_detr import build_fast_detr
from datasets.coco import build_coco
from utils.box_ops import box_cxcywh_to_xyxy


try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO_API = True
except ImportError:
    HAS_COCO_API = False
    print("Warning: pycocotools not installed. COCO metrics unavailable.")
    print("Install with: pip install pycocotools")


def get_args_parser():
    parser = argparse.ArgumentParser('FastDETR Evaluation')

    parser.add_argument('--config', type=str, default='configs/fast_detr_r50.yaml')
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--coco_path', type=str, default='data/coco')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--score_threshold', type=float, default=0.001,
                        help='Minimum score to keep a prediction')
    parser.add_argument('--max_detections', type=int, default=100,
                        help='Maximum detections per image')
    parser.add_argument('--tta', action='store_true',
                        help='Use test-time augmentation (horizontal flip)')
    parser.add_argument('--output_dir', type=str, default='outputs/eval',
                        help='Output directory for results')

    return parser


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    coco_gt: 'COCO',
    device: torch.device,
    args,
) -> Dict:
    """
    Run COCO evaluation.

    Args:
        model: FastDETR model
        data_loader: Validation data loader
        coco_gt: COCO ground truth object
        device: Device
        args: Arguments

    Returns:
        Dict with COCO metrics
    """
    model.eval()

    results = []
    image_ids = set()

    print("\nRunning inference on validation set...")
    for batch in tqdm(data_loader):
        images = torch.stack([item['image'] for item in batch]).to(device)
        targets = [item['target'] for item in batch]

        # Get image info
        for target in targets:
            image_ids.add(target['image_id'])

        # Forward pass
        outputs = model(images)
        pred_logits = outputs['pred_logits']  # (B, N_q, num_classes+1)
        pred_boxes = outputs['pred_boxes']    # (B, N_q, 4)

        # Post-processing
        probs = F.softmax(pred_logits, dim=-1)
        scores, labels = probs[..., :-1].max(dim=-1)  # exclude ∅ class

        for b in range(images.shape[0]):
            img_id = targets[b]['image_id']
            orig_h, orig_w = targets[b]['orig_size']

            # Filter by score threshold
            keep = scores[b] > args.score_threshold
            box_preds = pred_boxes[b][keep]
            score_preds = scores[b][keep]
            label_preds = labels[b][keep]

            if len(box_preds) == 0:
                continue

            # Convert boxes from normalized cxcywh → absolute xyxy
            boxes_xyxy = box_cxcywh_to_xyxy(box_preds)
            boxes_xyxy[:, [0, 2]] *= orig_w
            boxes_xyxy[:, [1, 3]] *= orig_h

            # TTA: horizontal flip averaging
            if args.tta:
                # Flip image
                images_flipped = torch.flip(images[b:b+1], dims=[-1])
                outputs_flipped = model(images_flipped)
                probs_flipped = F.softmax(outputs_flipped['pred_logits'], dim=-1)
                scores_f, labels_f = probs_flipped[..., :-1].max(dim=-1)

                keep_f = scores_f[0] > args.score_threshold
                boxes_f = box_cxcywh_to_xyxy(outputs_flipped['pred_boxes'][0][keep_f])
                # Unflip boxes
                boxes_f[:, [0, 2]] = orig_w - boxes_f[:, [0, 2]]
                # Swap left/right
                boxes_f[:, [0, 2]] = boxes_f[:, [2, 0]]

                # Combine predictions (weighted average)
                # Simple approach: concatenate and apply NMS
                boxes_xyxy = torch.cat([boxes_xyxy, boxes_f])
                score_preds = torch.cat([score_preds, scores_f[0][keep_f]])
                label_preds = torch.cat([label_preds, labels_f[0][keep_f]])

            # Convert to COCO format
            for box, score, label in zip(boxes_xyxy, score_preds, label_preds):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1

                results.append({
                    'image_id': int(img_id),
                    'category_id': int(label) + 1,  # COCO categories are 1-indexed
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score),
                    'area': float(w * h),
                })

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'coco_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    print(f"\nResults saved to {results_path}")
    print(f"Total predictions: {len(results)}")
    print(f"Unique images: {len(image_ids)}")

    # Evaluate with COCO API
    if HAS_COCO_API and len(results) > 0:
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = list(image_ids)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        metrics = {
            'AP': coco_eval.stats[0],
            'AP50': coco_eval.stats[1],
            'AP75': coco_eval.stats[2],
            'AP_S': coco_eval.stats[3],
            'AP_M': coco_eval.stats[4],
            'AP_L': coco_eval.stats[5],
            'AR_max1': coco_eval.stats[6],
            'AR_max10': coco_eval.stats[7],
            'AR_max100': coco_eval.stats[8],
            'AR_S': coco_eval.stats[9],
            'AR_M': coco_eval.stats[10],
            'AR_L': coco_eval.stats[11],
        }

        # Print metrics
        print("\n" + "="*60)
        print("COCO Evaluation Results")
        print("="*60)
        print(f"AP       (Avg Precision @ IoU=0.50:0.95) : {metrics['AP']:.3f}")
        print(f"AP50     (Avg Precision @ IoU=0.50)       : {metrics['AP50']:.3f}")
        print(f"AP75     (Avg Precision @ IoU=0.75)       : {metrics['AP75']:.3f}")
        print(f"AP_S     (Avg Precision - Small objects)  : {metrics['AP_S']:.3f}")
        print(f"AP_M     (Avg Precision - Medium objects) : {metrics['AP_M']:.3f}")
        print(f"AP_L     (Avg Precision - Large objects)  : {metrics['AP_L']:.3f}")
        print(f"AR_max1  (Avg Recall - 1 detection)       : {metrics['AR_max1']:.3f}")
        print(f"AR_max100 (Avg Recall - 100 detections)   : {metrics['AR_max100']:.3f}")
        print("="*60)

        return metrics

    return {}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load COCO ground truth
    if HAS_COCO_API:
        ann_file = os.path.join(args.coco_path, 'annotations', 'instances_val2017.json')
        coco_gt = COCO(ann_file)
        print(f"Loaded COCO ground truth: {ann_file}")
    else:
        coco_gt = None

    # Build model
    print("\nBuilding FastDETR model...")
    model = build_fast_detr(num_classes=91)
    model.to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded epoch {checkpoint.get('epoch', 'unknown')}")

    # Build dataset
    dataset_val = build_coco(
        root=args.coco_path,
        split='val2017',
    )

    data_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Run evaluation
    metrics = evaluate(model, data_loader, coco_gt, device, args)

    return metrics


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
