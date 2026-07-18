#!/usr/bin/env python3
"""
============================================================
  FastDETR 快速训练脚本 — 小数据集，几分钟出结果
  Quick Training with Small Datasets
============================================================

支持三种模式，按速度从快到慢：

  MODE 1: synthetic  — 合成数据 (无需下载, 3分钟)
  MODE 2: voc_sub    — VOC子集1000张 (需下载VOC, 5分钟)
  MODE 3: voc        — VOC完整数据集 (需下载VOC, 30-60分钟)

使用方法 (Usage):
  # 最快体验 — 合成数据, 不需要下载任何东西
  python train_quick.py --mode synthetic

  # VOC 子集快速测试
  python train_quick.py --mode voc_sub

  # VOC 完整训练 (需要先下载数据)
  python train_quick.py --mode voc

  # COCO 子集 (如果你已有 COCO 数据)
  python train_quick.py --mode coco_sub --coco_path ./data/coco

============================================================

数据集对比:
┌──────────────┬──────────┬──────────┬──────────┬─────────────┐
│ 数据集        │ 图片数   │ 类别数   │ 训练时间  │ 需要的磁盘   │
├──────────────┼──────────┼──────────┼──────────┼─────────────┤
│ synthetic    │ 200      │ 5        │ ~3 min   │ 0 MB        │
│ voc_sub      │ 1,000    │ 20       │ ~5 min   │ ~2 GB       │
│ voc (0712)   │ ~16,000  │ 20       │ ~40 min  │ ~2 GB       │
│ coco_sub     │ 5,000    │ 91       │ ~20 min  │ ~25 GB      │
│ coco (full)  │ 118,000  │ 91       │ ~7 hour  │ ~25 GB      │
└──────────────┴──────────┴──────────┴──────────┴─────────────┘

============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import sys
import time
import math
import urllib.request
import tarfile
import shutil
import json
import random
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fast_detr import build_fast_detr
from models.matcher import HungarianMatcher
from models.criterion import SetCriterion
from datasets.transforms import build_transforms
from utils.box_ops import box_cxcywh_to_xyxy
from utils.misc import save_checkpoint, SmoothedMetric


# ============================================================
#  合成数据集 (无需下载!)
# ============================================================
class SyntheticDataset(Dataset):
    """
    生成简单的合成检测数据。

    每张图是纯色背景上有随机颜色/大小的几何形状。
    目标: 让模型学会"找不同形状"这个基本能力。
    """
    SHAPES = ['circle', 'rectangle', 'triangle']
    CLASSES = ['circle', 'rectangle', 'triangle', 'star', 'cross']
    COLORS_HEX = {
        'circle': '#00ff00',
        'rectangle': '#00ccff',
        'triangle': '#ffaa00',
        'star': '#ff4444',
        'cross': '#ff00ff',
    }

    def __init__(
        self,
        num_samples: int = 200,
        img_size: int = 416,
        max_objects: int = 6,
        transforms=None,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.max_objects = max_objects
        self.transforms = transforms or build_transforms(is_train=True)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机背景
        bg = np.random.randint(20, 80, (self.img_size, self.img_size, 3),
                               dtype=np.uint8)

        # 随机生成 1~max_objects 个目标
        num_objs = random.randint(1, self.max_objects)
        boxes = []
        labels = []

        for _ in range(num_objs):
            cls = random.randint(0, len(self.CLASSES) - 1)

            # 随机大小 (占图 5%~30%)
            size = random.randint(
                int(self.img_size * 0.05), int(self.img_size * 0.3)
            )

            # 随机位置 (确保在图像内)
            x = random.randint(0, self.img_size - size - 1)
            y = random.randint(0, self.img_size - size - 1)

            # 随机颜色
            color = np.random.randint(100, 255, 3, dtype=np.uint8)

            # 画形状
            if cls == 0:  # circle
                cv2_circle(bg, (x + size // 2, y + size // 2),
                           size // 2, color)
            elif cls == 1:  # rectangle
                bg[y:y + size, x:x + size] = \
                    (bg[y:y + size, x:x + size] * 0.3 + color * 0.7
                     ).astype(np.uint8)
            elif cls == 2:  # triangle
                pts = np.array([
                    [x + size // 2, y],
                    [x, y + size],
                    [x + size, y + size],
                ])
                cv2_fill_poly(bg, [pts], color)
            elif cls == 3:  # star
                center = (x + size // 2, y + size // 2)
                r = size // 2
                pts = []
                for i in range(10):
                    angle = i * math.pi / 5 - math.pi / 2
                    rr = r if i % 2 == 0 else r * 0.4
                    pts.append([
                        int(center[0] + rr * math.cos(angle)),
                        int(center[1] + rr * math.sin(angle)),
                    ])
                cv2_fill_poly(bg, [np.array(pts)], color)
            else:  # cross
                cv2_rect(bg, (x, y, size, size), color, -1)
                cv2_rect(bg, (x + size // 4, y, size // 2, size),
                         (0, 0, 0), -1)
                cv2_rect(bg, (x, y + size // 4, size, size // 2),
                         (0, 0, 0), -1)

            # 归一化坐标 (cxcywh)
            boxes.append([
                (x + size / 2) / self.img_size,
                (y + size / 2) / self.img_size,
                size / self.img_size,
                size / self.img_size,
            ])
            labels.append(cls)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_id': int(idx),
            'area': torch.tensor([b[2] * b[3] for b in boxes],
                                 dtype=torch.float32),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.long),
            'orig_size': torch.tensor([self.img_size, self.img_size]),
        }

        # Convert numpy array to tensor directly (transforms expect tensors)
        image = torch.from_numpy(bg).float().permute(2, 0, 1) / 255.0

        if self.transforms:
            image, target = self.transforms(image, target)

        return {'image': image, 'target': target}


# OpenCV 兼容函数 (无需 import cv2)
def cv2_circle(img, center, radius, color):
    """纯 numpy 画圆"""
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    img[mask] = color


def cv2_fill_poly(img, pts_list, color):
    """纯 numpy 填充多边形 (简单实现)"""
    for pts in pts_list:
        pts = np.array(pts)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        yy, xx = np.mgrid[min_xy[1]:max_xy[1]+1, min_xy[0]:max_xy[0]+1]
        # 简化: 使用 bounding box 近似
        if yy.size > 0 and xx.size > 0:
            yy = yy.clip(0, img.shape[0]-1)
            xx = xx.clip(0, img.shape[1]-1)
            img[yy, xx] = color


def cv2_rect(img, rect, color, thickness):
    """纯 numpy 画矩形"""
    x, y, w, h = rect
    if thickness < 0:  # fill
        img[y:y+h, x:x+w] = color
    else:
        img[y:y+thickness, x:x+w] = color
        img[y+h-thickness:y+h, x:x+w] = color
        img[y:y+h, x:x+thickness] = color
        img[y:y+h, x+w-thickness:x+w] = color


# ============================================================
#  训练函数
# ============================================================
def quick_train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] {device}")

    # === 构建数据集 ===
    print(f"\n[INFO] 数据集模式: {args.mode}")

    if args.mode == 'synthetic':
        num_classes = len(SyntheticDataset.CLASSES)
        dataset_train = SyntheticDataset(
            num_samples=200, img_size=416, max_objects=6,
        )
        dataset_val = SyntheticDataset(
            num_samples=20, img_size=416, max_objects=6,
        )

    elif args.mode in ('voc', 'voc_sub'):
        # 检查/下载 VOC
        voc_root = args.voc_root or 'data/VOCdevkit'
        if not os.path.exists(os.path.join(voc_root, 'VOC2007')):
            print("\n[DOWNLOAD] 正在下载 Pascal VOC 2007...")
            print("  大小: ~860MB, 请稍候...")
            download_voc(voc_root)

        from datasets.voc import build_voc
        num_classes = 20
        max_s = 1000 if args.mode == 'voc_sub' else None
        dataset_train = build_voc(
            root=voc_root, year='0712', split='trainval',
            max_samples=max_s,
        )
        dataset_val = build_voc(
            root=voc_root, year='2007', split='val',
            max_samples=max_s // 5 if max_s else None,
        )

    elif args.mode == 'coco_sub':
        from datasets.coco import build_coco
        num_classes = 91
        dataset_train = build_coco(
            root=args.coco_path, split='train2017',
            max_samples=args.max_samples or 5000,
        )
        dataset_val = build_coco(
            root=args.coco_path, split='val2017',
            max_samples=500,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"  训练集: {len(dataset_train)} 张")
    print(f"  验证集: {len(dataset_val)} 张")
    print(f"  类别数: {num_classes}")

    def collate_batch(batch):
        """Pad images to same size within batch."""
        images = [item['image'] for item in batch]
        targets = [item['target'] for item in batch]
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        padded = []
        for img in images:
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            if pad_h > 0 or pad_w > 0:
                img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
            padded.append(img)
        return padded, targets

    loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_batch,
    )
    loader_val = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_batch,
    )

    # === 构建模型 ===
    print(f"\n[INFO] 构建 FastDETR (num_classes={num_classes})...")
    model = build_fast_detr(
        num_classes=num_classes,
        backbone_name='resnet50',
        num_encoder_layers=3,  # 浅层模型, 更快
        num_decoder_layers=3,
        num_queries=100,
        use_denoising=True,
        use_mixed_selection=True,
    )
    model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {params:,}")

    # === 损失 ===
    matcher = HungarianMatcher()
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict={
            'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0,
            'loss_ce_dn': 1.0, 'loss_bbox_dn': 5.0, 'loss_giou_dn': 2.0,
        },
        eos_coef=0.1,
        use_focal_loss=True,
    )
    criterion.to(device)

    # === 优化器 ===
    bb_params = []
    tr_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'backbone' in n:
                bb_params.append(p)
            else:
                tr_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': tr_params, 'lr': args.lr},
        {'params': bb_params, 'lr': args.lr * 0.1},
    ], weight_decay=1e-4)

    scaler = torch.amp.GradScaler('cuda') if args.use_amp and torch.cuda.is_available() else None

    # === 训练 ===
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  开始快速训练! {args.epochs} epochs")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader_train, desc=f'Epoch {epoch+1}/{args.epochs}',
                    ncols=100, ascii=True)

        for step, (images_list, targets_raw) in enumerate(pbar):
            images = torch.stack(images_list).to(device)
            targets = [{
                'boxes': t['boxes'].to(device),
                'labels': t['labels'].to(device),
            } for t in targets_raw]

            # Cosine LR with warmup
            progress = (epoch + step / len(loader_train)) / args.epochs
            if progress < 0.2:
                lr = args.lr * progress / 0.2
            else:
                lr = args.lr * 0.5 * (1 + math.cos(
                    math.pi * (progress - 0.2) / 0.8
                ))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(images, targets)
                    loss_dict = criterion(outputs, targets,
                                          outputs.get('dn_info'))
                    loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, targets)
                loss_dict = criterion(outputs, targets,
                                     outputs.get('dn_info'))
                loss = sum(loss_dict.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{total_loss/(step+1):.3f}',
                'lr': f'{lr:.2e}',
            })

        avg_loss = total_loss / len(loader_train)
        history['train_loss'].append(avg_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images_list, targets_raw in loader_val:
                images = torch.stack(images_list).to(device)
                targets_val = [{
                    'boxes': t['boxes'].to(device),
                    'labels': t['labels'].to(device),
                } for t in targets_raw]
                outputs = model(images)
                loss_dict = criterion(outputs, targets_val)
                val_loss += sum(loss_dict.values()).item()

        val_loss /= len(loader_val)
        history['val_loss'].append(val_loss)

        print(f"  Train Loss: {avg_loss:.4f}  |  Val Loss: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
            print(f"  >>> 最佳模型已保存: {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"  训练完成! 最佳 Val Loss: {best_loss:.4f}")
    print(f"  模型保存路径: {args.output_dir}/best_model.pth")
    print(f"{'='*60}")

    # 保存历史
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model


# ============================================================
#  自动下载 VOC
# ============================================================
def download_voc(root: str):
    """下载 Pascal VOC 2007 (最小的可用版本)"""
    os.makedirs(root, exist_ok=True)
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar'
    tar_path = os.path.join(root, 'voc2007.tar')

    print(f"  下载 {url} ...")
    print(f"  保存到 {tar_path}")

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            print(f'\r  进度: {pct:.0f}% ({downloaded/1024**2:.0f}/{total_size/1024**2:.0f} MB)', end='')

    try:
        urllib.request.urlretrieve(url, tar_path, reporthook=report)
        print()
        print("  解压中...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=root)
        os.remove(tar_path)
        print("  [OK] VOC 2007 下载完成!")
    except Exception as e:
        print(f"\n  [ERR] 下载失败: {e}")
        print("  请手动下载: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar")
        print(f"  解压到: {root}")


# ============================================================
#  入口
# ============================================================
def get_args():
    p = argparse.ArgumentParser(
        description='FastDETR 快速训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_quick.py --mode synthetic       # 最快, 3分钟
  python train_quick.py --mode voc_sub          # VOC子集, 5分钟
  python train_quick.py --mode voc              # 完整VOC, 40分钟
  python train_quick.py --mode coco_sub         # COCO子集, 20分钟
        """
    )

    p.add_argument('--mode', type=str, default='synthetic',
                   choices=['synthetic', 'voc_sub', 'voc', 'coco_sub'],
                   help='数据集模式')
    p.add_argument('--epochs', type=int, default=10,
                   help='训练轮数')
    p.add_argument('--batch_size', type=int, default=4,
                   help='Batch size')
    p.add_argument('--lr', type=float, default=1e-4,
                   help='学习率')
    p.add_argument('--output_dir', type=str, default='outputs/quick',
                   help='输出目录')
    p.add_argument('--coco_path', type=str, default='data/coco',
                   help='COCO 路径')
    p.add_argument('--voc_root', type=str, default='data/VOCdevkit',
                   help='VOC 路径')
    p.add_argument('--max_samples', type=int, default=5000,
                   help='COCO 子集样本数')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--use_amp', action='store_true', default=True)
    p.add_argument('--no_amp', action='store_false', dest='use_amp')

    return p.parse_args()


if __name__ == '__main__':
    args = get_args()
    quick_train(args)
