"""
============================================================
  FastDETR 单 GPU / CPU 训练脚本
  Single-GPU Training Script — 无需分布式，开箱即用
============================================================

使用方法 (Usage):
  # 单 GPU 训练 (推荐)
  python train_single_gpu.py --coco_path ./data/coco --epochs 50

  # CPU 训练 (慢，仅用于测试)
  python train_single_gpu.py --coco_path ./data/coco --epochs 1 --device cpu

  # 继续训练 (从 checkpoint 恢复)
  python train_single_gpu.py --coco_path ./data/coco --resume outputs/checkpoint.pth


硬件要求 (Requirements):
  GPU: 至少 6GB 显存 (RTX 2060 以上)
  RAM: 至少 16GB
  磁盘: 约 30GB (COCO 数据集)

训练时间估算 (单张 V100/RTX 3090):
  1 epoch  ≈ 8 分钟 (batch_size=2)
  50 epoch ≈ 6.5 小时
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fast_detr import FastDETR, build_fast_detr
from models.matcher import HungarianMatcher
from models.criterion import SetCriterion
from datasets.coco import build_coco
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from utils.misc import SmoothedMetric, save_checkpoint, load_checkpoint

# 尝试导入 COCO API (可选)
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_COCO_API = True
except ImportError:
    HAS_COCO_API = False
    print("[WARN] pycocotools 未安装，COCO 指标将不可用")
    print("      安装: pip install pycocotools")


# ============================================================
#  命令行参数
# ============================================================
def get_args():
    parser = argparse.ArgumentParser(
        description='FastDETR 单 GPU 训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_single_gpu.py --coco_path ./data/coco --epochs 50
  python train_single_gpu.py --coco_path ./data/coco --epochs 1 --device cpu --max_samples 100
  python train_single_gpu.py --coco_path ./data/coco --resume outputs/checkpoint.pth
        """
    )

    # === 路径 ===
    parser.add_argument('--coco_path', type=str, default='data/coco',
                        help='COCO 数据集路径')
    parser.add_argument('--output_dir', type=str, default='outputs/fastdetr_single',
                        help='输出目录')

    # === 模型 ===
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help='骨干网络 (resnet50 更快, resnet101 更准)')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_queries', type=int, default=300)
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)

    # === 训练 ===
    parser.add_argument('--epochs', type=int, default=50,
                        help='总训练轮数 (50 即可收敛, 原始 DETR 需要 500)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='单 GPU batch size (显存不够就减小)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (单 GPU 建议比多 GPU 稍小)')
    parser.add_argument('--lr_backbone', type=float, default=1e-5,
                        help='骨干网络学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    parser.add_argument('--grad_clip', type=float, default=0.1,
                        help='梯度裁剪阈值')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='使用混合精度 (省显存)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='关闭混合精度')

    # === 数据 ===
    parser.add_argument('--max_samples', type=int, default=None,
                        help='限制样本数 (调试用)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数')

    # === 设备 ===
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='训练设备')

    # === 恢复 ===
    parser.add_argument('--resume', type=str, default='',
                        help='从 checkpoint 恢复')

    # === 日志 ===
    parser.add_argument('--print_freq', type=int, default=10,
                        help='每 N 步打印日志')
    parser.add_argument('--eval_freq', type=int, default=5,
                        help='每 N 轮验证一次')

    return parser.parse_args()


# ============================================================
#  学习率调度 (Cosine + Warmup)
# ============================================================
def get_lr(epoch: int, batch_idx: int, total_batches: int,
           base_lr: float, warmup_epochs: int, total_epochs: int) -> float:
    """Cosine 学习率 + 线性预热"""
    global_step = epoch + batch_idx / total_batches
    if global_step < warmup_epochs:
        return base_lr * global_step / warmup_epochs
    progress = (global_step - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# ============================================================
#  训练一个 Epoch
# ============================================================
def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    model.train()
    total_loss = 0.0
    total_steps = len(data_loader)
    start_time = time.time()

    pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{args.epochs}',
                ncols=100, ascii=True)

    for step, batch in enumerate(pbar):
        # 手动处理 batch (list of dicts)
        images = torch.stack([item['image'] for item in batch]).to(device)
        targets = [{
            'boxes': item['target']['boxes'].to(device),
            'labels': item['target']['labels'].to(device),
        } for item in batch]

        # 学习率
        lr = get_lr(epoch, step, total_steps,
                    args.lr, args.warmup_epochs, args.epochs)
        lr_bb = get_lr(epoch, step, total_steps,
                       args.lr_backbone, args.warmup_epochs, args.epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_bb if 'backbone' in pg.get('name', '') else lr

        # 前向传播
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images, targets)
                loss_dict = criterion(outputs, targets, outputs.get('dn_info'))
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, targets)
            loss_dict = criterion(outputs, targets, outputs.get('dn_info'))
            loss = sum(loss_dict.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total_loss += loss.item()

        # 进度条
        avg_loss = total_loss / (step + 1)
        elapsed = time.time() - start_time
        eta = (elapsed / (step + 1)) * (total_steps - step - 1)
        pbar.set_postfix({
            'loss': f'{avg_loss:.3f}',
            'lr': f'{lr:.2e}',
            'ETA': f'{eta/60:.0f}m',
        })

        # 详细日志
        if (step + 1) % args.print_freq == 0:
            loss_str = '  '.join([
                f'{k}: {v.item():.4f}' for k, v in loss_dict.items()
            ])
            tqdm.write(f'  [Step {step+1:4d}/{total_steps}] {loss_str}')

    return total_loss / total_steps


# ============================================================
#  验证
# ============================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    loss_components = {}

    for batch in tqdm(data_loader, desc='Validating', ncols=80, ascii=True):
        images = torch.stack([item['image'] for item in batch]).to(device)
        targets = [{
            'boxes': item['target']['boxes'].to(device),
            'labels': item['target']['labels'].to(device),
        } for item in batch]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss = sum(loss_dict.values())
        total_loss += loss.item()

        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v.item()

    n = len(data_loader)
    return {
        'loss': total_loss / n,
        **{k: v / n for k, v in loss_components.items()}
    }


# ============================================================
#  COCO 评估 (需要 pycocotools)
# ============================================================
@torch.no_grad()
def coco_evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    coco_gt: Optional['COCO'],
    device: torch.device,
    output_file: str,
):
    """计算 COCO AP/AR 指标"""
    if not HAS_COCO_API or coco_gt is None:
        return None

    model.eval()
    results = []

    for batch in tqdm(data_loader, desc='COCO Eval', ncols=80, ascii=True):
        images = torch.stack([item['image'] for item in batch]).to(device)
        targets = [item['target'] for item in batch]

        outputs = model(images)
        probs = F.softmax(outputs['pred_logits'], dim=-1)
        scores, labels = probs[..., :-1].max(dim=-1)

        for b in range(images.shape[0]):
            img_id = targets[b]['image_id']
            orig_h, orig_w = targets[b]['orig_size']

            keep = scores[b] > 0.01
            box_preds = outputs['pred_boxes'][b][keep]
            score_preds = scores[b][keep]
            label_preds = labels[b][keep]

            if len(box_preds) == 0:
                continue

            boxes_xyxy = box_cxcywh_to_xyxy(box_preds)
            boxes_xyxy[:, [0, 2]] *= orig_w
            boxes_xyxy[:, [1, 3]] *= orig_h

            for box, score, label in zip(boxes_xyxy, score_preds, label_preds):
                x1, y1, x2, y2 = box.tolist()
                results.append({
                    'image_id': int(img_id),
                    'category_id': int(label) + 1,
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                    'score': float(score),
                })

    with open(output_file, 'w') as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_S': coco_eval.stats[3],
        'AP_M': coco_eval.stats[4],
        'AP_L': coco_eval.stats[5],
    }


# ============================================================
#  主函数
# ============================================================
def main():
    args = get_args()

    # 设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，切换到 CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    if args.device == 'cuda':
        args.use_amp = args.use_amp and torch.cuda.is_available()

    print("=" * 60)
    print("  FastDETR 单 GPU 训练")
    print("=" * 60)
    print(f"  设备:         {device}")
    print(f"  骨干网络:     {args.backbone}")
    print(f"  训练轮数:     {args.epochs}")
    print(f"  Batch Size:   {args.batch_size}")
    print(f"  学习率:       {args.lr}")
    print(f"  混合精度:     {'ON' if args.use_amp else 'OFF'}")
    print(f"  输出目录:     {args.output_dir}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # ========================
    # 构建模型
    # ========================
    print("\n>>> 构建模型...")
    model = build_fast_detr(
        num_classes=91,
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_queries=args.num_queries,
        use_denoising=True,
        use_mixed_selection=True,
    )
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {param_count:,}")

    # ========================
    # 损失 & 匹配器
    # ========================
    matcher = HungarianMatcher(
        cost_class=1.0, cost_bbox=5.0, cost_giou=2.0
    )
    criterion = SetCriterion(
        num_classes=91,
        matcher=matcher,
        weight_dict={
            'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0,
            'loss_ce_dn': 1.0, 'loss_bbox_dn': 5.0, 'loss_giou_dn': 2.0,
        },
        eos_coef=0.1,
        use_focal_loss=True,
    )
    criterion.to(device)

    # ========================
    # 优化器
    # ========================
    bb_params = []
    tr_params = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if 'backbone' in n:
                bb_params.append(p)
            else:
                tr_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': tr_params, 'lr': args.lr, 'name': 'transformer'},
        {'params': bb_params, 'lr': args.lr_backbone, 'name': 'backbone'},
    ], weight_decay=args.weight_decay)

    # 混合精度
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # ========================
    # 数据
    # ========================
    print("\n>>> 加载 COCO 数据集...")
    dataset_train = build_coco(
        root=args.coco_path, split='train2017',
        max_samples=args.max_samples,
    )
    dataset_val = build_coco(
        root=args.coco_path, split='val2017',
        max_samples=args.max_samples // 10 if args.max_samples else None,
    )

    loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == 'cuda'),
        collate_fn=lambda x: x,
    )
    loader_val = DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == 'cuda'),
        collate_fn=lambda x: x,
    )

    print(f"  训练集: {len(dataset_train)} 张图片, {len(loader_train)} 个 batch")
    print(f"  验证集: {len(dataset_val)} 张图片, {len(loader_val)} 个 batch")

    # COCO 评估
    coco_gt = None
    if HAS_COCO_API:
        ann_file = os.path.join(args.coco_path, 'annotations',
                                'instances_val2017.json')
        if os.path.exists(ann_file):
            coco_gt = COCO(ann_file)

    # ========================
    # 恢复
    # ========================
    start_epoch = 0
    best_ap = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"\n>>> 恢复训练: {args.resume}")
        checkpoint = load_checkpoint(model, optimizer, args.resume)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_ap = checkpoint.get('best_ap', 0.0)
        print(f"  从 epoch {start_epoch} 继续")

    # ========================
    # 训练循环
    # ========================
    print("\n" + "=" * 60)
    print("  开始训练!")
    print("=" * 60 + "\n")

    history = {'train_loss': [], 'val_loss': [], 'ap': []}

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # 训练
        train_loss = train_one_epoch(
            model, criterion, loader_train, optimizer,
            device, epoch, args, scaler
        )
        history['train_loss'].append(train_loss)

        # 日志
        elapsed = time.time() - epoch_start
        print(f'\n  Epoch {epoch+1}/{args.epochs} | '
              f'Loss: {train_loss:.4f} | '
              f'Time: {elapsed/60:.1f}min | '
              f'ETA: {elapsed*(args.epochs-epoch-1)/3600:.1f}h')

        # 保存 checkpoint
        checkpoint_path = os.path.join(
            args.output_dir, f'checkpoint_epoch{epoch+1:03d}.pth'
        )
        save_checkpoint(
            model, optimizer, epoch, train_loss, checkpoint_path,
            extra_info={'best_ap': best_ap}
        )

        # 验证
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_metrics = validate(model, criterion, loader_val, device)
            history['val_loss'].append(val_metrics['loss'])
            print(f'  Val Loss: {val_metrics["loss"]:.4f}')

            # COCO 评估
            if coco_gt is not None:
                coco_file = os.path.join(
                    args.output_dir, f'coco_results_epoch{epoch+1:03d}.json'
                )
                metrics = coco_evaluate(
                    model, loader_val, coco_gt, device, coco_file
                )
                if metrics:
                    history['ap'].append(metrics['AP'])
                    print(f'  COCO AP: {metrics["AP"]:.3f} | '
                          f'AP50: {metrics["AP50"]:.3f} | '
                          f'AP_S: {metrics["AP_S"]:.3f} | '
                          f'AP_L: {metrics["AP_L"]:.3f}')

                    if metrics['AP'] > best_ap:
                        best_ap = metrics['AP']
                        best_path = os.path.join(
                            args.output_dir, 'best_model.pth'
                        )
                        save_checkpoint(
                            model, optimizer, epoch, train_loss, best_path,
                            extra_info={'best_ap': best_ap}
                        )
                        print(f'  *** 最佳模型! AP={best_ap:.3f} ***')

    # ========================
    # 训练完成
    # ========================
    print("\n" + "=" * 60)
    print("  训练完成!")
    print(f"  最佳 AP: {best_ap:.4f}")
    print(f"  模型保存在: {args.output_dir}")
    print("=" * 60)

    # 保存训练历史
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model


if __name__ == '__main__':
    main()
