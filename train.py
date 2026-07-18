"""
Training script for FastDETR.

Usage:
    # Standard training (50 epochs, 8 GPUs)
    torchrun --nproc_per_node=8 train.py \
        --config configs/fast_detr_r50.yaml \
        --batch_size 16 \
        --epochs 50 \
        --output_dir outputs/fast_detr_r50_50ep

    # Extended training (108 epochs)
    torchrun --nproc_per_node=8 train.py \
        --config configs/fast_detr_r50.yaml \
        --batch_size 16 \
        --epochs 108 \
        --lr_drop 80 \
        --output_dir outputs/fast_detr_r50_108ep

    # Single GPU debugging
    python train.py \
        --config configs/fast_detr_r50.yaml \
        --batch_size 2 \
        --epochs 1 \
        --output_dir outputs/debug

Key training improvements over original DETR:
- AdamW optimizer with cosine LR schedule (vs. manual step decay)
- Denoising training provides clean gradients from epoch 1
- Mixed query selection gives decoder a "warm start"
- Gradient clipping prevents early instability
- Backbone LR is ~10× smaller than transformer LR for stability
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import argparse
import os
import sys
import time
import math
import yaml
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fast_detr import FastDETR, build_fast_detr
from models.matcher import HungarianMatcher
from models.criterion import SetCriterion
from datasets.coco import build_coco
from utils.misc import (
    MetricLogger, SmoothedMetric, save_checkpoint, load_checkpoint,
    reduce_dict, is_main_process, get_rank, get_world_size,
)


def get_args_parser():
    parser = argparse.ArgumentParser('FastDETR Training')

    # Configuration
    parser.add_argument('--config', type=str, default='configs/fast_detr_r50.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs/fast_detr',
                        help='Output directory for checkpoints and logs')

    # Model
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone network')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Transformer hidden dimension')
    parser.add_argument('--nheads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                        help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                        help='Number of decoder layers')
    parser.add_argument('--num_queries', type=int, default=300,
                        help='Max number of object queries')
    parser.add_argument('--n_points', type=int, default=4,
                        help='Sampling points per level in deformable attention')
    parser.add_argument('--num_feature_levels', type=int, default=4,
                        help='Number of FPN levels')

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Total batch size across all GPUs')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Base learning rate')
    parser.add_argument('--lr_backbone', type=float, default=2e-5,
                        help='Backbone learning rate (10× smaller)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--lr_drop', type=int, default=40,
                        help='Epoch to drop LR (cosine schedule is default)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--clip_max_norm', type=float, default=0.1,
                        help='Gradient clipping max norm')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')

    # Losses
    parser.add_argument('--focal_loss', action='store_true', default=True,
                        help='Use focal loss instead of CE')
    parser.add_argument('--eos_coef', type=float, default=0.1,
                        help='Relative weight for no-object class')
    parser.add_argument('--bbox_loss_coef', type=float, default=5.0,
                        help='L1 box loss weight')
    parser.add_argument('--giou_loss_coef', type=float, default=2.0,
                        help='GIoU loss weight')

    # Denoising
    parser.add_argument('--use_denoising', action='store_true', default=True,
                        help='Enable contrastive denoising training')
    parser.add_argument('--dn_groups', type=int, default=5,
                        help='Number of denoising groups')
    parser.add_argument('--dn_noise_scale', type=float, default=0.4,
                        help='Box noise scale for denoising')

    # Data
    parser.add_argument('--coco_path', type=str, default='data/coco',
                        help='Path to COCO dataset')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers per GPU')

    # Mixed precision + distributed
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')

    return parser


def main(args):
    # Initialize distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        print(f"[Rank {args.rank}] Initialized distributed training")
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        print("Running on single GPU/CPU")

    device = torch.device(f'cuda:{args.local_rank}')
    is_main = is_main_process()

    # ========================
    # Build Model
    # ========================
    if is_main:
        print("\n" + "="*60)
        print("Building FastDETR model...")
        print("="*60)

    model = build_fast_detr(
        num_classes=91,
        backbone_name=args.backbone,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_queries=args.num_queries,
        n_points=args.n_points,
        num_feature_levels=args.num_feature_levels,
        use_denoising=args.use_denoising,
        use_mixed_selection=True,
        use_dynamic_query=False,
    )
    model.to(device)

    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # ========================
    # Loss & Matcher
    # ========================
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=args.bbox_loss_coef,
        cost_giou=args.giou_loss_coef,
    )

    weight_dict = {
        'loss_ce': 1.0,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
        'loss_ce_dn': 1.0,
        'loss_bbox_dn': args.bbox_loss_coef,
        'loss_giou_dn': args.giou_loss_coef,
    }

    criterion = SetCriterion(
        num_classes=91,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        use_focal_loss=args.focal_loss,
    )
    criterion.to(device)

    # ========================
    # Optimizer
    # ========================
    # Separate backbone and transformer parameters
    param_dicts = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if 'backbone' not in n and p.requires_grad
            ],
            'lr': args.lr,
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if 'backbone' in n and p.requires_grad
            ],
            'lr': args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler (cosine with warmup)
    # We'll implement it manually in the training loop

    # ========================
    # Data
    # ========================
    if is_main:
        print("\nLoading COCO dataset...")

    dataset_train = build_coco(
        root=args.coco_path,
        split='train2017',
    )
    dataset_val = build_coco(
        root=args.coco_path,
        split='val2017',
    )

    if args.world_size > 1:
        train_sampler = DistributedSampler(dataset_train)
        val_sampler = DistributedSampler(dataset_val, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    batch_size_per_gpu = args.batch_size // args.world_size
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: x,  # We'll handle batching manually
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    if is_main:
        print(f"Training images: {len(dataset_train)}")
        print(f"Validation images: {len(dataset_val)}")
        print(f"Batch size per GPU: {batch_size_per_gpu}")
        print(f"Total batch size: {args.batch_size}")

    # ========================
    # Mixed Precision
    # ========================
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None

    # ========================
    # Distributed Wrapper
    # ========================
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # ========================
    # Resume
    # ========================
    start_epoch = 0
    if args.resume:
        if is_main:
            print(f"Resuming from {args.resume}")
        checkpoint = load_checkpoint(model_without_ddp, optimizer, args.resume)
        start_epoch = checkpoint['epoch'] + 1

    # ========================
    # Training Loop
    # ========================
    if is_main:
        print("\n" + "="*60)
        print("Starting training...")
        print("="*60 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        if args.world_size > 1:
            train_sampler.set_epoch(epoch)

        model.train()
        metric_logger = MetricLogger()
        header = f'Epoch [{epoch+1}/{args.epochs}]'

        for batch_idx, batch in enumerate(
            metric_logger.log_every(data_loader_train, 50, header)
        ):
            # Stack images from the batch
            images = torch.stack([item['image'] for item in batch]).to(device)
            targets = [{
                'boxes': item['target']['boxes'].to(device),
                'labels': item['target']['labels'].to(device),
            } for item in batch]

            # Set LR (cosine with warmup)
            current_lr = adjust_learning_rate(
                optimizer, epoch, batch_idx,
                len(data_loader_train), args
            )

            # Forward pass
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images, targets)
                    loss_dict = criterion(outputs, targets, outputs.get('dn_info'))
                    total_loss = sum(loss_dict.values())
            else:
                outputs = model(images, targets)
                loss_dict = criterion(outputs, targets, outputs.get('dn_info'))
                total_loss = sum(loss_dict.values())

            # Backward
            optimizer.zero_grad()
            if args.use_amp:
                scaler.scale(total_loss).backward()
                if args.clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_max_norm
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_max_norm
                    )
                optimizer.step()

            # Logging
            loss_dict_reduced = reduce_dict(loss_dict)
            metric_logger.update(
                loss=total_loss.item(),
                lr=current_lr,
                **{k: v.item() for k, v in loss_dict_reduced.items()},
            )

        # End of epoch
        if is_main:
            print(f"Epoch {epoch+1} summary: {metric_logger}")

            # Save checkpoint
            checkpoint_path = os.path.join(
                args.output_dir, f'checkpoint_epoch{epoch+1:03d}.pth'
            )
            save_checkpoint(
                model_without_ddp, optimizer, epoch,
                total_loss.item(), checkpoint_path,
                extra_info={'args': vars(args)},
            )
            print(f"Checkpoint saved: {checkpoint_path}")

            # Quick validation every 5 epochs
            if (epoch + 1) % 5 == 0:
                validate(model, criterion, data_loader_val, device, args)

    if is_main:
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)

    # Cleanup
    if args.world_size > 1:
        dist.destroy_process_group()


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    batch_idx: int,
    total_batches: int,
    args,
) -> float:
    """
    Cosine LR schedule with linear warmup.

    Warmup: LR linearly increases from 0 to base_lr over warmup_epochs
    Cosine: LR decays from base_lr to 0 following cosine curve
    """
    # Global step (in epochs, fractional)
    global_step = epoch + batch_idx / total_batches

    if global_step < args.warmup_epochs:
        # Linear warmup
        lr_scale = global_step / args.warmup_epochs
    else:
        # Cosine decay
        progress = (global_step - args.warmup_epochs) / \
                   (args.epochs - args.warmup_epochs)
        lr_scale = 0.5 * (1 + math.cos(math.pi * progress))

    lr = args.lr * lr_scale
    lr_backbone = args.lr_backbone * lr_scale

    for param_group in optimizer.param_groups:
        if param_group['lr'] < args.lr * 0.5:
            param_group['lr'] = lr_backbone
        else:
            param_group['lr'] = lr

    return lr


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: SetCriterion,
    data_loader: DataLoader,
    device: torch.device,
    args,
):
    """Run validation and compute COCO metrics."""
    model.eval()
    metric_logger = MetricLogger()

    print("\nRunning validation...")

    for batch in data_loader:
        images = torch.stack([item['image'] for item in batch]).to(device)
        targets = [{
            'boxes': item['target']['boxes'].to(device),
            'labels': item['target']['labels'].to(device),
        } for item in batch]

        outputs = model(images)
        loss_dict = criterion(outputs, targets)
        loss_dict_reduced = reduce_dict(loss_dict)

        metric_logger.update(
            **{k: v.item() for k, v in loss_dict_reduced.items()}
        )

    print(f"Validation: {metric_logger}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
