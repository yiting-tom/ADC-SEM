#!/usr/bin/env python3
"""
Fine-tune TinyViT for TIFF classification.

File naming: <label_id>_<...rest...>.tiff
  - Binary: label_id==1 => Normal (class 1), others => Abnormal (class 0)
  - Multiclass: label_id maps to class IDs discovered in the dataset
"""

import argparse
import os
import torch

from finetune_training import FinetuneConfig, FinetuneTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune TinyViT on multi-channel TIFF classification")
    p.add_argument("--data_path", type=str, required=True, help="Directory with .tiff/.tif files")
    p.add_argument("--task", type=str, default="multiclass", choices=["binary", "multiclass"])
    p.add_argument("--num_channels", type=int, default=15)
    p.add_argument("--model_name", type=str, default="tiny_vit_21m_512.dist_in22k_ft_in1k",
                   choices=[
                       'tiny_vit_5m_224.dist_in22k_ft_in1k',
                       'tiny_vit_11m_224.dist_in22k_ft_in1k',
                       'tiny_vit_21m_224.dist_in22k_ft_in1k',
                       'tiny_vit_21m_512.dist_in22k_ft_in1k'
                   ])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)

    # Pretrained control
    p.add_argument("--pretrained_path", type=str, default=None, help="Path to local TinyViT weights (state_dict)")
    p.add_argument("--no_timm_pretrained", action="store_true")

    # Output
    p.add_argument("--output_dir", type=str, default="./outputs")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"data_path must be a directory: {args.data_path}")
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = FinetuneConfig(
        data_path=args.data_path,
        task=args.task,
        num_channels=args.num_channels,
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_split=args.val_split,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
        pretrained_path=args.pretrained_path,
        timm_pretrained=(not args.no_timm_pretrained),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    print(f"🚀 Fine-tuning TinyViT for {args.task} classification on TIFFs")
    print(f"Data: {args.data_path} | Channels: {args.num_channels} | Model: {args.model_name}")
    print(f"Batch: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    print(f"Freeze backbone: {args.freeze_backbone} | Dropout: {args.dropout}")
    if args.pretrained_path:
        print(f"Using local pretrained weights: {args.pretrained_path}")

    FinetuneTrainer(cfg).train()


if __name__ == "__main__":
    main()

