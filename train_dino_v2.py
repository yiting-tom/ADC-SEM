#!/usr/bin/env python3
"""
DINOv2-style SSL training CLI for TinyViT with multi-channel input.

Examples:
  python train_dino_v2.py --data_path ./data --file_extension npy --num_channels 15
  python train_dino_v2.py --data_path ./tiff_data --file_extension tiff --epochs 50
  python train_dino_v2.py --data_path ./dataset.h5 --use_h5 --h5_dataset_key images
"""

import argparse
import os
import torch

from dino_training import DINOTrainConfig, DINOTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train TinyViT with DINOv2-style SSL")

    # Data
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--use_h5", action="store_true")
    p.add_argument("--h5_dataset_key", type=str, default="images")
    p.add_argument("--file_extension", type=str, default="npy", choices=["npy", "tiff", "tif"])

    # Model
    p.add_argument("--num_channels", type=int, default=15)
    p.add_argument("--model_name", type=str, default="tiny_vit_21m_512.dist_in22k_ft_in1k",
                   choices=[
                       'tiny_vit_5m_224.dist_in22k_ft_in1k',
                       'tiny_vit_11m_224.dist_in22k_ft_in1k',
                       'tiny_vit_21m_224.dist_in22k_ft_in1k',
                       'tiny_vit_21m_512.dist_in22k_ft_in1k'
                   ])

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--momentum_base", type=float, default=0.996)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--teacher_temp", type=float, default=0.07)
    p.add_argument("--warmup_teacher_temp", type=float, default=0.04)
    p.add_argument("--warmup_teacher_temp_epochs", type=int, default=10)
    p.add_argument("--center_momentum", type=float, default=0.9)
    p.add_argument("--local_crops", type=int, default=6)

    # Pretrained control
    p.add_argument("--pretrained_path", type=str, default=None,
                   help="Path to local pretrained TinyViT weights (state_dict)")
    p.add_argument("--no_timm_pretrained", action="store_true",
                   help="Disable timm pretrained weights to avoid network usage")

    # Output
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--save_name", type=str, default="tinyvit_dino_v2_backbone.pth")

    return p.parse_args()


def validate_args(args):
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")
    if args.use_h5 and not args.data_path.endswith((".h5", ".hdf5")):
        raise ValueError("--use_h5 requires an .h5 or .hdf5 file path")
    if not args.use_h5 and not os.path.isdir(args.data_path):
        raise ValueError("When not using --use_h5, data_path must be a directory")
    os.makedirs(args.output_dir, exist_ok=True)
    return args


def main():
    args = validate_args(parse_args())

    print("🚀 Starting DINOv2-style SSL Training")
    print(f"Data path: {args.data_path}")
    print(f"Model: {args.model_name}")
    print(f"Channels: {args.num_channels}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")

    cfg = DINOTrainConfig(
        data_path=args.data_path,
        use_h5=args.use_h5,
        h5_dataset_key=args.h5_dataset_key,
        file_extension=args.file_extension,
        num_channels=args.num_channels,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        proj_dim=args.proj_dim,
        momentum_base=args.momentum_base,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp=args.warmup_teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        center_momentum=args.center_momentum,
        local_crops=args.local_crops,
        pretrained_path=args.pretrained_path,
        timm_pretrained=(not args.no_timm_pretrained),
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )

    model, metrics = DINOTrainer(cfg).train()

    # Save backbone (student) for downstream tasks
    backbone_path = os.path.join(args.output_dir, args.save_name)
    torch.save(model.backbone.state_dict(), backbone_path)
    print(f"✅ Saved backbone (student) to {backbone_path}")

    # Optionally save student head
    head_path = backbone_path.replace("_backbone", "_head").replace(".pth", ".pth")
    torch.save(model.student_head.state_dict(), head_path)
    print(f"✅ Saved projection head to {head_path}")

    if metrics.get("epoch_loss"):
        print(f"Final epoch loss: {metrics['epoch_loss'][-1]:.5f}")


if __name__ == "__main__":
    main()
