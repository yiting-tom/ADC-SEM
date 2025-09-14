#!/usr/bin/env python3
"""
Lightweight smoke test for all SSL methods (SimCLR, MoCo v3, DINOv2, DINOv3).

Runs a single forward/backward/optimizer step per method on a tiny batch
loaded from the local dataset, without using timm's online pretrained weights.

Usage:
  python smoke_test.py --data_path ./example_data --file_extension npy
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch

from dataset import MultiChannelDataset, MultiChannelH5Dataset
from model import create_ssl_model_components, TinyViTBackbone
from ssl_training import MultiChannelCollateFunction
from lightly.loss import NTXentLoss, DINOLoss

from moco_training import MultiChannelMoCoCollate, MoCoTinyViT
from dino_training import MultiChannelDinoCollate, DinoTinyViT
from dino_v3_training import MultiChannelDinoV3Collate, DinoV3TinyViT


def load_small_batch(data_path: str, use_h5: bool, file_extension: str, channels: int) -> List[Tuple[torch.Tensor, int]]:
    if use_h5:
        ds = MultiChannelH5Dataset(h5_file_path=data_path, dataset_key='images', channels=channels)
    else:
        ds = MultiChannelDataset(root_dir=data_path, file_extension=file_extension, channels=channels, image_size=(460, 460))
    n = min(2, len(ds))
    return [ds[i] for i in range(n)]


def test_simclr(batch, channels: int = 15):
    print("[SimCLR] Building batch and model...")
    collate = MultiChannelCollateFunction(input_size=460, hf_prob=0.5, gaussian_blur=0.1)
    transforms, labels, _ = collate(batch)
    bsz = transforms.shape[0] // 2
    x0, x1 = transforms[:bsz], transforms[bsz:]

    backbone, proj = create_ssl_model_components(num_channels=channels, projection_dim=64, timm_pretrained=False)
    z0 = proj(backbone(x0))
    z1 = proj(backbone(x1))
    loss_fn = NTXentLoss(temperature=0.2)
    loss = loss_fn(z0, z1)
    loss.backward()
    print(f"[SimCLR] OK. loss={loss.item():.5f}")


def test_moco(batch, channels: int = 15):
    print("[MoCo] Building batch and model...")
    collate = MultiChannelMoCoCollate(num_channels=channels)
    (xq, xk), _ = collate(batch)
    model = MoCoTinyViT(TinyViTBackbone(num_channels=channels, timm_pretrained=False), proj_dim=64)
    criterion = NTXentLoss(memory_bank_size=(64, 64))
    q = model(xq)
    k = model.forward_momentum(xk)
    loss = criterion(q, k)
    loss.backward()
    print(f"[MoCo] OK. loss={loss.item():.5f}")


def test_dino_v2(batch, channels: int = 15):
    print("[DINOv2] Building batch and model...")
    collate = MultiChannelDinoCollate(num_channels=channels, num_local_crops=1)
    crops, _ = collate(batch)
    model = DinoTinyViT(TinyViTBackbone(num_channels=channels, timm_pretrained=False), proj_dim=128)
    with torch.no_grad():
        out_dim = model.forward_student(crops[0]).shape[-1]
    criterion = DINOLoss(output_dim=out_dim, warmup_teacher_temp=0.04, teacher_temp=0.07, warmup_teacher_temp_epochs=1, student_temp=0.1, center_momentum=0.9)
    student_outs = [model.forward_student(x) for x in crops]
    teacher_outs = [model.forward_teacher(crops[i]) for i in range(2)]
    loss = criterion(student_outs, teacher_outs, epoch=0)
    loss.backward()
    print(f"[DINOv2] OK. loss={loss.item():.5f}")


def test_dino_v3(batch, channels: int = 15):
    print("[DINOv3] Building batch and model...")
    collate = MultiChannelDinoV3Collate(num_channels=channels, num_local_crops=1)
    crops, _ = collate(batch)
    model = DinoV3TinyViT(TinyViTBackbone(num_channels=channels, timm_pretrained=False), proj_dim=192)
    with torch.no_grad():
        out_dim = model.forward_student(crops[0]).shape[-1]
    criterion = DINOLoss(output_dim=out_dim, warmup_teacher_temp=0.04, teacher_temp=0.07, warmup_teacher_temp_epochs=1, student_temp=0.2, center_momentum=0.9)
    student_outs = [model.forward_student(x) for x in crops]
    teacher_outs = [model.forward_teacher(crops[i]) for i in range(2)]
    loss = criterion(student_outs, teacher_outs, epoch=0)
    loss.backward()
    print(f"[DINOv3] OK. loss={loss.item():.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, default='./example_data')
    ap.add_argument('--file_extension', type=str, default='npy', choices=['npy', 'tiff', 'tif'])
    ap.add_argument('--use_h5', action='store_true')
    ap.add_argument('--num_channels', type=int, default=15)
    ap.add_argument('--methods', type=str, nargs='*', default=['simclr', 'moco', 'dino2', 'dino3'],
                    help='Subset of methods to test')
    args = ap.parse_args()

    batch = load_small_batch(args.data_path, args.use_h5, args.file_extension, channels=args.num_channels)

    if 'simclr' in args.methods:
        test_simclr(batch, channels=args.num_channels)
    if 'moco' in args.methods:
        test_moco(batch, channels=args.num_channels)
    if 'dino2' in args.methods:
        test_dino_v2(batch, channels=args.num_channels)
    if 'dino3' in args.methods:
        test_dino_v3(batch, channels=args.num_channels)

    print("✅ Smoke test completed.")


if __name__ == '__main__':
    main()
