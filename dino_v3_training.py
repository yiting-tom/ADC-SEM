"""
DINOv3-style SSL training for TinyViT with multi-channel input.

This mirrors our DINOv2 trainer but uses DINOv3-leaning defaults:
- More local crops by default (8)
- Larger projection dim (384)
- Cosine LR and EMA schedules
- Geometric-only transforms to support >3 channels
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from lightly.data import LightlyDataset
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum

from dataset import MultiChannelDataset, MultiChannelH5Dataset
from model import TinyViTBackbone


class MultiChannelDinoV3Collate:
    """Multi-crop collate for DINOv3-style training.

    Creates two global crops and N local crops with geometric transforms only.
    """

    def __init__(
        self,
        num_channels: int = 15,
        global_size: int = 460,
        local_size: int = 196,
        num_local_crops: int = 8,
        global_min_scale: float = 0.25,
        local_min_scale: float = 0.05,
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        rr_degrees: float = 0,
        gaussian_blur_p: float = 0.5,
        normalize: bool = True,
    ) -> None:
        def build_tx(size: int, min_scale: float):
            ops = [
                transforms.RandomResizedCrop(
                    size=size,
                    scale=(min_scale, 1.0),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
            ]
            if hf_prob > 0:
                ops.append(transforms.RandomHorizontalFlip(p=hf_prob))
            if vf_prob > 0:
                ops.append(transforms.RandomVerticalFlip(p=vf_prob))
            if rr_degrees > 0:
                ops.append(transforms.RandomRotation(degrees=rr_degrees))
            if gaussian_blur_p > 0:
                kernel = int(0.1 * size) // 2 * 2 + 1
                ops.append(
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=kernel, sigma=(0.1, 2.0))],
                        p=gaussian_blur_p,
                    )
                )
            if normalize:
                ops.append(
                    transforms.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)
                )
            return transforms.Compose(ops)

        self.global_transform = build_tx(global_size, global_min_scale)
        self.local_transform = build_tx(local_size, local_min_scale)
        self.num_local_crops = num_local_crops

    def __call__(self, batch) -> Tuple[List[torch.Tensor], torch.Tensor]:
        imgs = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch])
        crops: List[torch.Tensor] = []
        for img in imgs:
            crops.append(self.global_transform(img))
            crops.append(self.global_transform(img))
            for _ in range(self.num_local_crops):
                crops.append(self.local_transform(img))
        batch_size = len(imgs)
        num_crops = 2 + self.num_local_crops
        stacked: List[torch.Tensor] = []
        for i in range(num_crops):
            crop_i = torch.stack(crops[i::num_crops], dim=0)
            stacked.append(crop_i)
        return stacked, labels


class DinoV3TinyViT(nn.Module):
    """Student/Teacher TinyViT with DINO projection heads for DINOv3-style training."""

    def __init__(self, backbone: TinyViTBackbone, proj_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.backbone = backbone
        feat_dim = backbone.feature_dim
        if hidden_dim is None:
            hidden_dim = feat_dim
        self.student_head = DINOProjectionHead(feat_dim, hidden_dim, proj_dim)

        # Teacher momentum copies; create fresh head and load state dict
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = DINOProjectionHead(feat_dim, hidden_dim, proj_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict(), strict=True)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        return self.student_head(self.backbone(x))

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        return self.teacher_head(self.teacher_backbone(x)).detach()


@dataclass
class DINOv3TrainConfig:
    data_path: str
    use_h5: bool = False
    h5_dataset_key: str = "images"
    file_extension: str = "npy"
    num_channels: int = 15
    model_name: str = "tiny_vit_21m_512.dist_in22k_ft_in1k"
    pretrained_path: Optional[str] = None
    timm_pretrained: bool = True
    batch_size: int = 32
    num_workers: int = 8
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-6
    proj_dim: int = 384
    momentum_base: float = 0.996
    student_temp: float = 0.2
    teacher_temp: float = 0.07
    warmup_teacher_temp: float = 0.04
    warmup_teacher_temp_epochs: int = 10
    center_momentum: float = 0.9
    local_crops: int = 8
    device: Optional[str] = None


class DINOv3Trainer:
    def __init__(self, cfg: DINOv3TrainConfig) -> None:
        self.cfg = cfg

    def _build_dataloader(self) -> DataLoader:
        if self.cfg.use_h5:
            base_ds = MultiChannelH5Dataset(
                h5_file_path=self.cfg.data_path,
                dataset_key=self.cfg.h5_dataset_key,
                channels=self.cfg.num_channels,
            )
        else:
            base_ds = MultiChannelDataset(
                root_dir=self.cfg.data_path,
                channels=self.cfg.num_channels,
                image_size=(460, 460),
                file_extension=self.cfg.file_extension,
            )
        ds = LightlyDataset.from_torch_dataset(base_ds)
        collate = MultiChannelDinoV3Collate(num_channels=self.cfg.num_channels, num_local_crops=self.cfg.local_crops)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            collate_fn=collate,
        )

    def _build_model(self) -> DinoV3TinyViT:
        backbone = TinyViTBackbone(
            num_channels=self.cfg.num_channels,
            model_name=self.cfg.model_name,
            pretrained_path=self.cfg.pretrained_path,
            timm_pretrained=self.cfg.timm_pretrained,
        )
        model = DinoV3TinyViT(backbone=backbone, proj_dim=self.cfg.proj_dim)
        return model

    def train(self) -> Tuple[nn.Module, dict]:
        device = (
            self.cfg.device
            if self.cfg.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dataloader = self._build_dataloader()
        model = self._build_model().to(device)

        with torch.no_grad():
            dummy = torch.randn(1, self.cfg.num_channels, 460, 460, device=device)
            out_dim = model.forward_student(dummy).shape[-1]
        criterion = DINOLoss(
            output_dim=out_dim,
            warmup_teacher_temp=self.cfg.warmup_teacher_temp,
            teacher_temp=self.cfg.teacher_temp,
            warmup_teacher_temp_epochs=self.cfg.warmup_teacher_temp_epochs,
            student_temp=self.cfg.student_temp,
            center_momentum=self.cfg.center_momentum,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.lr * 0.01)

        print("Starting DINOv3-style training")
        metrics = {"epoch_loss": []}
        for epoch in range(self.cfg.epochs):
            model.train()
            total_loss = 0.0

            # Cosine EMA momentum schedule: base -> 1.0
            m_val = 1.0 - (1.0 - self.cfg.momentum_base) * (0.5 * (1.0 + torch.cos(torch.tensor(epoch / self.cfg.epochs * 3.1415926535))).item())

            for crops, _ in dataloader:
                # EMA teacher update
                update_momentum(model.backbone, model.teacher_backbone, m=m_val)
                update_momentum(model.student_head, model.teacher_head, m=m_val)

                # Student on all crops
                student_outs = [model.forward_student(x.to(device, non_blocking=True)) for x in crops]
                # Teacher on global crops only
                teacher_outs = [model.forward_teacher(crops[i].to(device, non_blocking=True)) for i in range(2)]

                loss = criterion(student_outs, teacher_outs, epoch)
                total_loss += float(loss.detach())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            scheduler.step()
            avg = total_loss / len(dataloader)
            metrics["epoch_loss"].append(avg)
            print(f"epoch: {epoch:02d}, loss: {avg:.5f}, m: {m_val:.5f}")

        return model, metrics
