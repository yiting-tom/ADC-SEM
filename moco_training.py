"""
MoCo-style SSL training for TinyViT with multi-channel input.

This adapts the official lightly example to:
- Use TinyViT multi-channel backbone from this repository
- Work with our MultiChannel* datasets and geometric-only augmentations
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from dataset import MultiChannelDataset, MultiChannelH5Dataset
from model import TinyViTBackbone


class MultiChannelMoCoCollate:
    """Geometric-only augmentations for multi-channel images.

    Produces two views per sample suitable for MoCo-style training.
    """

    def __init__(
        self,
        input_size: int = 460,
        num_channels: int = 15,
        min_scale: float = 0.08,
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        rr_degrees: float = 0,
        gaussian_blur: float = 0.5,
        normalize: bool = True,
    ) -> None:
        tx = [
            transforms.RandomResizedCrop(
                size=input_size,
                scale=(min_scale, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
        if hf_prob > 0:
            tx.append(transforms.RandomHorizontalFlip(p=hf_prob))
        if vf_prob > 0:
            tx.append(transforms.RandomVerticalFlip(p=vf_prob))
        if rr_degrees > 0:
            tx.append(transforms.RandomRotation(degrees=rr_degrees))
        if gaussian_blur > 0:
            kernel = int(0.1 * input_size) // 2 * 2 + 1
            tx.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=kernel, sigma=(0.1, 2.0))],
                    p=gaussian_blur,
                )
            )
        if normalize:
            tx.append(
                transforms.Normalize(
                    mean=[0.5] * num_channels,  # consider computing dataset stats
                    std=[0.5] * num_channels,
                )
            )
        self.transform = transforms.Compose(tx)

    def __call__(self, batch):
        # batch: list of tuples (C,H,W tensor, label)
        imgs = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch])
        q_list, k_list = [], []
        for img in imgs:
            q_list.append(self.transform(img))
            k_list.append(self.transform(img))
        x_query = torch.stack(q_list, dim=0)
        x_key = torch.stack(k_list, dim=0)
        return (x_query, x_key), labels


class MoCoTinyViT(nn.Module):
    """MoCo model with TinyViT backbone for multi-channel data."""

    def __init__(self, backbone: nn.Module, proj_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.backbone = backbone
        feat_dim = backbone.feature_dim if hasattr(backbone, "feature_dim") else None
        if feat_dim is None:
            raise ValueError("Backbone must expose feature_dim attribute")
        if hidden_dim is None:
            hidden_dim = feat_dim

        self.projection_head = MoCoProjectionHead(feat_dim, hidden_dim, proj_dim)

        # Momentum encoder copies
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.backbone(x)
        q = self.projection_head(q)
        return q

    @torch.no_grad()
    def forward_momentum(self, x: torch.Tensor) -> torch.Tensor:
        k = self.backbone_momentum(x)
        k = self.projection_head_momentum(k).detach()
        return k


@dataclass
class MoCoTrainConfig:
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
    lr: float = 6e-2
    weight_decay: float = 1e-4
    momentum_base: float = 0.996
    projection_dim: int = 128
    memory_bank_size: int = 4096
    device: Optional[str] = None


class MoCoTrainer:
    def __init__(self, cfg: MoCoTrainConfig) -> None:
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
        collate = MultiChannelMoCoCollate(num_channels=self.cfg.num_channels)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            drop_last=True,
            persistent_workers=True if self.cfg.num_workers > 0 else False,
            collate_fn=collate,
        )

    def _build_model(self) -> MoCoTinyViT:
        backbone = TinyViTBackbone(
            num_channels=self.cfg.num_channels,
            model_name=self.cfg.model_name,
            pretrained_path=self.cfg.pretrained_path,
            timm_pretrained=self.cfg.timm_pretrained,
        )
        model = MoCoTinyViT(backbone=backbone, proj_dim=self.cfg.projection_dim)
        return model

    def train(self) -> tuple[nn.Module, dict]:
        device = (
            self.cfg.device
            if self.cfg.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dataloader = self._build_dataloader()
        model = self._build_model().to(device)

        # NTXent with memory bank for negatives
        criterion = NTXentLoss(memory_bank_size=(self.cfg.memory_bank_size, self.cfg.projection_dim))
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay
        )

        print("Starting MoCo training")
        metrics = {"epoch_loss": []}
        for epoch in range(self.cfg.epochs):
            model.train()
            total_loss = 0.0
            m_val = cosine_schedule(epoch, self.cfg.epochs, self.cfg.momentum_base, 1.0)

            for (x_query, x_key), _ in dataloader:
                # Momentum update
                update_momentum(model.backbone, model.backbone_momentum, m=m_val)
                update_momentum(model.projection_head, model.projection_head_momentum, m=m_val)

                x_query = x_query.to(device, non_blocking=True)
                x_key = x_key.to(device, non_blocking=True)

                q = model(x_query)
                k = model.forward_momentum(x_key)
                loss = criterion(q, k)
                total_loss += float(loss.detach())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            metrics["epoch_loss"].append(avg_loss)
            print(f"epoch: {epoch:02d}, loss: {avg_loss:.5f}, m: {m_val:.5f}")

        # Save backbone and projection head state dicts for reuse
        return model, metrics
