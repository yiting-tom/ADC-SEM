"""
Fine-tuning TinyViT for classification on multi-channel TIFF images.

File naming convention (TIFF):
  <label_id>_<...rest...>.tiff

Examples:
  26_35_U2523.20_1368.tiff  -> multiclass label_id=26
  1_something_xxx.tiff      -> binary: Normal (id 1)
  any_other_label_xxx.tiff  -> binary: Abnormal
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

from model import TinyViTBackbone


def _load_tiff(path: str) -> np.ndarray:
    try:
        import tifffile
        img = tifffile.imread(path)
    except ImportError:
        from PIL import Image
        img = np.array(Image.open(path))
    return img


def _to_chw(img: np.ndarray, channels: int) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {img.shape}")
    if img.shape[0] == channels:
        return img
    if img.shape[-1] == channels:
        return np.transpose(img, (2, 0, 1))
    raise ValueError(f"Expected {channels} channels, got shape {img.shape}")


class LabeledTiffDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        channels: int = 15,
        image_size: Tuple[int, int] = (460, 460),
        task: str = "multiclass",  # or "binary"
        transforms: Optional[Callable] = None,
    ) -> None:
        self.root_dir = root_dir
        self.channels = channels
        self.image_size = image_size
        self.task = task
        self.transforms = transforms

        self.files = sorted(
            [p for p in glob.glob(os.path.join(root_dir, "*.tiff"))]
            + [p for p in glob.glob(os.path.join(root_dir, "*.tif"))]
        )
        if not self.files:
            raise ValueError(f"No .tiff/.tif files found in {root_dir}")

        # Parse labels
        self.raw_labels: List[int] = []
        for p in self.files:
            stem = os.path.splitext(os.path.basename(p))[0]
            first = stem.split("_")[0]
            try:
                self.raw_labels.append(int(first))
            except Exception as e:
                raise ValueError(f"Failed to parse label from '{p}': {e}")

        if task == "binary":
            # Map label id 1 => Normal (1), else Abnormal (0)
            self.classes = ["Abnormal", "Normal"]
            self.class_to_idx = {"Abnormal": 0, "Normal": 1}
        elif task == "multiclass":
            unique = sorted(set(self.raw_labels))
            self.classes = [str(u) for u in unique]
            self.class_to_idx = {str(u): i for i, u in enumerate(unique)}
        else:
            raise ValueError("task must be 'binary' or 'multiclass'")

    def __len__(self) -> int:
        return len(self.files)

    def _label_for_index(self, idx: int) -> int:
        raw = self.raw_labels[idx]
        if self.task == "binary":
            return 1 if raw == 1 else 0
        return self.class_to_idx[str(raw)]

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = _load_tiff(path)
        img = _to_chw(img, self.channels)

        # Resize if needed
        if img.shape[1:] != self.image_size:
            from scipy.ndimage import zoom
            h_factor = self.image_size[0] / img.shape[1]
            w_factor = self.image_size[1] / img.shape[2]
            img = zoom(img, (1, h_factor, w_factor), order=1)

        if img.max() > 1.0:
            img = img / 255.0

        x = torch.from_numpy(img.copy()).float()
        if self.transforms is not None:
            x = self.transforms(x)

        y = self._label_for_index(idx)
        return x, y


def default_train_transforms(num_channels: int = 15, size: int = 460) -> T.Compose:
    kernel = int(0.1 * size) // 2 * 2 + 1
    return T.Compose(
        [
            T.RandomResizedCrop(size=size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=kernel, sigma=(0.1, 2.0))], p=0.2),
            T.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels),
        ]
    )


def default_val_transforms(num_channels: int = 15, size: int = 460) -> T.Compose:
    return T.Compose([T.Normalize(mean=[0.5] * num_channels, std=[0.5] * num_channels)])


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, p_drop: float = 0.0):
        super().__init__()
        if p_drop > 0:
            self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(in_dim, num_classes))
        else:
            self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FinetuneModel(nn.Module):
    def __init__(
        self,
        num_channels: int,
        model_name: str,
        num_classes: int,
        pretrained_path: Optional[str] = None,
        timm_pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = TinyViTBackbone(
            num_channels=num_channels,
            model_name=model_name,
            pretrained_path=pretrained_path,
            timm_pretrained=timm_pretrained,
        )
        self.classifier = ClassificationHead(self.backbone.feature_dim, num_classes, p_drop=dropout)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.classifier(feats)


@dataclass
class FinetuneConfig:
    data_path: str
    task: str = "multiclass"  # or "binary"
    num_channels: int = 15
    model_name: str = "tiny_vit_21m_512.dist_in22k_ft_in1k"
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-6
    num_workers: int = 8
    val_split: float = 0.1
    freeze_backbone: bool = False
    dropout: float = 0.0
    pretrained_path: Optional[str] = None
    timm_pretrained: bool = True
    device: Optional[str] = None


class FinetuneTrainer:
    def __init__(self, cfg: FinetuneConfig) -> None:
        self.cfg = cfg

    def _build_datasets(self) -> Tuple[Dataset, Dataset, Dict[str, int]]:
        train_tf = default_train_transforms(self.cfg.num_channels)
        val_tf = default_val_transforms(self.cfg.num_channels)
        full = LabeledTiffDataset(
            root_dir=self.cfg.data_path,
            channels=self.cfg.num_channels,
            task=self.cfg.task,
            transforms=train_tf,
        )
        # Use a copy for val with val transforms
        val_ds = LabeledTiffDataset(
            root_dir=self.cfg.data_path,
            channels=self.cfg.num_channels,
            task=self.cfg.task,
            transforms=val_tf,
        )
        # Split indices
        n_total = len(full)
        n_val = max(1, int(n_total * self.cfg.val_split))
        n_train = n_total - n_val

        # Use Subset to share label parsing and file lists
        g = torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(range(n_total), [n_train, n_val], generator=g)

        class_to_idx = full.class_to_idx

        # Wrap subsets with lambda-style indexing
        class Proxy(Dataset):
            def __init__(self, base: LabeledTiffDataset, indices: List[int]):
                self.base = base
                self.indices = list(indices)
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, i):
                return self.base[self.indices[i]]

        return Proxy(full, train_subset.indices), Proxy(val_ds, val_subset.indices), class_to_idx

    def train(self) -> Dict[str, List[float]]:
        device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        train_ds, val_ds, class_to_idx = self._build_datasets()
        num_classes = 2 if self.cfg.task == "binary" else len(class_to_idx)

        model = FinetuneModel(
            num_channels=self.cfg.num_channels,
            model_name=self.cfg.model_name,
            num_classes=num_classes,
            pretrained_path=self.cfg.pretrained_path,
            timm_pretrained=self.cfg.timm_pretrained,
            freeze_backbone=self.cfg.freeze_backbone,
            dropout=self.cfg.dropout,
        ).to(device)

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers, drop_last=False
        )

        if self.cfg.task == "binary":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epochs, eta_min=self.cfg.lr * 0.01)

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        for epoch in range(self.cfg.epochs):
            model.train()
            running = 0.0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                if self.cfg.task == "binary":
                    loss = criterion(logits.squeeze(dim=-1), y.float())
                else:
                    loss = criterion(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += float(loss.detach())
            scheduler.step()
            train_loss = running / max(1, len(train_loader))

            # Validation
            model.eval()
            v_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    logits = model(x)
                    if self.cfg.task == "binary":
                        loss = criterion(logits.squeeze(dim=-1), y.float())
                        preds = (torch.sigmoid(logits).squeeze(dim=-1) >= 0.5).long()
                    else:
                        loss = criterion(logits, y)
                        preds = logits.argmax(dim=-1)
                    v_loss += float(loss)
                    correct += int((preds == y).sum().item())
                    total += int(y.numel())
            val_loss = v_loss / max(1, len(val_loader))
            val_acc = correct / max(1, total)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

        # Save final model
        os.makedirs("outputs", exist_ok=True)
        save_name = f"tinyvit_finetune_{self.cfg.task}.pth"
        torch.save(model.state_dict(), os.path.join("outputs", save_name))
        print(f"✅ Saved fine-tuned model to outputs/{save_name}")
        return history

