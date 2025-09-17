import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from lightly.data import LightlyDataset
from lightly.data.collate import SimCLRCollateFunction, BaseCollateFunction
from lightly.models.simclr import SimCLR
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms import SimCLRTransform
from lightly.loss import NTXentLoss

from dataset import MultiChannelDataset, MultiChannelH5Dataset
from model import create_ssl_model_components, TinyViTBackbone


class MultiChannelCollateFunction(BaseCollateFunction):
    """
    Custom collate function for multi-channel data that applies
    only geometric transformations (no color augmentations).
    """

    def __init__(
        self,
        input_size: int = 460,
        min_scale: float = 0.08,
        normalize: bool = True,
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        rr_degrees: float = 0,
        gaussian_blur: float = 0.5,
    ):
        # Create transforms that work for multi-channel data
        transform = [
            transforms.RandomResizedCrop(
                size=input_size,
                scale=(min_scale, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
        ]

        if hf_prob > 0:
            transform.append(transforms.RandomHorizontalFlip(p=hf_prob))

        if vf_prob > 0:
            transform.append(transforms.RandomVerticalFlip(p=vf_prob))

        if rr_degrees > 0:
            transform.append(transforms.RandomRotation(degrees=rr_degrees))

        if gaussian_blur > 0:
            # Note: GaussianBlur works on multi-channel tensors
            transform.append(transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=int(0.1 * input_size) // 2 * 2 + 1, sigma=(0.1, 2.0))
            ], p=gaussian_blur))

        if normalize:
            # For multi-channel data, we'll normalize per-channel
            # You may need to compute your own mean/std for 15 channels
            transform.append(transforms.Normalize(
                mean=[0.5] * 15,  # Placeholder - compute actual stats
                std=[0.5] * 15
            ))

        self.transform = transforms.Compose(transform)

    def __call__(self, batch):
        """Apply transformations and create positive pairs."""
        batch_size = len(batch)

        # Extract images and labels
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])

        # Create two augmented views for each image
        transforms = []
        for img in images:
            transform1 = self.transform(img)
            transform2 = self.transform(img)
            transforms.extend([transform1, transform2])

        # Stack all transforms
        transforms = torch.stack(transforms)

        # Create labels for contrastive learning
        labels = torch.cat([labels, labels])

        return transforms, labels, []  # Empty list for filenames (not used)


class SSLTrainingModule(pl.LightningModule):
    """
    PyTorch Lightning module for SSL training with TinyViT.
    """

    def __init__(
        self,
        num_channels: int = 15,
        model_name: str = 'tiny_vit_21m_512.dist_in22k_ft_in1k',
        projection_dim: int = 128,
        projection_head: str = 'mlp',
        kan_num_knots: int = 16,
        kan_x_min: float = -3.0,
        kan_x_max: float = 3.0,
        kan_use_skip: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        temperature: float = 0.1,
        max_epochs: int = 100,
        pretrained_path: str | None = None,
        timm_pretrained: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model components
        self.backbone, projection_head = create_ssl_model_components(
            num_channels=num_channels,
            model_name=model_name,
            projection_dim=projection_dim,
            pretrained_path=pretrained_path,
            timm_pretrained=timm_pretrained,
            projection_head_type=projection_head,
            kan_num_knots=kan_num_knots,
            kan_x_min=kan_x_min,
            kan_x_max=kan_x_max,
            kan_use_skip=kan_use_skip,
        )

        # Create SSL model - manually combine backbone and projection head
        self.backbone_model = self.backbone
        self.projection_head = projection_head

        # Loss function
        self.criterion = NTXentLoss(temperature=temperature)

        # Training metrics
        self.train_losses = []

    def forward(self, x):
        features = self.backbone_model(x)
        projections = self.projection_head(features)
        return projections

    def training_step(self, batch, batch_idx):
        transforms, labels, filenames = batch

        # Split into two views for contrastive learning
        batch_size = transforms.shape[0] // 2
        x0 = transforms[:batch_size]
        x1 = transforms[batch_size:]

        # Get features
        z0 = self.forward(x0)
        z1 = self.forward(x1)

        # Compute contrastive loss
        loss = self.criterion(z0, z1)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_losses.append(loss.item())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.learning_rate * 0.01
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def on_train_epoch_end(self):
        if len(self.train_losses) > 0:
            avg_loss = sum(self.train_losses[-50:]) / min(50, len(self.train_losses))
            print(f"Epoch {self.current_epoch}: Average Loss = {avg_loss:.4f}")


class SSLTrainer:
    """
    High-level trainer class for SSL training.
    """

    def __init__(
        self,
        data_path: str,
        num_channels: int = 15,
        batch_size: int = 32,
        num_workers: int = 8,
        max_epochs: int = 100,
        learning_rate: float = 1e-3,
        model_name: str = 'tiny_vit_21m_512.dist_in22k_ft_in1k',
        projection_head: str = 'mlp',
        kan_num_knots: int = 16,
        kan_x_min: float = -3.0,
        kan_x_max: float = 3.0,
        kan_use_skip: bool = True,
        use_h5: bool = False,
        h5_dataset_key: str = 'images',
        file_extension: str = 'npy',
        pretrained_path: str | None = None,
        timm_pretrained: bool = True,
    ):
        self.data_path = data_path
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.projection_head = projection_head
        self.kan_num_knots = kan_num_knots
        self.kan_x_min = kan_x_min
        self.kan_x_max = kan_x_max
        self.kan_use_skip = kan_use_skip
        self.use_h5 = use_h5
        self.h5_dataset_key = h5_dataset_key
        self.file_extension = file_extension
        self.pretrained_path = pretrained_path
        self.timm_pretrained = timm_pretrained

    def create_dataloader(self):
        """Create the data loader for training."""
        # Create dataset
        if self.use_h5:
            dataset = MultiChannelH5Dataset(
                h5_file_path=self.data_path,
                dataset_key=self.h5_dataset_key,
                channels=self.num_channels
            )
        else:
            dataset = MultiChannelDataset(
                root_dir=self.data_path,
                channels=self.num_channels,
                image_size=(460, 460),
                file_extension=self.file_extension
            )

        # Wrap with LightlyDataset
        lightly_dataset = LightlyDataset.from_torch_dataset(dataset)

        # Create collate function
        collate_fn = MultiChannelCollateFunction(
            input_size=460,
            hf_prob=0.5,
            gaussian_blur=0.5
        )

        # Create dataloader
        dataloader = DataLoader(
            lightly_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

        return dataloader

    def train(self, accelerator: str = 'auto', devices: int = 1, save_path: str = 'tinyvit_ssl_model.ckpt'):
        """Run the SSL training."""
        # Create dataloader
        dataloader = self.create_dataloader()
        print(f"Created dataloader with {len(dataloader)} batches")

        # Create model
        model = SSLTrainingModule(
            num_channels=self.num_channels,
            model_name=self.model_name,
            projection_head=self.projection_head,
            kan_num_knots=self.kan_num_knots,
            kan_x_min=self.kan_x_min,
            kan_x_max=self.kan_x_max,
            kan_use_skip=self.kan_use_skip,
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            pretrained_path=self.pretrained_path,
            timm_pretrained=self.timm_pretrained,
        )

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision='16-mixed' if torch.cuda.is_available() else '32',
            gradient_clip_val=1.0,
            log_every_n_steps=50,
        )

        # Train the model
        print("🚀 Starting SSL training...")
        trainer.fit(model, dataloader)

        # Save the trained model
        trainer.save_checkpoint(save_path)
        print(f"✅ Model saved to {save_path}")

        # Also save just the backbone for downstream tasks
        backbone_path = save_path.replace('.ckpt', '_backbone.pth')
        torch.save(model.backbone.state_dict(), backbone_path)
        print(f"✅ Backbone saved to {backbone_path}")

        return model, trainer
