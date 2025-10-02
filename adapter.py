import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from typing import Optional, Dict, Tuple
import copy


# ============================================================================
# ADAPTER MODULE
# ============================================================================

class Adapter(nn.Module):
    """Adapter module to be inserted in ViT blocks"""
    def __init__(self, dim: int, reduction_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim // reduction_factor
        
        self.down_proj = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return x + residual


# ============================================================================
# MODIFY VIT BLOCK TO INCLUDE ADAPTERS
# ============================================================================

class ViTBlockWithAdapters(nn.Module):
    """Wrapper for ViT block with adapters"""
    def __init__(self, original_block, dim: int, reduction_factor: int = 4):
        super().__init__()
        self.original_block = original_block
        self.adapter_attn = Adapter(dim, reduction_factor)
        self.adapter_mlp = Adapter(dim, reduction_factor)
        
    def forward(self, x):
        # Attention + Adapter
        x = x + self.original_block.attn(self.original_block.norm1(x))
        x = self.adapter_attn(x)
        
        # MLP + Adapter
        x = x + self.original_block.mlp(self.original_block.norm2(x))
        x = self.adapter_mlp(x)
        
        return x


def add_adapters_to_vit(model, reduction_factor: int = 4):
    """Add adapters to all ViT blocks"""
    # Get the embedding dimension
    if hasattr(model, 'embed_dim'):
        dim = model.embed_dim
    elif hasattr(model, 'blocks') and len(model.blocks) > 0:
        # Try to infer from first block
        first_block = model.blocks[0]
        if hasattr(first_block, 'attn') and hasattr(first_block.attn, 'qkv'):
            dim = first_block.attn.qkv.in_features
        else:
            raise ValueError("Cannot infer embedding dimension")
    else:
        raise ValueError("Cannot find blocks in model")
    
    print(f"Adding adapters with dim={dim}, reduction_factor={reduction_factor}")
    
    # Replace blocks with adapter-enabled blocks
    new_blocks = nn.ModuleList([
        ViTBlockWithAdapters(block, dim, reduction_factor)
        for block in model.blocks
    ])
    model.blocks = new_blocks
    
    return model


# ============================================================================
# MODEL WITH CLASSIFICATION HEAD
# ============================================================================

class DINOv3WithAdapters(nn.Module):
    """DINOv3 model with adapters and classification head"""
    def __init__(self, backbone, num_classes: int, embed_dim: int = 768):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        
        # If output is a dict (some DINOv2 versions), extract the cls token
        if isinstance(features, dict):
            features = features['x_norm_clstoken']
        
        # Apply classification head
        return self.head(features)


# ============================================================================
# ADAPTER WEIGHT MANAGEMENT
# ============================================================================

class AdapterManager:
    """Utility class for saving and loading adapter weights"""
    
    @staticmethod
    def save(model, path: str, metadata: Optional[Dict] = None):
        """Save only adapter weights"""
        adapter_params = {}
        
        for name, param in model.named_parameters():
            if 'adapter' in name or ('head' in name and param.requires_grad):
                adapter_params[name] = param.cpu().clone()
        
        checkpoint = {
            'adapters': adapter_params,
            'num_params': sum(p.numel() for p in adapter_params.values()),
            'metadata': metadata or {}
        }
        
        torch.save(checkpoint, path)
        print(f"✓ Saved {len(adapter_params)} adapter modules ({checkpoint['num_params']:,} params)")
        print(f"✓ File size: {os.path.getsize(path) / (1024**2):.2f} MB")
        return checkpoint['num_params']
    
    @staticmethod
    def load(model, path: str, device: str = 'cpu', strict: bool = False):
        """Load adapter weights"""
        checkpoint = torch.load(path, map_location=device)
        
        if 'adapters' in checkpoint:
            adapter_params = checkpoint['adapters']
            metadata = checkpoint.get('metadata', {})
            if metadata:
                print(f"Loading adapters with metadata: {metadata}")
        else:
            adapter_params = checkpoint
        
        # Load into model
        current_state = model.state_dict()
        loaded_count = 0
        
        for name, param in adapter_params.items():
            if name in current_state:
                current_state[name] = param.to(device)
                loaded_count += 1
            elif not strict:
                print(f"Warning: {name} not found in model")
        
        model.load_state_dict(current_state, strict=False)
        print(f"✓ Loaded {loaded_count} adapter parameters")
        
        return model
    
    @staticmethod
    def freeze_base_model(model):
        """Freeze all parameters except adapters and head"""
        # First freeze everything
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze adapters
        for name, param in model.named_parameters():
            if 'adapter' in name or 'head' in name:
                param.requires_grad = True
        
        # Count trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    @staticmethod
    def get_param_groups(model, lr: float = 1e-3, weight_decay: float = 0.01):
        """Get parameter groups for optimizer"""
        adapter_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'head' in name:
                head_params.append(param)
            elif 'adapter' in name:
                adapter_params.append(param)
        
        return [
            {'params': adapter_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr * 10, 'weight_decay': weight_decay}
        ]


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc.item(), images.size(0))
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc.item(), images.size(0))
    
    return loss_meter.avg, acc_meter.avg


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def create_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """Create train and validation dataloaders"""
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, len(train_dataset.classes)


def train_dinov3_with_adapters(
    data_dir: str,
    output_dir: str = './checkpoints',
    model_name: str = 'dinov2_vitb14',
    reduction_factor: int = 4,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Complete training pipeline"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, num_classes = create_dataloaders(
        data_dir, batch_size
    )
    print(f"Number of classes: {num_classes}")
    
    # Load pre-trained DINOv2/v3 model
    print(f"Loading {model_name}...")
    try:
        backbone = torch.hub.load('facebookresearch/dinov2', model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have internet connection for first-time download")
        return
    
    # Get embedding dimension
    embed_dim = backbone.embed_dim if hasattr(backbone, 'embed_dim') else 768
    
    # Add adapters
    print("Adding adapters...")
    backbone = add_adapters_to_vit(backbone, reduction_factor)
    
    # Create model with classification head
    model = DINOv3WithAdapters(backbone, num_classes, embed_dim)
    model = model.to(device)
    
    # Freeze base model, only train adapters and head
    AdapterManager.freeze_base_model(model)
    
    # Setup optimizer and scheduler
    param_groups = AdapterManager.get_param_groups(model, lr, weight_decay)
    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)
        
        if is_best:
            print(f"New best model! Val Acc: {val_acc:.4f}")
            AdapterManager.save(
                model,
                os.path.join(output_dir, 'best_adapters.pt'),
                metadata={
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'num_classes': num_classes,
                    'model_name': model_name,
                    'reduction_factor': reduction_factor
                }
            )
        
        # Save last checkpoint
        if epoch % 5 == 0 or epoch == num_epochs:
            AdapterManager.save(
                model,
                os.path.join(output_dir, f'adapters_epoch_{epoch}.pt'),
                metadata={'epoch': epoch, 'val_acc': val_acc}
            )
    
    print(f"\nTraining completed! Best Val Acc: {best_val_acc:.4f}")
    return model


# ============================================================================
# INFERENCE
# ============================================================================

def load_model_for_inference(
    adapter_path: str,
    num_classes: int,
    model_name: str = 'dinov2_vitb14',
    reduction_factor: int = 4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Load model with trained adapters for inference"""
    
    # Load base model
    print(f"Loading {model_name}...")
    backbone = torch.hub.load('facebookresearch/dinov2', model_name)
    embed_dim = backbone.embed_dim if hasattr(backbone, 'embed_dim') else 768
    
    # Add adapters
    backbone = add_adapters_to_vit(backbone, reduction_factor)
    
    # Create model with head
    model = DINOv3WithAdapters(backbone, num_classes, embed_dim)
    
    # Load adapter weights
    model = AdapterManager.load(model, adapter_path, device)
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Train from scratch
    train_dinov3_with_adapters(
        data_dir='./data',  # Should have 'train' and 'val' subdirectories
        output_dir='./checkpoints',
        model_name='dinov2_vitb14',  # or 'dinov2_vits14', 'dinov2_vitl14', 'dinov2_vitg14'
        reduction_factor=4,
        batch_size=32,
        num_epochs=10,
        lr=1e-3,
        weight_decay=0.01
    )
    
    # Example 2: Load and use trained model
    """
    model = load_model_for_inference(
        adapter_path='./checkpoints/best_adapters.pt',
        num_classes=10,  # Your number of classes
        model_name='dinov2_vitb14',
        reduction_factor=4
    )
    
    # Inference
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    from PIL import Image
    image = Image.open('test_image.jpg')
    image_tensor = transform(image).unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1)
        print(f"Predicted class: {pred.item()}")
    """