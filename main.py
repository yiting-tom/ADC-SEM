#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Channel TIFF Image Classification with PyTorch Lightning
Supports both binary and multiclass classification using AlexNet and EfficientNet-B2
"""

# %%
# Import required libraries
import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import tifffile
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import alexnet
import timm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set random seeds for reproducibility
pl.seed_everything(42)

# %%
# Configuration
CONFIG = {
    'data_folder': 'path/to/your/tiff/images',  # Update this path
    'batch_size': 16,
    'num_workers': 4,
    'max_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'model_type': 'efficientnet',  # 'alexnet' or 'efficientnet'
    'classification_type': 'multiclass',  # 'binary' or 'multiclass'
    'target_channels': 15,  # Number of channels to use/pad to
    'image_size': 224,  # Input image size
    'test_split': 0.2,
    'val_split': 0.1,
}

# %%
# Custom Dataset Class
class TIFFDataset(Dataset):
    def __init__(self, file_paths: List[str], labels: List[int], 
                 target_channels: int = 15, image_size: int = 224, 
                 transform=None, is_training: bool = True):
        """
        Dataset for multi-channel TIFF images
        
        Args:
            file_paths: List of TIFF file paths
            labels: List of corresponding labels
            target_channels: Target number of channels (pad/truncate to this)
            image_size: Target image size
            transform: Image transformations
            is_training: Whether this is training data (for augmentation)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.target_channels = target_channels
        self.image_size = image_size
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.file_paths)
    
    def load_tiff_stack(self, file_path: str) -> np.ndarray:
        """Load TIFF stack and return as numpy array"""
        try:
            # Method 1: Using tifffile (recommended)
            images = tifffile.imread(file_path)
            if images.ndim == 2:
                images = images[np.newaxis, ...]  # Add channel dimension
            elif images.ndim == 3 and images.shape[0] > images.shape[2]:
                # Assume first dimension is channels
                pass
            else:
                # Assume last dimension is channels, transpose
                images = images.transpose(2, 0, 1)
        except:
            # Method 2: Using PIL as fallback
            with Image.open(file_path) as img:
                images = []
                try:
                    for i in range(20):  # Max expected frames
                        img.seek(i)
                        images.append(np.array(img))
                except EOFError:
                    pass
                images = np.stack(images, axis=0)
        
        return images
    
    def preprocess_channels(self, images: np.ndarray) -> np.ndarray:
        """Preprocess channels to match target number"""
        current_channels = images.shape[0]
        
        if current_channels == self.target_channels:
            return images
        elif current_channels > self.target_channels:
            # Truncate to target channels
            return images[:self.target_channels]
        else:
            # Pad with zeros
            padding = self.target_channels - current_channels
            pad_images = np.zeros((padding, images.shape[1], images.shape[2]), dtype=images.dtype)
            return np.concatenate([images, pad_images], axis=0)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load TIFF stack
        images = self.load_tiff_stack(file_path)
        
        # Preprocess channels
        images = self.preprocess_channels(images)
        
        # Resize images if needed
        if images.shape[1] != self.image_size or images.shape[2] != self.image_size:
            resized_images = []
            for i in range(images.shape[0]):
                img = Image.fromarray(images[i].astype(np.uint8))
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                resized_images.append(np.array(img))
            images = np.stack(resized_images, axis=0)
        
        # Convert to float and normalize
        images = images.astype(np.float32) / 255.0
        
        # Convert to tensor
        images = torch.from_numpy(images)
        
        # Apply transforms if provided
        if self.transform:
            images = self.transform(images)
        
        return images, torch.tensor(label, dtype=torch.long)

# %%
# Data preprocessing functions
def parse_filename(filename: str) -> Tuple[int, str]:
    """Parse filename to extract label_id and image_set_id"""
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    parts = name_without_ext.split('_')
    
    if len(parts) >= 5:
        label_id = int(parts[0])
        image_set_id = '_'.join(parts[1:5])
        return label_id, image_set_id
    else:
        raise ValueError(f"Invalid filename format: {filename}")

def prepare_data(data_folder: str, classification_type: str = 'multiclass'):
    """Prepare data from folder of TIFF images"""
    # Get all TIFF files
    tiff_files = glob.glob(os.path.join(data_folder, '*.tiff')) + \
                 glob.glob(os.path.join(data_folder, '*.tif'))
    
    if len(tiff_files) == 0:
        raise ValueError(f"No TIFF files found in {data_folder}")
    
    # Parse filenames
    file_info = []
    for file_path in tiff_files:
        try:
            label_id, image_set_id = parse_filename(file_path)
            file_info.append({
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'label_id': label_id,
                'image_set_id': image_set_id
            })
        except ValueError as e:
            print(f"Skipping file due to error: {e}")
    
    df = pd.DataFrame(file_info)
    
    # Prepare labels based on classification type
    if classification_type == 'binary':
        # Binary: Normal (label_id = 1) vs Abnormal (label_id != 1)
        df['label'] = (df['label_id'] == 1).astype(int)
        num_classes = 2
        print(f"Binary classification - Normal: {sum(df['label'] == 1)}, Abnormal: {sum(df['label'] == 0)}")
    else:
        # Multiclass: use label_id directly
        unique_labels = sorted(df['label_id'].unique())
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        df['label'] = df['label_id'].map(label_to_idx)
        num_classes = len(unique_labels)
        print(f"Multiclass classification - {num_classes} classes: {unique_labels}")
        print(f"Label distribution:\n{df['label_id'].value_counts().sort_index()}")
    
    return df, num_classes

# %%
# Model definitions
class MultiChannelAlexNet(nn.Module):
    def __init__(self, num_channels: int = 15, num_classes: int = 2):
        super(MultiChannelAlexNet, self).__init__()
        
        # Load pretrained AlexNet and modify first layer
        self.alexnet = alexnet(pretrained=True)
        
        # Replace first conv layer to accept multi-channel input
        self.alexnet.features[0] = nn.Conv2d(
            num_channels, 64, kernel_size=11, stride=4, padding=2
        )
        
        # Replace classifier
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)
        
    def forward(self, x):
        return self.alexnet(x)

class MultiChannelEfficientNet(nn.Module):
    def __init__(self, num_channels: int = 15, num_classes: int = 2):
        super(MultiChannelEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B2
        self.efficientnet = timm.create_model('efficientnet_b2', pretrained=True)
        
        # Replace first conv layer
        self.efficientnet.conv_stem = nn.Conv2d(
            num_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Replace classifier
        self.efficientnet.classifier = nn.Linear(
            self.efficientnet.classifier.in_features, num_classes
        )
        
    def forward(self, x):
        return self.efficientnet(x)

# %%
# Lightning Module
class ImageClassifier(pl.LightningModule):
    def __init__(self, model_type: str = 'efficientnet', num_channels: int = 15, 
                 num_classes: int = 2, learning_rate: float = 1e-4, 
                 weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_type = model_type
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize model
        if model_type == 'alexnet':
            self.model = MultiChannelAlexNet(num_channels, num_classes)
        elif model_type == 'efficientnet':
            self.model = MultiChannelEfficientNet(num_channels, num_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.train_acc = []
        self.val_acc = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': accuracy}
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)
        
        return {'test_loss': loss, 'test_acc': accuracy, 'predictions': predicted, 'labels': labels}
    
    def test_epoch_end(self, outputs):
        # Collect all predictions and labels
        all_predictions = torch.cat([x['predictions'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])
        
        # Store for later use in main function
        self.test_predictions = all_predictions.cpu().numpy()
        self.test_labels = all_labels.cpu().numpy()
    
    def predict_step(self, batch, batch_idx):
        images, _ = batch
        outputs = self(images)
        predictions = torch.argmax(outputs, dim=1)
        return predictions
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
            },
        }

# %%
# Data Module
class TIFFDataModule(pl.LightningDataModule):
    def __init__(self, data_df: pd.DataFrame, config: dict):
        super().__init__()
        self.data_df = data_df
        self.config = config
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            trnasforms.
        ])
        
        self.val_transform = None  # No augmentation for validation/test
        
    def setup(self, stage: Optional[str] = None):
        # Split data
        train_val_df, test_df = train_test_split(
            self.data_df, 
            test_size=self.config['test_split'], 
            random_state=42, 
            stratify=self.data_df['label']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=self.config['val_split']/(1-self.config['test_split']), 
            random_state=42, 
            stratify=train_val_df['label']
        )
        
        # Create datasets
        self.train_dataset = TIFFDataset(
            train_df['file_path'].tolist(),
            train_df['label'].tolist(),
            target_channels=self.config['target_channels'],
            image_size=self.config['image_size'],
            transform=self.train_transform,
            is_training=True
        )
        
        self.val_dataset = TIFFDataset(
            val_df['file_path'].tolist(),
            val_df['label'].tolist(),
            target_channels=self.config['target_channels'],
            image_size=self.config['image_size'],
            transform=self.val_transform,
            is_training=False
        )
        
        self.test_dataset = TIFFDataset(
            test_df['file_path'].tolist(),
            test_df['label'].tolist(),
            target_channels=self.config['target_channels'],
            image_size=self.config['image_size'],
            transform=self.val_transform,
            is_training=False
        )
        
        # Store test filenames for prediction output
        self.test_filenames = test_df['filename'].tolist()
        
        print(f"Dataset splits - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            persistent_workers=True if self.config['num_workers'] > 0 else False
        )

# %%
# Main execution
def main():
    print("Starting TIFF Image Classification Training...")
    print(f"Configuration: {CONFIG}")
    
    # Prepare data
    print("\n1. Preparing data...")
    data_df, num_classes = prepare_data(CONFIG['data_folder'], CONFIG['classification_type'])
    
    # Create data module
    print("\n2. Creating data module...")
    data_module = TIFFDataModule(data_df, CONFIG)
    data_module.setup()
    
    # Create model
    print(f"\n3. Creating {CONFIG['model_type']} model...")
    model = ImageClassifier(
        model_type=CONFIG['model_type'],
        num_channels=CONFIG['target_channels'],
        num_classes=num_classes,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints',
        filename=f"{CONFIG['model_type']}-{CONFIG['classification_type']}-{{epoch:02d}}-{{val_acc:.3f}}",
        save_top_k=1,
        mode='max'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )
    
    # Create trainer
    trainer = Trainer(
        max_epochs=CONFIG['max_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
    )
    
    # Train model
    print("\n4. Training model...")
    trainer.fit(model, data_module)
    
    # Test model
    print("\n5. Testing model...")
    test_results = trainer.test(model, data_module)
    
    # Load best model for evaluation
    best_model = ImageClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model_type=CONFIG['model_type'],
        num_channels=CONFIG['target_channels'],
        num_classes=num_classes,
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Get test predictions and labels for detailed evaluation
    print("\n6. Generating detailed evaluation metrics...")
    best_model.eval()
    test_predictions = []
    test_labels_list = []
    test_filenames_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_module.test_dataloader()):
            images, labels = batch
            outputs = best_model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            test_predictions.extend(predictions.cpu().numpy())
            test_labels_list.extend(labels.cpu().numpy())
            
            # Get corresponding filenames for this batch
            batch_size = len(labels)
            start_idx = batch_idx * CONFIG['batch_size']
            end_idx = start_idx + batch_size
            test_filenames_list.extend(data_module.test_filenames[start_idx:end_idx])
    
    test_predictions = np.array(test_predictions)
    test_labels_array = np.array(test_labels_list)
    
    # Print detailed metrics
    print("\n" + "="*80)
    print("DETAILED EVALUATION METRICS")
    print("="*80)
    
    # Test accuracy
    test_accuracy = accuracy_score(test_labels_array, test_predictions)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print("-" * 50)
    
    # Prepare class names for report
    if CONFIG['classification_type'] == 'binary':
        target_names = ['Abnormal', 'Normal']
    else:
        # Get original label mapping from data
        unique_labels = sorted(data_df['label_id'].unique())
        target_names = [f'Class_{label}' for label in unique_labels]
    
    print(classification_report(
        test_labels_array, 
        test_predictions, 
        target_names=target_names,
        digits=4
    ))
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print("-" * 30)
    cm = confusion_matrix(test_labels_array, test_predictions)
    print(cm)
    
    # Pretty print confusion matrix with labels
    print(f"\nConfusion Matrix (with labels):")
    print("-" * 50)
    cm_df = pd.DataFrame(cm, 
                        index=[f'True_{name}' for name in target_names],
                        columns=[f'Pred_{name}' for name in target_names])
    print(cm_df)
    
    # Per-class accuracy
    print(f"\nPer-class Accuracy:")
    print("-" * 30)
    for i, class_name in enumerate(target_names):
        class_mask = test_labels_array == i
        if np.sum(class_mask) > 0:
            class_acc = accuracy_score(
                test_labels_array[class_mask], 
                test_predictions[class_mask]
            )
            print(f"{class_name}: {class_acc:.4f} ({np.sum(class_mask)} samples)")
    
    # Generate predictions for CSV output
    print("\n7. Generating prediction file...")
    
    # For CSV output, convert back to original label space if needed
    if CONFIG['classification_type'] == 'binary':
        # Convert back: 0 -> not 1 (use most common non-1 label), 1 -> 1
        csv_predictions = []
        non_normal_label = data_df[data_df['label_id'] != 1]['label_id'].mode().iloc[0] if len(data_df[data_df['label_id'] != 1]) > 0 else 0
        for pred in test_predictions:
            if pred == 1:  # Normal
                csv_predictions.append(1)
            else:  # Abnormal
                csv_predictions.append(non_normal_label)
    else:
        # Convert back to original label_ids
        unique_labels = sorted(data_df['label_id'].unique())
        idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
        csv_predictions = [idx_to_label[pred] for pred in test_predictions]
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'filename': test_filenames_list,
        'prediction': csv_predictions
    })
    
    # Save predictions
    predictions_df.to_csv('predictions.csv', index=False)
    print(f"Predictions saved to 'predictions.csv'")
    
    # Save detailed results
    results_summary = {
        'model_type': CONFIG['model_type'],
        'classification_type': CONFIG['classification_type'],
        'test_accuracy': test_accuracy,
        'num_classes': num_classes,
        'total_test_samples': len(test_predictions),
        'confusion_matrix': cm.tolist(),
        'class_names': target_names
    }
    
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Detailed evaluation results saved to 'evaluation_results.json'")
    
    print("\n" + "="*80)
    
    return best_model, trainer, predictions_df, test_predictions, test_labels_array

# %%
# Run the training
if __name__ == "__main__":
    # Update the data folder path before running
    if not os.path.exists(CONFIG['data_folder']):
        print(f"Please update CONFIG['data_folder'] to point to your TIFF images directory")
        print(f"Current path: {CONFIG['data_folder']}")
    else:
        model, trainer, predictions_df = main()
        
        # Display some predictions
        print("\nSample predictions:")
        print(predictions_df.head(10))