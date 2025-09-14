#!/usr/bin/env python3
"""
SSL Training Script for TinyViT with 15-channel input
Usage: python train_ssl.py --data_path /path/to/data --batch_size 32 --epochs 100
"""

import argparse
import os
import torch
from ssl_training import SSLTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train TinyViT with SSL on 15-channel data')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data directory (for image files) or .h5 file')
    parser.add_argument('--use_h5', action='store_true',
                        help='Use HDF5 dataset instead of directory of files')
    parser.add_argument('--h5_dataset_key', type=str, default='images',
                        help='Key for dataset in HDF5 file')
    parser.add_argument('--file_extension', type=str, default='npy',
                        choices=['npy', 'tiff', 'tif'],
                        help='File extension for image files (npy, tiff, tif)')

    # Model arguments
    parser.add_argument('--num_channels', type=int, default=15,
                        help='Number of input channels')
    parser.add_argument('--model_name', type=str, default='tiny_vit_21m_512.dist_in22k_ft_in1k',
                        choices=[
                            'tiny_vit_5m_224.dist_in22k_ft_in1k',
                            'tiny_vit_11m_224.dist_in22k_ft_in1k',
                            'tiny_vit_21m_224.dist_in22k_ft_in1k',
                            'tiny_vit_21m_512.dist_in22k_ft_in1k'
                        ],
                        help='TinyViT model variant')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loader workers')

    # Hardware arguments
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu'],
                        help='Accelerator type')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices to use')

    # Output arguments
    parser.add_argument('--save_path', type=str, default='tinyvit_ssl_model.ckpt',
                        help='Path to save the trained model')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for logs and models')

    # Pretrained weights (local or disable timm downloads)
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to local pretrained TinyViT weights (state_dict).')
    parser.add_argument('--no_timm_pretrained', action='store_true',
                        help='Disable timm pretrained weights to avoid network usage.')

    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    # Check data path exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path does not exist: {args.data_path}")

    # Check if using H5 file
    if args.use_h5 and not args.data_path.endswith(('.h5', '.hdf5')):
        raise ValueError("When using --use_h5, data_path must be an .h5 or .hdf5 file")

    # Check if using directory
    if not args.use_h5 and not os.path.isdir(args.data_path):
        raise ValueError("When not using --use_h5, data_path must be a directory")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Update save path to be in output directory
    if not os.path.dirname(args.save_path):
        args.save_path = os.path.join(args.output_dir, args.save_path)

    return args


def main():
    # Parse arguments
    args = parse_args()
    args = validate_args(args)

    print("🚀 Starting SSL Training with TinyViT")
    print(f"Data path: {args.data_path}")
    print(f"Model: {args.model_name}")
    print(f"Channels: {args.num_channels}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Accelerator: {args.accelerator}")
    print(f"Save path: {args.save_path}")
    if args.pretrained_path:
        print(f"Using local pretrained weights: {args.pretrained_path}")
    print("-" * 50)

    # Create trainer
    trainer = SSLTrainer(
        data_path=args.data_path,
        num_channels=args.num_channels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        model_name=args.model_name,
        use_h5=args.use_h5,
        h5_dataset_key=args.h5_dataset_key,
        file_extension=args.file_extension,
        pretrained_path=args.pretrained_path,
        timm_pretrained=not args.no_timm_pretrained
    )

    try:
        # Run training
        model, lightning_trainer = trainer.train(
            accelerator=args.accelerator,
            devices=args.devices,
            save_path=args.save_path
        )

        print("🎉 Training completed successfully!")
        print(f"Model saved to: {args.save_path}")
        print(f"Backbone saved to: {args.save_path.replace('.ckpt', '_backbone.pth')}")

        # Print final training stats
        if hasattr(model, 'train_losses') and model.train_losses:
            final_loss = model.train_losses[-1]
            print(f"Final training loss: {final_loss:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
