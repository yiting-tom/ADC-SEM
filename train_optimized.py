#!/usr/bin/env python3
"""
Train SSL model with optimized hyperparameters from Optuna.
Usage: python train_optimized.py --config_path optimization_results/best_config.json
"""

import argparse
import json
import os
from ssl_training import SSLTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train SSL model with optimized hyperparameters')

    # Required arguments
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to best_config.json from optimization')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data')

    # Optional overrides
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs (default from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (default from config)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Override save path (default: optimized_model.ckpt)')

    # Training arguments
    parser.add_argument('--use_h5', action='store_true',
                        help='Use HDF5 dataset format')
    parser.add_argument('--h5_dataset_key', type=str, default='images',
                        help='Key for dataset in HDF5 file')
    parser.add_argument('--file_extension', type=str, default='npy',
                        choices=['npy', 'tiff', 'tif'],
                        help='File extension for image files (npy, tiff, tif)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loader workers')

    # Hardware arguments
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator type')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices to use')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load the best configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def main():
    # Parse arguments
    args = parse_args()

    # Load optimized configuration
    config = load_config(args.config_path)
    best_params = config['best_params']

    print("🚀 Training with Optimized Hyperparameters")
    print(f"Configuration: {args.config_path}")
    print(f"Best trial: {config['best_trial']}")
    print(f"Best objective value: {config['best_value']:.4f}")
    print("-" * 60)

    print("📊 Optimized hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("-" * 60)

    # Apply overrides
    final_params = best_params.copy()
    if args.epochs is not None:
        final_params['epochs'] = args.epochs
        print(f"🔧 Override epochs: {args.epochs}")
    else:
        final_params['epochs'] = 200  # Default for final training

    if args.batch_size is not None:
        final_params['batch_size'] = args.batch_size
        print(f"🔧 Override batch size: {args.batch_size}")

    save_path = args.save_path or 'optimized_ssl_model.ckpt'
    print(f"💾 Save path: {save_path}")
    print("-" * 60)

    # Create trainer with optimized parameters
    trainer = SSLTrainer(
        data_path=args.data_path,
        num_channels=15,  # Fixed for this setup
        batch_size=final_params['batch_size'],
        num_workers=args.num_workers,
        max_epochs=final_params['epochs'],
        learning_rate=final_params['learning_rate'],
        model_name=final_params['model_name'],
        use_h5=args.use_h5,
        h5_dataset_key=args.h5_dataset_key,
        file_extension=args.file_extension
    )

    # Override trainer's create_dataloader to use optimized augmentation parameters
    def create_optimized_dataloader():
        from ssl_training import MultiChannelCollateFunction
        from lightly.data import LightlyDataset
        from torch.utils.data import DataLoader
        from dataset import MultiChannelDataset, MultiChannelH5Dataset

        # Create dataset
        if args.use_h5:
            dataset = MultiChannelH5Dataset(
                h5_file_path=args.data_path,
                dataset_key=args.h5_dataset_key,
                channels=15
            )
        else:
            dataset = MultiChannelDataset(
                root_dir=args.data_path,
                channels=15,
                image_size=(460, 460),
                file_extension=args.file_extension
            )

        # Wrap with LightlyDataset
        lightly_dataset = LightlyDataset.from_torch_dataset(dataset)

        # Create optimized collate function
        collate_fn = MultiChannelCollateFunction(
            input_size=460,
            min_scale=final_params.get('min_scale', 0.08),
            hf_prob=final_params.get('horizontal_flip_prob', 0.5),
            gaussian_blur=final_params.get('gaussian_blur_prob', 0.5)
        )

        # Create dataloader
        dataloader = DataLoader(
            lightly_dataset,
            batch_size=final_params['batch_size'],
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            persistent_workers=True if args.num_workers > 0 else False
        )

        return dataloader

    # Replace the trainer's method
    trainer.create_dataloader = create_optimized_dataloader

    try:
        # Run training with optimized parameters
        print("🏋️ Starting optimized training...")
        model, lightning_trainer = trainer.train(
            accelerator=args.accelerator,
            devices=args.devices,
            save_path=save_path
        )

        print("🎉 Optimized training completed successfully!")
        print(f"Model saved to: {save_path}")
        print(f"Backbone saved to: {save_path.replace('.ckpt', '_backbone.pth')}")

        # Print final training stats
        if hasattr(model, 'train_losses') and model.train_losses:
            final_loss = model.train_losses[-1]
            print(f"Final training loss: {final_loss:.4f}")

        # Save final results
        results = {
            'config_used': config,
            'final_loss': final_loss if hasattr(model, 'train_losses') and model.train_losses else None,
            'model_path': save_path,
            'backbone_path': save_path.replace('.ckpt', '_backbone.pth')
        }

        results_path = save_path.replace('.ckpt', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"📊 Training results saved to: {results_path}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()