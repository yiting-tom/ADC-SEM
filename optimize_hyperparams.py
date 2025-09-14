#!/usr/bin/env python3
"""
CLI script for hyperparameter optimization using Optuna.
Usage: python optimize_hyperparams.py --data_path /path/to/data --n_trials 100
"""

import argparse
import os
import json
from optuna_optimization import HyperparameterOptimizer


def parse_args():
    parser = argparse.ArgumentParser(description='Optimize SSL hyperparameters with Optuna')

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

    # Optimization arguments
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum epochs per trial')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--study_name', type=str, default='ssl_optimization',
                        help='Name of the optimization study')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g., sqlite:///optuna.db)')

    # Hardware arguments
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu'],
                        help='Accelerator type')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./optimization_results',
                        help='Output directory for results')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save optimization plots')

    # Advanced optimization settings
    parser.add_argument('--sampler', type=str, default='tpe',
                        choices=['tpe', 'random', 'cmaes'],
                        help='Optuna sampler type')
    parser.add_argument('--pruner', type=str, default='median',
                        choices=['median', 'successive_halving', 'hyperband', 'none'],
                        help='Optuna pruner type')

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

    return args


def create_sampler(sampler_type: str):
    """Create Optuna sampler based on type."""
    import optuna

    if sampler_type == 'tpe':
        return optuna.samplers.TPESampler(seed=42)
    elif sampler_type == 'random':
        return optuna.samplers.RandomSampler(seed=42)
    elif sampler_type == 'cmaes':
        return optuna.samplers.CmaEsSampler(seed=42)
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def create_pruner(pruner_type: str):
    """Create Optuna pruner based on type."""
    import optuna

    if pruner_type == 'median':
        return optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
    elif pruner_type == 'successive_halving':
        return optuna.pruners.SuccessiveHalvingPruner()
    elif pruner_type == 'hyperband':
        return optuna.pruners.HyperbandPruner()
    elif pruner_type == 'none':
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner type: {pruner_type}")


def save_best_config(study, output_path: str):
    """Save the best configuration to JSON file."""
    best_config = {
        'best_value': study.best_value,
        'best_trial': study.best_trial.number,
        'best_params': study.best_params,
        'study_name': study.study_name,
        'n_trials': len(study.trials)
    }

    with open(output_path, 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"💾 Best configuration saved to {output_path}")


def print_training_command(best_params: dict, data_path: str):
    """Print the command to run training with best parameters."""
    cmd = f"python train_ssl.py \\\n"
    cmd += f"    --data_path {data_path} \\\n"
    cmd += f"    --model_name {best_params['model_name']} \\\n"
    cmd += f"    --batch_size {best_params['batch_size']} \\\n"
    cmd += f"    --lr {best_params['learning_rate']:.6f} \\\n"
    cmd += f"    --epochs 200 \\\n"
    cmd += f"    --num_workers 8"

    print("\n🚀 To train with optimal hyperparameters, run:")
    print("-" * 60)
    print(cmd)
    print("-" * 60)


def main():
    # Parse and validate arguments
    args = parse_args()
    args = validate_args(args)

    print("🔍 Hyperparameter Optimization with Optuna")
    print(f"Data path: {args.data_path}")
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Max epochs per trial: {args.max_epochs}")
    print(f"Sampler: {args.sampler}")
    print(f"Pruner: {args.pruner}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Create optimizer
    optimizer = HyperparameterOptimizer(
        data_path=args.data_path,
        study_name=args.study_name,
        storage=args.storage,
        num_channels=args.num_channels,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        use_h5=args.use_h5,
        h5_dataset_key=args.h5_dataset_key,
        accelerator=args.accelerator,
        devices=args.devices,
        num_workers=args.num_workers,
        file_extension=args.file_extension
    )

    # Create sampler and pruner
    sampler = create_sampler(args.sampler)
    pruner = create_pruner(args.pruner)

    try:
        # Run optimization
        study = optimizer.optimize(
            direction='minimize',
            sampler=sampler,
            pruner=pruner
        )

        # Save results
        config_path = os.path.join(args.output_dir, 'best_config.json')
        save_best_config(study, config_path)

        # Create visualizations
        if args.save_plots:
            plot_dir = os.path.join(args.output_dir, 'plots')
            optimizer.create_visualization(study, plot_dir)

        # Print training command
        print_training_command(study.best_params, args.data_path)

        print(f"\n🎉 Optimization completed successfully!")
        print(f"Results saved to: {args.output_dir}")

    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"❌ Optimization failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()