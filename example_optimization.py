#!/usr/bin/env python3
"""
Example script demonstrating Optuna hyperparameter optimization workflow.
This creates synthetic data and runs a quick optimization for demonstration.
"""

import os
import numpy as np
from optuna_optimization import HyperparameterOptimizer


def create_example_data(num_samples=50, save_dir="example_data"):
    """Create synthetic 15-channel data for optimization demo."""
    print(f"📊 Creating {num_samples} synthetic 15-channel images...")

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_samples):
        # Create realistic-looking 15-channel data
        image = np.zeros((460, 460, 15), dtype=np.float32)

        for c in range(15):
            # Add different patterns for different channels
            if c < 5:  # Low-frequency channels
                freq = np.random.uniform(0.01, 0.05)
                phase = np.random.uniform(0, 2*np.pi)
                y, x = np.ogrid[:460, :460]
                pattern = np.sin(freq * x + phase) * np.cos(freq * y + phase)
            elif c < 10:  # Mid-frequency channels
                freq = np.random.uniform(0.05, 0.15)
                phase = np.random.uniform(0, 2*np.pi)
                y, x = np.ogrid[:460, :460]
                pattern = np.sin(freq * x + phase) * np.sin(freq * y + phase)
            else:  # High-frequency channels (noise-like)
                pattern = np.random.randn(460, 460) * 0.1

            # Add some spatial structure
            pattern += np.random.randn(460, 460) * 0.05

            # Normalize to [0, 1]
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
            image[:, :, c] = pattern

        # Save as .npy file
        np.save(os.path.join(save_dir, f"sample_{i:03d}.npy"), image)

    print(f"✅ Created {num_samples} samples in {save_dir}/")
    return save_dir


def run_quick_optimization():
    """Run a quick optimization demo with synthetic data."""
    print("🚀 Optuna Hyperparameter Optimization Demo")
    print("=" * 60)

    # Create synthetic data
    data_dir = create_example_data(num_samples=30)  # Small dataset for demo

    try:
        # Create optimizer with reduced parameters for quick demo
        optimizer = HyperparameterOptimizer(
            data_path=data_dir,
            study_name="demo_ssl_optimization",
            num_channels=15,
            n_trials=10,  # Fewer trials for demo
            max_epochs=5,  # Fewer epochs for demo
            early_stopping_patience=3,
            num_workers=0,  # Avoid multiprocessing issues
            accelerator='cpu'  # Use CPU for demo
        )

        # Run optimization
        study = optimizer.optimize(direction='minimize')

        # Print results
        print("\n🎉 Demo Optimization Results:")
        print(f"Best loss: {study.best_value:.4f}")
        print(f"Best trial: {study.best_trial.number}")
        print("\n📊 Best hyperparameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")

        # Create visualization
        output_dir = "demo_optimization_results"
        os.makedirs(output_dir, exist_ok=True)

        # Save best config
        best_config = optimizer.get_best_config(study)
        import json
        with open(os.path.join(output_dir, 'best_config.json'), 'w') as f:
            json.dump(best_config, f, indent=2)

        print(f"\n💾 Results saved to {output_dir}/")

        # Create plots if possible
        try:
            optimizer.create_visualization(study, os.path.join(output_dir, 'plots'))
        except Exception as e:
            print(f"⚠️ Could not create plots: {e}")

        # Print example training command
        print("\n🚀 To train with these optimized parameters, run:")
        print("-" * 40)
        print(f"python train_optimized.py \\")
        print(f"    --config_path {output_dir}/best_config.json \\")
        print(f"    --data_path {data_dir} \\")
        print(f"    --epochs 100")
        print("-" * 40)

        return study

    finally:
        # Clean up synthetic data
        import shutil
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"\n🧹 Cleaned up synthetic data: {data_dir}")


if __name__ == '__main__':
    try:
        study = run_quick_optimization()
        print("\n✅ Demo completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        # Don't raise - this is just a demo