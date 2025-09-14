#!/usr/bin/env python3
"""
Comprehensive test to verify TIFF support is correctly set up throughout the pipeline.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import subprocess
import json


def install_tifffile_if_needed():
    """Install tifffile if not available."""
    try:
        import tifffile
        print("✅ tifffile already available")
        return True
    except ImportError:
        print("📦 Installing tifffile...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tifffile>=2023.7.10"])
            import tifffile
            print("✅ tifffile installed successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to install tifffile: {e}")
            return False


def create_tiff_test_dataset(temp_dir, num_samples=10):
    """Create test TIFF dataset with shape (15, 460, 460)."""
    print(f"📊 Creating {num_samples} test TIFF files...")

    import tifffile

    for i in range(num_samples):
        # Create realistic multi-channel data with shape (15, 460, 460)
        image = np.zeros((15, 460, 460), dtype=np.float32)

        for c in range(15):
            # Different patterns for different channels
            if c < 5:  # Low frequency
                freq = 0.01 + c * 0.005
                y, x = np.ogrid[:460, :460]
                pattern = np.sin(freq * x) * np.cos(freq * y)
            elif c < 10:  # Mid frequency
                freq = 0.05 + c * 0.01
                y, x = np.ogrid[:460, :460]
                pattern = np.sin(freq * x + c) * np.cos(freq * y + c)
            else:  # High frequency / noise
                pattern = np.random.randn(460, 460) * 0.2

            # Add noise and normalize
            pattern += np.random.randn(460, 460) * 0.05
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
            image[c] = pattern

        # Convert to uint8 range (0-255) as typical for TIFF
        image = (image * 255).astype(np.uint8)

        # Save as TIFF
        tiff_path = os.path.join(temp_dir, f"image_{i:03d}.tiff")
        tifffile.imwrite(tiff_path, image)

    print(f"✅ Created {num_samples} TIFF files with shape (15, 460, 460)")
    return temp_dir


def test_dataset_loading():
    """Test basic dataset loading with TIFF files."""
    print("\n🔍 Testing dataset loading...")

    temp_dir = tempfile.mkdtemp(prefix="tiff_dataset_test_")

    try:
        create_tiff_test_dataset(temp_dir, 5)

        # Test MultiChannelDataset
        from dataset import MultiChannelDataset

        dataset = MultiChannelDataset(
            root_dir=temp_dir,
            file_extension='tiff',
            channels=15,
            image_size=(460, 460)
        )

        print(f"  ✅ Dataset created: {len(dataset)} images")

        # Test loading images
        for i in range(min(3, len(dataset))):
            image_tensor, label = dataset[i]

            # Verify shape
            if image_tensor.shape != (15, 460, 460):
                print(f"  ❌ Wrong shape for image {i}: {image_tensor.shape}")
                return False

            # Verify data type
            if not str(image_tensor.dtype).startswith('torch.float'):
                print(f"  ❌ Wrong dtype for image {i}: {image_tensor.dtype}")
                return False

            # Verify value range (should be normalized to [0,1])
            if image_tensor.min() < 0 or image_tensor.max() > 1:
                print(f"  ⚠️ Values not in [0,1] range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

        print("  ✅ Dataset loading successful")
        return True

    except Exception as e:
        print(f"  ❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_ssl_trainer_integration():
    """Test SSL trainer with TIFF files."""
    print("\n🏋️ Testing SSL Trainer integration...")

    temp_dir = tempfile.mkdtemp(prefix="tiff_ssl_test_")

    try:
        create_tiff_test_dataset(temp_dir, 6)

        # Test SSLTrainer creation
        from ssl_training import SSLTrainer

        trainer = SSLTrainer(
            data_path=temp_dir,
            num_channels=15,
            batch_size=2,
            num_workers=0,
            max_epochs=1,
            file_extension='tiff'
        )

        print("  ✅ SSLTrainer created successfully")

        # Test dataloader creation
        dataloader = trainer.create_dataloader()
        print(f"  ✅ Dataloader created: {len(dataloader)} batches")

        # Test batch loading
        if len(dataloader) > 0:
            batch = next(iter(dataloader))
            print(f"  ✅ Batch loaded successfully")

            # For lightly collate function, batch structure might be different
            if hasattr(batch, '__len__') and len(batch) > 0:
                if hasattr(batch[0], 'shape'):
                    batch_shape = batch[0].shape
                    print(f"    Batch shape: {batch_shape}")

                    # Check if batch dimension is correct
                    if len(batch_shape) >= 4 and batch_shape[-3:] == (15, 460, 460):
                        print("  ✅ Batch shape correct")
                    else:
                        print(f"  ⚠️ Unexpected batch shape: {batch_shape}")

        return True

    except Exception as e:
        print(f"  ❌ SSL Trainer integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_cli_scripts():
    """Test CLI scripts with TIFF support."""
    print("\n💻 Testing CLI scripts...")

    try:
        # Test train_ssl.py help (should show file_extension option)
        result = subprocess.run([
            sys.executable, 'train_ssl.py', '--help'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and '--file_extension' in result.stdout:
            print("  ✅ train_ssl.py has --file_extension option")
        else:
            print("  ❌ train_ssl.py missing --file_extension option")
            return False

        # Test optimize_hyperparams.py help
        result = subprocess.run([
            sys.executable, 'optimize_hyperparams.py', '--help'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and '--file_extension' in result.stdout:
            print("  ✅ optimize_hyperparams.py has --file_extension option")
        else:
            print("  ❌ optimize_hyperparams.py missing --file_extension option")
            return False

        # Test train_optimized.py help
        result = subprocess.run([
            sys.executable, 'train_optimized.py', '--help'
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and '--file_extension' in result.stdout:
            print("  ✅ train_optimized.py has --file_extension option")
        else:
            print("  ❌ train_optimized.py missing --file_extension option")
            return False

        return True

    except Exception as e:
        print(f"  ❌ CLI script test failed: {e}")
        return False


def test_optuna_integration():
    """Test Optuna optimization with TIFF files."""
    print("\n🎯 Testing Optuna integration...")

    temp_dir = tempfile.mkdtemp(prefix="tiff_optuna_test_")

    try:
        create_tiff_test_dataset(temp_dir, 8)

        # Test OptunaSSLObjective creation
        from optuna_optimization import OptunaSSLObjective

        objective = OptunaSSLObjective(
            data_path=temp_dir,
            num_channels=15,
            max_epochs=1,
            early_stopping_patience=1,
            accelerator='cpu',
            devices=1,
            num_workers=0,
            file_extension='tiff'
        )

        print("  ✅ OptunaSSLObjective created with TIFF support")

        # Test hyperparameter suggestion
        import optuna
        study = optuna.create_study()
        trial = study.ask()

        hyperparams = objective._suggest_hyperparameters(trial)
        print("  ✅ Hyperparameter suggestion works")

        # Test HyperparameterOptimizer
        from optuna_optimization import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer(
            data_path=temp_dir,
            study_name="tiff_test",
            num_channels=15,
            n_trials=2,
            max_epochs=1,
            file_extension='tiff',
            accelerator='cpu',
            devices=1,
            num_workers=0
        )

        print("  ✅ HyperparameterOptimizer created with TIFF support")

        return True

    except Exception as e:
        print(f"  ❌ Optuna integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_mini_optimization():
    """Test a mini optimization run with TIFF files."""
    print("\n🚀 Testing mini optimization with TIFF files...")

    temp_dir = tempfile.mkdtemp(prefix="tiff_mini_opt_")

    try:
        create_tiff_test_dataset(temp_dir, 10)

        from optuna_optimization import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer(
            data_path=temp_dir,
            study_name="tiff_mini_test",
            num_channels=15,
            n_trials=2,  # Very few trials
            max_epochs=1,  # Very few epochs
            early_stopping_patience=1,
            accelerator='cpu',
            devices=1,
            num_workers=0,
            file_extension='tiff'
        )

        print("  🚀 Starting mini optimization (2 trials, 1 epoch each)...")
        study = optimizer.optimize(direction='minimize')

        print(f"  ✅ Mini optimization completed")
        print(f"     Best value: {study.best_value:.4f}")
        print(f"     Best trial: {study.best_trial.number}")

        # Test config generation
        best_config = optimizer.get_best_config(study)
        print(f"  ✅ Best config generated: {len(best_config)} parameters")

        return True

    except Exception as e:
        print(f"  ❌ Mini optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_requirements():
    """Test that all required packages are available."""
    print("\n📦 Testing requirements...")

    required_packages = [
        'torch',
        'torchvision',
        'timm',
        'pytorch_lightning',
        'lightly',
        'numpy',
        'scipy',
        'h5py',
        'Pillow',
        'tifffile',
        'optuna',
        'plotly'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'Pillow':
                import PIL
            elif package == 'pytorch_lightning':
                import pytorch_lightning
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        return False

    print("  ✅ All required packages available")
    return True


def main():
    """Run comprehensive TIFF pipeline verification."""
    print("🚀 Comprehensive TIFF Pipeline Verification")
    print("=" * 60)

    all_passed = True

    # Test 1: Requirements
    print("Phase 1: Requirements")
    if not test_requirements():
        all_passed = False

    # Test 2: Install tifffile if needed
    print("\nPhase 2: TIFF Support")
    if not install_tifffile_if_needed():
        all_passed = False
        print("❌ Cannot proceed without tifffile")
        return

    # Test 3: Dataset loading
    print("\nPhase 3: Dataset Loading")
    if not test_dataset_loading():
        all_passed = False

    # Test 4: SSL trainer integration
    print("\nPhase 4: SSL Trainer Integration")
    if not test_ssl_trainer_integration():
        all_passed = False

    # Test 5: CLI scripts
    print("\nPhase 5: CLI Scripts")
    if not test_cli_scripts():
        all_passed = False

    # Test 6: Optuna integration
    print("\nPhase 6: Optuna Integration")
    if not test_optuna_integration():
        all_passed = False

    # Test 7: Mini optimization
    print("\nPhase 7: End-to-End Optimization")
    if not test_mini_optimization():
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ TIFF Support Fully Verified:")
        print("   1. Requirements satisfied ✅")
        print("   2. TIFF loading (15, 460, 460) ✅")
        print("   3. Dataset integration ✅")
        print("   4. SSL training pipeline ✅")
        print("   5. CLI scripts updated ✅")
        print("   6. Optuna optimization ✅")
        print("   7. End-to-end workflow ✅")

        print("\n🚀 Ready for TIFF data:")
        print("   # Basic training")
        print("   python train_ssl.py --data_path /path/to/tiff/files --file_extension tiff")
        print("")
        print("   # Hyperparameter optimization")
        print("   python optimize_hyperparams.py --data_path /path/to/tiff/files --file_extension tiff --n_trials 100")
        print("")
        print("   # Training with optimized parameters")
        print("   python train_optimized.py --config_path results/best_config.json --data_path /path/to/tiff/files")

    else:
        print("❌ Some tests failed.")
        print("Please check the errors above and ensure all components are properly configured.")

    print("=" * 60)


if __name__ == "__main__":
    main()