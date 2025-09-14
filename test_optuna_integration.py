#!/usr/bin/env python3
"""
Test script to verify Optuna integration works correctly.
"""

import sys
import traceback

def test_imports():
    """Test all Optuna-related imports."""
    print("🔍 Testing Optuna imports...")

    try:
        import optuna
        print(f"  ✅ optuna: {optuna.__version__}")
    except ImportError as e:
        print(f"  ❌ optuna: {e}")
        return False

    try:
        import plotly
        print(f"  ✅ plotly: {plotly.__version__}")
    except ImportError as e:
        print(f"  ❌ plotly: {e}")
        return False

    try:
        import plotly.io as pio
        print(f"  ✅ plotly.io imported")
    except ImportError as e:
        print(f"  ❌ plotly.io: {e}")
        return False

    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice
        )
        print(f"  ✅ optuna.visualization imported")
    except ImportError as e:
        print(f"  ❌ optuna.visualization: {e}")
        return False

    try:
        from optuna_optimization import (
            OptunaSSLObjective,
            OptunaCallback,
            HyperparameterOptimizer
        )
        print(f"  ✅ Custom optuna modules imported")
    except ImportError as e:
        print(f"  ❌ Custom optuna modules: {e}")
        return False

    return True

def test_basic_optuna_functionality():
    """Test basic Optuna study creation and optimization."""
    print("\n🧪 Testing basic Optuna functionality...")

    try:
        import optuna

        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            y = trial.suggest_float('y', -10, 10)
            return x**2 + y**2

        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=5)

        print(f"  ✅ Basic optimization works: best_value={study.best_value:.4f}")
        print(f"  ✅ Best params: {study.best_params}")

        return True

    except Exception as e:
        print(f"  ❌ Basic Optuna test failed: {e}")
        return False

def test_ssl_objective_creation():
    """Test SSL objective function creation."""
    print("\n🎯 Testing SSL objective creation...")

    try:
        from optuna_optimization import OptunaSSLObjective

        # Create objective with dummy parameters
        objective = OptunaSSLObjective(
            data_path="./dummy_path",  # Won't be used in this test
            num_channels=15,
            max_epochs=1,
            early_stopping_patience=1,
            accelerator='cpu',
            devices=1,
            num_workers=0
        )

        print(f"  ✅ OptunaSSLObjective created successfully")

        # Test hyperparameter suggestion with a mock trial
        import optuna

        # Create a study and get a proper trial
        study = optuna.create_study()
        trial = study.ask()

        hyperparams = objective._suggest_hyperparameters(trial)
        print(f"  ✅ Hyperparameter suggestion works")
        print(f"     Sample params: lr={hyperparams['learning_rate']:.6f}, "
              f"batch={hyperparams['batch_size']}, temp={hyperparams['temperature']:.3f}")

        return True

    except Exception as e:
        print(f"  ❌ SSL objective test failed: {e}")
        traceback.print_exc()
        return False

def test_hyperparameter_optimizer():
    """Test HyperparameterOptimizer class."""
    print("\n🔧 Testing HyperparameterOptimizer...")

    try:
        from optuna_optimization import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer(
            data_path="./dummy_path",
            study_name="test_study",
            num_channels=15,
            n_trials=3,  # Very small for testing
            max_epochs=1
        )

        print(f"  ✅ HyperparameterOptimizer created")
        print(f"     Study name: {optimizer.study_name}")
        print(f"     Channels: {optimizer.num_channels}")
        print(f"     Trials: {optimizer.n_trials}")

        return True

    except Exception as e:
        print(f"  ❌ HyperparameterOptimizer test failed: {e}")
        traceback.print_exc()
        return False

def test_visualization_functions():
    """Test visualization functions."""
    print("\n📊 Testing visualization functions...")

    try:
        import optuna
        from optuna.visualization import plot_optimization_history

        # Create a dummy study with some trials
        def dummy_objective(trial):
            x = trial.suggest_float('x', 0, 1)
            return x**2

        study = optuna.create_study()
        study.optimize(dummy_objective, n_trials=10)

        # Try to create a plot
        fig = plot_optimization_history(study)
        print(f"  ✅ Optimization history plot created")

        # Test if we can convert to HTML
        import plotly.io as pio
        html_str = pio.to_html(fig)
        print(f"  ✅ Plot to HTML conversion works ({len(html_str)} chars)")

        return True

    except Exception as e:
        print(f"  ❌ Visualization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Optuna Integration Pipeline")
    print("=" * 50)

    all_passed = True

    # Test 1: Imports
    if not test_imports():
        all_passed = False
        print("\n❌ Import test failed. Please check dependencies.")
        return

    # Test 2: Basic Optuna functionality
    if not test_basic_optuna_functionality():
        all_passed = False

    # Test 3: SSL objective creation
    if not test_ssl_objective_creation():
        all_passed = False

    # Test 4: HyperparameterOptimizer
    if not test_hyperparameter_optimizer():
        all_passed = False

    # Test 5: Visualizations
    if not test_visualization_functions():
        all_passed = False

    # Final result
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All Optuna integration tests passed!")
        print("\nNext steps:")
        print("1. Run: python example_optimization.py")
        print("2. Or run: python optimize_hyperparams.py --data_path /path/to/data")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    print("=" * 50)

if __name__ == "__main__":
    main()