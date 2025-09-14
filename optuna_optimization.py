#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for TinyViT SSL training.
"""

import os
import tempfile
import shutil
from typing import Dict, Any, Optional
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ssl_training import SSLTrainer, SSLTrainingModule


class OptunaSSLObjective:
    """
    Optuna objective function for SSL hyperparameter optimization.
    """

    def __init__(
        self,
        data_path: str,
        num_channels: int = 15,
        max_epochs: int = 50,
        early_stopping_patience: int = 10,
        use_h5: bool = False,
        h5_dataset_key: str = 'images',
        accelerator: str = 'auto',
        devices: int = 1,
        num_workers: int = 4,
        file_extension: str = 'npy'
    ):
        self.data_path = data_path
        self.num_channels = num_channels
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.use_h5 = use_h5
        self.h5_dataset_key = h5_dataset_key
        self.accelerator = accelerator
        self.devices = devices
        self.num_workers = num_workers
        self.file_extension = file_extension

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna for each trial.

        Args:
            trial: Optuna trial object

        Returns:
            float: Objective value (lower is better - we use final loss)
        """
        # Suggest hyperparameters
        hyperparams = self._suggest_hyperparameters(trial)

        # Create temporary directory for this trial
        trial_dir = tempfile.mkdtemp(prefix=f"optuna_trial_{trial.number}_")

        try:
            # Train model with suggested hyperparameters
            final_loss = self._train_model(trial, hyperparams, trial_dir)
            return final_loss

        except Exception as e:
            # Handle training failures gracefully
            print(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.TrialPruned()

        finally:
            # Clean up temporary directory
            if os.path.exists(trial_dir):
                shutil.rmtree(trial_dir)

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define hyperparameter search space.

        Args:
            trial: Optuna trial object

        Returns:
            Dict of suggested hyperparameters
        """
        return {
            # Learning rate - log scale between 1e-5 and 1e-1
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),

            # Batch size - power of 2 values
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128]),

            # Temperature for contrastive loss
            'temperature': trial.suggest_float('temperature', 0.05, 0.5),

            # Projection dimension
            'projection_dim': trial.suggest_categorical('projection_dim', [64, 128, 256, 512]),

            # Weight decay
            'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True),

            # Augmentation parameters
            'horizontal_flip_prob': trial.suggest_float('horizontal_flip_prob', 0.0, 1.0),
            'gaussian_blur_prob': trial.suggest_float('gaussian_blur_prob', 0.0, 1.0),
            'min_scale': trial.suggest_float('min_scale', 0.05, 0.3),

            # Model architecture
            'model_name': trial.suggest_categorical('model_name', [
                'tiny_vit_21m_512.dist_in22k_ft_in1k',
                'tiny_vit_11m_224.dist_in22k_ft_in1k',
                'tiny_vit_21m_224.dist_in22k_ft_in1k'
            ])
        }

    def _train_model(
        self,
        trial: optuna.Trial,
        hyperparams: Dict[str, Any],
        trial_dir: str
    ) -> float:
        """
        Train model with given hyperparameters.

        Args:
            trial: Optuna trial object
            hyperparams: Dictionary of hyperparameters
            trial_dir: Temporary directory for this trial

        Returns:
            float: Final validation loss
        """
        # Create SSL training module with optimized hyperparameters
        ssl_model = SSLTrainingModule(
            num_channels=self.num_channels,
            model_name=hyperparams['model_name'],
            projection_dim=hyperparams['projection_dim'],
            learning_rate=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay'],
            temperature=hyperparams['temperature'],
            max_epochs=self.max_epochs
        )

        # Create data loader with optimized parameters
        trainer_config = self._create_trainer_config(hyperparams, trial_dir)

        # Create custom dataloader with optimized augmentation parameters
        dataloader = self._create_optimized_dataloader(hyperparams)

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='train_loss_epoch',
                patience=self.early_stopping_patience,
                mode='min'
            ),
            ModelCheckpoint(
                dirpath=trial_dir,
                filename=f'best_model_trial_{trial.number}',
                monitor='train_loss_epoch',
                mode='min',
                save_top_k=1
            ),
            OptunaCallback(trial, monitor='train_loss_epoch')
        ]

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            callbacks=callbacks,
            logger=False,  # Disable logging for optimization
            enable_checkpointing=True,
            enable_progress_bar=False,  # Reduce output clutter
            **trainer_config
        )

        # Train the model
        trainer.fit(ssl_model, dataloader)

        # Return the best validation loss
        best_loss = trainer.callback_metrics.get('train_loss_epoch', float('inf'))
        return float(best_loss)

    def _create_trainer_config(
        self,
        hyperparams: Dict[str, Any],
        trial_dir: str
    ) -> Dict[str, Any]:
        """
        Create trainer configuration based on hyperparameters.

        Args:
            hyperparams: Hyperparameter dictionary
            trial_dir: Directory for this trial

        Returns:
            Trainer configuration dictionary
        """
        config = {
            'default_root_dir': trial_dir,
        }

        # Add precision based on model size
        if 'tiny_vit_21m' in hyperparams['model_name']:
            config['precision'] = '16-mixed' if torch.cuda.is_available() else '32'

        return config

    def _create_optimized_dataloader(self, hyperparams: Dict[str, Any]):
        """
        Create dataloader with optimized augmentation parameters.

        Args:
            hyperparams: Hyperparameter dictionary

        Returns:
            Optimized dataloader
        """
        from ssl_training import MultiChannelCollateFunction
        from lightly.data import LightlyDataset
        from torch.utils.data import DataLoader
        from dataset import MultiChannelDataset, MultiChannelH5Dataset

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

        # Create optimized collate function
        collate_fn = MultiChannelCollateFunction(
            input_size=460,
            min_scale=hyperparams['min_scale'],
            hf_prob=hyperparams['horizontal_flip_prob'],
            gaussian_blur=hyperparams['gaussian_blur_prob']
        )

        # Create dataloader
        dataloader = DataLoader(
            lightly_dataset,
            batch_size=hyperparams['batch_size'],
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

        return dataloader


class OptunaCallback(pl.Callback):
    """
    PyTorch Lightning callback for Optuna integration.
    """

    def __init__(self, trial: optuna.Trial, monitor: str):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_train_epoch_end(self, trainer, pl_module):
        # Report intermediate value to Optuna
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is not None:
            self.trial.report(float(current_score), step=trainer.current_epoch)

            # Prune trial if not promising
            if self.trial.should_prune():
                raise optuna.TrialPruned()


class HyperparameterOptimizer:
    """
    High-level interface for SSL hyperparameter optimization.
    """

    def __init__(
        self,
        data_path: str,
        study_name: str = "ssl_optimization",
        storage: Optional[str] = None,
        num_channels: int = 15,
        n_trials: int = 100,
        max_epochs: int = 50,
        **kwargs
    ):
        self.data_path = data_path
        self.study_name = study_name
        self.storage = storage
        self.num_channels = num_channels
        self.n_trials = n_trials
        self.max_epochs = max_epochs
        self.kwargs = kwargs

        # Create objective
        self.objective = OptunaSSLObjective(
            data_path=data_path,
            num_channels=num_channels,
            max_epochs=max_epochs,
            **kwargs
        )

    def optimize(
        self,
        direction: str = 'minimize',
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            direction: Optimization direction ('minimize' or 'maximize')
            sampler: Optuna sampler (default: TPESampler)
            pruner: Optuna pruner (default: MedianPruner)

        Returns:
            Completed Optuna study
        """
        # Default sampler and pruner
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=42)

        if pruner is None:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )

        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True
        )

        print(f"🔍 Starting hyperparameter optimization with {self.n_trials} trials")
        print(f"Study name: {self.study_name}")
        print(f"Direction: {direction}")
        print(f"Data path: {self.data_path}")
        print("-" * 60)

        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials)

        # Print results
        self._print_optimization_results(study)

        return study

    def _print_optimization_results(self, study: optuna.Study):
        """Print optimization results."""
        print("\n" + "=" * 60)
        print("🎉 Optimization Complete!")
        print(f"Best value: {study.best_value:.4f}")
        print(f"Best trial: {study.best_trial.number}")

        print("\n📊 Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Print trial statistics
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

        print(f"\n📈 Trial statistics:")
        print(f"  Complete trials: {len(complete_trials)}")
        print(f"  Pruned trials: {len(pruned_trials)}")
        print(f"  Failed trials: {len(failed_trials)}")

        print("=" * 60)

    def create_visualization(
        self,
        study: optuna.Study,
        output_dir: str = "optuna_plots"
    ):
        """
        Create optimization visualizations.

        Args:
            study: Completed Optuna study
            output_dir: Directory to save plots
        """
        import plotly.io as pio
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice
        )

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Optimization history
            fig = plot_optimization_history(study)
            pio.write_html(fig, os.path.join(output_dir, "optimization_history.html"))

            # Parameter importance
            fig = plot_param_importances(study)
            pio.write_html(fig, os.path.join(output_dir, "param_importance.html"))

            # Parallel coordinate plot
            fig = plot_parallel_coordinate(study)
            pio.write_html(fig, os.path.join(output_dir, "parallel_coordinate.html"))

            # Parameter slice plots
            fig = plot_slice(study)
            pio.write_html(fig, os.path.join(output_dir, "param_slices.html"))

            print(f"📊 Visualizations saved to {output_dir}/")

        except Exception as e:
            print(f"⚠️ Could not create visualizations: {e}")

    def get_best_config(self, study: optuna.Study) -> Dict[str, Any]:
        """
        Get the best configuration for training.

        Args:
            study: Completed Optuna study

        Returns:
            Best configuration dictionary
        """
        best_params = study.best_params.copy()
        best_params['objective_value'] = study.best_value
        best_params['trial_number'] = study.best_trial.number

        return best_params