"""
Training utilities for triplet loss learning with PyTorch Lightning.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateMonitor,
    Callback
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.triplet_model import TripletModel
from src.data.food_datamodule import FoodDataModule


class MetricsSaver(Callback):
    """Callback to save training metrics to JSON files."""
    
    def __init__(self, save_dir: str, experiment_name: str):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "epoch": []
        }
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.save_dir / f"{experiment_name}_{timestamp}_metrics.json"
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Save metrics at the end of each epoch."""
        epoch = trainer.current_epoch
        

        # Get logged metrics (try multiple possible names)
        train_loss = (trainer.callback_metrics.get("train_loss_epoch", None) or 
                     trainer.callback_metrics.get("train_loss", None))
        train_acc = (trainer.callback_metrics.get("train_accuracy_epoch", None) or 
                    trainer.callback_metrics.get("train_accuracy", None))
        val_loss = trainer.callback_metrics.get("val_loss", None)
        val_acc = trainer.callback_metrics.get("val_accuracy", None)
        
        # Get learning rate
        lr = trainer.optimizers[0].param_groups[0]['lr']
        
        # Append to metrics
        self.metrics["epoch"].append(epoch)
        self.metrics["train_loss"].append(float(train_loss) if train_loss is not None else None)
        self.metrics["train_accuracy"].append(float(train_acc) if train_acc is not None else None)
        self.metrics["val_loss"].append(float(val_loss) if val_loss is not None else None)
        self.metrics["val_accuracy"].append(float(val_acc) if val_acc is not None else None)
        self.metrics["learning_rate"].append(lr)
        
        # Save to file
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


class TqdmProgressBar(Callback):
    """Custom progress bar using tqdm."""
    
    def __init__(self):
        super().__init__()
        self.train_progress_bar = None
        self.val_progress_bar = None
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Initialize training progress bar."""
        num_training_batches = getattr(trainer, 'num_training_batches', None)
        if num_training_batches is None:
            num_training_batches = len(trainer.train_dataloader) if trainer.train_dataloader else 0
        elif isinstance(num_training_batches, list):
            num_training_batches = sum(num_training_batches) if num_training_batches else 0
            
        self.train_progress_bar = tqdm(
            total=num_training_batches,
            desc=f"Epoch {trainer.current_epoch}",
            leave=False,
            dynamic_ncols=True
        )
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update training progress bar."""
        if self.train_progress_bar:
            # Get current metrics
            train_loss = trainer.callback_metrics.get("train_loss", 0)
            train_acc = trainer.callback_metrics.get("train_accuracy", 0)
            
            self.train_progress_bar.set_postfix({
                "loss": f"{train_loss:.4f}",
                "acc": f"{train_acc:.4f}"
            })
            self.train_progress_bar.update(1)
            
    def on_validation_epoch_start(self, trainer, pl_module):
        """Initialize validation progress bar."""
        # Get number of validation batches
        num_val_batches = getattr(trainer, 'num_val_batches', None)
        if num_val_batches is None:
            num_val_batches = len(trainer.val_dataloaders) if trainer.val_dataloaders else 0
        elif isinstance(num_val_batches, list):
            num_val_batches = sum(num_val_batches) if num_val_batches else 0
            
        if num_val_batches > 0:
            self.val_progress_bar = tqdm(
                total=num_val_batches,
                desc="Validating",
                leave=False,
                dynamic_ncols=True
            )
            
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Update validation progress bar."""
        if self.val_progress_bar:
            self.val_progress_bar.update(1)
            
    def on_train_epoch_end(self, trainer, pl_module):
        """Close training progress bar and show epoch summary."""
        if self.train_progress_bar:
            self.train_progress_bar.close()
            
            # Print epoch summary
            val_loss = trainer.callback_metrics.get("val_loss", 0)
            val_acc = trainer.callback_metrics.get("val_accuracy", 0)
            train_loss = (trainer.callback_metrics.get("train_loss_epoch", None) or 
                         trainer.callback_metrics.get("train_loss", 0))
            train_acc = (trainer.callback_metrics.get("train_accuracy_epoch", None) or 
                        trainer.callback_metrics.get("train_accuracy", 0))
            lr = trainer.optimizers[0].param_groups[0]['lr']
            
            print(f"Epoch {trainer.current_epoch}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, lr={lr:.2e}")
            
    def on_validation_epoch_end(self, trainer, pl_module):
        """Close validation progress bar."""
        if self.val_progress_bar:
            self.val_progress_bar.close()


class TripletTrainer:
    """
    Comprehensive trainer for triplet loss learning on food images.
    
    Handles PyTorch Lightning setup, logging, callbacks, and training orchestration.
    """
    
    def __init__(
        self,
        # Model parameters
        input_dim: int = 768,   # CLIP embedding dimension
        hidden_dim: int = 256,  # Hidden layer dimension
        margin: float = 0.3,    # Triplet loss margin
        dropout_prob: float = 0.3,  # Dropout probability
        learning_rate: float = 1e-4,  # Learning rate for SGD
        normalize_embeddings: bool = True,  # L2 normalization
        
        # Data parameters (CLIP embeddings in triplet format)
        train_embeddings_file: str = "src/data/embeddings/X_train_clip.pt",
        val_embeddings_file: str = "src/data/embeddings/X_val_clip.pt",
        test_embeddings_file: str = "src/data/embeddings/X_test_clip.pt",
        batch_size: int = 1024,  # Batch size
        num_workers: int = 8,
        
        # Training parameters
        max_epochs: int = 100,
        accelerator: str = "auto",
        devices: str = "auto",
        precision: str = "32-true",
        
        # Logging and checkpointing
        log_dir: str = "logs",
        experiment_name: str = "triplet_taste_simple",
        save_top_k: int = 3,
        monitor_metric: str = "val_accuracy",  # Monitor accuracy
        monitor_mode: str = "max",  # Maximize accuracy
        patience: int = 10,
        
        # Other parameters
        seed: int = 42,
    ):
        """
        Initialize the TripletTrainer.
        
        Args:
            # Model parameters
            input_dim: Input embedding dimension (768 for CLIP)
            hidden_dim: Hidden layer dimension
            output_dim: Final embedding dimension
            margin: Triplet loss margin
            dropout_prob: Dropout probability
            learning_rate: Learning rate
            weight_decay: Weight decay
            normalize_embeddings: Whether to normalize embeddings
            
            # Data parameters
            embeddings_file: Path to precomputed embeddings
            train_triplets_file: Path to training triplets
            val_triplets_file: Path to validation triplets
            test_triplets_file: Path to test triplets
            class_assignments_file: Path to pre-computed class assignments
            batch_size: Batch size for validation/test
            num_workers: Number of data loading workers
            
            # Balanced batching parameters
            use_balanced_batches: Enable food-balanced batching for training
            k_foods: Number of food categories per training batch
            m_images: Number of images per food category in training batches
            
            # Training parameters
            max_epochs: Maximum number of epochs
            accelerator: Accelerator type ("auto", "cpu", "gpu")
            devices: Number of devices
            precision: Training precision
            
            # Logging and checkpointing
            log_dir: Directory for logs
            experiment_name: Name for this experiment
            save_top_k: Number of best models to save
            monitor_metric: Metric to monitor for callbacks
            monitor_mode: Mode for monitoring ("min" or "max")
            patience: Early stopping patience
            
            # Other parameters
            seed: Random seed
            fast_dev_run: Run single batch for debugging
            limit_train_batches: Limit training batches (for debugging)
            limit_val_batches: Limit validation batches (for debugging)
        """
        self.model_params = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "margin": margin,
            "dropout_prob": dropout_prob,
            "learning_rate": learning_rate,
            "normalize_embeddings": normalize_embeddings,
        }
        
        self.data_params = {
            "train_embeddings_file": train_embeddings_file,
            "val_embeddings_file": val_embeddings_file,
            "test_embeddings_file": test_embeddings_file,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "embedding_dim": input_dim,
            "seed": seed,
        }
        
        self.training_params = {
            "max_epochs": max_epochs,
            "accelerator": accelerator,
            "devices": devices,
            "precision": precision,
        }
        
        self.logging_params = {
            "log_dir": log_dir,
            "experiment_name": experiment_name,
            "save_top_k": save_top_k,
            "monitor_metric": monitor_metric,
            "monitor_mode": monitor_mode,
            "patience": patience,
        }
        
        self.other_params = {
            "seed": seed
        }
        
        # Initialize components
        self.model = None
        self.datamodule = None
        self.trainer = None
        
        # Set seed for reproducibility
        seed_everything(seed, workers=True)
    
    def setup_model(self) -> TripletModel:
        """Set up the triplet loss model."""
        print("Setting up TripletModel...")
        print(f"Model configuration: {self.model_params}")
        
        self.model = TripletModel(**self.model_params)
        
        print(f"Model created with {self.model.num_trainable_params:,} trainable parameters")
        return self.model
    
    def setup_data(self) -> FoodDataModule:
        """Set up the data module."""
        print("Setting up FoodDataModule...")
        
        self.datamodule = FoodDataModule(**self.data_params)
        
        print("Data module created successfully")
        return self.datamodule
    
    def setup_callbacks(self) -> List[pl.Callback]:
        """Set up training callbacks."""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.logging_params["log_dir"], "checkpoints"),
            filename="{epoch:02d}-{" + self.logging_params["monitor_metric"].replace('_', '-') + ":.4f}",
            save_top_k=self.logging_params["save_top_k"],
            monitor=self.logging_params["monitor_metric"],
            mode=self.logging_params["monitor_mode"],
            save_last=True,
            save_weights_only=False,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=self.logging_params["monitor_metric"],
            mode=self.logging_params["monitor_mode"],
            patience=self.logging_params["patience"],
            verbose=True,
            strict=True,
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitoring
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)
        
        # Custom tqdm progress bar
        progress_bar = TqdmProgressBar()
        callbacks.append(progress_bar)
        
        # Metrics saver
        metrics_saver = MetricsSaver(
            save_dir=self.logging_params["log_dir"],
            experiment_name=self.logging_params["experiment_name"]
        )
        callbacks.append(metrics_saver)
        
        return callbacks
    
    def setup_logger(self) -> CSVLogger:
        """Set up CSV logger."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{self.logging_params['experiment_name']}_{timestamp}"
        
        logger = CSVLogger(
            save_dir=self.logging_params["log_dir"],
            name="csv_logs",
            version=version,
        )
        
        return logger
    
    def setup_trainer(self) -> Trainer:
        """Set up PyTorch Lightning trainer."""
        print("Setting up PyTorch Lightning Trainer...")
        
        # Setup callbacks and logger
        callbacks = self.setup_callbacks()
        logger = self.setup_logger()
        
        # Create trainer
        trainer_args = {
            **self.training_params,
            "callbacks": callbacks,
            "logger": logger,
            "enable_checkpointing": True,
            "enable_progress_bar": False,  # Disabled to use custom tqdm progress bar
            "enable_model_summary": True,
            "deterministic": True,
            "benchmark": False,
            "log_every_n_steps": 1,
            "check_val_every_n_epoch": 1,
            "num_sanity_val_steps": 2,
        }
        
        self.trainer = Trainer(**trainer_args)
        
        return self.trainer
    
    def train(self) -> Dict[str, Any]:
        """
        Run the full training pipeline.
        
        Returns:
            Training results dictionary
        """
        print("="*60)
        print("Starting Triplet Loss Training")
        print("="*60)
        
        # Setup components
        model = self.setup_model()
        datamodule = self.setup_data()
        trainer = self.setup_trainer()
        
        # Log experiment details
        print(f"\nExperiment: {self.logging_params['experiment_name']}")
        print(f"Log directory: {self.logging_params['log_dir']}")
        print(f"Seed: {self.other_params['seed']}")
        
        # Start training
        print("\nStarting training...")
        trainer.fit(model, datamodule)
        
        # Test the best model
        print("\nTesting best model...")
        test_results = trainer.test(model, datamodule, ckpt_path="best")
        
        # Get metrics file path from MetricsSaver callback
        metrics_file = None
        for callback in trainer.callbacks:
            if isinstance(callback, MetricsSaver):
                metrics_file = str(callback.metrics_file)
                break
        
        # Training summary
        results = {
            "best_model_path": trainer.checkpoint_callback.best_model_path,
            "best_model_score": trainer.checkpoint_callback.best_model_score,
            "test_results": test_results,
            "total_epochs": trainer.current_epoch,
            "global_step": trainer.global_step,
            "metrics_file": metrics_file,
        }
        
        print("="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Best model: {results['best_model_path']}")
        print(f"Best {self.logging_params['monitor_metric']}: {results['best_model_score']:.4f}")
        print(f"Test results: {test_results}")
        if metrics_file:
            print(f"Metrics saved to: {metrics_file}")
        print(f"CSV logs saved to: {self.logging_params['log_dir']}/csv_logs/")
        
        return results
    
    def load_best_model(self) -> TripletModel:
        """Load the best model from checkpoint."""
        if not self.trainer or not self.trainer.checkpoint_callback.best_model_path:
            raise ValueError("No trained model found. Run train() first.")
        
        model = TripletModel.load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path
        )
        
        print(f"Loaded best model from: {self.trainer.checkpoint_callback.best_model_path}")
        return model


def main():
    """Main training function."""
    # Create trainer with default parameters
    trainer = TripletTrainer()
    
    # Run training
    results = trainer.train()
    
    return results


if __name__ == "__main__":
    main() 