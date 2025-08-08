"""
PyTorch Lightning DataModule for food image triplet learning using precomputed CLIP embeddings.
"""

import os
from typing import Optional, List, Set, Tuple, Dict
from pathlib import Path
import random

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset



class FoodDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for food image triplet learning.
    
    Uses precomputed CLIP embeddings in triplet format for fast training.
    """
    
    def __init__(
        self,
        train_embeddings_file: str = "embeddings/X_train_clip.pt",
        val_embeddings_file: str = "embeddings/X_val_clip.pt", 
        test_embeddings_file: str = "embeddings/X_test_clip.pt",
        batch_size: int = 1024,  # Batch size
        num_workers: int = 8,
        embedding_dim: int = 768,  # CLIP embedding dimension
        seed: int = 42,
    ):
        """
        Initialize the FoodDataModule.
        
        Args:
            train_embeddings_file: Path to precomputed training embeddings (.pt)
            val_embeddings_file: Path to precomputed validation embeddings (.pt)  
            test_embeddings_file: Path to precomputed test embeddings (.pt)
            batch_size: Batch size for all splits
            num_workers: Number of workers for data loading
            embedding_dim: Dimension of CLIP embeddings (768)
            seed: Random seed for reproducible splits
        """
        super().__init__()
        
        # Store hyperparameters
        self.save_hyperparameters()
        
        self.train_embeddings_file = Path(train_embeddings_file)
        self.val_embeddings_file = Path(val_embeddings_file)
        self.test_embeddings_file = Path(test_embeddings_file)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.embedding_dim = embedding_dim
        self.seed = seed
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Set random seed
        torch.manual_seed(seed)
        random.seed(seed)
    
    def prepare_data(self) -> None:
        """
        Check that all required embedding files exist.
        
        This method is called only once and on a single process.
        """
        # Check if embedding files exist
        for file_path, name in [
            (self.train_embeddings_file, "training embeddings"),
            (self.val_embeddings_file, "validation embeddings"),
            (self.test_embeddings_file, "test embeddings")
        ]:
            if not file_path.exists():
                print(f"Warning: {name} file not found: {file_path}")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for training, validation, and testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None for all)
        """
        print(f"Setting up data for stage: {stage}")
        
        # Setup datasets based on stage
        if stage == "fit" or stage is None:
            # Setup training dataset
            if self.train_embeddings_file.exists():
                print(f"Loading training embeddings from {self.train_embeddings_file}")
                train_embeddings = torch.load(self.train_embeddings_file, map_location='cpu', weights_only=False)
                print(f"Loaded training embeddings: {train_embeddings.shape}")
                
                self.train_dataset = TensorDataset(train_embeddings)
                print(f"Created training dataset with {len(self.train_dataset)} samples")
            else:
                raise FileNotFoundError(f"Training embeddings file not found: {self.train_embeddings_file}")
        
        if stage == "fit" or stage == "validate" or stage is None:
            # Setup validation dataset
            if self.val_embeddings_file.exists():
                print(f"Loading validation embeddings from {self.val_embeddings_file}")
                val_embeddings = torch.load(self.val_embeddings_file, map_location='cpu', weights_only=False)
                print(f"Loaded validation embeddings: {val_embeddings.shape}")
                
                self.val_dataset = TensorDataset(val_embeddings)
                print(f"Created validation dataset with {len(self.val_dataset)} samples")
            else:
                print("Warning: No validation embeddings file found, validation will be skipped")
                self.val_dataset = None
        
        if stage == "test" or stage is None:
            # Setup test dataset
            if self.test_embeddings_file.exists():
                print(f"Loading test embeddings from {self.test_embeddings_file}")
                test_embeddings = torch.load(self.test_embeddings_file, map_location='cpu', weights_only=False)
                print(f"Loaded test embeddings: {test_embeddings.shape}")
                
                self.test_dataset = TensorDataset(test_embeddings)
                print(f"Created test dataset with {len(self.test_dataset)} samples")
            else:
                print("Warning: No test embeddings file found, testing will be skipped")
                self.test_dataset = None
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Create validation dataloader."""
        if self.val_dataset is None:
            return None
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader."""
        if self.test_dataset is None:
            return None
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    