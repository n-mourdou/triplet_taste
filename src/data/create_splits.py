#!/usr/bin/env python3
"""
Module to split train triplets into train/validation sets at image level.

This module splits triplets by first splitting image IDs, then filtering triplets
to ensure no image appears in both train and validation sets.
"""

import random
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional


class TripletSplitter:
    """
    A class to handle splitting triplet data into train/validation sets.
    
    This splitter ensures no image leakage between train and validation sets
    by first splitting unique image IDs, then filtering triplets accordingly.
    """
    
    def __init__(self, split_ratio: float = 0.8, seed: int = 42, verbose: bool = True):
        """
        Initialize the TripletSplitter.
        
        Args:
            split_ratio: Ratio for train split (default 0.8)
            seed: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        self.split_ratio = split_ratio
        self.seed = seed
        self.verbose = verbose
        self._train_ids: Optional[Set[str]] = None
        self._val_ids: Optional[Set[str]] = None
        
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _read_triplets(self, file_path: Path) -> List[Tuple[str, str, str]]:
        """Read triplets from file."""
        with open(file_path, "r") as f:
            return [tuple(line.strip().split()) for line in f if line.strip()]
    
    def _save_triplets(self, triplets: List[Tuple[str, str, str]], output_path: Path) -> None:
        """Save triplets to file."""
        with open(output_path, "w") as f:
            for triplet in triplets:
                f.write(" ".join(triplet) + "\n")
    
    def _split_image_ids(self, triplets: List[Tuple[str, str, str]]) -> Tuple[Set[str], Set[str]]:
        """Split image IDs into train and validation sets."""
        # Extract all unique image IDs from triplets
        all_ids = {img_id for triplet in triplets for img_id in triplet}
        all_ids = list(all_ids)
        
        # Shuffle with seed for reproducibility
        random.seed(self.seed)
        random.shuffle(all_ids)
        
        # Split at specified ratio
        split_point = int(len(all_ids) * self.split_ratio)
        train_ids = set(all_ids[:split_point])
        val_ids = set(all_ids[split_point:])
        
        return train_ids, val_ids
    
    def _filter_triplets(self, triplets: List[Tuple[str, str, str]], 
                        train_ids: Set[str], val_ids: Set[str]) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], int]:
        """Filter triplets based on train/val image ID sets."""
        train_triplets = []
        val_triplets = []
        discarded = 0
        
        for anchor, positive, negative in triplets:
            triplet_ids = {anchor, positive, negative}
            
            if triplet_ids <= train_ids:  # All IDs in train set
                train_triplets.append((anchor, positive, negative))
            elif triplet_ids <= val_ids:  # All IDs in val set
                val_triplets.append((anchor, positive, negative))
            else:  # Mixed or missing IDs
                discarded += 1
        
        return train_triplets, val_triplets, discarded

    
    def create_splits(self, train_triplets_file: str, output_dir: str) -> Dict[str, any]:
        """
        Create train/validation splits from train triplets file.
        
        We discard triplets that have an image in both train and val, ensuring no leakage.
        
        Args:
            train_triplets_file: Path to original train triplets file 
            output_dir: Directory to save split files
            
        Returns:
            Dictionary with split statistics
        """
        # Convert to Path objects
        train_file = Path(train_triplets_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self._log("CREATING TRAIN/VALIDATION SPLITS")
        self._log("=" * 50)
        
        # Read triplets
        self._log(f"Reading train triplets from: {train_file}")
        train_triplets_raw = self._read_triplets(train_file)
        self._log(f"Loaded {len(train_triplets_raw)} train triplets")
        
        # Split image IDs
        self._log(f"Splitting image IDs with ratio {self.split_ratio}")
        self._train_ids, self._val_ids = self._split_image_ids(train_triplets_raw)
        self._log(f"Train image IDs: {len(self._train_ids)}")
        self._log(f"Validation image IDs: {len(self._val_ids)}")
        
        # Filter triplets based on image ID splits
        self._log("Filtering triplets...")
        train_triplets, val_triplets, discarded = self._filter_triplets(
            train_triplets_raw, self._train_ids, self._val_ids
        )
        
        # Save split files
        train_output = output_path / "train_triplets_clean.txt"
        val_output = output_path / "val_triplets_clean.txt"
        
        self._save_triplets(train_triplets, train_output)
        self._save_triplets(val_triplets, val_output)
        
        # Print statistics
        self._log("\nSPLIT STATISTICS:")
        self._log("-" * 30)
        self._log(f"Original train triplets: {len(train_triplets_raw)}")
        self._log(f"Clean train triplets: {len(train_triplets)}")
        self._log(f"Clean validation triplets: {len(val_triplets)}")
        self._log(f"Discarded triplets: {discarded}")
        
        self._log(f"\nSaved files:")
        self._log(f"  {train_output}")
        self._log(f"  {val_output}")
        
        # Return statistics
        return {
            "original_train": len(train_triplets_raw),
            "clean_train": len(train_triplets),
            "clean_val": len(val_triplets),
            "discarded": discarded,
            "train_ids": len(self._train_ids),
            "val_ids": len(self._val_ids),
            "split_ratio": self.split_ratio,
            "seed": self.seed
        }
    
    @property
    def train_ids(self) -> Optional[Set[str]]:
        """Get the train image IDs after splitting."""
        return self._train_ids
    
    @property
    def val_ids(self) -> Optional[Set[str]]:
        """Get the validation image IDs after splitting."""
        return self._val_ids



if __name__ == "__main__":

    splitter = TripletSplitter(split_ratio=0.8, seed=42, verbose=True)
    stats = splitter.create_splits(
        train_triplets_file="src/data/splits/train_triplets.txt",
        output_dir="src/data/splits"
    )
    print(f"Number of train image IDs: {len(splitter.train_ids)}")
    print(f"Number of val image IDs: {len(splitter.val_ids)}")
    print("\nSplit creation completed successfully!")