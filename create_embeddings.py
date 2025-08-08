#!/usr/bin/env python3
"""
Complete pipeline to create CLIP embeddings from triplet files.

This script combines the splitting and embedding extraction steps:
1. Splits train triplets into train/val at image level, ensuring that there is no leakage, 
by making sure that the same image is not in both train and val.
2. Extracts CLIP embeddings for all images 
3. Creates tensor files ready for training
"""

from pathlib import Path

from src.data.create_splits import TripletSplitter
from src.data.extract_clip_embeddings import extract_embeddings_from_splits


def create_clip_embeddings(
    train_triplets_file: str = "src/data/splits/train_triplets.txt",
    images_dir: str = "src/data/food_images",
    splits_dir: str = "src/data/splits",
    embeddings_dir: str = "src/data/embeddings",
    split_ratio: float = 0.8,
    model_name: str = "openai/clip-vit-large-patch14-336",
    batch_size: int = 256,
    device: str = "auto",
    seed: int = 42
):
    """Complete pipeline for creating CLIP embeddings."""
    
    print("Creating train/validation splits...")
    
    splitter = TripletSplitter(split_ratio=split_ratio, seed=seed, verbose=False)
    split_stats = splitter.create_splits(
        train_triplets_file=train_triplets_file,
        output_dir=splits_dir
    )
    print(f"✓ Split {split_stats['original_train']} triplets → {split_stats['clean_train']} train, {split_stats['clean_val']} val ({split_stats['discarded']} discarded, no leakage)")
    
    print("Extracting CLIP embeddings...")
    
    # Use the split files as input for embedding extraction
    splits_path = Path(splits_dir)
    embeddings_stats = extract_embeddings_from_splits(
        train_triplets_file=str(splits_path / "train_triplets_clean.txt"),
        val_triplets_file=str(splits_path / "val_triplets_clean.txt"),
        test_triplets_file=str(splits_path / "test_triplets.txt"),  # Assumes test file exists separately
        images_dir=images_dir,
        output_dir=embeddings_dir,
        model_name=model_name,
        batch_size=batch_size,
        device=device
    )
    
    # Concise summary
    embeddings_path = Path(embeddings_dir)
    print(f"✓ Processed {embeddings_stats['total_images']} images → X_train: {embeddings_stats['X_train_shape']}, X_val: {embeddings_stats['X_val_shape']}, X_test: {embeddings_stats['X_test_shape']}")
    print(f"\nPipeline completed! Files saved to: {embeddings_path}/")
    print(f"Ready for training with {embeddings_stats['model']} embeddings (dim: {embeddings_stats['embedding_dim']})")
    
    return split_stats, embeddings_stats


if __name__ == "__main__":
    create_clip_embeddings()
