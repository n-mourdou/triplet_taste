#!/usr/bin/env python3
"""
Module to extract CLIP embeddings from triplet files.

This module takes clean triplet files (output from create_splits.py) and extracts
CLIP embeddings for all images, then converts triplets to tensor format.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Set
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel


def read_triplets(file_path: Path) -> List[Tuple[str, str, str]]:
    """Read triplets from file."""
    with open(file_path, "r") as f:
        return [tuple(line.strip().split()) for line in f if line.strip()]


def get_all_image_ids(triplet_files: List[Path]) -> Set[str]:
    """Extract all unique image IDs from triplet files."""
    all_ids = set()
    for file_path in triplet_files:
        triplets = read_triplets(file_path)
        for triplet in triplets:
            all_ids.update(triplet)
    return all_ids


def load_images_batch(image_paths: List[Path]) -> Tuple[List[Image.Image], List[str]]:
    """Load a batch of images."""
    images = []
    valid_ids = []
    
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_ids.append(path.stem)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
    
    return images, valid_ids


def extract_clip_embeddings(
    image_ids: List[str],
    images_dir: Path,
    model_name: str = "openai/clip-vit-large-patch14-336",
    batch_size: int = 256,
    device: str = "auto"
) -> Dict[str, torch.Tensor]:
    """
    Extract CLIP embeddings for given image IDs.
    
    Args:
        image_ids: List of image IDs to process
        images_dir: Directory containing image files
        model_name: CLIP model name from HuggingFace
        batch_size: Batch size for processing
        device: Device to use ("auto", "cpu", "cuda")
        
    Returns:
        Dictionary mapping image IDs to embeddings
    """
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    print(f"Loading CLIP model: {model_name}")
    print(f"Using device: {device}")
    
    # Load model and processor
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    print(f"Extracting embeddings for {len(image_ids)} images")
    
    embeddings = {}
    image_paths = [images_dir / f"{img_id}.jpg" for img_id in image_ids]
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images, valid_ids = load_images_batch(batch_paths)
        
        if not batch_images:
            continue
        
        # Process batch
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            features = model.get_image_features(**inputs).cpu()
        
        # Store embeddings
        for img_id, embedding in zip(valid_ids, features):
            embeddings[img_id] = embedding
        
        # Clear GPU cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    print(f"Successfully extracted embeddings for {len(embeddings)} images")
    return embeddings


def triplets_to_tensor(
    triplets: List[Tuple[str, str, str]], 
    embedding_dict: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    Convert triplets to tensor format using embedding dictionary.
    
    Args:
        triplets: List of (anchor, positive, negative) triplets
        embedding_dict: Dictionary mapping image IDs to embeddings
        
    Returns:
        Tensor of shape (num_triplets, 3, embedding_dim)
    """
    tensor_triplets = []
    skipped = 0
    
    for anchor, positive, negative in triplets:
        try:
            anchor_emb = embedding_dict[anchor]
            positive_emb = embedding_dict[positive]
            negative_emb = embedding_dict[negative]
            
            # Stack as (3, embedding_dim)
            triplet_tensor = torch.stack([anchor_emb, positive_emb, negative_emb])
            tensor_triplets.append(triplet_tensor)
            
        except KeyError as e:
            skipped += 1
            print(f"Skipping triplet due to missing embedding: {e}")
    
    if skipped > 0:
        print(f"Skipped {skipped} triplets due to missing embeddings")
    
    if not tensor_triplets:
        raise ValueError("No valid triplets found")
    
    # Stack to (num_triplets, 3, embedding_dim)
    return torch.stack(tensor_triplets)


def extract_embeddings_from_splits(
    train_triplets_file: str,
    val_triplets_file: str,
    test_triplets_file: str,
    images_dir: str,
    output_dir: str,
    model_name: str = "openai/clip-vit-large-patch14-336",
    batch_size: int = 256,
    device: str = "auto"
) -> dict:
    """
    Extract CLIP embeddings and create tensor files for train/val/test splits.
    
    Args:
        train_triplets_file: Path to clean train triplets file
        val_triplets_file: Path to clean validation triplets file  
        test_triplets_file: Path to clean test triplets file
        images_dir: Directory containing image files
        output_dir: Directory to save embedding files
        model_name: CLIP model name
        batch_size: Batch size for processing
        device: Device to use
        
    Returns:
        Dictionary with processing statistics
    """
    # Convert to Path objects
    train_file = Path(train_triplets_file)
    val_file = Path(val_triplets_file)
    test_file = Path(test_triplets_file)
    images_path = Path(images_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("EXTRACTING CLIP EMBEDDINGS")
    print("=" * 50)
    
    # Read all triplets
    print("Reading triplet files...")
    train_triplets = read_triplets(train_file)
    val_triplets = read_triplets(val_file)
    test_triplets = read_triplets(test_file)
    
    print(f"Train triplets: {len(train_triplets)}")
    print(f"Validation triplets: {len(val_triplets)}")
    print(f"Test triplets: {len(test_triplets)}")
    
    # Get all unique image IDs
    all_image_ids = get_all_image_ids([train_file, val_file, test_file])
    print(f"Total unique images: {len(all_image_ids)}")
    
    # Extract embeddings for all images
    embedding_dict = extract_clip_embeddings(
        image_ids=list(all_image_ids),
        images_dir=images_path,
        model_name=model_name,
        batch_size=batch_size,
        device=device
    )
    
    # Convert triplets to tensors
    print("Converting triplets to tensors...")
    X_train = triplets_to_tensor(train_triplets, embedding_dict)
    X_val = triplets_to_tensor(val_triplets, embedding_dict)
    X_test = triplets_to_tensor(test_triplets, embedding_dict)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Save tensor files
    train_output = output_path / "X_train_clip.pt"
    val_output = output_path / "X_val_clip.pt"
    test_output = output_path / "X_test_clip.pt"
    
    torch.save(X_train, train_output)
    torch.save(X_val, val_output)
    torch.save(X_test, test_output)
    
    print(f"\nSaved tensor files:")
    print(f"  {train_output}")
    print(f"  {val_output}")
    print(f"  {test_output}")
    
    # Save metadata
    embedding_dim = list(embedding_dict.values())[0].shape[0] if embedding_dict else 0
    metadata = {
        "model": model_name,
        "embedding_dim": embedding_dim,
        "train_triplets": len(train_triplets),
        "val_triplets": len(val_triplets),
        "test_triplets": len(test_triplets),
        "total_images": len(all_image_ids),
        "successful_embeddings": len(embedding_dict),
        "X_train_shape": tuple(X_train.shape),
        "X_val_shape": tuple(X_val.shape),
        "X_test_shape": tuple(X_test.shape),
        "batch_size": batch_size,
        "images_dir": str(images_path),
        "device": str(device)
    }
    
    metadata_file = output_path / "metadata_clip.pt"
    torch.save(metadata, metadata_file)
    print(f"  {metadata_file}")
    
    print("\nEmbedding extraction completed successfully!")
    return metadata


if __name__ == "__main__":
    # Example usage
    metadata = extract_embeddings_from_splits(
        train_triplets_file="splits/train_triplets_clean.txt",
        val_triplets_file="splits/val_triplets_clean.txt",
        test_triplets_file="splits/test_triplets_clean.txt",
        images_dir="src/data/food_images",
        output_dir="embeddings",
        model_name="openai/clip-vit-large-patch14-336",
        batch_size=256,
        device="auto"
    )