#!/usr/bin/env python3
"""
Training script for the triplet-taste project.

This script provides a convenient entry point for training the triplet loss model
on food image embeddings.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.trainer import TripletTrainer


def main():
    """Main training function with default parameters optimized for CPU training."""
    
    # Create trainer
    trainer = TripletTrainer(
        # Model parameters
        input_dim=768,       # CLIP embedding dimension
        hidden_dim=256,      # Hidden layer size
        margin=0.4,          # Triplet loss margin
        dropout_prob=0.1,    # Dropout for regularization
        learning_rate=1e-4,  # Learning rate for SGD
        normalize_embeddings=True,  # L2 normalization 
        
        # Data parameters (using CLIP embeddings)
        train_embeddings_file="src/data/embeddings/X_train_clip.pt",
        val_embeddings_file="src/data/embeddings/X_val_clip.pt",
        test_embeddings_file="src/data/embeddings/X_test_clip.pt",
        batch_size=1024,     # Batch size
        num_workers=8,       # CPU workers
        
        # Training parameters
        max_epochs=100,      # Maximum epochs
        accelerator="auto",  # Auto-detect CPU/GPU
        devices="auto",      # Auto-detect devices
        precision="32-true", # Full precision for stability
        
        # Logging and checkpointing
        log_dir="logs",
        experiment_name="triplet_taste",
        save_top_k=3,        # Save 3 best models
        monitor_metric="val_accuracy",  # Monitor accuracy instead of loss
        monitor_mode="max",  # Maximize accuracy
        patience=10,         # Early stopping patience
        
        # Other parameters
        seed=42,
    )
    
    # Run training
    print("Starting training with default configuration...")
    results = trainer.train()
    
    print("\n" + "="*60)
    print("Training Summary:")
    print("="*60)
    print(f"Best model path: {results['best_model_path']}")
    print(f"Best validation loss: {results['best_model_score']:.4f}")
    print(f"Total epochs: {results['total_epochs']}")
    print(f"Test results: {results['test_results']}")
    if results.get('metrics_file'):
        print(f"Metrics saved to: {results['metrics_file']}")
    print("\nTo analyze training results, open the analysis notebook:")
    print("jupyter notebook analysis.ipynb")
    
    return results


if __name__ == "__main__":
    main() 