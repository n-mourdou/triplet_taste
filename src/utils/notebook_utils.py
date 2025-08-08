"""
Utility functions for the triplet learning demonstration notebook.

This module contains helper functions for data loading and visualization
used in the educational notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Optional, Sequence, Set


def load_triplet_data(file_path: str) -> pd.DataFrame:
    """
    Load triplet data from file with error handling.
    
    Args:
        file_path: Path to the triplet file
        
    Returns:
        DataFrame with columns: anchor, positive, negative
    """
    data = []
    skipped_lines = 0
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Split by whitespace (space or tab)
                parts = line.split()
                if len(parts) >= 3:
                    anchor, positive, negative = parts[0], parts[1], parts[2]
                    data.append({
                        'anchor': anchor.strip(),
                        'positive': positive.strip(),
                        'negative': negative.strip()
                    })
                else:
                    print(f"Warning: Skipping line {line_num} in {file_path}: '{line}' (expected 3 values, got {len(parts)})")
                    skipped_lines += 1
                    
        if skipped_lines > 0:
            print(f"Skipped {skipped_lines} invalid lines in {file_path}")
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return pd.DataFrame()
        
    return pd.DataFrame(data)


def plot_single_triplet_before_after():
    """Create 2D plots showing a single triplet before and after training."""
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # BEFORE TRAINING: Random positions
    anchor_before = np.array([0, 0])
    positive_before = np.array([1.7, 1.3])  # Random position
    negative_before = np.array([-0.8, 1.8])  # Random position
    
    # Plot before training
    ax1.scatter(*anchor_before, c='blue', s=200, label='Anchor', marker='o', edgecolor='black', linewidth=2)
    ax1.scatter(*positive_before, c='green', s=200, label='Positive (same category)', marker='^', edgecolor='black', linewidth=2)
    ax1.scatter(*negative_before, c='red', s=200, label='Negative (different category)', marker='s', edgecolor='black', linewidth=2)
    
    # Draw distance lines (before)
    ax1.plot([anchor_before[0], positive_before[0]], [anchor_before[1], positive_before[1]], 
             'g-', linewidth=3, alpha=0.7, label=f'd(a,p) = {np.linalg.norm(positive_before - anchor_before):.2f}')
    ax1.plot([anchor_before[0], negative_before[0]], [anchor_before[1], negative_before[1]], 
             'r-', linewidth=3, alpha=0.7, label=f'd(a,n) = {np.linalg.norm(negative_before - anchor_before):.2f}')
    
    ax1.set_title('Before Training', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Embedding Dimension 1', fontsize=12)
    ax1.set_ylabel('Embedding Dimension 2', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    
    # AFTER TRAINING: Optimized positions
    anchor_after = np.array([0, 0])  # Anchor stays as reference
    positive_after = np.array([0.4, 0.3])  # Much closer to anchor
    negative_after = np.array([-2.1, 1.8])  # Much farther from anchor
    
    # Plot after training
    ax2.scatter(*anchor_after, c='blue', s=200, label='Anchor', marker='o', edgecolor='black', linewidth=2)
    ax2.scatter(*positive_after, c='green', s=200, label='Positive (same category)', marker='^', edgecolor='black', linewidth=2)
    ax2.scatter(*negative_after, c='red', s=200, label='Negative (different category)', marker='s', edgecolor='black', linewidth=2)
    
    # Draw distance lines (after)
    ax2.plot([anchor_after[0], positive_after[0]], [anchor_after[1], positive_after[1]], 
             'g-', linewidth=3, alpha=0.7, label=f'd(a,p) = {np.linalg.norm(positive_after - anchor_after):.2f}')
    ax2.plot([anchor_after[0], negative_after[0]], [anchor_after[1], negative_after[1]], 
             'r-', linewidth=3, alpha=0.7, label=f'd(a,n) = {np.linalg.norm(negative_after - anchor_after):.2f}')
    
    ax2.set_title('After Training', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Embedding Dimension 1', fontsize=12)
    ax2.set_ylabel('Embedding Dimension 2', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Print the learning outcome
    pos_dist_before = np.linalg.norm(positive_before - anchor_before)
    neg_dist_before = np.linalg.norm(negative_before - anchor_before)
    pos_dist_after = np.linalg.norm(positive_after - anchor_after)
    neg_dist_after = np.linalg.norm(negative_after - anchor_after)
    
    print("Triplet Learning Outcome:")
    print("=" * 40)
    print(f"BEFORE: d(a,p) = {pos_dist_before:.2f}, d(a,n) = {neg_dist_before:.2f}")
    print(f"        Correctly ordered? {pos_dist_before < neg_dist_before}")
    print(f"AFTER:  d(a,p) = {pos_dist_after:.2f}, d(a,n) = {neg_dist_after:.2f}")
    print(f"        Correctly ordered? {pos_dist_after < neg_dist_after}")
    print(f"\nGoal achieved: d(a,p) < d(a,n) ✓")


def visualize_triplet(anchor_id: str, positive_id: str, negative_id: str, 
                     images_dir: str = 'src/data/food_images/') -> None:
    """
    Visualize a triplet of images.
    
    Args:
        anchor_id: ID of anchor image
        positive_id: ID of positive image  
        negative_id: ID of negative image
        images_dir: Directory containing images
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Load and display images
    images = [anchor_id, positive_id, negative_id]
    titles = ['Anchor', 'Positive (Same Category)', 'Negative (Different Category)']
    colors = ['blue', 'green', 'red']
    
    for i, (img_id, title, color) in enumerate(zip(images, titles, colors)):
        try:
            img_path = Path(images_dir) / f"{img_id}.jpg"
            if img_path.exists():
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(f"{title}\nID: {img_id}", color=color, fontweight='bold')
            else:
                axes[i].text(0.5, 0.5, f"Image {img_id}\nnot found", 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{title}\nID: {img_id}", color=color, fontweight='bold')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading\n{img_id}", 
                       ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"{title}\nID: {img_id}", color=color, fontweight='bold')
        
        axes[i].axis('off')
    
    plt.suptitle('Triplet Example: Learning Relative Similarity', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def ids_from_df(df: pd.DataFrame, triplet_cols: Optional[Sequence[str]] = None) -> Set[str]:
    """
    Extract unique sample IDs from a triplet DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame where each row contains a triplet of IDs (e.g., anchor, positive, negative).
    triplet_cols : Sequence[str], optional
        Column names holding the triplet IDs (e.g., ["anchor", "positive", "negative"]).
        If None, the first three columns of `df` are used.

    Returns
    -------
    set[str]
        Unique IDs as strings (leading zeros preserved). NaNs are ignored.

    Notes
    -----
    - Values are cast to str to avoid int/string mismatches.
    - If none of the specified columns exist, returns an empty set.
    """
    if triplet_cols is None:
        triplet_cols = list(df.columns[:3])
    series_list = [df[c].dropna().astype(str) for c in triplet_cols if c in df.columns]
    if not series_list:
        return set()
    return set(pd.unique(pd.concat(series_list, ignore_index=True)))