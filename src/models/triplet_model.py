"""
PyTorch Lightning model for triplet loss learning on food image embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np


class TripletModel(pl.LightningModule):
    """
    PyTorch Lightning model for triplet loss learning.
    
    Takes precomputed CLIP embeddings in triplet format [anchor, positive, negative]
    and applies learned transformations to optimize triplet loss for food similarity.
    
    - Architecture: 768 → 256 → 256
    - L2 normalization of final embeddings
    - SGD optimizer
    - Triplet loss
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # CLIP embedding dimension
        hidden_dim: int = 256,  # Hidden layer dimension
        margin: float = 0.3,  # Triplet loss margin (alpha)
        dropout_prob: float = 0.3,  # Dropout probability
        learning_rate: float = 1e-4,  # Learning rate for SGD
        normalize_embeddings: bool = True,  # L2 normalize final embeddings
    ):
        """
        Initialize the triplet loss model.
        
        Args:
            input_dim: Input embedding dimension (768 for CLIP)
            hidden_dim: Hidden layer dimension 
            margin: Triplet loss margin (alpha)
            dropout_prob: Dropout probability
            learning_rate: Learning rate for SGD optimizer
            normalize_embeddings: Whether to L2 normalize final embeddings
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.margin = margin
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.normalize_embeddings = normalize_embeddings
        
        # Build the  embedding network
        self.embedding_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 768 → 256
            nn.PReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.AlphaDropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),  # 256 → 256
        )
            
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedding network for triplet input.
        
        Args:
            x: Input tensor of shape (batch_size, 3, input_dim) - triplet format
            
        Returns:
            Stacked embeddings of shape (3, batch_size, hidden_dim) - [anchor, positive, negative]
        """
        # Extract anchor, positive, negative from triplet input
        x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]  # Each is (batch_size, input_dim)
        
        # Apply embedding network to each component
        x0 = self.embedding_network(x0)
        x1 = self.embedding_network(x1) 
        x2 = self.embedding_network(x2)
        
        # L2 normalize embeddings if enabled
        if self.normalize_embeddings:
            x0 = F.normalize(x0, p=2, dim=1)
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
        
        # Return stacked triplet embeddings: (3, batch_size, hidden_dim)
        return torch.stack((x0, x1, x2))
    
    def compute_distance(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean (L2) distance between two embeddings.
        
        Args:
            emb1: First embedding tensor
            emb2: Second embedding tensor
            
        Returns:
            Distance tensor
        """
        return torch.norm(emb1 - emb2, p=2, dim=1)
    
    def triplet_loss(self, output: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            output: Stacked embeddings from forward pass (3, batch_size, hidden_dim)
            
        Returns:
            Triplet loss tensor
        """
        anchor, positive, negative = output[0], output[1], output[2]
        
        # Compute distances
        dp = self.compute_distance(anchor, positive)  # Distance anchor-positive
        dn = self.compute_distance(anchor, negative)  # Distance anchor-negative
        
        # Triplet loss: max(0, dp - dn + margin)
        loss = torch.relu(dp - dn + self.margin).mean()
        
        return loss
    
    def compute_accuracy(self, output: torch.Tensor, use_margin: bool = True) -> torch.Tensor:
        """
        Compute triplet accuracy with optional margin.
        
        Args:
            output: Stacked embeddings from forward pass (3, batch_size, hidden_dim)
            use_margin: If True, use training margin; if False, use strict comparison
            
        Returns:
            Accuracy tensor
        """
        with torch.no_grad():
            anchor, positive, negative = output[0], output[1], output[2]
            
            dp = self.compute_distance(anchor, positive)
            dn = self.compute_distance(anchor, negative)
            
            if use_margin:
                # Training/validation: use margin for robustness
                correct = (dp < dn + self.margin).float()
            else:
                # Test/inference: strict comparison (literature standard)
                correct = (dp < dn).float()
            
            return correct.mean()
    
    def training_step(self, batch, batch_idx: int):
        """
        Training step.
        
        Args:
            batch: Tuple containing (triplet_embeddings,) where triplet_embeddings
                   has shape (batch_size, 3, input_dim)
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        triplet_embeddings = batch[0]  # Shape: (batch_size, 3, input_dim)
        
        # Forward pass
        output = self.forward(triplet_embeddings)  # Shape: (3, batch_size, hidden_dim)
        
        # Compute loss and accuracy
        loss = self.triplet_loss(output)
        accuracy = self.compute_accuracy(output.detach())
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        triplet_embeddings = batch[0]  # Shape: (batch_size, 3, input_dim)
        
        # Forward pass
        output = self.forward(triplet_embeddings)  # Shape: (3, batch_size, hidden_dim)
        
        # Compute loss and accuracy
        loss = self.triplet_loss(output)
        accuracy = self.compute_accuracy(output.detach())
        
        # Log metrics  
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        triplet_embeddings = batch[0]  # Shape: (batch_size, 3, input_dim)
        
        # Forward pass
        output = self.forward(triplet_embeddings)  # Shape: (3, batch_size, hidden_dim)
        
        # Compute loss and accuracy
        loss = self.triplet_loss(output)
        accuracy = self.compute_accuracy(output.detach())
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure AdamW optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )
        
        return {'optimizer': optimizer}
    
    def predict_triplet(self, triplet_embeddings: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
        """
        Predict if anchor-positive distance < anchor-negative distance for triplets.
        
        Args:
            triplet_embeddings: Input triplet embeddings (batch_size, 3, input_dim)
            alpha: Additional margin for prediction (default 0.0 for strict comparison)
            
        Returns:
            Binary predictions (1 if dp < dn + alpha, 0 otherwise)
        """
        with torch.no_grad():
            output = self.forward(triplet_embeddings)
            anchor, positive, negative = output[0], output[1], output[2]
            
            dp = self.compute_distance(anchor, positive)
            dn = self.compute_distance(anchor, negative)
            
            # Strict comparison for inference (literature standard): dp < dn + alpha
            predictions = (dp < dn + alpha).int()
            
            return predictions
    
    @property
    def num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        params = filter(lambda p: p.requires_grad, self.parameters())
        return sum([np.prod(p.size()) for p in params])
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for individual input tensor (single embedding, not triplet).
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            Embedding tensor (batch_size, hidden_dim)
        """
        with torch.no_grad():
            embedding = self.embedding_network(x)
            
            if self.normalize_embeddings:
                embedding = F.normalize(embedding, p=2, dim=1)
                
            return embedding 