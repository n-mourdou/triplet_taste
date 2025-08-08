"""
Food similarity prediction using trained triplet loss model with CLIP embeddings.
"""

import os
import sys
from pathlib import Path
from typing import Union, Tuple, Dict, Any, List
import requests
from io import BytesIO
import numpy as np

import torch
import torch.nn as nn
import clip
from PIL import Image

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.triplet_model import TripletModel


class FoodSimilarityPredictor:
    """
    Inference class for predicting food image similarity using trained triplet loss model.
    
    Can load images from URLs or file paths, extract CLIP embeddings, and predict
    if two food images belong to the same category.
    
    This version works with the migrated approach using CLIP embeddings.
    """
    
    def __init__(
        self,
        model_checkpoint_path: str,
        similarity_threshold: float = 0.5,
        device: str = "auto",
    ):
        """
        Initialize the food similarity predictor.
        
        Args:
            model_checkpoint_path: Path to trained TripletModel checkpoint
            similarity_threshold: Threshold for binary classification (0-1)
            device: Device to use ("auto", "cpu", "cuda")
        """
        self.model_checkpoint_path = Path(model_checkpoint_path)
        self.similarity_threshold = similarity_threshold
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.clip_model = None
        self.clip_preprocess = None
        self.triplet_model = None
        
        # Setup inference pipeline
        self._setup_clip_encoder()
        self._load_triplet_model()
        
        print("Food similarity predictor initialized successfully!")
    
    def _setup_clip_encoder(self):
        """Setup CLIP encoder for feature extraction."""
        print("Loading CLIP encoder...")
        
        # Load CLIP model (ViT-L/14 for 768-dimensional embeddings)
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=self.device)
        self.clip_model.eval()
        
        # Freeze parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        print("CLIP ViT-L/14 encoder loaded successfully (768-dim embeddings)")
    
    def _load_triplet_model(self):
        """Load trained triplet loss model from checkpoint."""
        if not self.model_checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_checkpoint_path}")
        
        print(f"Loading triplet model from: {self.model_checkpoint_path}")
        
        # Load model from checkpoint
        self.triplet_model = TripletModel.load_from_checkpoint(
            str(self.model_checkpoint_path),
            map_location=self.device
        )
        
        self.triplet_model.eval()
        self.triplet_model = self.triplet_model.to(self.device)
        
        print("Triplet model loaded successfully")
        print(f"Model parameters: {self.triplet_model.num_trainable_params:,}")
    
    def load_image_from_url(self, url: str) -> Image.Image:
        """
        Load image from URL.
        
        Args:
            url: Image URL
            
        Returns:
            PIL Image object
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to load image from URL {url}: {str(e)}")
    
    def load_image_from_path(self, path: Union[str, Path]) -> Image.Image:
        """
        Load image from file path.
        
        Args:
            path: Path to image file
            
        Returns:
            PIL Image object
        """
        try:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            
            image = Image.open(path).convert('RGB')
            return image
            
        except Exception as e:
            raise ValueError(f"Failed to load image from path {path}: {str(e)}")
    
    def load_image(self, source: Union[str, Path]) -> Image.Image:
        """
        Load image from URL or file path.
        
        Args:
            source: Image URL or file path
            
        Returns:
            PIL Image object
        """
        source_str = str(source)
        
        # Check if it's a URL
        if source_str.startswith(('http://', 'https://')):
            return self.load_image_from_url(source_str)
        else:
            return self.load_image_from_path(source)
    
    def extract_clip_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Extract CLIP embedding from image.
        
        Args:
            image: PIL Image object
            
        Returns:
            768-dimensional embedding tensor
        """
        # Preprocess image
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.clip_model.encode_image(image_tensor)
            embedding = embedding.float().squeeze()
        
        return embedding
    
    def get_learned_embedding(self, clip_embedding: torch.Tensor) -> torch.Tensor:
        """
        Get learned embedding from CLIP embedding using the trained model.
        
        Args:
            clip_embedding: CLIP embedding tensor (768-dim)
            
        Returns:
            Learned embedding tensor (256-dim)
        """
        with torch.no_grad():
            learned_embedding = self.triplet_model.get_embedding(clip_embedding.unsqueeze(0))
            return learned_embedding.squeeze()
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute similarity between two learned embeddings.
        
        Args:
            embedding1: First learned embedding
            embedding2: Second learned embedding
            
        Returns:
            Similarity score (higher = more similar)
        """
        # Compute Euclidean distance
        distance = torch.norm(embedding1 - embedding2, p=2).item()
        
        # Convert to similarity score (inverse of distance)
        similarity = 1.0 / (1.0 + distance)
        
        return similarity
    
    def predict_same_category(
        self, 
        image1_source: Union[str, Path], 
        image2_source: Union[str, Path],
        return_details: bool = False
    ) -> Union[bool, Dict[str, Any]]:
        """
        Predict if two images belong to the same food category.
        
        Args:
            image1_source: First image (URL or file path)
            image2_source: Second image (URL or file path)
            return_details: Whether to return detailed results
            
        Returns:
            Boolean prediction or detailed results dict
        """
        try:
            # Load images
            print(f"Loading image 1: {image1_source}")
            image1 = self.load_image(image1_source)
            
            print(f"Loading image 2: {image2_source}")
            image2 = self.load_image(image2_source)
            
            # Extract CLIP embeddings
            print("Extracting CLIP embeddings...")
            clip_emb1 = self.extract_clip_embedding(image1)
            clip_emb2 = self.extract_clip_embedding(image2)
            
            # Get learned embeddings
            print("Computing learned embeddings...")
            learned_emb1 = self.get_learned_embedding(clip_emb1)
            learned_emb2 = self.get_learned_embedding(clip_emb2)
            
            # Compute similarity
            similarity_score = self.compute_similarity(learned_emb1, learned_emb2)
            
            # Make prediction
            same_category = similarity_score >= self.similarity_threshold
            
            if return_details:
                return {
                    'same_category': same_category,
                    'similarity_score': similarity_score,
                    'threshold': self.similarity_threshold,
                    'image1_source': str(image1_source),
                    'image2_source': str(image2_source),
                    'image1_size': image1.size,
                    'image2_size': image2.size,
                }
            else:
                return same_category
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            if return_details:
                return {
                    'same_category': False,
                    'error': str(e)
                }
            else:
                return False
    
    def batch_predict(
        self, 
        image_pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
        return_details: bool = False
    ) -> List[Union[bool, Dict[str, Any]]]:
        """
        Predict similarity for multiple image pairs.
        
        Args:
            image_pairs: List of (image1, image2) pairs
            return_details: Whether to return detailed results
            
        Returns:
            List of predictions
        """
        results = []
        
        for i, (image1, image2) in enumerate(image_pairs):
            print(f"Processing pair {i+1}/{len(image_pairs)}")
            try:
                result = self.predict_same_category(image1, image2, return_details)
                results.append(result)
            except Exception as e:
                print(f"Error processing pair {i+1}: {str(e)}")
                if return_details:
                    results.append({'same_category': False, 'error': str(e)})
                else:
                    results.append(False)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict with model information
        """
        return {
            'model_checkpoint_path': str(self.model_checkpoint_path),
            'model_parameters': self.triplet_model.num_trainable_params,
            'device': str(self.device),
            'similarity_threshold': self.similarity_threshold,
            'embedding_dimension': self.triplet_model.input_dim,
            'output_dimension': self.triplet_model.hidden_dim,
        }