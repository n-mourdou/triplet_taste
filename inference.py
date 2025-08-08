#!/usr/bin/env python3
"""
Food Image Similarity Inference Script

This script allows you to compare two food images and predict if they belong
to the same category using a trained triplet loss model.

Command line usage:
    python inference.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg --model logs/checkpoints/best_model.ckpt
    python inference.py --image1 https://example.com/pizza.jpg --image2 https://example.com/burger.jpg --model logs/checkpoints/best_model.ckpt

Programmatic usage:
    from inference import predict_similarity
    
    predict_similarity(
        image1_path="path/to/image1.jpg",
        image2_path="path/to/image2.jpg", 
        model_path="logs/checkpoints/best_model.ckpt"
    )
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.inference.predictor import FoodSimilarityPredictor


def predict_similarity(
    image1_path: str,
    image2_path: str, 
    model_path: str,
    threshold: float = 0.5,
    show_details: bool = False,
    device: str = "auto"
):
    """
    Compare two food images and predict if they're in the same category.
    
    Args:
        image1_path: Path to first image (file path or URL)
        image2_path: Path to second image (file path or URL)
        model_path: Path to trained model checkpoint
        threshold: Similarity threshold for binary classification (0-1)
        show_details: Whether to show detailed results
        device: Device to use ("auto", "cpu", "cuda")
    """
    # Validate threshold
    if not 0 <= threshold <= 1:
        print("Error: Threshold must be between 0 and 1")
        return
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model checkpoint not found: {model_path}")
        print("Make sure to train a model first using: python train.py")
        return
    
    try:
        # Initialize predictor
        print("Initializing food similarity predictor...")
        predictor = FoodSimilarityPredictor(
            model_checkpoint_path=model_path,
            similarity_threshold=threshold,
            device=device
        )
        
        # Show model info
        model_info = predictor.get_model_info()
        print(f"Model loaded: {model_info['model_parameters']:,} parameters")
        print(f"Similarity threshold: {threshold}")
        print(f"Device: {model_info['device']}")
        
        print("\n" + "="*60)
        print("Comparing Food Images")
        print("="*60)
        
        # Make prediction
        result = predictor.predict_same_category(
            image1_path, 
            image2_path, 
            return_details=show_details
        )
        
        print("\n" + "="*60)
        print("Results")
        print("="*60)
        
        if show_details:
            # Show detailed results
            print(f"Image 1: {result['image1_source']}")
            print(f"Image 2: {result['image2_source']}")
            print(f"Image 1 Size: {result['image1_size']}")
            print(f"Image 2 Size: {result['image2_size']}")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
            print(f"Threshold: {result['threshold']}")
            print(f"Similarity Formula: 1 / (1 + euclidean_distance)")
            print(f"Similarity Range: [0.0, 1.0] (higher = more similar)")
            
            # Final prediction
            if result['same_category']:
                print(f"Prediction: SAME CATEGORY (similarity: {result['similarity_score']:.4f} >= {result['threshold']})")
            else:
                print(f"Prediction: DIFFERENT CATEGORIES (similarity: {result['similarity_score']:.4f} < {result['threshold']})")
        else:
            if result:
                print("Prediction: Images are likely from the SAME CATEGORY")
            else:
                print("Prediction: Images are likely from DIFFERENT CATEGORIES")
        
        print("\nTips:")
        print("  - Higher similarity scores indicate more similar images")
        print("  - You can adjust the threshold parameter")
        print("  - Use show_details=True for more information about the prediction")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return False
    
    return True


def batch_inference(image_pairs, model_path, threshold=0.5, device="auto"):
    """
    Run batch inference on multiple image pairs.
    
    Args:
        image_pairs: List of (image1_path, image2_path) tuples
        model_path: Path to trained model checkpoint
        threshold: Similarity threshold
        device: Device to use
    """
    print("Batch inference")
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Please train a model first using: python train.py")
        return
    
    predictor = FoodSimilarityPredictor(model_path, similarity_threshold=threshold, device=device)
    
    # Run batch prediction
    results = predictor.batch_predict(image_pairs, return_details=True)
    
    # Display results
    print("\n" + "="*60)
    print("Batch Results")
    print("="*60)
    
    for i, result in enumerate(results):
        if result and 'same_category' in result:
            status = "SAME" if result['same_category'] else "DIFFERENT"
            print(f"Pair {i+1}: {status} (similarity: {result['similarity_score']:.4f})")
        else:
            print(f"Pair {i+1}: ERROR")
    
    return results


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Compare two food images and predict if they're in the same category",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare two local images
    python inference.py --image1 food/00001.jpg --image2 food/00002.jpg --model logs/checkpoints/best_model.ckpt
    
    # Compare images from URLs
    python inference.py --image1 https://example.com/pizza.jpg --image2 https://example.com/burger.jpg --model logs/checkpoints/best_model.ckpt
    
    # Get detailed results
    python inference.py --image1 food/00001.jpg --image2 food/00002.jpg --model logs/checkpoints/best_model.ckpt --details
    
    # Use custom threshold
    python inference.py --image1 food/00001.jpg --image2 food/00002.jpg --model logs/checkpoints/best_model.ckpt --threshold 0.7
        """
    )
    
    # Required arguments
    parser.add_argument("--image1", required=True, help="First image (file path or URL)")
    parser.add_argument("--image2", required=True, help="Second image (file path or URL)")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    
    # Optional arguments
    parser.add_argument("--threshold", type=float, default=0.5, 
                       help="Similarity threshold for binary classification (0-1, default: 0.5)")
    parser.add_argument("--details", action="store_true", 
                       help="Show detailed results including confidence scores")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to use for inference (default: auto)")
    
    args = parser.parse_args()
    
    # Run prediction using the function
    success = predict_similarity(
        image1_path=args.image1,
        image2_path=args.image2,
        model_path=args.model,
        threshold=args.threshold,
        show_details=args.details,
        device=args.device
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main()) 