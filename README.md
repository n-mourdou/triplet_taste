# Triplet Taste

A deep learning project for food image similarity using triplet loss and CLIP embeddings. This project trains a neural network to determine whether two food images belong to the same category by learning from triplet data (anchor, positive, negative examples).

## Overview

The project combines pre-trained CLIP embeddings with a custom triplet loss neural network to create a food image similarity model. The approach ensures robust feature learning by leveraging CLIP's visual understanding while fine-tuning for food-specific similarity tasks.

### Key Features

- **CLIP Integration**: Uses OpenAI's CLIP model for high-quality image embeddings
- **Triplet Loss Learning**: Learns similarity through triplet relationships
- **Data Leakage Prevention**: Ensures proper train/validation splits at the image level
- **GPU/CPU Support**: Automatic device detection with optimized training
- **Comprehensive Evaluation**: Detailed metrics and analysis tools
- **Production Ready**: Easy-to-use inference API for new images

## Architecture

The system consists of three main components:

1. **Data Pipeline**: Processes triplet data and extracts CLIP embeddings
2. **Triplet Network**: A neural network trained with triplet loss on CLIP embeddings
3. **Inference Engine**: Predicts similarity between new food images

### Model Architecture

```
Input: CLIP Embeddings (768-dim)
    ↓
Hidden Layer (256-dim) + Dropout + ReLU
    ↓
Hidden Layer (128-dim) + Dropout + ReLU  
    ↓
Output Embedding (64-dim) + L2 Normalization
    ↓
Triplet Loss (margin=0.4)
```

## Project Structure

```
triplet-taste/
├── src/
│   ├── data/
│   │   ├── create_splits.py          # Train/validation splitting
│   │   ├── extract_clip_embeddings.py # CLIP embedding extraction
│   │   ├── food_datamodule.py        # PyTorch Lightning data module
│   │   ├── embeddings/               # Generated CLIP embeddings (LFS)
│   │   ├── food_images/              # Food image dataset
│   │   └── splits/                   # Train/val/test triplet files
│   ├── models/
│   │   └── triplet_model.py          # Triplet loss neural network
│   ├── training/
│   │   └── trainer.py                # Training logic and configuration
│   └── inference/
│       └── predictor.py              # Inference and similarity prediction
├── logs/                             # Training logs and checkpoints
├── train.py                          # Main training script
├── inference.py                      # Main inference script  
├── create_embeddings.py              # Complete embedding pipeline
├── pyproject.toml                    # Poetry dependencies
├── LICENSE                           # MIT license
└── README.md                         # This file
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, but recommended)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/n-mourdou/triplet_taste_v3.git
cd triplet_taste_v3
```

2. **Install dependencies using Poetry**:
```bash
# Install Poetry if you haven't already
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

3. **Activate the environment**:
```bash
poetry env activate
```

4. **Download food images**:
Download the food images from [Google Drive](https://drive.google.com/drive/folders/13y7IQ7GqZXWgEyK2z1aH7TabCQQFeAHB?usp=drive_link) and place them in `src/data/food_images/` directory.


## Quick Start

### 1. Prepare Data

#### Starting Point Files

The project requires three fundamental files to get started:

- **Food Images**: Download from [Google Drive](https://drive.google.com/drive/folders/13y7IQ7GqZXWgEyK2z1aH7TabCQQFeAHB?usp=drive_link)
- **train_triplets.txt**: Training triplet relationships (anchor,positive,negative)
- **test_triplets.txt**: Test triplet relationships

#### Included Pre-processed Files

This repository includes the processed files needed for immediate training:

- **Clean train/validation splits**: `train_triplets_clean.txt` and `val_triplets_clean.txt` (no data leakage)
- **CLIP embeddings**: Pre-extracted embeddings for all images in `src/data/embeddings/`

You can start training immediately with the included files, or regenerate them following the instructions below.

#### Data Structure

Ensure your data structure follows this format:

```
src/data/
├── food_images/              # Food images (download from Google Drive link above)
├── splits/
│   ├── train_triplets.txt    # Original training triplets
│   ├── test_triplets.txt     # Test triplets
│   ├── train_triplets_clean.txt  # Clean train split (included)
│   └── val_triplets_clean.txt    # Clean validation split (included)
└── embeddings/               # Pre-extracted CLIP embeddings (included)
    ├── X_train_clip.pt
    ├── X_val_clip.pt
    ├── X_test_clip.pt
    └── metadata_clip_split.pt
```

### 2. Create Embeddings (Optional)

**Note**: Pre-extracted embeddings are already included in the repository. You only need to run this step if you want to regenerate the embeddings or use different settings.

Generate CLIP embeddings from your triplet data:

```bash
python create_embeddings.py
```

This script will:
- Split training data into train/validation sets (preventing data leakage)
- Extract CLIP embeddings for all images
- Save embeddings as PyTorch tensors ready for training

### 3. Train the Model

Start training with default parameters:

```bash
python train.py
```

The training script uses optimized defaults:
- Batch size: 1024
- Learning rate: 1e-4
- Hidden dimensions: 256, 128
- Triplet margin: 0.4
- Early stopping: 10 epochs patience

### 4. Run Inference

Compare two food images:

```bash
python inference.py --image1 path/to/image1.jpg --image2 path/to/image2.jpg --model logs/checkpoints/best_model.ckpt
```

#### Inference Examples

**Local images**:
```bash
python inference.py \
    --image1 src/data/inference_images/steak.jpeg \
    --image2 src/data/inference_images/millefeuille.jpg \
    --model logs/checkpoints/last.ckpt \
    --details
```

**URL images**:
```bash
python inference.py \
    --image1 https://example.com/pizza.jpg \
    --image2 https://example.com/burger.jpg \
    --model logs/checkpoints/last.ckpt \
    --threshold 0.7
```


## Configuration

### Training Parameters

Key training parameters can be modified in `train.py`:

```python
trainer = TripletTrainer(
    # Model architecture
    input_dim=768,              # CLIP embedding dimension
    hidden_dim=256,             # Hidden layer size
    margin=0.4,                 # Triplet loss margin
    dropout_prob=0.1,           # Dropout rate
    learning_rate=1e-4,         # Learning rate
    
    # Training setup
    batch_size=1024,            # Batch size
    max_epochs=100,             # Maximum epochs
    patience=10,                # Early stopping patience
    
    # Hardware
    accelerator="auto",         # auto/cpu/gpu
    devices="auto",             # auto/1/2/etc
)
```

### Embedding Configuration

CLIP model and processing settings in `create_embeddings.py`:

```python
create_clip_embeddings(
    model_name="openai/clip-vit-large-patch14-336",  # CLIP model
    batch_size=256,                                  # Processing batch size
    split_ratio=0.8,                                 # Train/val split ratio
    device="auto"                                    # Processing device
)
```

## Data Format

### Triplet Files

Triplet files should contain one triplet per line in CSV format:

```
anchor_image_id,positive_image_id,negative_image_id
00001,00123,00456
00002,00124,00789
...
```

Where:
- `anchor_image_id`: Reference image
- `positive_image_id`: Image from same category as anchor
- `negative_image_id`: Image from different category than anchor

### Image Organization

Images should be organized in a single directory with consistent naming:

```
src/data/food_images/
├── 00001.jpg
├── 00002.jpg
├── 00123.jpg
└── ...
```

Image IDs in triplet files must match the filenames (without extension).

## Model Performance

The model learns to predict food image similarity with the following metrics:

- **Accuracy**: Percentage of correct same/different category predictions
- **Triplet Accuracy**: Percentage of triplets where anchor is closer to positive than negative
- **Embedding Quality**: Measured through clustering and nearest neighbor analysis

### Training Monitoring

Training progress is logged to:
- **Console**: Real-time loss and accuracy metrics
- **CSV**: Detailed epoch-by-epoch metrics in `logs/csv_logs/`
- **Checkpoints**: Best models saved in `logs/checkpoints/`

## Advanced Usage

### Programmatic Training

```python
from src.training.trainer import TripletTrainer

trainer = TripletTrainer(
    input_dim=768,
    hidden_dim=256,
    margin=0.4,
    learning_rate=1e-4,
    batch_size=512,
    max_epochs=50
)

results = trainer.train()
print(f"Best model: {results['best_model_path']}")
print(f"Test accuracy: {results['test_results']['accuracy']:.4f}")
```

### Programmatic Inference

```python
from src.inference.predictor import FoodSimilarityPredictor

# Initialize predictor
predictor = FoodSimilarityPredictor(
    model_checkpoint_path="logs/checkpoints/best_model.ckpt",
    similarity_threshold=0.5
)

# Single prediction
result = predictor.predict_same_category(
    "image1.jpg", 
    "image2.jpg", 
    return_details=True
)

print(f"Same category: {result['same_category']}")
print(f"Similarity: {result['similarity_score']:.4f}")

# Batch prediction
image_pairs = [("img1.jpg", "img2.jpg"), ("img3.jpg", "img4.jpg")]
results = predictor.batch_predict(image_pairs)
```

### Custom Data Pipeline

```python
from src.data.create_splits import TripletSplitter
from src.data.extract_clip_embeddings import extract_embeddings_from_splits

# Create custom splits
splitter = TripletSplitter(split_ratio=0.85, seed=123)
splitter.create_splits(
    train_triplets_file="custom_triplets.txt",
    output_dir="custom_splits/"
)

# Extract embeddings with custom settings
extract_embeddings_from_splits(
    train_triplets_file="custom_splits/train_triplets_clean.txt",
    val_triplets_file="custom_splits/val_triplets_clean.txt", 
    test_triplets_file="custom_splits/test_triplets.txt",
    images_dir="custom_images/",
    model_name="openai/clip-vit-base-patch32",
    batch_size=128
)
```



## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Reduce batch size
python train.py  # Edit batch_size in train.py

# Or force CPU training
python train.py  # Edit accelerator="cpu" in train.py
```

**Missing Images**:
```bash
# Ensure image paths are correct
ls src/data/food_images/ | head -5
```

**Import Errors**:
```bash
# Ensure you're in the project root directory
pwd  # Should end in triplet-taste
python -c "import src.models.triplet_model"  # Should not error
```

### Performance Tips

- Use GPU training for faster convergence
- Increase batch size if you have sufficient memory
- Use larger CLIP models (e.g., ViT-Large) for better embeddings
- Monitor validation metrics to prevent overfitting

For additional support, please open an issue on the GitHub repository.
