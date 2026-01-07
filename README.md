# PPI-ViT: A Structural Point Cloud Imaging Approach for Protein Interaction Prediction with Vision Transformers

A deep learning framework for predicting protein-protein interactions using 3D structural information. This project converts 3D protein structures into 2D images and uses Vision Transformer models for binary classification.

## Table of Contents

- [Overview](#overview)
- [installation](#requirements)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)


## Overview

This project implements a novel approach for protein-protein interaction prediction by:

1. **Parsing PDB files** to extract 3D atomic coordinates
2. **Projecting 3D structures to 2D images** using various projection methods
3. **Training CNN models** for binary classification (interacting vs non-interacting)

The main advantage of this approach is that it leverages spatial structural information while being compatible with standard computer vision architectures.


## Installation

### Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Namyanyo/PPI-ViT.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

The project requires the following main packages:
- PyTorch >= 2.0.0
- BioPython >= 1.79 (for PDB parsing)
- NumPy, SciPy, scikit-learn
- Matplotlib, Pillow
- TensorBoard (for training visualization)
- tqdm (for progress bars)

## Data Preparation

### Data Format

The dataset should be in JSON format with the following structure:

```json
[
  {
    "pdb_path": "path/to/protein1.pdb",
    "label": 1
  },
  {
    "pdb_path": "path/to/protein2.pdb",
    "label": 0
  }
]
```

Where:
- `pdb_path`: Path to the PDB structure file
- `label`: Binary label (1 = interacting, 0 = non-interacting)


## Usage

### Training

Train a model using the training script:

```bash
python train.py \
    --train_file dataset/train.json \
    --val_file dataset/val.json \
    --model vit_base \
    --image_size 224 \
    --batch_size 128 \
    --num_epochs 100 \
    --learning_rate 0.0001 \
    --checkpoint_dir checkpoints
```

#### Key Training Arguments

- `--model`: Model architecture (`cnn` or `resnet`)
- `--image_size`: Size of projected images (default: 224)
- `--projection_method`: Projection method (`density_map`, `scatter`, or `voxel`)
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Initial learning rate
- `--optimizer`: Optimizer (`adam` or `sgd`)
- `--cache_dir`: Directory to cache preprocessed images (speeds up training)



### Prediction

Use a trained model to predict protein interactions:


```bash
python predict.py \
    --checkpoint checkpoints/best_model.pth \
    --pdb_file path/to/protein.pdb \
    --output predictions.json
```


## License

This project is licensed under the MIT License.


**Note**: This is a research tool. Results should be validated with experimental methods before drawing biological conclusions.
