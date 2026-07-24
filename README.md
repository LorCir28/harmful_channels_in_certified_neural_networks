# Harmful Channels in Certified Neural Networks

This repository is for the purpose of paper review submission and contains the codebase and experimental workflows for analyzing the existence of harmful channels in convolutional certified neural networks.

This project uses the CTRAIN library for certification training and evaluation: https://github.com/ADA-research/CTRAIN

---

# Note on Usage

Permission is granted exclusively to reviewers and editors of this paper submission for the purpose of review. All rights reserved.


---

## Requirements & Setup

### Prerequisites
Make sure system dependencies and packages are installed. You can set up the environment with the following commands:

```bash
# System updates and Git installation
apt update
apt install git

# Initialize submodules
git submodule init
git submodule update

# Install specialized dependencies without dependencies override
pip install --no-deps git+https://github.com/KaidiXu/onnx2pytorch@8447c42c3192dad383e5598edc74dddac5706ee2
pip install --no-deps git+https://github.com/Verified-Intelligence/auto_LiRPA.git@cf0169ce6bfb4fddd82cfff5c259c162a23ad03c

# Install ctrain package in editable mode with development dependencies
pip install -e ".[dev]"

# Install additional required Python packages
pip install appdirs
pip install graphviz
pip install smac
pip install onnx
pip install onnxruntime
pip install onnxoptimizer
pip install skl2onnx
```

---

## Repository Structure

```
cert_rob_github/
├── ctrain/               # Core Python package containing models, bounds, losses, wrappers, and utilities
├── data/                 # Datasets directory (e.g., CIFAR-10, MNIST)
├── jsons/                # Pre-computed channel scores and configuration output files
├── model_weights/        # Trained model checkpoints (.pt)
├── source/               # Experimental scripts
│   ├── training/         # Certified training scripts (CNN7 & ResNet18 on MNIST & CIFAR-10)
│   ├── scoring/          # Channel bound-impact / bounds-explosion scoring scripts
│   ├── regularization/   # Selective L2 regularization training scripts using computed scores
│   ├── zero_ablation/    # Zero-ablation analysis scripts
│   └── testing/          # Standard and certified accuracy evaluation / testing scripts
├── setup.py              # Package setup file
└── pyproject.toml        # Project metadata and build configurations
```

### Directory Details
- **`ctrain/`**: Core library supporting interval bound calculations, custom loss functions, model definitions (`CNN7`, `ResNet18`), and data loaders.
- **`source/training/`**: Standard baseline training scripts for certified robustness methods (**IBP**, **CROWN-IBP**, **MTL-IBP**, **SABR**).
- **`source/scoring/`**: Scripts that compute scores for individual channels across convolutional layers.
- **`source/regularization/`**: Targeted training scripts that apply channel-specific weighted L2 regularization based on computed scores to suppress bound explosion.
- **`source/zero_ablation/`**: Evaluation scripts that identify the harmful channels.
- **`source/testing/`**: Scripts to test standard and certified accuracy on trained model checkpoints.
- **`jsons/`**: Stores output JSON files containing channel scores across models and layers and harmful channels list.
- **`model_weights/`**: Directory where `.pt` checkpoint files are saved and loaded.

---

## Execution & Workflow Commands

### 1. Baseline Certified Training
Train certified models (i.e. CNN7 or ResNet18 on CIFAR-10 / MNIST) using IBP, CROWN-IBP, MTL-IBP, and SABR methods:

```bash
# Train CNN7 on CIFAR-10 using CROWN-IBP
python source/training/cnn7/cifar/cnn7_cifar_crown_ibp_training.py

# Train CNN7 on CIFAR-10 using IBP
python source/training/cnn7/cifar/cnn7_cifar_ibp_training.py

# Train CNN7 on CIFAR-10 using MTL-IBP
python source/training/cnn7/cifar/cnn7_cifar_mtl_ibp_training.py

# Train CNN7 on CIFAR-10 using SABR
python source/training/cnn7/cifar/cnn7_cifar_sabr_training.py
```
*(Similar training scripts exist under `source/training/cnn7/mnist/`, `source/training/resnet18/cifar/`, and `source/training/resnet18/mnist/`)*

---

### 2. Compute Channel Impact Scores
Run scoring scripts on trained baseline checkpoints to identify channels that contribute significantly to bound explosion:

```bash
# Score CNN7 channels on CIFAR-10
python source/scoring/cnn7_cifar_scores.py

# Score CNN7 channels on MNIST
python source/scoring/cnn7_mnist_scores.py

# Score ResNet18 channels on CIFAR-10
python source/scoring/resnet18_cifar_scores.py

# Score ResNet18 channels on MNIST
python source/scoring/resnet18_mnist_scores.py
```

---

### 3. Targeted Regularization Training
Train models using targeted L2 channel regularization based on the generated channel scores:

```bash
# CNN7 on CIFAR-10 with targeted L2 regularization
python source/regularization/cnn7/cifar/cnn7_cifar_crown_ibp_l2.py
python source/regularization/cnn7/cifar/cnn7_cifar_ibp_l2.py

# ResNet18 on CIFAR-10 with targeted L2 regularization
python source/regularization/resnet18/cifar/resnet18_cifar_crown_ibp_l2.py
```

---

### 4. Evaluation & Zero-Ablation Analysis
Evaluate standard and certified performance, or analyze the effect of channel zero-ablation:

```bash
# Evaluate CNN7 zero-ablation on CIFAR-10
python source/zero_ablation/cnn7/cifar/cnn7_cifar_crown_ibp_zero_ablation.py

# Test standard and certified accuracy on model checkpoints
python source/testing/cnn7/cifar/cnn7_cifar_test.py
```
