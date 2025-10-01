# Machine Learning Assignment 3: Active Learning

This repository contains implementations and experiments for active learning methods in both classification and regression tasks.

## Overview

The project compares different active learning strategies:
- Passive Learning: Traditional supervised learning with all labeled data
- Uncertainty-based Active Learning: Query samples based on model uncertainty
- Sensitivity-based Active Learning: Query samples based on gradient sensitivity

## Project Structure

```
├── src/
│   ├── combined_run_cls.py      # Classification experiments runner
│   ├── combined_run_reg.py      # Regression experiments runner
│   ├── compare_classification.ipynb
│   ├── compare_regression.ipynb
│   └── nn/                       # Neural network modules
│       ├── data.py              # Data loading and preprocessing
│       ├── evaluation.py        # Model evaluation metrics
│       ├── experiments.py       # Active learning experiments
│       ├── models.py            # Neural network models
│       ├── strategies.py        # Active learning strategies
│       └── training.py          # Training utilities
├── data/                        # Experiment results and checkpoints
├── report/                      # LaTeX report and figures
└── requirements.txt             # Python dependencies
```

## Datasets

### Classification
- Iris
- Wine
- Breast Cancer

### Regression
- Diabetes
- Linnerud
- California Housing

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run Classification Experiments
```bash
python src/combined_run_cls.py --method all
```

### Run Regression Experiments
```bash
python src/combined_run_reg.py --method all
```

### Run Specific Methods
```bash
# Only passive learning
python src/combined_run_cls.py --method passive

# Only uncertainty-based active learning
python src/combined_run_cls.py --method uncertainty

# Only sensitivity-based active learning
python src/combined_run_cls.py --method sensitivity
```

## Active Learning Strategies

### Uncertainty Sampling
- Entropy: Select samples with highest prediction entropy
- Margin: Select samples with smallest margin between top-2 classes
- Least Confidence: Select samples with lowest maximum probability

### Sensitivity Sampling
- Select samples with highest gradient sensitivity
- Measures how much the model output changes with input perturbations

## Results

Results are automatically saved to:
- `data/` directory for JSON results
- `report/figures/` directory for plots

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- matplotlib
- numpy
- tqdm

See `requirements.txt` for complete dependency list.

## Report

A detailed LaTeX report is available in the `report/` directory with experimental results and analysis.
