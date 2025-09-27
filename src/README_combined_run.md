# Combined Classification Tuning Script

## Overview

The `combined_run_cls.py` script provides a unified interface for running all three classification approaches:
- **Passive Learning**: Standard supervised learning with hyperparameter tuning
- **Uncertainty-based Active Learning**: Uses uncertainty sampling strategies (entropy, margin, least confidence)
- **Sensitivity-based Active Learning**: Uses sensitivity sampling for query selection

## Features

✅ **Consistent Cross-Validation**: All methods use the same CV procedure (5 trials × 5 folds)  
✅ **Checkpoint System**: Resume interrupted runs automatically  
✅ **Progress Tracking**: Real-time progress bars with tqdm  
✅ **Structured Results**: Organized output with mean ± std metrics  
✅ **Modular Design**: Run individual methods or all together  
✅ **Command Line Interface**: Flexible execution options  

## Usage

### Run All Methods
```bash
python3 combined_run_cls.py
# or explicitly:
python3 combined_run_cls.py --method all
```

### Run Individual Methods
```bash
# Passive learning only
python3 combined_run_cls.py --method passive

# Uncertainty-based active learning only
python3 combined_run_cls.py --method uncertainty

# Sensitivity-based active learning only
python3 combined_run_cls.py --method sensitivity
```

### Custom Datasets
```bash
python3 combined_run_cls.py --datasets iris wine --method passive
```

## Configuration

The script uses these default hyperparameters:

### Passive Learning
- **Learning Rates**: [3e-3, 1e-2, 3e-2]
- **Weight Decay**: [0.0, 1e-5, 1e-4]
- **Hidden Units**: [32, 64, 128]
- **Batch Sizes**: [32, 64]
- **Total Configs**: 3 × 3 × 3 × 2 = 54 per dataset

### Active Learning
- **Same hyperparameters as passive**, plus:
- **Initial Labeled**: [10, 20, 40]
- **Query Batch Sizes**: [5, 10, 20]
- **Total Configs**: 54 × 3 × 3 = 486 per method per dataset
- **Budgets**: [40, 80, 120, 160, 200] labels

## Evaluation Protocol

### Cross-Validation
- **5 trials** with different random seeds (42, 43, 44, 45, 46)
- **5-fold stratified CV** per trial
- **Total evaluations per config**: 5 trials × 5 folds = 25

### Metrics
- **Primary**: Classification accuracy
- **Secondary**: Precision, recall, F1-score
- **Reporting**: Mean ± standard deviation across trials

## Checkpoint System

The script automatically saves progress and can resume from interruptions:

### Passive Learning
- Checkpoint: `../data/passive_cls_checkpoint.json`
- Results: `../data/passive_cls_best.json`

### Uncertainty-based Active Learning
- Main checkpoint: `../data/cls_uncertainty_main_checkpoint.json`
- Hyperparameter checkpoints: `../data/cls_uncertainty_{dataset}_{method}_checkpoint.json`
- Curve checkpoints: `../data/cls_uncertainty_{dataset}_{method}_curve_checkpoint.json`
- Results: `../data/cls_uncertainty_results.json`

### Sensitivity-based Active Learning
- Main checkpoint: `../data/cls_sensitivity_main_checkpoint.json`
- Hyperparameter checkpoints: `../data/cls_sensitivity_{dataset}_checkpoint.json`
- Curve checkpoints: `../data/cls_sensitivity_{dataset}_curve_checkpoint.json`
- Results: `../data/cls_sensitivity_results.json`

## Output Structure

### Passive Results
```json
{
  "iris": {
    "best_cfg": {"lr": 0.01, "wd": 1e-5, "hidden": 64, "bs": 32},
    "best_metric": 0.9733,
    "history": [...]
  }
}
```

### Active Learning Results
```json
{
  "iris": {
    "entropy": {
      "40": {"accuracy_mean": 0.85, "accuracy_std": 0.02, ...},
      "80": {"accuracy_mean": 0.91, "accuracy_std": 0.01, ...},
      ...
    }
  }
}
```

## Performance Estimates

### Time Requirements
- **Passive Learning**: ~2-4 hours for all datasets
- **Uncertainty Active Learning**: ~6-12 hours for all datasets
- **Sensitivity Active Learning**: ~4-8 hours for all datasets
- **Total**: ~12-24 hours for complete run

### Computational Cost
- **Passive**: 3 datasets × 54 configs × 25 evaluations = 4,050 evaluations
- **Uncertainty**: 3 datasets × 3 methods × 486 configs × 25 evaluations = 109,350 evaluations
- **Sensitivity**: 3 datasets × 486 configs × 25 evaluations = 36,450 evaluations

## Class Structure

```python
class ClassificationTuner:
    def run_passive_tuning()      # Passive learning hyperparameter tuning
    def run_uncertainty_tuning()  # Uncertainty-based active learning
    def run_sensitivity_tuning()  # Sensitivity-based active learning
    def run_all()                 # Run all three methods
    def plot_results()            # Generate result plots
```

## Dependencies

- torch
- sklearn
- numpy
- matplotlib
- tqdm
- alnn (local module)

## Example Output

```
================================================================================
COMBINED CLASSIFICATION HYPERPARAMETER TUNING
================================================================================
Datasets: ['iris', 'wine', 'breast_cancer']
Methods: Passive, Uncertainty-based, Sensitivity-based
Trials per config: 5
CV folds: 5
================================================================================

============================================================
PASSIVE LEARNING HYPERPARAMETER TUNING
============================================================
Starting fresh run

=== Tuning iris ===
Starting iris from config 1/54
iris configs: 100%|██████████| 54/54 [15:30<00:00, 17.23s/it]
Best config for iris: {'lr': 0.01, 'wd': 1e-05, 'hidden': 64, 'bs': 32} (accuracy: 0.9733)

Total time: 0.26 hours
Average time per config: 17.23 seconds
```

This script provides a comprehensive and robust framework for comparing different learning approaches with consistent evaluation protocols.
