# Combined Regression Tuning Script

## Overview

The `combined_run_reg.py` script provides a unified interface for running all three regression approaches:
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
python3 combined_run_reg.py
# or explicitly:
python3 combined_run_reg.py --method all
```

### Run Individual Methods
```bash
# Passive learning only
python3 combined_run_reg.py --method passive

# Uncertainty-based active learning only
python3 combined_run_reg.py --method uncertainty

# Sensitivity-based active learning only
python3 combined_run_reg.py --method sensitivity
```

### Custom Datasets
```bash
python3 combined_run_reg.py --datasets diabetes california --method passive
```

## Configuration

The script uses these default hyperparameters:

### Passive Learning
- **Learning Rates**: [1e-4, 3e-4, 1e-3, 3e-3] (more conservative for regression)
- **Weight Decay**: [0.0, 1e-5, 1e-4]
- **Hidden Units**: [32, 64, 128]
- **Batch Sizes**: [32, 64]
- **Total Configs**: 4 × 3 × 3 × 2 = 72 per dataset

### Active Learning
- **Same hyperparameters as passive**, plus:
- **Initial Labeled**: [10, 20, 40]
- **Query Batch Sizes**: [5, 10, 20]
- **Total Configs**: 72 × 3 × 3 = 648 per method per dataset
- **Budgets**: [40, 80, 120, 160, 200] labels

## Datasets

### Supported Datasets
- **diabetes**: Diabetes dataset (442 samples, 10 features)
- **linnerud**: Linnerud dataset (20 samples, 3 features, using Weight target)
- **california**: California housing dataset (20640 samples, 8 features)

## Evaluation Protocol

### Cross-Validation
- **5 trials** with different random seeds (42, 43, 44, 45, 46)
- **5-fold KFold CV** per trial (stratified for classification, standard for regression)
- **Total evaluations per config**: 5 trials × 5 folds = 25

### Metrics
- **Primary**: Root Mean Square Error (RMSE)
- **Secondary**: Mean Absolute Error (MAE), R² score
- **Reporting**: Mean ± standard deviation across trials

## Checkpoint System

The script automatically saves progress and can resume from interruptions:

### Passive Learning
- Checkpoint: `../data/passive_reg_checkpoint.json`
- Results: `../data/passive_reg_best.json`

### Uncertainty-based Active Learning
- Main checkpoint: `../data/reg_uncertainty_main_checkpoint.json`
- Hyperparameter checkpoints: `../data/reg_uncertainty_{dataset}_{method}_checkpoint.json`
- Curve checkpoints: `../data/reg_uncertainty_{dataset}_{method}_curve_checkpoint.json`
- Results: `../data/reg_uncertainty_results.json`

### Sensitivity-based Active Learning
- Main checkpoint: `../data/reg_sensitivity_main_checkpoint.json`
- Hyperparameter checkpoints: `../data/reg_sensitivity_{dataset}_checkpoint.json`
- Curve checkpoints: `../data/reg_sensitivity_{dataset}_curve_checkpoint.json`
- Results: `../data/reg_sensitivity_results.json`

## Output Structure

### Passive Results
```json
{
  "diabetes": {
    "best_cfg": {"lr": 0.001, "wd": 1e-5, "hidden": 64, "bs": 32},
    "best_metric": 55.23,
    "history": [...]
  }
}
```

### Active Learning Results
```json
{
  "diabetes": {
    "entropy": {
      "40": {"rmse_mean": 62.45, "rmse_std": 3.21, ...},
      "80": {"rmse_mean": 58.12, "rmse_std": 2.87, ...},
      ...
    }
  }
}
```

## Performance Estimates

### Time Requirements
- **Passive Learning**: ~3-5 hours for all datasets
- **Uncertainty Active Learning**: ~8-15 hours for all datasets
- **Sensitivity Active Learning**: ~6-12 hours for all datasets
- **Total**: ~17-32 hours for complete run

### Computational Cost
- **Passive**: 3 datasets × 72 configs × 25 evaluations = 5,400 evaluations
- **Uncertainty**: 3 datasets × 3 methods × 648 configs × 25 evaluations = 145,800 evaluations
- **Sensitivity**: 3 datasets × 648 configs × 25 evaluations = 48,600 evaluations

## Class Structure

```python
class RegressionTuner:
    def run_passive_tuning()      # Passive learning hyperparameter tuning
    def run_uncertainty_tuning()  # Uncertainty-based active learning
    def run_sensitivity_tuning()  # Sensitivity-based active learning
    def run_all()                 # Run all three methods
    def plot_results()            # Generate result plots
```

## Key Differences from Classification

| Aspect | Classification | Regression |
|--------|----------------|------------|
| **CV Method** | StratifiedKFold | KFold |
| **Primary Metric** | Accuracy (maximize) | RMSE (minimize) |
| **Loss Function** | CrossEntropyLoss | MSELoss |
| **Output Dimension** | Number of classes | 1 |
| **Learning Rates** | [3e-3, 1e-2, 3e-2] | [1e-4, 3e-4, 1e-3, 3e-3] |
| **Data Preprocessing** | StandardScaler only | StandardScaler + target handling |

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
COMBINED REGRESSION HYPERPARAMETER TUNING
================================================================================
Datasets: ['diabetes', 'linnerud', 'california']
Methods: Passive, Uncertainty-based, Sensitivity-based
Trials per config: 5
CV folds: 5
================================================================================

============================================================
PASSIVE LEARNING HYPERPARAMETER TUNING
============================================================
Starting fresh run

=== Tuning diabetes ===
Starting diabetes from config 1/72
diabetes configs: 100%|██████████| 72/72 [25:30<00:00, 21.25s/it]
Best config for diabetes: {'lr': 0.001, 'wd': 1e-05, 'hidden': 64, 'bs': 32} (RMSE: 55.23)

Total time: 0.42 hours
Average time per config: 21.25 seconds
```

This script provides a comprehensive and robust framework for comparing different learning approaches with consistent evaluation protocols for regression tasks.
