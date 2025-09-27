# Cross-Validation and Multiple Trials Improvements

This document summarizes the improvements made to all notebooks to include proper cross-validation and multiple trials for more robust hyperparameter tuning and evaluation.

## Key Improvements

### 1. Cross-Validation Implementation
- **Classification**: Uses `StratifiedKFold` (5 folds) to maintain class distribution across splits
- **Regression**: Uses `KFold` (5 folds) for proper train/validation splits
- Each configuration is evaluated across all CV folds before averaging

### 2. Multiple Trials
- **5 trials** per configuration with different random seeds (42, 43, 44, 45, 46)
- Each trial uses different CV splits to reduce variance
- Final metrics are averaged across trials with standard deviation reported

### 3. Updated Notebooks

#### Passive Learning Tuning
- `passive_cls_tuning.ipynb`: Classification hyperparameter tuning with CV + trials
- `passive_reg_tuning.ipynb`: Regression hyperparameter tuning with CV + trials
- More conservative learning rate range for regression (1e-4 to 3e-3)

#### Active Learning Exploration
- `explore_cls_uncertainty.ipynb`: Classification uncertainty methods with CV + trials
- `explore_reg_uncertainty.ipynb`: Regression uncertainty methods with CV + trials
- `explore_cls_sensitivity.ipynb`: Classification sensitivity method with CV + trials
- `explore_reg_sensitivity.ipynb`: Regression sensitivity method with CV + trials

#### Comparison Notebooks
- `compare_classification.ipynb`: Compare all strategies with error bars and summary tables
- `compare_regression.ipynb`: Compare all strategies with error bars and summary tables

## Evaluation Methodology

### For Each Configuration:
1. **5 Trials**: Different random seeds for initialization and data splits
2. **5-Fold CV**: Each trial uses different train/validation splits
3. **Total Evaluations**: 5 trials × 5 folds = 25 evaluations per configuration
4. **Metrics**: Mean ± standard deviation across all evaluations

### Benefits:
- **Reduced Variance**: Multiple trials and CV reduce overfitting to specific splits
- **Better Generalization**: More robust hyperparameter selection
- **Statistical Significance**: Standard deviations provide confidence intervals
- **Reproducibility**: Fixed random seeds ensure reproducible results

## Usage

1. **Run Tuning Notebooks First**:
   ```bash
   # Run passive tuning
   jupyter notebook passive_cls_tuning.ipynb
   jupyter notebook passive_reg_tuning.ipynb
   
   # Run active learning exploration
   jupyter notebook explore_cls_uncertainty.ipynb
   jupyter notebook explore_reg_uncertainty.ipynb
   jupyter notebook explore_cls_sensitivity.ipynb
   jupyter notebook explore_reg_sensitivity.ipynb
   ```

2. **Run Comparison Notebooks**:
   ```bash
   jupyter notebook compare_classification.ipynb
   jupyter notebook compare_regression.ipynb
   ```

## Output Files

### Results Files:
- `passive_cls_best.json`: Best configurations for classification
- `passive_reg_best.json`: Best configurations for regression
- `cls_uncertainty_results.json`: Uncertainty method results
- `reg_uncertainty_results.json`: Uncertainty method results
- `cls_sensitivity_results.json`: Sensitivity method results
- `reg_sensitivity_results.json`: Sensitivity method results

### Summary Files:
- `cls_comparison_summary.csv`: Classification comparison table
- `reg_comparison_summary.csv`: Regression comparison table

### Figures:
- All plots now include error bars showing ±1 standard deviation
- Comparison plots show all methods with confidence intervals
- Summary tables provide comprehensive performance metrics

## Computational Considerations

- **Runtime**: ~25x longer per configuration due to CV + trials
- **Memory**: Minimal increase (same model size, just more evaluations)
- **Storage**: Results files are larger due to additional metrics

## Quality Improvements

1. **Statistical Robustness**: Results are now statistically meaningful
2. **Better Hyperparameter Selection**: CV prevents overfitting to validation set
3. **Confidence Intervals**: Standard deviations show result reliability
4. **Comprehensive Evaluation**: All datasets and methods compared fairly
5. **Reproducible Results**: Fixed seeds ensure consistent results across runs