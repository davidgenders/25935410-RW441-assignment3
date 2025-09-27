# Proper Data Splitting for Hyperparameter Tuning

## Overview

Both `combined_run_cls.py` and `combined_run_reg.py` now implement proper data splitting to ensure that hyperparameter tuning is done on a reasonable training set, while keeping the test set completely separate for final evaluation in the comparison notebooks.

## Data Splitting Strategy

### Three-Way Split
```
Full Dataset
├── Train + Validation (80%) → Used for hyperparameter tuning
│   ├── Train (80% of 80% = 64%) → Model training
│   └── Validation (20% of 80% = 16%) → Hyperparameter evaluation
└── Test (20%) → Held out for final evaluation in compare_*.ipynb
```

### Implementation Details

#### 1. **Initial Split** (`_get_data_splits` method)
```python
# Split into train+val (80%) and test (20%) - test set is held out for final evaluation
# Use fixed random state (42) so test set is always the same across all trials
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify for classification
)
```

#### 2. **Cross-Validation on Train+Val Only**
```python
# Use CV on train+val split only (test is completely untouched)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + trial)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
```

#### 3. **Standardization Protocol**
```python
# Fit scaler on train data only, apply to validation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

## Benefits of This Approach

### ✅ **Prevents Data Leakage**
- Test set is never seen during hyperparameter tuning
- No information from test set influences model selection

### ✅ **Realistic Evaluation**
- Hyperparameter tuning uses realistic data sizes
- Final test evaluation in comparison notebooks is unbiased

### ✅ **Consistent Splits**
- Fixed random state (42) ensures **identical test set** across all trials
- Test set is **completely isolated** and never seen during hyperparameter tuning
- Stratified splits maintain class distribution (classification)

### ✅ **Proper Preprocessing**
- StandardScaler fitted only on training data
- No information leakage from validation/test sets

## Data Flow

### Hyperparameter Tuning Phase (combined_run_*.py)
```
Dataset → Split (80/20) → CV on 80% → Best hyperparameters
                                ↓
                          Train: 64% of original
                          Val: 16% of original
```

### Final Evaluation Phase (compare_*.ipynb)
```
Dataset → Same Split (80/20) → Train on 80% → Evaluate on 20%
                                ↓
                          Use best hyperparameters from tuning
```

## Key Changes Made

### 1. **Added `_get_data_splits` Method**
- Handles the initial 80/20 split
- Consistent across all evaluation methods
- Returns both train+val and test sets

### 2. **Modified CV Evaluation**
- All cross-validation now operates on train+val split only
- Test set is completely untouched during tuning

### 3. **Updated Active Learning**
- Active learning simulation now happens within train+val split
- Query selection and model training use only training portion
- Final evaluation on validation portion

### 4. **Fixed Test Set**
- Test set split uses fixed random state (42) - **identical across all trials**
- Train+val split is the same for all hyperparameter tuning
- Test set is **completely unseen** during all tuning phases

## Usage in Comparison Notebooks

When you run the comparison notebooks (`compare_classification.ipynb` and `compare_regression.ipynb`), they should:

1. **Use the same data splitting logic** to get the same 80/20 split
2. **Load the best hyperparameters** from the tuning results
3. **Train on the full 80%** using the best hyperparameters
4. **Evaluate on the held-out 20%** for unbiased final results

## Example Data Sizes

### Classification Datasets
- **Iris**: 150 samples → Train+Val: 120, Test: 30
- **Wine**: 178 samples → Train+Val: 142, Test: 36  
- **Breast Cancer**: 569 samples → Train+Val: 455, Test: 114

### Regression Datasets
- **Diabetes**: 442 samples → Train+Val: 354, Test: 88
- **Linnerud**: 20 samples → Train+Val: 16, Test: 4
- **California**: 20,640 samples → Train+Val: 16,512, Test: 4,128

## Validation Strategy

### During Hyperparameter Tuning
- **5 trials** with different random seeds
- **5-fold CV** on train+val split (80% of data)
- **Total evaluations per config**: 5 × 5 = 25

### Final Evaluation
- **Single train/test split** using best hyperparameters
- **Train on 80%**, **evaluate on 20%**
- **Unbiased estimate** of true performance

This approach ensures that your hyperparameter tuning results are realistic and that your final evaluation in the comparison notebooks provides unbiased estimates of model performance.
