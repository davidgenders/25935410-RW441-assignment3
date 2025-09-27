# Datasets and Problem Descriptions

This folder documents the datasets used for both classification and function approximation (regression) tasks. We include at least three problems per type with varying complexity to enable meaningful comparisons across active learning strategies.

## Classification (increasing complexity)

1) Iris (very low complexity)
- Inputs: 4 continuous features (sepal/petal lengths and widths)
- Classes: 3 balanced classes
- Notes: Linearly separable for some pairs; small dataset (~150). Good sanity check.

2) Wine (moderate complexity)
- Inputs: 13 physicochemical features
- Classes: 3 classes, moderately separable
- Notes: Slight class imbalance; nonlinear boundaries benefit from MLP capacity.

3) Breast Cancer Wisconsin (higher complexity)
- Inputs: 30 features derived from cell nuclei images
- Classes: 2 (benign/malignant) with class imbalance
- Notes: More complex feature interactions; regularization and early stopping help.

## Regression / Function Approximation (increasing complexity)

1) Diabetes (low-to-moderate complexity)
- Inputs: 10 standardized features
- Target: Disease progression (scalar)
- Notes: Small dataset; moderate noise; useful for baseline.

2) Linnerud (moderate, with lower intrinsic dimensionality)
- Inputs: 3 exercise features
- Target: We use a single target (Weight) for scalar regression
- Notes: Simple relationships; tests capacity to avoid overfitting with small data.

3) California Housing (higher complexity)
- Inputs: 8 features about districts in California
- Target: Median house value (scalar)
- Notes: Larger dataset with heteroscedasticity and nonlinearity; benefits from deeper models and careful training.

## Rationale for selection
- Breadth of complexity: we span small/clean to large/noisy datasets to examine behavior across regimes.
- Feature types: mostly continuous standardized features, minimizing preprocessing confounds.
- Practicality: datasets are available via scikit-learn loaders, ensuring reproducibility.

## Usage
These datasets are automatically loaded via scikit-learn in `src/alnn/data.py`. No manual downloads are required.
