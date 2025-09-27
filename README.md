# 25935410-rw441-assignment3
This is the repo for 25935410 RW441 Assignment 3

Usage
-----

Create a virtual environment and install requirements:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run experiments:

```
python -m alnn.cli passive-cls iris --device cpu
python -m alnn.cli active-cls iris --strategy uncertainty
python -m alnn.cli active-cls iris --strategy sensitivity
python -m alnn.cli passive-reg diabetes
python -m alnn.cli active-reg diabetes --strategy uncertainty
```

Results are printed as JSON.

Hyperparameter Tuning
---------------------

Random search examples (print JSON per trial and best summary at end):

Classification (passive):
```
python -m alnn.tune passive-cls iris --search random --trials 20 --device cpu
```

Classification (active, uncertainty entropy):
```
python -m alnn.tune active-cls iris --strategy uncertainty --method entropy --search random --trials 20 --device cpu
```

Classification (active, sensitivity):
```
python -m alnn.tune active-cls iris --strategy sensitivity --search random --trials 20 --device cpu
```

Regression (passive):
```
python -m alnn.tune passive-reg diabetes --search random --trials 20 --device cpu
```

Regression (active, uncertainty margin):
```
python -m alnn.tune active-reg diabetes --strategy uncertainty --method margin --search random --trials 20 --device cpu
```

Flags to control search spaces (examples):
```
--lrs 0.001 0.01 0.1 --wds 0 0.0001 --bss 32 64 128 --hidden 32 64 128 --patience 10 20 40 --inits 10 20 40 --queries 5 10 20 --max_labels 200
```

Performance Criteria and Empirical Protocol
------------------------------------------

Classification metrics (reported on validation during tuning; test once at the end):
- Accuracy: overall correctness
- Macro-F1: balances precision/recall across classes, handles imbalance
- AUROC (OvR): discrimination across thresholds (binary: AUROC of positive class)
- Log-loss: probabilistic calibration (lower is better)

Regression metrics:
- RMSE: error magnitude (sensitive to large errors)
- MAE: robust average absolute error
- R2: explained variance

Active learning efficiency metrics:
- ALC (Area under Learning Curve): integrate performance vs. labels acquired
- Label-efficiency at fixed budget: metric value at N labeled points (e.g., 50, 100, 200)

Empirical protocol (statistically sound):
1) Data splits: fixed train/val/test (or 5-fold CV for small datasets). Validation drives model selection; test is used once.
2) Hyperparameter search: random or grid search using `alnn.tune`, same trial budget per algorithm.
3) Multiple seeds: repeat each configuration (e.g., 3 runs) to average over initialization and sampling variance.
4) Model selection: choose params maximizing val accuracy/macro-F1 (classification) or minimizing val RMSE (regression); for AL, use ALC as primary.
5) Final fit: retrain best configuration on train+val; evaluate once on test. Report mean±std across seeds.
6) Statistical comparison: paired tests across seeds (e.g., Wilcoxon signed-rank on per-seed test metrics) to compare algorithms.

Notes:
- Keep labeling budgets identical across active strategies.
- Track the full performance-vs-label curve to compute ALC.
Datasets
--------

See `data/README.md` for the list of classification and regression problems (with complexity notes) used in experiments. All datasets are loaded via scikit-learn; no manual downloads are needed.
