"""
Combined Regression Tuning Script
=================================

This script combines passive, uncertainty-based, and sensitivity-based active learning
for regression tasks with consistent cross-validation and checkpoint handling.

Features:
- Cross-validation with multiple trials for robust evaluation
- Checkpoint system for resuming interrupted runs
- Consistent hyperparameter tuning across all methods
- Progress tracking with tqdm
- Results visualization and saving
"""

import os
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple
from tqdm import tqdm
import time
import argparse
import math

from nn.models import OneHiddenMLP
from nn.training import train_passive, TrainConfig
from nn.evaluation import evaluate_regression
from nn.experiments import ActiveConfig, run_active_regression
from nn.data import make_regression_split, to_datasets

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Configuration
FIGURES_DIR = os.path.join('..', 'report', 'figures')
DATA_DIR = os.path.join('..', 'data')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = ['diabetes', 'linnerud', 'california']
METHODS = ['entropy', 'margin', 'least_confidence']
LR = [3e-4, 3e-3, 1e-2, 3e-2]
WD = [1e-5, 1e-4, 1e-3]
HIDDEN = [64]
BS = [64]
BUDGETS = [200]
N_TRIALS = 5
N_FOLDS = 5
INITS = [20]
QUERIES = [10]


# DATASETS = ['diabetes', 'linnerud', 'california']
# METHODS = ['entropy', 'margin', 'least_confidence']
# LR = [3e-3]  # More conservative LR range for regression
# WD = [1e-4]
# HIDDEN = [32]
# BS = [32]
# BUDGETS = [40]
# N_TRIALS = 5  # Number of random seeds for each config
# N_FOLDS = 5  # Number of CV folds
# INITS = [10]
# QUERIES = [5]

def nan_to_none(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [nan_to_none(v) for v in obj]
    return obj


class RegressionTuner:
    """Main class for regression hyperparameter tuning."""
    
    def __init__(self, datasets: List[str] = None):
        self.datasets = datasets or DATASETS
        self.results = {}
    
    def _serialize_config(self, train_config, active_config, hidden_units):
        """Convert config objects to JSON-serializable format."""
        return {
            'train_config': {
                'learning_rate': train_config.learning_rate,
                'weight_decay': train_config.weight_decay,
                'batch_size': train_config.batch_size,
                'max_epochs': train_config.max_epochs,
                'patience': train_config.patience,
            },
            'active_config': {
                'initial_labeled': active_config.initial_labeled,
                'query_batch': active_config.query_batch,
                'max_labels': active_config.max_labels,
            },
            'hidden_units': hidden_units
        }
    
    def _deserialize_config(self, serialized_config):
        """Convert JSON-serializable format back to config objects."""
        train_config = TrainConfig(
            learning_rate=serialized_config['train_config']['learning_rate'],
            weight_decay=serialized_config['train_config']['weight_decay'],
            batch_size=serialized_config['train_config']['batch_size'],
            max_epochs=serialized_config['train_config']['max_epochs'],
            patience=serialized_config['train_config']['patience'],
        )
        active_config = ActiveConfig(
            initial_labeled=serialized_config['active_config']['initial_labeled'],
            query_batch=serialized_config['active_config']['query_batch'],
            max_labels=serialized_config['active_config']['max_labels'],
        )
        hidden_units = serialized_config['hidden_units']
        return train_config, active_config, hidden_units
        
    def _get_data_splits(self, dataset: str):
        """Get train/validation/test splits for hyperparameter tuning."""
        # Load data
        if dataset == "diabetes":
            ds = datasets.load_diabetes()
            y = ds.target.astype(np.float32)
        elif dataset == "linnerud":
            ds = datasets.load_linnerud()
            y = ds.target[:, 0].astype(np.float32)  # use one target (Weight)
        elif dataset == "california":
            ds = datasets.fetch_california_housing()
            y = ds.target.astype(np.float32)
        
        X = ds.data.astype(np.float32)
        
        # Split into train+val (80%) and test (20%) - test set is held out for final evaluation
        # Use fixed random state (42) so test set is always the same across all trials
        from sklearn.model_selection import train_test_split
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train_val, X_test, y_train_val, y_test

    def evaluate_config_cv_passive(self, dataset: str, lr: float, wd: float, hidden: int, bs: int) -> Dict[str, float]:
        """Evaluate a passive learning configuration using cross-validation across multiple trials."""
        # Get the fixed train+val split (test is held out and never seen)
        X_train_val, X_test, y_train_val, y_test = self._get_data_splits(dataset)
        
        all_metrics = []
        
        for trial in range(N_TRIALS):
            # Set random seed for reproducibility
            torch.manual_seed(42 + trial)
            np.random.seed(42 + trial)
            
            trial_metrics = []
            
            # Use KFold for regression on train+val only (test is completely held out)
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + trial)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
                X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
                y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
                
                # Standardize features using train data only
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Convert to tensors
                X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
                
                # Create datasets
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                
                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
                
                # Train model
                model = OneHiddenMLP(input_dim=X_train_scaled.shape[1], hidden_units=hidden, output_dim=1)
                loss_fn = nn.MSELoss()
                config = TrainConfig(learning_rate=lr, weight_decay=wd, batch_size=bs, max_epochs=200, patience=20)
                
                train_passive(model, train_loader, val_loader, loss_fn, config)
                
                # Evaluate on validation set only
                metrics = evaluate_regression(model, val_loader)
                trial_metrics.append(metrics)
            
            # Average across folds for this trial
            trial_avg = {}
            for key in trial_metrics[0].keys():
                trial_avg[key] = np.mean([m[key] for m in trial_metrics])
            all_metrics.append(trial_avg)
        
        # Average across trials and compute std
        final_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            final_metrics[f'{key}_mean'] = float(np.mean(values))
            final_metrics[f'{key}_std'] = float(np.std(values, ddof=1))
        
        return final_metrics

    def evaluate_config_cv_active(self, dataset: str, strategy: str, method: str, lr: float, wd: float, 
                                 hidden: int, bs: int, init: int, query: int, budget: int) -> Dict[str, float]:
        """Evaluate an active learning configuration using cross-validation across multiple trials."""
        # Get the fixed train+val split (test is held out and never seen)
        X_train_val, X_test, y_train_val, y_test = self._get_data_splits(dataset)
        
        all_metrics = []
        
        for trial in range(N_TRIALS):
            # Set random seed for reproducibility
            torch.manual_seed(42 + trial)
            np.random.seed(42 + trial)
            
            trial_metrics = []
            
            # Use KFold for regression on train+val only (test is completely held out)
            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42 + trial)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
                X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
                y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
                
                # Standardize features using train data only
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Convert to tensors
                X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
                X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
                
                # Simulate active learning on the train set
                train_config = TrainConfig(learning_rate=lr, weight_decay=wd, batch_size=bs, 
                                         max_epochs=200, patience=20)
                
                # Create initial labeled pool
                num_train = X_train_scaled.shape[0]
                labeled_indices = torch.randperm(num_train)[:init]
                unlabeled_indices = torch.tensor([i for i in range(num_train) if i not in labeled_indices.tolist()], dtype=torch.long)
                
                x_pool = torch.tensor(X_train_scaled, dtype=torch.float32)
                y_pool = y_train_tensor.clone()
                val_subset = TensorDataset(X_val_tensor, y_val_tensor)
                
                # Active learning loop
                while labeled_indices.numel() < min(budget, num_train):
                    # Train model on current labeled set
                    train_subset = TensorDataset(x_pool[labeled_indices], y_pool[labeled_indices])
                    
                    train_loader = DataLoader(train_subset, batch_size=bs, shuffle=True)
                    val_loader = DataLoader(val_subset, batch_size=bs, shuffle=False)
                    
                    model = OneHiddenMLP(input_dim=X_train_scaled.shape[1], hidden_units=hidden, output_dim=1)
                    loss_fn = nn.MSELoss()
                    
                    train_passive(model, train_loader, val_loader, loss_fn, train_config)
                    
                    if unlabeled_indices.numel() == 0:
                        break
                    
                    # Query selection
                    if strategy == 'uncertainty':
                        from nn.strategies import uncertainty_sampling, UncertaintySamplingConfig
                        sel = uncertainty_sampling(
                            model,
                            x_pool[unlabeled_indices].to("cpu"),
                            query,
                            UncertaintySamplingConfig(mode="regression", method=method),
                        )
                    elif strategy == 'sensitivity':
                        from nn.strategies import sensitivity_sampling
                        sel = sensitivity_sampling(model, x_pool[unlabeled_indices].to("cpu"), query)
                    
                    # Update labeled and unlabeled sets
                    newly_selected = unlabeled_indices[sel]
                    labeled_indices = torch.unique(torch.cat([labeled_indices, newly_selected]))
                    mask = torch.ones_like(unlabeled_indices, dtype=torch.bool)
                    mask[sel] = False
                    unlabeled_indices = unlabeled_indices[mask]
                    
                    if labeled_indices.numel() >= budget:
                        break
                
                # Final evaluation on validation set
                final_train_subset = TensorDataset(x_pool[labeled_indices], y_pool[labeled_indices])
                final_train_loader = DataLoader(final_train_subset, batch_size=bs, shuffle=True)
                final_val_loader = DataLoader(val_subset, batch_size=bs, shuffle=False)
                
                final_model = OneHiddenMLP(input_dim=X_train_scaled.shape[1], hidden_units=hidden, output_dim=1)
                loss_fn = nn.MSELoss()
                
                train_passive(final_model, final_train_loader, final_val_loader, loss_fn, train_config)
                
                # Evaluate on validation set
                metrics = evaluate_regression(final_model, final_val_loader)
                trial_metrics.append(metrics)

            # Average across folds for this trial
            trial_avg = {}
            for key in trial_metrics[0].keys():
                trial_avg[key] = np.mean([m[key] for m in trial_metrics])
            all_metrics.append(trial_avg)
        
        # Average across trials and folds
        final_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            final_metrics[f'{key}_mean'] = float(np.mean(values))
            final_metrics[f'{key}_std'] = float(np.std(values, ddof=1))
        
        return final_metrics

    def run_passive_tuning(self):
        """Run passive learning hyperparameter tuning."""
        print("\n" + "="*60)
        print("PASSIVE LEARNING HYPERPARAMETER TUNING")
        print("="*60)
        
        # Load checkpoint if exists
        checkpoint_file = os.path.join(DATA_DIR, 'passive_reg_checkpoint.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Resuming from checkpoint: {checkpoint.get('completed_configs', 0)} configs completed")
            self.results['passive'] = checkpoint.get('results', {})
            
            # Determine which dataset to resume from
            dataset_configs = len(LR) * len(WD) * len(HIDDEN) * len(BS)
            completed_configs = checkpoint.get('completed_configs', 0)
            
            # Find which dataset we should resume from
            resume_dataset_idx = completed_configs // dataset_configs
            resume_config_idx = completed_configs % dataset_configs
            
            print(f"Resuming from dataset {resume_dataset_idx} ({self.datasets[resume_dataset_idx] if resume_dataset_idx < len(self.datasets) else 'completed'}), config {resume_config_idx}")
            
        else:
            checkpoint = {'completed_configs': 0, 'results': {}}
            self.results['passive'] = {}
            resume_dataset_idx = 0
            resume_config_idx = 0
            print("Starting fresh run")

        # Calculate total configs
        total_configs = len(self.datasets) * len(LR) * len(WD) * len(HIDDEN) * len(BS)
        start_time = time.time()

        # Process datasets starting from the resume point
        for dataset_idx, dataset in enumerate(self.datasets):
            if dataset not in self.results['passive']:
                self.results['passive'][dataset] = {"best_cfg": None, "best_metric": np.inf, "history": []}
            
            print(f"\n=== Tuning {dataset} ===")
            best_metric = self.results['passive'][dataset]["best_metric"]
            best_cfg = self.results['passive'][dataset]["best_cfg"]
            hist = self.results['passive'][dataset]["history"]
            
            dataset_configs = len(LR) * len(WD) * len(HIDDEN) * len(BS)
            
            # Determine starting point for this dataset
            if dataset_idx < resume_dataset_idx:
                # This dataset is already completed, skip it
                print(f"Skipping {dataset} (already completed)")
                continue
            elif dataset_idx == resume_dataset_idx:
                # This is the dataset we need to resume from
                start_config_idx = resume_config_idx
                print(f"Resuming {dataset} from config {start_config_idx + 1}/{dataset_configs}")
            else:
                # This dataset hasn't been started yet
                start_config_idx = 0
                print(f"Starting {dataset} from config 1/{dataset_configs}")
            
            # Create progress bar for this dataset
            pbar = tqdm(total=dataset_configs, desc=f"{dataset} configs", 
                        initial=len(hist), position=0, leave=True)
            
            config_count = 0
            for lr, wd, hidden, bs in itertools.product(LR, WD, HIDDEN, BS):
                # Skip configs that were already completed
                if config_count < start_config_idx:
                    config_count += 1
                    continue
                    
                config_idx = len(hist) + 1
                print(f'Config {config_idx}/{dataset_configs}: lr={lr}, wd={wd}, hidden={hidden}, bs={bs}')
                
                res = self.evaluate_config_cv_passive(dataset, lr, wd, hidden, bs)
                res.update({"lr": lr, "wd": wd, "hidden": hidden, "bs": bs})
                hist.append(res)
                
                if res['rmse_mean'] < best_metric:
                    best_metric = res['rmse_mean']
                    best_cfg = {"lr": lr, "wd": wd, "hidden": hidden, "bs": bs}
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'best_rmse': f"{best_metric:.4f}"})
                
                # Save checkpoint after each config
                checkpoint['completed_configs'] += 1
                self.results['passive'][dataset] = {"best_cfg": best_cfg, "best_metric": best_metric, "history": hist}
                checkpoint['results'] = self.results['passive']
                with open(checkpoint_file, 'w') as f:
                    data = nan_to_none(checkpoint)
                    json.dump(data, f, indent=2)
                
                config_count += 1
            
            pbar.close()
            self.results['passive'][dataset] = {"best_cfg": best_cfg, "best_metric": best_metric, "history": hist}
            print(f"Best config for {dataset}: {best_cfg} (RMSE: {best_metric:.4f})")

        # Save final results
        with open(os.path.join(DATA_DIR, 'passive_reg_best.json'), 'w') as f:
            data = nan_to_none(self.results['passive'])
            json.dump(data, f, indent=2)

        # Clean up checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time/3600:.2f} hours")
        print(f"Average time per config: {total_time/total_configs:.2f} seconds")

    def run_uncertainty_tuning(self):
        """Run uncertainty-based active learning hyperparameter tuning."""
        print("\n" + "="*60)
        print("UNCERTAINTY-BASED ACTIVE LEARNING HYPERPARAMETER TUNING")
        print("="*60)
        
        # Load checkpoint if exists
        checkpoint_file = os.path.join(DATA_DIR, 'reg_uncertainty_main_checkpoint.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Resuming from checkpoint: {checkpoint['completed_datasets']} datasets completed")
            self.results['uncertainty'] = checkpoint.get('results', {})
        else:
            checkpoint = {'completed_datasets': 0, 'results': {}}
            self.results['uncertainty'] = {}
            print("Starting fresh run")

        start_time = time.time()

        # in run_uncertainty_tuning(...)
        for dataset_idx, dataset in enumerate(self.datasets):
            if dataset in self.results.get('uncertainty', {}):
                print(f"\n=== Skipping {dataset} (already completed) ===")
                continue

            print(f"\n=== Processing {dataset} ===")
            self.results['uncertainty'][dataset] = {}

            for method in METHODS:
                print(f"\n--- Method: {method} ---")
                result = self.evaluate_curve_uncertainty(dataset, method, BUDGETS)
                # result has: best_cfg, best_metric (RMSE), curve
                self.results['uncertainty'][dataset][method] = result

            checkpoint['completed_datasets'] += 1
            checkpoint['results'] = self.results['uncertainty']
            with open(checkpoint_file, 'w') as f:
                json.dump(nan_to_none(checkpoint), f, indent=2)

        with open(os.path.join(DATA_DIR, 'reg_uncertainty_results.json'), 'w') as f:
            json.dump(nan_to_none(self.results['uncertainty']), f, indent=2)


        # Clean up checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time/3600:.2f} hours")
        print(f'\nSaved figures and results to {DATA_DIR}')
        print(f'Used {N_TRIALS} trials per config')

    def evaluate_curve_uncertainty(self, dataset: str, method: str, budgets: List[int]) -> Dict:
        tune_budget = sorted(budgets)[len(budgets)//2]

        # get best config + best rmse at tune_budget
        (tcfg, acfg_base, hidden_units), best_rmse = self.tune_hparams_uncertainty(dataset, method, tune_budget)

        checkpoint_file = os.path.join(DATA_DIR, f'reg_uncertainty_{dataset}_{method}_curve_checkpoint.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Resuming curve evaluation from checkpoint: {len(checkpoint.get('curve', {}))} budgets completed")
            results = {
                'best_cfg': checkpoint.get('best_cfg'),
                'best_metric': checkpoint.get('best_metric', best_rmse),  # RMSE
                'curve': checkpoint.get('curve', {})
            }
        else:
            results = {
                'best_cfg': self._serialize_config(tcfg, acfg_base, hidden_units),
                'best_metric': float(best_rmse),
                'curve': {}
            }
            print("Starting fresh curve evaluation")

        pbar = tqdm(total=len(budgets), desc=f"Curve {dataset}-{method}",
                    initial=len(results['curve']), position=0, leave=True)

        for max_labels in budgets:
            if str(max_labels) in results['curve']:
                pbar.update(1)
                continue

            print(f'Evaluating {dataset}-{method} at budget {max_labels}')
            metrics = []
            for seed in range(N_TRIALS):
                torch.manual_seed(42 + seed)
                acfg = ActiveConfig(
                    initial_labeled=acfg_base.initial_labeled,
                    query_batch=acfg_base.query_batch,
                    max_labels=max_labels,
                )
                res = run_active_regression(
                    dataset_name=dataset,
                    strategy='uncertainty',
                    uncertainty_method=method,
                    hidden_units=hidden_units,
                    train_config=tcfg,
                    active_config=acfg
                )
                metrics.append(res)

            keys = metrics[0].keys()
            results['curve'][str(max_labels)] = {
                **{f'{k}_mean': float(np.mean([m[k] for m in metrics])) for k in keys},
                **{f'{k}_std': float(np.std([m[k] for m in metrics], ddof=1)) for k in keys}
            }

            pbar.update(1)

            # persist checkpoint with best fields
            with open(checkpoint_file, 'w') as f:
                json.dump(nan_to_none(results), f, indent=2)

        pbar.close()
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        return results  # { best_cfg, best_metric, curve }


    def tune_hparams_uncertainty(self, dataset: str, method: str, tune_budget: int) -> tuple:
        """Tune hyperparameters for uncertainty-based active learning."""
        best_rmse = float('inf')
        best_config = None
        
        total_configs = len(LR) * len(WD) * len(HIDDEN) * len(BS) * len(INITS) * len(QUERIES)
        
        # Load checkpoint if exists
        checkpoint_file = os.path.join(DATA_DIR, f'reg_uncertainty_{dataset}_{method}_checkpoint.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Resuming hyperparameter tuning from checkpoint: {checkpoint['completed_configs']} configs completed")
            best_rmse = checkpoint.get('best_rmse', float('inf'))
            serialized_config = checkpoint.get('best_config', None)
            best_config = None
            if serialized_config is not None:
                best_config = self._deserialize_config(serialized_config)
            completed_configs = checkpoint['completed_configs']
        else:
            checkpoint = {'completed_configs': 0, 'best_rmse': float('inf'), 'best_config': None}
            completed_configs = 0
            print("Starting fresh hyperparameter tuning")
        
        # Create progress bar
        pbar = tqdm(total=total_configs, desc=f"Tuning {dataset}-{method}", 
                    initial=completed_configs, position=0, leave=True)
        
        config_idx = completed_configs
        for lr, wd, hidden, bs, init, query in itertools.product(LR, WD, HIDDEN, BS, INITS, QUERIES):
            if config_idx < completed_configs:
                config_idx += 1
                pbar.update(1)
                continue
                
            print(f'Tuning config {config_idx+1}/{total_configs}: lr={lr}, wd={wd}, hidden={hidden}, bs={bs}, init={init}, query={query}')
            
            # Evaluate this configuration using CV
            metrics = self.evaluate_config_cv_active(dataset, 'uncertainty', method, lr, wd, hidden, bs, init, query, tune_budget)
            avg_rmse = metrics['rmse_mean']
            
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                train_cfg = TrainConfig(learning_rate=lr, weight_decay=wd, batch_size=bs, max_epochs=200, patience=20)
                active_cfg = ActiveConfig(initial_labeled=init, query_batch=query, max_labels=tune_budget)
                best_config = (train_cfg, active_cfg, hidden)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'best_rmse': f"{best_rmse:.4f}"})
            
            # Save checkpoint after each config
            checkpoint['completed_configs'] = config_idx + 1
            checkpoint['best_rmse'] = best_rmse
            if best_config is not None:
                checkpoint['best_config'] = self._serialize_config(best_config[0], best_config[1], best_config[2])
            with open(checkpoint_file, 'w') as f:
                data = nan_to_none(checkpoint)
                json.dump(data, f, indent=2)
            
            config_idx += 1
        
        pbar.close()
        
        # Clean up checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        print(f"Best config for {dataset}-{method}: RMSE={best_rmse:.4f}")
        return best_config, best_rmse

    def run_sensitivity_tuning(self):
        """Run sensitivity-based active learning hyperparameter tuning."""
        print("\n" + "="*60)
        print("SENSITIVITY-BASED ACTIVE LEARNING HYPERPARAMETER TUNING")
        print("="*60)
        
        # Load checkpoint if exists
        checkpoint_file = os.path.join(DATA_DIR, 'reg_sensitivity_main_checkpoint.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Resuming from checkpoint: {checkpoint['completed_datasets']} datasets completed")
            self.results['sensitivity'] = checkpoint.get('results', {})
        else:
            checkpoint = {'completed_datasets': 0, 'results': {}}
            self.results['sensitivity'] = {}
            print("Starting fresh run")

        start_time = time.time()

        # in run_sensitivity_tuning(...)
        for dataset_idx, dataset in enumerate(self.datasets):
            if dataset in self.results.get('sensitivity', {}):
                print(f"\n=== Skipping {dataset} (already completed) ===")
                continue

            print(f"\n=== Processing {dataset} ===")
            result = self.evaluate_curve_sensitivity(dataset, BUDGETS)
            # result has: best_cfg, best_metric (RMSE), curve
            self.results['sensitivity'][dataset] = result

            checkpoint['completed_datasets'] += 1
            checkpoint['results'] = self.results['sensitivity']
            with open(checkpoint_file, 'w') as f:
                json.dump(nan_to_none(checkpoint), f, indent=2)

        with open(os.path.join(DATA_DIR, 'reg_sensitivity_results.json'), 'w') as f:
            json.dump(nan_to_none(self.results['sensitivity']), f, indent=2)


        # Clean up checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time/3600:.2f} hours")
        print(f'\nSaved figures and results to {DATA_DIR}')
        print(f'Used {N_TRIALS} trials per config')

    # in evaluate_curve_sensitivity(...)
    def evaluate_curve_sensitivity(self, dataset: str, budgets: List[int]) -> Dict:
        tune_budget = sorted(budgets)[len(budgets)//2]

        (tcfg, acfg_base, hidden_units), best_rmse = self.tune_hparams_sensitivity(dataset, tune_budget)

        checkpoint_file = os.path.join(DATA_DIR, f'reg_sensitivity_{dataset}_curve_checkpoint.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Resuming curve evaluation from checkpoint: {len(checkpoint.get('curve', {}))} budgets completed")
            results = {
                'best_cfg': checkpoint.get('best_cfg'),
                'best_metric': checkpoint.get('best_metric', best_rmse),  # RMSE
                'curve': checkpoint.get('curve', {})
            }
        else:
            results = {
                'best_cfg': self._serialize_config(tcfg, acfg_base, hidden_units),
                'best_metric': float(best_rmse),
                'curve': {}
            }
            print("Starting fresh curve evaluation")

        pbar = tqdm(total=len(budgets), desc=f"Curve {dataset}-sensitivity",
                    initial=len(results['curve']), position=0, leave=True)

        for max_labels in budgets:
            if str(max_labels) in results['curve']:
                pbar.update(1)
                continue

            print(f'Evaluating {dataset}-sensitivity at budget {max_labels}')
            metrics = []
            for seed in range(N_TRIALS):
                torch.manual_seed(42 + seed)
                acfg = ActiveConfig(
                    initial_labeled=acfg_base.initial_labeled,
                    query_batch=acfg_base.query_batch,
                    max_labels=max_labels,
                )
                res = run_active_regression(
                    dataset_name=dataset,
                    strategy='sensitivity',
                    hidden_units=hidden_units,
                    train_config=tcfg,
                    active_config=acfg
                )
                metrics.append(res)

            keys = metrics[0].keys()
            results['curve'][str(max_labels)] = {
                **{f'{k}_mean': float(np.mean([m[k] for m in metrics])) for k in keys},
                **{f'{k}_std': float(np.std([m[k] for m in metrics], ddof=1)) for k in keys}
            }

            pbar.update(1)

            with open(checkpoint_file, 'w') as f:
                json.dump(nan_to_none(results), f, indent=2)

        pbar.close()
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        return results


    def tune_hparams_sensitivity(self, dataset: str, tune_budget: int) -> tuple:
        """Tune hyperparameters for sensitivity-based active learning."""
        best_rmse = float('inf')
        best_config = None
        
        total_configs = len(LR) * len(WD) * len(HIDDEN) * len(BS) * len(INITS) * len(QUERIES)
        
        # Load checkpoint if exists
        checkpoint_file = os.path.join(DATA_DIR, f'reg_sensitivity_{dataset}_checkpoint.json')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"Resuming hyperparameter tuning from checkpoint: {checkpoint['completed_configs']} configs completed")
            best_rmse = checkpoint.get('best_rmse', float('inf'))
            serialized_config = checkpoint.get('best_config', None)
            best_config = None
            if serialized_config is not None:
                best_config = self._deserialize_config(serialized_config)
            completed_configs = checkpoint['completed_configs']
        else:
            checkpoint = {'completed_configs': 0, 'best_rmse': float('inf'), 'best_config': None}
            completed_configs = 0
            print("Starting fresh hyperparameter tuning")
        
        # Create progress bar
        pbar = tqdm(total=total_configs, desc=f"Tuning {dataset}-sensitivity", 
                    initial=completed_configs, position=0, leave=True)
        
        config_idx = completed_configs
        for lr, wd, hidden, bs, init, query in itertools.product(LR, WD, HIDDEN, BS, INITS, QUERIES):
            if config_idx < completed_configs:
                config_idx += 1
                pbar.update(1)
                continue
                
            print(f'Tuning config {config_idx+1}/{total_configs}: lr={lr}, wd={wd}, hidden={hidden}, bs={bs}, init={init}, query={query}')
            
            # Evaluate this configuration using CV
            metrics = self.evaluate_config_cv_active(dataset, 'sensitivity', '', lr, wd, hidden, bs, init, query, tune_budget)
            avg_rmse = metrics['rmse_mean']
            
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                train_cfg = TrainConfig(learning_rate=lr, weight_decay=wd, batch_size=bs, max_epochs=200, patience=20)
                active_cfg = ActiveConfig(initial_labeled=init, query_batch=query, max_labels=tune_budget)
                best_config = (train_cfg, active_cfg, hidden)
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'best_rmse': f"{best_rmse:.4f}"})
            
            # Save checkpoint after each config
            checkpoint['completed_configs'] = config_idx + 1
            checkpoint['best_rmse'] = best_rmse
            if best_config is not None:
                checkpoint['best_config'] = self._serialize_config(best_config[0], best_config[1], best_config[2])
            with open(checkpoint_file, 'w') as f:
                data = nan_to_none(checkpoint)
                json.dump(data, f, indent=2)
            
            config_idx += 1
        
        pbar.close()
        
        # Clean up checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        print(f"Best config for {dataset}-sensitivity: RMSE={best_rmse:.4f}")
        return best_config, best_rmse

    def plot_results(self):
        """Plot and save results for all methods."""
        print("\n" + "="*60)
        print("GENERATING RESULTS PLOTS")
        print("="*60)
        
        # Plot passive learning results
        if 'passive' in self.results:
            self.plot_passive_results()
        
        # Plot active learning results
        if 'uncertainty' in self.results:
            self.plot_uncertainty_results()
        
        if 'sensitivity' in self.results:
            self.plot_sensitivity_results()

    def plot_passive_results(self):
        """Plot passive learning results."""
        plt.figure(figsize=(8, 5))
        datasets_plot = []
        means_plot = []
        stds_plot = []

        for dataset in self.datasets:
            if dataset in self.results['passive']:
                best_idx = None
                best_rmse = np.inf
                for i, h in enumerate(self.results['passive'][dataset]['history']):
                    if h['rmse_mean'] < best_rmse:
                        best_rmse = h['rmse_mean']
                        best_idx = i
                
                datasets_plot.append(dataset)
                means_plot.append(self.results['passive'][dataset]['history'][best_idx]['rmse_mean'])
                stds_plot.append(self.results['passive'][dataset]['history'][best_idx]['rmse_std'])

        plt.errorbar(datasets_plot, means_plot, yerr=stds_plot, fmt='o', capsize=5, capthick=2)
        plt.ylabel('RMSE (best ± std)')
        plt.title('Passive Regression Best RMSE (CV + Multiple Trials)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'passive_reg_best_rmse.png'), dpi=200)
        plt.show()

        print(f'\nSaved passive regression tuning results to {FIGURES_DIR}')
        print(f'Used {N_TRIALS} trials × {N_FOLDS} folds = {N_TRIALS * N_FOLDS} evaluations per config')

    def plot_uncertainty_results(self):
        """Plot uncertainty-based active learning results."""
        # Implementation for plotting uncertainty results
        # This would create learning curves for each dataset and method
        pass

    def plot_sensitivity_results(self):
        """Plot sensitivity-based active learning results."""
        # Implementation for plotting sensitivity results
        # This would create learning curves for each dataset
        pass

    def run_all(self):
        """Run all three methods: passive, uncertainty, and sensitivity."""
        print("="*80)
        print("COMBINED REGRESSION HYPERPARAMETER TUNING")
        print("="*80)
        print(f"Datasets: {self.datasets}")
        print(f"Methods: Passive, Uncertainty-based, Sensitivity-based")
        print(f"Trials per config: {N_TRIALS}")
        print(f"CV folds: {N_FOLDS}")
        print("="*80)
        
        # Run passive learning
        self.run_passive_tuning()
        
        # Run uncertainty-based active learning
        self.run_uncertainty_tuning()
        
        # Run sensitivity-based active learning
        self.run_sensitivity_tuning()
        
        # Plot all results
        self.plot_results()
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED!")
        print("="*80)


def main():
    """Main function to run the combined regression tuning."""
    parser = argparse.ArgumentParser(description='Combined Regression Hyperparameter Tuning')
    parser.add_argument('--datasets', nargs='+', default=DATASETS, 
                       help='Datasets to use for tuning')
    parser.add_argument('--method', choices=['passive', 'uncertainty', 'sensitivity', 'all'], 
                       default='all', help='Which method to run')
    
    args = parser.parse_args()
    
    # Create tuner instance
    tuner = RegressionTuner(datasets=args.datasets)
    
    # Run specified method(s)
    # if args.method == 'passive':
    #     tuner.run_passive_tuning()
    if args.method == 'uncertainty':
        tuner.run_uncertainty_tuning()
    elif args.method == 'sensitivity':
        tuner.run_sensitivity_tuning()
    elif args.method == 'all':
        tuner.run_all()
    
    # Always plot results
    tuner.plot_results()


if __name__ == "__main__":
    main()
