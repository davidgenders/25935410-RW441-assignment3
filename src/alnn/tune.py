import argparse
import itertools
import json
import math
import random
from typing import Dict, List, Tuple

from .experiments import (
    ActiveConfig,
    run_active_classification,
    run_active_regression,
    run_passive_classification,
    run_passive_regression,
)
from .training import TrainConfig


def _product(sample_space: Dict[str, List]):
    keys = list(sample_space.keys())
    for values in itertools.product(*[sample_space[k] for k in keys]):
        yield dict(zip(keys, values))


def _random_choices(sample_space: Dict[str, List], n: int) -> List[Dict]:
    keys = list(sample_space.keys())
    return [
        {k: random.choice(sample_space[k]) for k in keys}
        for _ in range(n)
    ]


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuner for ALNN (random/grid search)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Passive classification
    pcls = subparsers.add_parser("passive-cls")
    pcls.add_argument("dataset", choices=["iris", "wine", "breast_cancer"]) 
    # Active classification
    acls = subparsers.add_parser("active-cls")
    acls.add_argument("dataset", choices=["iris", "wine", "breast_cancer"]) 
    acls.add_argument("--strategy", choices=["uncertainty", "sensitivity"], default="uncertainty")
    acls.add_argument("--method", choices=["entropy", "margin", "least_confidence"], default="entropy")

    # Passive regression
    preg = subparsers.add_parser("passive-reg")
    preg.add_argument("dataset", choices=["diabetes", "linnerud", "california"]) 
    # Active regression
    areg = subparsers.add_parser("active-reg")
    areg.add_argument("dataset", choices=["diabetes", "linnerud", "california"]) 
    areg.add_argument("--strategy", choices=["uncertainty", "sensitivity"], default="uncertainty")
    areg.add_argument("--method", choices=["entropy", "margin", "least_confidence"], default="entropy")

    # Shared search config
    parser.add_argument("--search", choices=["grid", "random"], default="random")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials for random search")
    parser.add_argument("--device", type=str, default="cpu")

    # Train config search spaces
    parser.add_argument("--lrs", type=float, nargs="*", default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    parser.add_argument("--wds", type=float, nargs="*", default=[0.0, 1e-5, 1e-4, 1e-3])
    parser.add_argument("--bss", type=int, nargs="*", default=[32, 64, 128])
    parser.add_argument("--hidden", type=int, nargs="*", default=[32, 64, 128])
    parser.add_argument("--patience", type=int, nargs="*", default=[10, 20, 40])

    # Active config search spaces
    parser.add_argument("--inits", type=int, nargs="*", default=[10, 20, 40])
    parser.add_argument("--queries", type=int, nargs="*", default=[5, 10, 20])
    parser.add_argument("--max_labels", type=int, default=200)

    args = parser.parse_args()

    random.seed(42)

    # Assemble search space
    train_space = {
        "lr": args.lrs,
        "wd": args.wds,
        "bs": args.bss,
        "hidden": args.hidden,
        "pat": args.patience,
    }

    active_space = {
        "init": args.inits,
        "query": args.queries,
    }

    if args.search == "grid":
        hp_iter = list(_product({**train_space, **active_space})) if args.mode.startswith("active") else list(_product(train_space))
    else:
        size = args.trials
        hp_iter = _random_choices({**train_space, **active_space}, size) if args.mode.startswith("active") else _random_choices(train_space, size)

    results: List[Tuple[Dict, Dict]] = []

    for hp in hp_iter:
        tcfg = TrainConfig(
            learning_rate=hp["lr"],
            weight_decay=hp["wd"],
            batch_size=hp["bs"],
            max_epochs=200,
            patience=hp["pat"],
            device=args.device,
        )

        if args.mode == "passive-cls":
            res = run_passive_classification(args.dataset, hidden_units=hp["hidden"], config=tcfg)
        elif args.mode == "passive-reg":
            res = run_passive_regression(args.dataset, hidden_units=hp["hidden"], config=tcfg)
        elif args.mode == "active-cls":
            acfg = ActiveConfig(initial_labeled=hp["init"], query_batch=hp["query"], max_labels=args.max_labels, device=args.device)
            res = run_active_classification(
                args.dataset,
                strategy=args.strategy,
                uncertainty_method=args.method,
                hidden_units=hp["hidden"],
                train_config=tcfg,
                active_config=acfg,
            )
        else:
            acfg = ActiveConfig(initial_labeled=hp["init"], query_batch=hp["query"], max_labels=args.max_labels, device=args.device)
            res = run_active_regression(
                args.dataset,
                strategy=args.strategy,
                uncertainty_method=args.method,
                hidden_units=hp["hidden"],
                train_config=tcfg,
                active_config=acfg,
            )

        results.append((hp, res))
        out = {**hp, **res}
        print(json.dumps(out))

    # Optionally pick best by a metric
    # We will select by the first metric key present in results for simplicity
    if results:
        metric_key = next(iter(results[0][1].keys()))
        reverse = metric_key not in ("train_loss", "val_loss", "best_val_loss", "rmse")
        best_hp, best_res = sorted(results, key=lambda x: x[1][metric_key], reverse=reverse)[0]
        summary = {"best": {**best_hp, **best_res}, "metric": metric_key}
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


