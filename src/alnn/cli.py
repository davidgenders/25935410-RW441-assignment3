import argparse
import json

from .experiments import (
    ActiveConfig,
    run_active_classification,
    run_active_regression,
    run_passive_classification,
    run_passive_regression,
)
from .training import TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Active Learning for Neural Networks Experiments")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Passive
    p_cls = subparsers.add_parser("passive-cls")
    p_cls.add_argument("dataset", choices=["iris", "wine", "breast_cancer"])
    p_cls.add_argument("--hidden", type=int, default=64)
    p_cls.add_argument("--lr", type=float, default=1e-2)
    p_cls.add_argument("--wd", type=float, default=1e-4)
    p_cls.add_argument("--epochs", type=int, default=200)
    p_cls.add_argument("--bs", type=int, default=64)
    p_cls.add_argument("--device", type=str, default="cpu")

    p_reg = subparsers.add_parser("passive-reg")
    p_reg.add_argument("dataset", choices=["diabetes", "linnerud", "california"]) 
    p_reg.add_argument("--hidden", type=int, default=64)
    p_reg.add_argument("--lr", type=float, default=1e-2)
    p_reg.add_argument("--wd", type=float, default=1e-4)
    p_reg.add_argument("--epochs", type=int, default=200)
    p_reg.add_argument("--bs", type=int, default=64)
    p_reg.add_argument("--device", type=str, default="cpu")

    # Active
    a_cls = subparsers.add_parser("active-cls")
    a_cls.add_argument("dataset", choices=["iris", "wine", "breast_cancer"]) 
    a_cls.add_argument("--strategy", choices=["uncertainty", "sensitivity"], default="uncertainty")
    a_cls.add_argument("--method", choices=["entropy", "margin", "least_confidence"], default="entropy", help="Uncertainty method when strategy=uncertainty")
    a_cls.add_argument("--hidden", type=int, default=64)
    a_cls.add_argument("--init", type=int, default=20)
    a_cls.add_argument("--query", type=int, default=10)
    a_cls.add_argument("--max_labels", type=int, default=200)
    a_cls.add_argument("--lr", type=float, default=1e-2)
    a_cls.add_argument("--wd", type=float, default=1e-4)
    a_cls.add_argument("--epochs", type=int, default=200)
    a_cls.add_argument("--bs", type=int, default=64)
    a_cls.add_argument("--device", type=str, default="cpu")

    a_reg = subparsers.add_parser("active-reg")
    a_reg.add_argument("dataset", choices=["diabetes", "linnerud", "california"]) 
    a_reg.add_argument("--strategy", choices=["uncertainty", "sensitivity"], default="uncertainty")
    a_reg.add_argument("--method", choices=["entropy", "margin", "least_confidence"], default="entropy", help="Uncertainty method when strategy=uncertainty")
    a_reg.add_argument("--hidden", type=int, default=64)
    a_reg.add_argument("--init", type=int, default=20)
    a_reg.add_argument("--query", type=int, default=10)
    a_reg.add_argument("--max_labels", type=int, default=200)
    a_reg.add_argument("--lr", type=float, default=1e-2)
    a_reg.add_argument("--wd", type=float, default=1e-4)
    a_reg.add_argument("--epochs", type=int, default=200)
    a_reg.add_argument("--bs", type=int, default=64)
    a_reg.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    tcfg = TrainConfig(
        learning_rate=args.lr,
        weight_decay=args.wd,
        batch_size=args.bs,
        max_epochs=args.epochs,
        device=args.device,
    )

    if args.mode == "passive-cls":
        res = run_passive_classification(args.dataset, hidden_units=args.hidden, config=tcfg)
    elif args.mode == "passive-reg":
        res = run_passive_regression(args.dataset, hidden_units=args.hidden, config=tcfg)
    elif args.mode == "active-cls":
        acfg = ActiveConfig(initial_labeled=args.init, query_batch=args.query, max_labels=args.max_labels, device=args.device)
        res = run_active_classification(args.dataset, strategy=args.strategy, uncertainty_method=args.method, hidden_units=args.hidden, train_config=tcfg, active_config=acfg)
    else:
        acfg = ActiveConfig(initial_labeled=args.init, query_batch=args.query, max_labels=args.max_labels, device=args.device)
        res = run_active_regression(args.dataset, strategy=args.strategy, uncertainty_method=args.method, hidden_units=args.hidden, train_config=tcfg, active_config=acfg)

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()


