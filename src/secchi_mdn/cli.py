"""CLI for training Secchi Mixture Density Networks."""

from __future__ import annotations

import argparse
import json
import sys

from .trainer import TrainingConfig, train_many


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Maciel et al. (2023)-style Secchi MDN models from the supplied XLSX files."
    )
    parser.add_argument(
        "command",
        choices=["train"],
        help="Only 'train' is currently implemented.",
    )
    parser.add_argument("--dataset-dir", required=False, help="Directory containing the Maciel XLSX files.")
    parser.add_argument("--output-dir", default="outputs", help="Directory where models and metrics are saved.")
    parser.add_argument("--sensor", choices=["tm", "etm", "oli", "all"], default="all")
    parser.add_argument("--include-ratios", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-coastal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--monte-carlo-runs", type=int, default=50)
    parser.add_argument("--ensemble-size", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--n-mix", type=int, default=5)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[100, 100, 100, 100, 100])
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--bagging-fraction", type=float, default=0.75)
    parser.add_argument("--prediction-mode", choices=["top", "mean"], default="top")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "train":
        parser.error(f"Unsupported command: {args.command}")

    if not args.dataset_dir:
        parser.error("--dataset-dir is required for training.")

    sensors = ["tm", "etm", "oli"] if args.sensor == "all" else [args.sensor]
    base_config = TrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sensor=sensors[0],
        include_ratios=args.include_ratios,
        include_coastal=args.include_coastal,
        monte_carlo_runs=args.monte_carlo_runs,
        ensemble_size=args.ensemble_size,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        n_mix=args.n_mix,
        hidden_dims=tuple(args.hidden_dims),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        bagging_fraction=args.bagging_fraction,
        prediction_mode=args.prediction_mode,
        seed=args.seed,
    )

    try:
        results = train_many(base_config, sensors)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(results, indent=2))
    return 0
