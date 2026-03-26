#!/usr/bin/env python3
"""Plot measured-vs-estimated Secchi for saved final-model training outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.secchi_mdn.plotting import save_scatterplot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs_final", help="Directory containing sensor output folders.")
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["tm", "etm", "oli"],
        choices=["tm", "etm", "oli"],
        help="Sensors to plot.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Directory where plots are saved. Defaults to <output-root>/plots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    matplotlib.use("Agg")
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).expanduser()
    plot_dir = Path(args.plot_dir).expanduser() if args.plot_dir else output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for sensor in args.sensors:
        predictions_path = output_root / sensor / "final_model" / "full_dataset_predictions.csv"
        frame = pd.read_csv(predictions_path)
        save_scatterplot(
            frame["observed_secchi_m"],
            frame["predicted_secchi_m"],
            plot_dir / f"{sensor}_measured_vs_estimated_secchi.png",
            title=f"{sensor.upper()} Final MDN: Measured vs Estimated Secchi",
            xlabel="Measured Secchi depth (m)",
            ylabel="Estimated Secchi depth (m)",
        )

    print(f"Saved plots to {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
