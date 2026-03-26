#!/usr/bin/env python3
"""Summarize ACOLITE validation results across raw and filtered matchup sets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TABLE1_VALID_MATCHUPS = {
    "tm": 629,
    "etm": 1856,
    "oli": 1049,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        default="outputs_final_20260326",
        help="Root directory containing acolite_validation outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the comparison report. Defaults to <model-root>/validation_comparison.",
    )
    return parser


def _load_metrics(path: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return (
        frame.loc[
            (frame["reference"] == "observed_secchi_m")
            & (frame["model"] == model)
            & (frame["subset"] == "positive_observed_only")
        ]
        .loc[:, ["sensor", "n", "mae", "rmse", "r2", "mape_percent", "epsilon_percent", "beta_percent"]]
        .copy()
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_root = Path(args.model_root).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else model_root / "validation_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_summary = model_root / "acolite_validation" / "comparison_metrics_summary.csv"
    filtered_summary = model_root / "acolite_validation_paper_filters" / "comparison_metrics_summary.csv"
    filtered_counts = model_root / "acolite_validation_paper_filters" / "paper_filter_summary.csv"

    ours_raw = _load_metrics(raw_summary, "our_final_mdn").rename(
        columns={
            "n": "raw_n",
            "mae": "raw_mae",
            "rmse": "raw_rmse",
            "r2": "raw_r2",
            "mape_percent": "raw_mape_percent",
            "epsilon_percent": "raw_epsilon_percent",
            "beta_percent": "raw_beta_percent",
        }
    )
    ours_filtered = _load_metrics(filtered_summary, "our_final_mdn").rename(
        columns={
            "n": "filtered_n",
            "mae": "filtered_mae",
            "rmse": "filtered_rmse",
            "r2": "filtered_r2",
            "mape_percent": "filtered_mape_percent",
            "epsilon_percent": "filtered_epsilon_percent",
            "beta_percent": "filtered_beta_percent",
        }
    )
    maciel_filtered = _load_metrics(filtered_summary, "maciel_2023_prediction").rename(
        columns={
            "n": "maciel_filtered_n",
            "mae": "maciel_filtered_mae",
            "rmse": "maciel_filtered_rmse",
            "r2": "maciel_filtered_r2",
            "mape_percent": "maciel_filtered_mape_percent",
            "epsilon_percent": "maciel_filtered_epsilon_percent",
            "beta_percent": "maciel_filtered_beta_percent",
        }
    )
    counts = pd.read_csv(filtered_counts).rename(
        columns={
            "raw_rows": "acolite_raw_rows",
            "kept_rows": "paper_filter_kept_rows",
            "removed_rows": "paper_filter_removed_rows",
        }
    )

    summary = ours_raw.merge(ours_filtered, on="sensor").merge(maciel_filtered, on="sensor").merge(
        counts.loc[:, ["sensor", "acolite_raw_rows", "paper_filter_kept_rows", "paper_filter_removed_rows"]],
        on="sensor",
    )
    summary["table1_valid_matchups"] = summary["sensor"].map(TABLE1_VALID_MATCHUPS)
    summary["gap_vs_table1_after_filter"] = summary["paper_filter_kept_rows"] - summary["table1_valid_matchups"]

    ordered_columns = [
        "sensor",
        "table1_valid_matchups",
        "acolite_raw_rows",
        "paper_filter_kept_rows",
        "gap_vs_table1_after_filter",
        "raw_n",
        "raw_mae",
        "raw_rmse",
        "raw_r2",
        "filtered_n",
        "filtered_mae",
        "filtered_rmse",
        "filtered_r2",
        "maciel_filtered_n",
        "maciel_filtered_mae",
        "maciel_filtered_rmse",
        "maciel_filtered_r2",
    ]
    summary = summary.loc[:, ordered_columns].sort_values("sensor")
    summary.to_csv(output_dir / "validation_comparison_summary.csv", index=False)

    lines = [
        "# Validation Comparison",
        "",
        "| Sensor | Table 1 Valid Matchups | Raw ACOLITE Rows | Paper-Filter Rows | Gap vs Table 1 | Our Raw RMSE | Our Filtered RMSE | Maciel Filtered RMSE | Our Raw R2 | Our Filtered R2 | Maciel Filtered R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.sensor.upper()} | {row.table1_valid_matchups} | {row.acolite_raw_rows} | {row.paper_filter_kept_rows} | "
            f"{row.gap_vs_table1_after_filter:+d} | {row.raw_rmse:.4f} | {row.filtered_rmse:.4f} | "
            f"{row.maciel_filtered_rmse:.4f} | {row.raw_r2:.4f} | {row.filtered_r2:.4f} | {row.maciel_filtered_r2:.4f} |"
        )
    (output_dir / "validation_comparison_summary.md").write_text("\n".join(lines) + "\n")

    print(f"Saved comparison report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
