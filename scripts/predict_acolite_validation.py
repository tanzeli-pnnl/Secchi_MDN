#!/usr/bin/env python3
"""Apply saved final MDNs to the ACOLITE matchup files and compare with references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

from src.secchi_mdn.inference import load_final_ensemble, predict_from_frame
from src.secchi_mdn.metrics import summarize_regression
from src.secchi_mdn.plotting import save_three_panel_comparison


SENSOR_FILES = {
    "tm": "tm_acolite_maciel_v3.xlsx",
    "etm": "etm_acolite_maciel_v3.xlsx",
    "oli": "oli_acolite_maciel_v3.xlsx",
}

RENAME_BY_SENSOR = {
    "tm": {
        "secchi.(m)": "secchi_m",
        "blue.(sr-1)": "blue",
        "green.(sr-1)": "green",
        "red.(sr-1)": "red",
        "nir.(sr-1)": "nir",
        "Secchi.predicted.(m)": "maciel_predicted_secchi_m",
        "Image_Name": "image_name",
    },
    "etm": {
        "secchi.(m)": "secchi_m",
        "blue.(sr-1)": "blue",
        "green.(sr-1)": "green",
        "red.(sr-1)": "red",
        "nir.(sr-1)": "nir",
        "Secchi.predicted.(m)": "maciel_predicted_secchi_m",
        "Image_name": "image_name",
    },
    "oli": {
        "secchi.(m)": "secchi_m",
        "coastal.(sr-1)": "coastal",
        "blue.(sr-1)": "blue",
        "green.(sr-1)": "green",
        "red.(sr-1)": "red",
        "nir.(sr-1)": "nir",
        "Secchi.predicted.(m)": "maciel_predicted_secchi_m",
        "Image_Name": "image_name",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", default="outputs_final", help="Directory containing saved final MDN models.")
    parser.add_argument("--data-dir", required=True, help="Directory containing the ACOLITE matchup XLSX files.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for comparison CSVs and plots. Defaults to <model-root>/acolite_validation.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["tm", "etm", "oli"],
        choices=["tm", "etm", "oli"],
        help="Sensors to evaluate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    matplotlib.use("Agg")
    args = build_parser().parse_args(argv)

    model_root = Path(args.model_root).expanduser()
    data_dir = Path(args.data_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else model_root / "acolite_validation"
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for sensor in args.sensors:
        bundle = load_final_ensemble(model_root, sensor)
        frame = pd.read_excel(data_dir / SENSOR_FILES[sensor], engine="openpyxl").rename(columns=RENAME_BY_SENSOR[sensor])
        aligned, pred_secchi = predict_from_frame(bundle, frame)

        aligned["our_predicted_secchi_m"] = pred_secchi
        aligned["prediction_difference_vs_maciel_m"] = aligned["our_predicted_secchi_m"] - aligned["maciel_predicted_secchi_m"]
        aligned["abs_error_ours_vs_observed_m"] = (aligned["our_predicted_secchi_m"] - aligned["secchi_m"]).abs()
        aligned["abs_error_maciel_vs_observed_m"] = (aligned["maciel_predicted_secchi_m"] - aligned["secchi_m"]).abs()

        sensor_dir = output_dir / sensor
        sensor_dir.mkdir(parents=True, exist_ok=True)
        aligned.to_csv(sensor_dir / "predictions_with_comparison.csv", index=False)

        metrics_frame = pd.DataFrame(
            [
                {
                    "sensor": sensor,
                    "reference": "observed_secchi_m",
                    "model": "our_final_mdn",
                    **summarize_regression(aligned["secchi_m"].to_numpy(), aligned["our_predicted_secchi_m"].to_numpy()),
                },
                {
                    "sensor": sensor,
                    "reference": "observed_secchi_m",
                    "model": "maciel_2023_prediction",
                    **summarize_regression(aligned["secchi_m"].to_numpy(), aligned["maciel_predicted_secchi_m"].to_numpy()),
                },
                {
                    "sensor": sensor,
                    "reference": "maciel_predicted_secchi_m",
                    "model": "our_final_mdn",
                    **summarize_regression(
                        aligned["maciel_predicted_secchi_m"].to_numpy(),
                        aligned["our_predicted_secchi_m"].to_numpy(),
                    ),
                },
            ]
        )
        metrics_frame.to_csv(sensor_dir / "comparison_metrics.csv", index=False)
        summary_rows.extend(metrics_frame.to_dict(orient="records"))

        save_three_panel_comparison(
            plot_dir / f"{sensor}_comparison.png",
            sensor=sensor,
            observed=aligned["secchi_m"],
            ours=aligned["our_predicted_secchi_m"],
            maciel=aligned["maciel_predicted_secchi_m"],
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "comparison_metrics_summary.csv", index=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "data_root": str(data_dir),
                "model_root": str(model_root),
                "sensors": list(args.sensors),
            },
            indent=2,
        )
    )
    print(f"Saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
