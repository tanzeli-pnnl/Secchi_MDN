"""Dataset loading helpers for the Maciel et al. spreadsheets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SensorSpec:
    sensor: str
    filename: str
    input_bands: tuple[str, ...]
    default_bands: tuple[str, ...]


SENSOR_SPECS: dict[str, SensorSpec] = {
    "tm": SensorSpec(
        sensor="tm",
        filename="rrs_tm_v3.xlsx",
        input_bands=("blue", "green", "red", "nir"),
        default_bands=("blue", "green", "red"),
    ),
    "etm": SensorSpec(
        sensor="etm",
        filename="rrs_etm_v3.xlsx",
        input_bands=("blue", "green", "red", "nir"),
        default_bands=("blue", "green", "red"),
    ),
    "oli": SensorSpec(
        sensor="oli",
        filename="rrs_oli_v3.xlsx",
        input_bands=("coastal", "blue", "green", "red", "nir"),
        default_bands=("blue", "green", "red"),
    ),
}

COMMON_COLUMNS = {
    "station_id": "station_id",
    "region": "region",
    "local": "local",
    "lat": "lat",
    "long": "lon",
    "date": "date",
    "secchi (m)": "secchi_m",
}


def _band_column_name(band: str) -> str:
    return f"{band} (sr-1)"


def load_sensor_dataframe(dataset_dir: str | Path, sensor: str) -> pd.DataFrame:
    """Load one of the provided sensor spreadsheets into a standard schema."""
    key = sensor.lower()
    if key not in SENSOR_SPECS:
        raise ValueError(f"Unknown sensor '{sensor}'. Expected one of {sorted(SENSOR_SPECS)}.")

    spec = SENSOR_SPECS[key]
    path = Path(dataset_dir).expanduser() / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"Spreadsheet not found: {path}")

    rename_map = {**COMMON_COLUMNS, **{_band_column_name(b): b for b in spec.input_bands}}
    # Normalize the spreadsheet-specific labels into a stable schema used by the trainer.
    frame = pd.read_excel(path, engine="openpyxl").rename(columns=rename_map)
    expected_columns = list(rename_map.values())
    missing = [column for column in expected_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path.name}: {missing}")

    frame = frame.loc[:, expected_columns].copy()
    frame["sensor"] = key
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "secchi_m", *spec.default_bands])
    frame = frame[frame["secchi_m"] > 0].copy()
    # The published scripts split by a combined location/month key to reduce leakage.
    frame["group_key"] = frame["local"].astype(str) + "_" + frame["date"].dt.to_period("M").astype(str)
    frame["row_id"] = frame.index.astype(str)
    return frame.reset_index(drop=True)
