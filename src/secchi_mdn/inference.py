"""Inference helpers for saved final MDN ensembles."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build_feature_matrix
from .model import SecchiMDNFactory, require_torch
from .trainer import _inverse_target, _predict_ensemble


def load_final_ensemble(model_root: str | Path, sensor: str):
    """Load a saved final ensemble and its preprocessing bundle."""
    torch, _ = require_torch()
    sensor_key = sensor.lower()
    model_dir = Path(model_root).expanduser() / sensor_key / "final_model"
    prep = joblib.load(model_dir / "preprocessing.joblib")
    config = prep["config"]

    models = []
    for member_path in sorted(model_dir.glob("member_*.pt")):
        _, model = SecchiMDNFactory.create(
            input_dim=len(prep["feature_names"]),
            n_mix=config["n_mix"],
            hidden_dims=list(config["hidden_dims"]),
        )
        state = torch.load(member_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    return {
        "model_dir": model_dir,
        "config": config,
        "feature_names": tuple(prep["feature_names"]),
        "band_columns": tuple(prep["band_columns"]),
        "x_scaler": prep["x_scaler"],
        "y_scaler": prep["y_scaler"],
        "models": models,
    }


def predict_from_frame(bundle: dict[str, object], frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Build features for a dataframe and predict Secchi with a saved ensemble."""
    features, _ = build_feature_matrix(
        frame,
        band_columns=tuple(bundle["band_columns"]),
        include_ratios=bool(bundle["config"]["include_ratios"]),
    )
    aligned = frame.loc[features.index].copy()
    X_scaled = bundle["x_scaler"].transform(features.to_numpy(dtype=float)).astype(np.float32)
    pred_scaled = _predict_ensemble(bundle["models"], X_scaled, bundle["config"]["prediction_mode"])
    pred_secchi = np.clip(_inverse_target(bundle["y_scaler"], pred_scaled), a_min=1.0e-6, a_max=None)
    return aligned, pred_secchi
