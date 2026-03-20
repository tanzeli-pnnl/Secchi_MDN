"""Training orchestration for the Secchi MDN workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from .data import SENSOR_SPECS, load_sensor_dataframe
from .features import build_feature_matrix
from .metrics import summarize_regression
from .model import MDNOutputs, SecchiMDNFactory, mdn_nll_loss, mdn_predict, require_torch


@dataclass
class TrainingConfig:
    dataset_dir: str
    output_dir: str
    sensor: str
    fit_mode: str = "final"
    include_ratios: bool = True
    include_coastal: bool = False
    monte_carlo_runs: int = 50
    ensemble_size: int = 10
    test_fraction: float = 0.30
    validation_fraction: float = 0.15
    n_mix: int = 5
    hidden_dims: tuple[int, ...] = (100, 100, 100, 100, 100)
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-3
    epsilon: float = 1.0e-3
    batch_size: int = 128
    max_epochs: int = 500
    patience: int = 40
    bagging_fraction: float = 0.75
    prediction_mode: str = "top"
    seed: int = 42


def _selected_bands(sensor: str, include_coastal: bool) -> tuple[str, ...]:
    spec = SENSOR_SPECS[sensor]
    if sensor == "oli" and include_coastal:
        return ("coastal", "blue", "green", "red")
    return spec.default_bands


def _align_target(features: pd.DataFrame, target: pd.Series, meta: pd.DataFrame):
    aligned_target = target.loc[features.index]
    aligned_meta = meta.loc[features.index]
    return aligned_target, aligned_meta


def _fit_scalers(X_train: np.ndarray, y_train: np.ndarray):
    x_scaler = RobustScaler()
    y_log = np.log(y_train).reshape(-1, 1)
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaler.fit(X_train)
    y_scaler.fit(y_log)
    return x_scaler, y_scaler


def _transform_xy(x_scaler, y_scaler, X: np.ndarray, y: np.ndarray | None = None):
    X_scaled = x_scaler.transform(X).astype(np.float32)
    if y is None:
        return X_scaled, None
    y_scaled = y_scaler.transform(np.log(y).reshape(-1, 1)).reshape(-1).astype(np.float32)
    return X_scaled, y_scaled


def _inverse_target(y_scaler, y_scaled: np.ndarray) -> np.ndarray:
    clipped = np.clip(y_scaled, -1.0, 1.0)
    return np.exp(y_scaler.inverse_transform(clipped.reshape(-1, 1)).reshape(-1))


def _build_loader(torch, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_member(
    config: TrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
):
    torch, _ = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if 0 < config.bagging_fraction < 1:
        rng = np.random.default_rng(seed)
        subset_size = max(2, int(len(X_train) * config.bagging_fraction))
        subset_idx = rng.choice(len(X_train), size=subset_size, replace=False)
        X_fit = X_train[subset_idx]
        y_fit = y_train[subset_idx]
    else:
        X_fit = X_train
        y_fit = y_train

    torch, model = SecchiMDNFactory.create(
        input_dim=X_train.shape[1],
        n_mix=config.n_mix,
        hidden_dims=list(config.hidden_dims),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    train_loader = _build_loader(torch, X_fit, y_fit, config.batch_size, shuffle=True)
    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    X_valid_tensor = torch.from_numpy(X_valid)
    y_valid_tensor = torch.from_numpy(y_valid)

    for _epoch in range(config.max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = mdn_nll_loss(outputs, batch_y, epsilon=config.epsilon)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = float(mdn_nll_loss(model(X_valid_tensor), y_valid_tensor, epsilon=config.epsilon).item())

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_loss


def _predict_ensemble(models, X_scaled: np.ndarray, prediction_mode: str) -> np.ndarray:
    torch, _ = require_torch()
    predictions = []
    X_tensor = torch.from_numpy(X_scaled)
    for model in models:
        with torch.no_grad():
            output: MDNOutputs = model(X_tensor)
            predictions.append(mdn_predict(output, mode=prediction_mode).cpu().numpy())
    return np.median(np.vstack(predictions), axis=0)


def train_sensor_model(config: TrainingConfig) -> dict[str, object]:
    """Train a grouped Monte Carlo MDN for one sensor."""
    sensor = config.sensor.lower()
    if sensor not in SENSOR_SPECS:
        raise ValueError(f"Unknown sensor '{config.sensor}'. Expected one of {sorted(SENSOR_SPECS)}.")

    frame = load_sensor_dataframe(config.dataset_dir, sensor)
    band_columns = _selected_bands(sensor, config.include_coastal)
    feature_frame, feature_set = build_feature_matrix(frame, band_columns, include_ratios=config.include_ratios)
    target, meta = _align_target(feature_frame, frame["secchi_m"], frame)

    X_all = feature_frame.to_numpy(dtype=float)
    y_all = target.to_numpy(dtype=float)
    groups = meta["group_key"].to_numpy()

    splitter = GroupShuffleSplit(
        n_splits=config.monte_carlo_runs,
        test_size=config.test_fraction,
        random_state=config.seed,
    )

    output_root = Path(config.output_dir).expanduser() / sensor
    output_root.mkdir(parents=True, exist_ok=True)

    torch, _ = require_torch()

    if config.fit_mode == "final":
        valid_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=config.validation_fraction,
            random_state=config.seed,
        )
        fit_idx, valid_idx = next(valid_splitter.split(X_all, y_all, groups=groups))
        X_fit_raw = X_all[fit_idx]
        y_fit_raw = y_all[fit_idx]
        X_valid_raw = X_all[valid_idx]
        y_valid_raw = y_all[valid_idx]

        x_scaler, y_scaler = _fit_scalers(X_fit_raw, y_fit_raw)
        X_fit, y_fit = _transform_xy(x_scaler, y_scaler, X_fit_raw, y_fit_raw)
        X_valid, y_valid = _transform_xy(x_scaler, y_scaler, X_valid_raw, y_valid_raw)
        X_all_scaled, _ = _transform_xy(x_scaler, y_scaler, X_all)

        models = []
        member_losses = []
        for member_index in range(config.ensemble_size):
            member_seed = config.seed + member_index
            model, val_loss = _train_member(config, X_fit, y_fit, X_valid, y_valid, member_seed)
            models.append(model)
            member_losses.append(val_loss)

        valid_pred_scaled = _predict_ensemble(models, X_valid, config.prediction_mode)
        valid_pred = np.clip(_inverse_target(y_scaler, valid_pred_scaled), a_min=1.0e-6, a_max=None)
        valid_metrics = summarize_regression(y_valid_raw, valid_pred)
        valid_metrics.update(
            {
                "mean_member_val_nll": float(np.mean(member_losses)),
                "n_features": len(feature_set.feature_names),
            }
        )

        full_pred_scaled = _predict_ensemble(models, X_all_scaled, config.prediction_mode)
        full_pred = np.clip(_inverse_target(y_scaler, full_pred_scaled), a_min=1.0e-6, a_max=None)

        final_dir = output_root / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_names": feature_set.feature_names,
                "band_columns": feature_set.band_columns,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
                "config": asdict(config),
            },
            final_dir / "preprocessing.joblib",
        )
        for member_index, model in enumerate(models, start=1):
            torch.save(model.state_dict(), final_dir / f"member_{member_index:02d}.pt")

        validation_frame = meta.iloc[valid_idx].copy()
        validation_frame["observed_secchi_m"] = y_valid_raw
        validation_frame["predicted_secchi_m"] = valid_pred
        validation_frame.to_csv(final_dir / "validation_predictions.csv", index=False)

        full_frame = meta.copy()
        full_frame["observed_secchi_m"] = y_all
        full_frame["predicted_secchi_m"] = full_pred
        full_frame.to_csv(final_dir / "full_dataset_predictions.csv", index=False)
        pd.DataFrame([valid_metrics]).to_csv(final_dir / "validation_metrics.csv", index=False)

        summary = {
            "sensor": sensor,
            "fit_mode": config.fit_mode,
            "rows": int(len(frame)),
            "groups": int(pd.Series(groups).nunique()),
            "feature_names": list(feature_set.feature_names),
            "validation_metrics": valid_metrics,
        }
        (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    run_summaries: list[dict[str, float]] = []
    all_predictions: list[pd.DataFrame] = []

    for run_index, (train_idx, test_idx) in enumerate(splitter.split(X_all, y_all, groups=groups), start=1):
        X_train_full = X_all[train_idx]
        y_train_full = y_all[train_idx]
        train_groups = groups[train_idx]
        X_test = X_all[test_idx]
        y_test = y_all[test_idx]

        valid_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=config.validation_fraction,
            random_state=config.seed + run_index,
        )
        fit_idx, valid_idx = next(valid_splitter.split(X_train_full, y_train_full, groups=train_groups))
        X_fit_raw = X_train_full[fit_idx]
        y_fit_raw = y_train_full[fit_idx]
        X_valid_raw = X_train_full[valid_idx]
        y_valid_raw = y_train_full[valid_idx]

        x_scaler, y_scaler = _fit_scalers(X_fit_raw, y_fit_raw)
        X_fit, y_fit = _transform_xy(x_scaler, y_scaler, X_fit_raw, y_fit_raw)
        X_valid, y_valid = _transform_xy(x_scaler, y_scaler, X_valid_raw, y_valid_raw)
        X_test_scaled, _ = _transform_xy(x_scaler, y_scaler, X_test)

        models = []
        member_losses = []
        for member_index in range(config.ensemble_size):
            member_seed = config.seed + (run_index * 1_000) + member_index
            model, val_loss = _train_member(config, X_fit, y_fit, X_valid, y_valid, member_seed)
            models.append(model)
            member_losses.append(val_loss)

        pred_scaled = _predict_ensemble(models, X_test_scaled, config.prediction_mode)
        pred_secchi = np.clip(_inverse_target(y_scaler, pred_scaled), a_min=1.0e-6, a_max=None)
        metrics = summarize_regression(y_test, pred_secchi)
        metrics.update(
            {
                "run": run_index,
                "mean_member_val_nll": float(np.mean(member_losses)),
                "n_features": len(feature_set.feature_names),
            }
        )
        run_summaries.append(metrics)

        test_meta = meta.iloc[test_idx].copy()
        test_meta["observed_secchi_m"] = y_test
        test_meta["predicted_secchi_m"] = pred_secchi
        test_meta["run"] = run_index
        all_predictions.append(test_meta)

        run_dir = output_root / f"run_{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_names": feature_set.feature_names,
                "band_columns": feature_set.band_columns,
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
                "config": asdict(config),
            },
            run_dir / "preprocessing.joblib",
        )
        for member_index, model in enumerate(models, start=1):
            torch.save(model.state_dict(), run_dir / f"member_{member_index:02d}.pt")
        pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
        test_meta.to_csv(run_dir / "predictions.csv", index=False)

    metrics_frame = pd.DataFrame(run_summaries)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    metrics_frame.to_csv(output_root / "monte_carlo_metrics.csv", index=False)
    predictions_frame.to_csv(output_root / "monte_carlo_predictions.csv", index=False)

    summary = {
        "sensor": sensor,
        "fit_mode": config.fit_mode,
        "rows": int(len(frame)),
        "groups": int(pd.Series(groups).nunique()),
        "feature_names": list(feature_set.feature_names),
        "metrics_mean": metrics_frame.mean(numeric_only=True).to_dict(),
        "metrics_std": metrics_frame.std(numeric_only=True).to_dict(),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def train_many(config: TrainingConfig, sensors: list[str]) -> list[dict[str, object]]:
    return [train_sensor_model(TrainingConfig(**{**asdict(config), "sensor": sensor})) for sensor in sensors]
