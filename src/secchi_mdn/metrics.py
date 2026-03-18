"""Metrics aligned with the paper's reporting style."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def median_symmetric_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ratio = np.log(np.asarray(y_pred) / np.asarray(y_true))
    return float(100.0 * (np.exp(np.median(np.abs(ratio))) - 1.0))


def symmetric_signed_percentage_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ratio = np.log(np.asarray(y_pred) / np.asarray(y_true))
    median = np.median(ratio)
    return float(np.sign(median) * 100.0 * (np.exp(np.abs(median)) - 1.0))


def summarize_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(y_true.size),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape_percent": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0),
        "rmsle": float(np.sqrt(np.mean((np.log(y_pred) - np.log(y_true)) ** 2))),
        "r2": float(r2_score(y_true, y_pred)),
        "epsilon_percent": median_symmetric_accuracy(y_true, y_pred),
        "beta_percent": symmetric_signed_percentage_bias(y_true, y_pred),
    }
