"""Feature engineering for the Secchi MDN workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSet:
    feature_names: tuple[str, ...]
    band_columns: tuple[str, ...]


def _line_height(left: np.ndarray, middle: np.ndarray, right: np.ndarray) -> np.ndarray:
    return middle - 0.5 * (left + right)


def build_feature_matrix(
    frame: pd.DataFrame,
    band_columns: tuple[str, ...],
    include_ratios: bool = True,
) -> tuple[pd.DataFrame, FeatureSet]:
    """Build the raw-band and ratio-based features used for training."""
    features = frame.loc[:, band_columns].astype(float).copy()

    if include_ratios:
        # Maciel et al. (2023) use the visible bands, three forward band ratios,
        # and a line-height term derived from the visible triplet.
        if {"blue", "green"}.issubset(features.columns):
            features["green_over_blue"] = features["green"] / features["blue"]
        if {"blue", "red"}.issubset(features.columns):
            features["blue_over_red"] = features["blue"] / features["red"]
        if {"green", "red"}.issubset(features.columns):
            features["green_over_red"] = features["green"] / features["red"]
        if {"blue", "green", "red"}.issubset(features.columns):
            features["line_height_blue_green_red"] = _line_height(
                features["blue"].to_numpy(),
                features["green"].to_numpy(),
                features["red"].to_numpy(),
            )

    # Any invalid ratio propagates to NaN, so keep only rows with a complete feature vector.
    features = features.replace([np.inf, -np.inf], np.nan).dropna().copy()
    feature_names = tuple(features.columns.tolist())
    return features, FeatureSet(feature_names=feature_names, band_columns=band_columns)
