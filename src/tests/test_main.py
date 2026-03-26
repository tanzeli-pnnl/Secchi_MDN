"""Tests for the Secchi MDN helpers that do not require torch."""

from __future__ import annotations

import unittest

import pandas as pd

from src.secchi_mdn.cli import build_parser
from src.secchi_mdn.features import build_feature_matrix


class TestFeatureEngineering(unittest.TestCase):
    def test_feature_builder_adds_expected_ratios(self):
        frame = pd.DataFrame(
            {
                "blue": [0.01, 0.02],
                "green": [0.02, 0.03],
                "red": [0.03, 0.04],
                "coastal": [0.005, 0.006],
            }
        )

        features, feature_set = build_feature_matrix(
            frame,
            band_columns=("coastal", "blue", "green", "red"),
            include_ratios=True,
        )

        self.assertEqual(
            tuple(features.columns),
            (
                "coastal",
                "blue",
                "green",
                "red",
                "green_over_blue",
                "blue_over_red",
                "green_over_red",
                "line_height_blue_green_red",
            ),
        )
        self.assertEqual(tuple(features.columns), feature_set.feature_names)

    def test_parser_accepts_train_command(self):
        parser = build_parser()
        args = parser.parse_args(["train", "--dataset-dir", "/tmp/data"])
        self.assertEqual(args.command, "train")
        self.assertEqual(args.sensor, "all")


if __name__ == "__main__":
    unittest.main()
