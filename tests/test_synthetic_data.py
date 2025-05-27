import unittest
import pandas as pd
import numpy as np
import torch
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader.synthetic_data import (
    generate_synthetic_market_data,
    create_features_and_targets_from_synthetic,
)


class TestSyntheticData(unittest.TestCase):
    def test_generate_synthetic_data_shapes_and_types(self):
        n_samples = 100
        n_assets = 5
        n_factors = 2
        seed = 42

        asset_returns_df, factor_returns_df, factor_betas_true, sigma_np_true = (
            generate_synthetic_market_data(
                n_samples=n_samples, n_assets=n_assets, n_factors=n_factors, seed=seed
            )
        )

        self.assertIsInstance(asset_returns_df, pd.DataFrame)
        self.assertEqual(asset_returns_df.shape, (n_samples, n_assets))

        self.assertIsInstance(factor_returns_df, pd.DataFrame)
        self.assertEqual(factor_returns_df.shape, (n_samples, n_factors))

        self.assertIsInstance(factor_betas_true, np.ndarray)
        self.assertEqual(factor_betas_true.shape, (n_assets, n_factors))

        self.assertIsInstance(sigma_np_true, np.ndarray)
        self.assertEqual(sigma_np_true.shape, (n_assets, n_assets))

        # Check PSD for sigma_np_true
        eigvals = np.linalg.eigvalsh(sigma_np_true)
        self.assertTrue(np.all(eigvals >= -1e-9))

    def test_generate_synthetic_data_reproducibility(self):
        # Generate twice with the same seed
        asset_returns_df1, _, _, sigma1 = generate_synthetic_market_data(seed=123)
        asset_returns_df2, _, _, sigma2 = generate_synthetic_market_data(seed=123)

        pd.testing.assert_frame_equal(asset_returns_df1, asset_returns_df2)
        np.testing.assert_array_almost_equal(sigma1, sigma2)

    def test_create_features_from_synthetic(self):
        n_samples = 50
        n_assets = 3
        asset_returns_df, _, _, _ = generate_synthetic_market_data(
            n_samples=n_samples, n_assets=n_assets
        )

        n_lags = 5
        pred_horizon = 1

        X, Y = create_features_and_targets_from_synthetic(
            asset_returns_df, n_lags=n_lags, pred_horizon=pred_horizon
        )

        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(Y, torch.Tensor)

        expected_samples = n_samples - n_lags - pred_horizon + 1
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(Y.shape[0], expected_samples)
        self.assertEqual(X.shape[1], n_assets * n_lags)
        self.assertEqual(Y.shape[1], n_assets)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
