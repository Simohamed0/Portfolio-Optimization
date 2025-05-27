import unittest
import pandas as pd
import numpy as np
import torch
import os
import shutil  # For managing temporary test data directories

# Adjust the path to import from the parent directory if tests/ is a sibling to data_loader/
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader.real_data import (
    load_and_process_real_stock_data,
    calculate_covariance_from_real_returns,
    create_features_and_targets_from_real_data,
)


class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test CSV files
        cls.test_data_dir = os.path.join(
            os.path.dirname(__file__), "temp_real_data_test_dir"
        )
        os.makedirs(cls.test_data_dir, exist_ok=True)

        # Create dummy CSV files
        dates1 = pd.to_datetime(
            ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        )
        stock1_data = pd.DataFrame(
            {
                "Date": dates1,
                "Open": [10, 11, 10, 12, 13],
                "Close": [10, 11, 10.5, 12, 11.5],
            }
        )
        stock1_data.to_csv(os.path.join(cls.test_data_dir, "STOCKA.csv"), index=False)

        dates2 = pd.to_datetime(
            ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
        )
        stock2_data = pd.DataFrame(
            {
                "Date": dates2,
                "Open": [20, 20, 21, 22, 20],
                "Close": [20, 20.5, 21, 20.8, 21.2],
            }
        )
        stock2_data.to_csv(os.path.join(cls.test_data_dir, "STOCKB.csv"), index=False)

        # Data for testing lags/horizon (more data points)
        dates_long = pd.date_range(
            start="2023-01-01", periods=20, freq="B"
        )  # Business days
        cls.stock_long_data_df = pd.DataFrame(
            {
                "Date": dates_long,
                "Close": np.arange(100, 100 + len(dates_long))
                + np.random.randn(len(dates_long)) * 0.1,
            }
        )
        cls.stock_long_data_df.to_csv(
            os.path.join(cls.test_data_dir, "STOCKL.csv"), index=False
        )

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary directory and its contents
        if os.path.exists(cls.test_data_dir):
            shutil.rmtree(cls.test_data_dir)

    def test_load_and_process_valid_tickers(self):
        tickers = ["STOCKA", "STOCKB"]
        returns_df, loaded_tickers = load_and_process_real_stock_data(
            tickers=tickers,
            data_dir=self.test_data_dir,
            price_column="Close",
            start_date_str="2023-01-01",
            end_date_str="2023-01-05",
        )
        self.assertEqual(len(loaded_tickers), 2)
        self.assertListEqual(sorted(loaded_tickers), sorted(tickers))
        self.assertIsInstance(returns_df, pd.DataFrame)
        self.assertFalse(returns_df.empty)
        self.assertEqual(returns_df.shape[0], 4)  # 5 days -> 4 returns
        self.assertEqual(returns_df.shape[1], 2)
        self.assertTrue(
            not returns_df.isnull().values.any()
        )  # Ensure no NaNs after processing

    def test_load_handles_missing_ticker(self):
        tickers = ["STOCKA", "MISSING"]
        returns_df, loaded_tickers = load_and_process_real_stock_data(
            tickers=tickers, data_dir=self.test_data_dir
        )
        self.assertEqual(len(loaded_tickers), 1)
        self.assertEqual(loaded_tickers[0], "STOCKA")
        self.assertEqual(returns_df.shape[1], 1)

    def test_load_date_filtering(self):
        tickers = ["STOCKA"]
        returns_df, _ = load_and_process_real_stock_data(
            tickers=tickers,
            data_dir=self.test_data_dir,
            start_date_str="2023-01-02",  # Start later
            end_date_str="2023-01-04",  # End earlier
        )
        self.assertEqual(returns_df.shape[0], 2)  # 3 days of prices -> 2 returns

    def test_calculate_covariance_valid_input(self):
        # Manually create a small returns DataFrame for predictable covariance
        data = {
            "AssetA": [0.01, -0.005, 0.015, 0.002],
            "AssetB": [0.005, 0.000, -0.01, 0.008],
        }
        returns_df = pd.DataFrame(data)

        sigma_np = calculate_covariance_from_real_returns(
            returns_df, annualization_factor=1
        )  # Daily
        self.assertIsNotNone(sigma_np)
        self.assertEqual(sigma_np.shape, (2, 2))
        # Check for positive semi-definiteness (all eigenvalues non-negative)
        eigvals = np.linalg.eigvalsh(sigma_np)
        self.assertTrue(
            np.all(eigvals >= -1e-9)
        )  # Allow for tiny numerical precision issues

        # Expected daily covariance (calculated manually or with numpy.cov)
        # np.cov(returns_df, rowvar=False, ddof=1) # ddof=1 for sample covariance like pandas .cov()
        # Expected:
        # [[7.29166667e-05, 2.08333333e-05],
        #  [2.08333333e-05, 6.02083333e-05]]
        # Note: our function adds regularization, so it won't be exact.
        # We mainly check if it's PSD and shape is correct. More precise value checks are harder with regularization.
        self.assertGreater(sigma_np[0, 0], 0)  # Variance of AssetA should be positive
        self.assertGreater(sigma_np[1, 1], 0)  # Variance of AssetB should be positive

    def test_calculate_covariance_empty_input(self):
        returns_df = pd.DataFrame()
        sigma_np = calculate_covariance_from_real_returns(returns_df)
        self.assertIsNone(sigma_np)

    def test_calculate_covariance_too_short_input(self):
        returns_df = pd.DataFrame({"AssetA": [0.01]})  # Only one row
        sigma_np = calculate_covariance_from_real_returns(returns_df)
        self.assertIsNone(sigma_np)  # Should return None as len < 2

    def test_create_features_and_targets(self):
        # Use the STOCKL data created in setUpClass
        # This requires loading it first to get a returns_df
        temp_returns_df, _ = load_and_process_real_stock_data(
            ["STOCKL"], data_dir=self.test_data_dir
        )

        n_lags = 5
        pred_horizon = 1
        n_assets = temp_returns_df.shape[1]  # Should be 1 for STOCKL

        X, Y = create_features_and_targets_from_real_data(
            temp_returns_df, n_lags=n_lags, pred_horizon=pred_horizon
        )

        self.assertIsInstance(X, torch.Tensor)
        self.assertIsInstance(Y, torch.Tensor)

        expected_samples = len(temp_returns_df) - n_lags - pred_horizon + 1
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(Y.shape[0], expected_samples)
        self.assertEqual(X.shape[1], n_assets * n_lags)
        self.assertEqual(Y.shape[1], n_assets)

    def test_create_features_insufficient_data(self):
        # Only 4 returns from STOCKA by default
        temp_returns_df, _ = load_and_process_real_stock_data(
            ["STOCKA"], data_dir=self.test_data_dir
        )

        n_lags = 10  # Too many lags for the data
        pred_horizon = 1

        X, Y = create_features_and_targets_from_real_data(
            temp_returns_df, n_lags=n_lags, pred_horizon=pred_horizon
        )
        self.assertTrue(X.numel() == 0)  # Should be empty
        self.assertTrue(Y.numel() == 0)  # Should be empty


if __name__ == "__main__":
    unittest.main(
        argv=["first-arg-is-ignored"], exit=False
    )  # For running in environments like notebooks
    # For command line:
    # unittest.main()
