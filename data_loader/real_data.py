# portfolio_optimizer_project/data_loader/real_data.py
import pandas as pd
import numpy as np
import torch
import os
from typing import List, Tuple, Optional
# Plotting libraries are removed from core functions, only used in __main__ if re-added

# Default values (can be overridden by Hydra config when called from main.py)
DEFAULT_DATA_DIR_REAL = "data"
DEFAULT_PRICE_COLUMN = "Close"
DEFAULT_N_LAGS_REAL = 10
DEFAULT_PRED_HORIZON_REAL = 1


def load_and_process_real_stock_data(
    tickers: List[str],
    data_dir: str = DEFAULT_DATA_DIR_REAL,
    price_column: str = DEFAULT_PRICE_COLUMN,
    start_date_str: Optional[str] = None,
    end_date_str: Optional[str] = None,
    seed: int = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if seed is not None:
        np.random.seed(seed)

    print(f"\n--- Loading and Processing Real Stock Data ---")
    print(f"Tickers: {tickers}, Data directory: {data_dir}")
    print(f"Price column: {price_column}, Start: {start_date_str}, End: {end_date_str}")

    all_prices_dict = {}
    loaded_tickers = []

    for ticker in tickers:
        file_ticker = ticker.replace(".", "-")
        file_path = os.path.join(data_dir, f"{file_ticker}.csv")

        if os.path.exists(file_path):
            try:
                df_ticker = pd.read_csv(file_path, index_col="Date", parse_dates=True)
                if price_column in df_ticker.columns:
                    series_ticker = df_ticker[price_column]
                    if start_date_str:
                        series_ticker = series_ticker[
                            series_ticker.index >= pd.to_datetime(start_date_str)
                        ]
                    if end_date_str:
                        series_ticker = series_ticker[
                            series_ticker.index <= pd.to_datetime(end_date_str)
                        ]

                    if series_ticker.empty:
                        print(
                            f"Warning: No data for ticker {ticker} within date range {start_date_str}-{end_date_str}. Skipping."
                        )
                        continue

                    all_prices_dict[ticker] = series_ticker
                    loaded_tickers.append(ticker)
                else:
                    print(
                        f"Warning: Price column '{price_column}' not found in {file_path} for ticker {ticker}. Skipping."
                    )
            except Exception as e:
                print(
                    f"Error loading or parsing {file_path} for ticker {ticker}: {e}. Skipping."
                )
        else:
            print(
                f"Warning: CSV file not found for ticker {ticker} at {file_path}. Skipping."
            )

    if not all_prices_dict:
        print(
            "Error: No data loaded for any of the specified tickers. Returning empty DataFrame."
        )
        return pd.DataFrame(), []

    adj_close_df_unfilled = pd.DataFrame(all_prices_dict)
    print(
        f"\nShape of combined price data (already date filtered): {adj_close_df_unfilled.shape}"
    )
    if adj_close_df_unfilled.empty:
        print("Error: Combined price DataFrame is empty. Exiting processing.")
        return pd.DataFrame(), []

    nan_counts_before_fill = adj_close_df_unfilled.isnull().sum()
    if nan_counts_before_fill.sum() > 0:
        print("\n--- Analyzing NaNs Before Filling ---")
        print("NaN counts per ticker BEFORE ffill/bfill:")
        print(nan_counts_before_fill[nan_counts_before_fill > 0])
        print(f"Total NaNs BEFORE ffill/bfill: {nan_counts_before_fill.sum()}")

    original_columns = adj_close_df_unfilled.columns.tolist()
    adj_close_df_filled = adj_close_df_unfilled.ffill().bfill()
    # print(f"\nShape after ffill/bfill: {adj_close_df_filled.shape}") # Less verbose

    adj_close_df_final = adj_close_df_filled.dropna(axis=1, how="all")
    final_loaded_tickers = adj_close_df_final.columns.tolist()
    if len(final_loaded_tickers) < len(original_columns):
        dropped_tickers = list(set(original_columns) - set(final_loaded_tickers))
        print(f"Dropped entirely NaN tickers after fill: {dropped_tickers}")
    # print(f"Shape after dropping all-NaN columns: {adj_close_df_final.shape}") # Less verbose

    if adj_close_df_final.empty:
        print(
            "Error: DataFrame is empty after NaN handling. Returning empty DataFrame."
        )
        return pd.DataFrame(), []

    daily_returns_df = adj_close_df_final.pct_change()
    # print(f"\nShape after pct_change: {daily_returns_df.shape}") # Less verbose
    # print(f"NaNs in returns before dropping first row (from pct_change): {daily_returns_df.isnull().sum().sum()}")

    daily_returns_df = daily_returns_df.dropna(how="all", axis=0)
    # print(f"Shape after dropping all-NaN rows (post pct_change): {daily_returns_df.shape}")

    nan_in_returns_after_dropna = daily_returns_df.isnull().sum()
    if nan_in_returns_after_dropna.sum() > 0:
        print(
            f"Total NaNs in returns AFTER dropping first row: {nan_in_returns_after_dropna.sum()}"
        )
        print("Warning: Individual NaNs still present in daily_returns_df.")
        print(
            "Summary of NaNs per ticker in returns:\n",
            nan_in_returns_after_dropna[nan_in_returns_after_dropna > 0],
        )
        # Consider: daily_returns_df = daily_returns_df.fillna(0)

    if daily_returns_df.empty:
        print(
            "Error: DataFrame of returns is empty. Check data variability and length."
        )
        return pd.DataFrame(), []

    print(
        f"\nSuccessfully loaded and processed data for {len(final_loaded_tickers)} tickers: {final_loaded_tickers}"
    )
    print(f"Final Daily returns DataFrame shape: {daily_returns_df.shape}")
    if (
        not daily_returns_df.empty and len(daily_returns_df) < 20
    ):  # Print head only if short
        print("Daily returns head:\n", daily_returns_df.head())
    return daily_returns_df, final_loaded_tickers


def calculate_covariance_from_real_returns(
    real_returns_df: pd.DataFrame,
    annualization_factor: int = 252,
    # plot_dir and asset_labels removed as parameters for plotting
) -> Optional[np.ndarray]:
    print("\n--- Calculating Covariance Matrix from Real Returns ---")
    if not isinstance(real_returns_df, pd.DataFrame) or real_returns_df.empty:
        print(
            "Input real_returns_df is not a valid DataFrame or is empty. Cannot calculate covariance."
        )
        return None
    print(
        f"Input real_returns_df to calculate_covariance_from_real_returns - shape: {real_returns_df.shape}"
    )

    nan_counts_input = real_returns_df.isnull().sum()
    total_nans_in_input = nan_counts_input.sum()
    if total_nans_in_input > 0:
        print(
            f"WARNING: Total {total_nans_in_input} NaNs found in input returns data for covariance."
        )
        print(
            f"NaN count per column in input real_returns_df:\n{nan_counts_input[nan_counts_input > 0]}"
        )
        print("Pandas .cov() will use pairwise deletion.")

    if len(real_returns_df) < 2:
        print(
            "Warning: Returns DataFrame has less than 2 samples. Covariance calculation will likely fail or be meaningless. Returning None."
        )
        return None
    if len(real_returns_df) < real_returns_df.shape[1]:
        print(
            f"Warning: Number of samples ({len(real_returns_df)}) is less than number of assets ({real_returns_df.shape[1]}). Covariance matrix might be singular or unstable."
        )

    cov_matrix_pd = real_returns_df.cov()
    nan_in_cov_pd = cov_matrix_pd.isnull().sum().sum()

    if nan_in_cov_pd > 0:
        print("ERROR: NaNs found in pandas covariance matrix BEFORE annualization!")
        print(
            "Columns with NaNs in cov_matrix_pd:",
            cov_matrix_pd.columns[cov_matrix_pd.isnull().any()].tolist(),
        )
        print(
            "Attempting to fill NaNs in cov_matrix_pd with 0 to proceed (problematic)."
        )
        cov_matrix_pd = cov_matrix_pd.fillna(0)

    cov_matrix_annualized_pd = cov_matrix_pd * annualization_factor
    sigma_np = cov_matrix_annualized_pd.values.astype(np.float64)

    if np.isnan(sigma_np).all() or not np.isfinite(sigma_np).all():
        print(
            "ERROR: sigma_np is all NaN or contains non-finite values before regularization. Returning None."
        )
        return None
    if np.isnan(sigma_np).any():
        print(
            "WARNING: NaNs exist in sigma_np before regularization. Filling with 0 for regularization (problematic)."
        )
        sigma_np = np.nan_to_num(
            sigma_np,
            nan=0.0,
            posinf=np.finfo(np.float64).max,
            neginf=np.finfo(np.float64).min,
        )

    regularization_term = np.eye(sigma_np.shape[0]) * 1e-9
    sigma_np += regularization_term

    try:
        eigvals = np.linalg.eigvalsh(sigma_np)
        min_eigval = eigvals.min()
        if np.any(eigvals < -1e-7):
            print(
                f"WARNING: sigma_np might not be positive semi-definite. Smallest eigenvalue: {min_eigval:.2e}"
            )
        else:
            print(
                f"Sigma_np appears to be positive semi-definite. Smallest eigenvalue: {min_eigval:.2e}"
            )
    except np.linalg.LinAlgError as e:
        print(
            f"ERROR calculating eigenvalues for final sigma_np: {e}. The matrix might be badly conditioned."
        )
        return None

    print(f"Calculated covariance matrix sigma_np shape: {sigma_np.shape}")
    return sigma_np


def create_features_and_targets_from_real_data(
    real_returns_df: pd.DataFrame,
    n_lags: int = DEFAULT_N_LAGS_REAL,
    pred_horizon: int = DEFAULT_PRED_HORIZON_REAL,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if real_returns_df.empty:
        return torch.empty(0, 0, dtype=torch.float32), torch.empty(
            0, 0, dtype=torch.float32
        )

    n_assets = real_returns_df.shape[1]
    input_dim_expected = n_assets * n_lags
    X_list, Y_list = [], []
    total_timesteps = len(real_returns_df)

    if total_timesteps < n_lags + pred_horizon:
        return torch.empty(0, input_dim_expected, dtype=torch.float32), torch.empty(
            0, n_assets, dtype=torch.float32
        )

    for t in range(n_lags, total_timesteps - pred_horizon + 1):
        feature_window = real_returns_df.iloc[t - n_lags : t]
        if feature_window.isnull().values.any():
            continue
        features_t = feature_window.values.flatten()

        target_t = real_returns_df.iloc[t + pred_horizon - 1].values
        if np.isnan(target_t).any():
            continue
        X_list.append(features_t)
        Y_list.append(target_t)

    if not X_list:
        return torch.empty(0, input_dim_expected, dtype=torch.float32), torch.empty(
            0, n_assets, dtype=torch.float32
        )

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32)
    # print(f"Real Data Features (X_train) shape: {X_tensor.shape}") # Less verbose
    # print(f"Real Data Targets (Y_train) shape: {Y_tensor.shape}")
    return X_tensor, Y_tensor


if __name__ == "__main__":
    print(
        "--- Running real_data.py directly for testing & debugging (plots removed from functions) ---"
    )

    test_tickers = ["AAPL", "MSFT", "GOOGL", "BRK-B", "JPM", "NVDA", "NONEXISTENT"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    test_data_dir = os.path.join(project_root, "data")

    # plot_dir is not used by the functions anymore, but can be defined if you add plotting back here
    # debug_plot_dir = os.path.join(script_dir, "debug_plots_real_data_no_plots_in_func")
    # os.makedirs(debug_plot_dir, exist_ok=True)
    # print(f"Debug plots would be saved to: {debug_plot_dir} (if enabled)")

    overall_start_date = "2012-01-01"
    overall_end_date = "2023-12-31"
    sigma_period_start_date = "2015-01-01"
    sigma_period_end_date = "2019-12-31"
    annualization_factor_for_sigma = 252
    train_period_start_date = "2020-01-01"
    train_period_end_date = "2022-12-31"
    n_lags_for_train = 20
    pred_horizon_for_train = 1
    # --- End Configuration ---

    print(f"\n--- 1. Testing load_and_process_real_stock_data ---")
    all_daily_returns_df, loaded_tickers_list = load_and_process_real_stock_data(
        tickers=test_tickers,
        data_dir=test_data_dir,
        start_date_str=overall_start_date,
        end_date_str=overall_end_date,
        # plot_dir=debug_plot_dir, # plot_dir argument removed from function
    )

    if not all_daily_returns_df.empty:
        print(f"\nSuccessfully loaded daily returns for: {loaded_tickers_list}")
        print(f"Full loaded returns DataFrame shape: {all_daily_returns_df.shape}")

        print(f"\n--- 2. Preparing data for Sigma calculation ---")
        print(f"Sigma period: {sigma_period_start_date} to {sigma_period_end_date}")

        sigma_returns_df = all_daily_returns_df.loc[
            sigma_period_start_date:sigma_period_end_date
        ].copy()
        sigma_returns_df.dropna(axis=1, how="all", inplace=True)

        if sigma_returns_df.empty:
            print(
                f"ERROR: sigma_returns_df is EMPTY for dates {sigma_period_start_date} to {sigma_period_end_date}."
            )
        else:
            print(
                f"Shape of returns data for Sigma calculation (sigma_returns_df): {sigma_returns_df.shape}"
            )
            print(
                "\n--- INSPECTING sigma_returns_df before covariance calculation (in real_data.py direct run) ---"
            )
            print(f"sigma_returns_df head:\n{sigma_returns_df.head()}")
            print(f"sigma_returns_df describe:\n{sigma_returns_df.describe()}")
            for col in sigma_returns_df.columns:
                if sigma_returns_df[col].nunique() <= 2:
                    print(
                        f"WARNING (sigma_returns_df): Column {col} has <= 2 unique values! Values: {sigma_returns_df[col].unique()}"
                    )
                current_std = sigma_returns_df[col].std()
                if pd.isna(current_std) or current_std < 1e-7:
                    print(
                        f"WARNING (sigma_returns_df): Column {col} has problematic standard deviation: {current_std}"
                    )
            print("--- END INSPECTION ---")

            print(f"\n--- 3. Testing calculate_covariance_from_real_returns ---")
            sigma_real_test = calculate_covariance_from_real_returns(
                sigma_returns_df,
                annualization_factor=annualization_factor_for_sigma,
                # plot_dir and asset_labels removed
            )
            if sigma_real_test is not None:
                print(
                    f"\nSuccessfully calculated Sigma matrix. Shape: {sigma_real_test.shape}"
                )
                print(
                    f"Sample of Sigma (first 3x3 or less):\n{sigma_real_test[: min(3, sigma_real_test.shape[0]), : min(3, sigma_real_test.shape[1])]}"
                )

            else:
                print("\nFailed to calculate Sigma matrix.")

        print(f"\n--- 4. Preparing data for Feature/Target creation (Training) ---")
        print(
            f"Training data period: {train_period_start_date} to {train_period_end_date}"
        )

        train_returns_df = all_daily_returns_df.loc[
            train_period_start_date:train_period_end_date
        ].copy()
        train_returns_df.dropna(axis=1, how="all", inplace=True)

        if train_returns_df.empty:
            print(
                f"ERROR: train_returns_df is EMPTY for dates {train_period_start_date} to {train_period_end_date}."
            )
        else:
            print(
                f"Shape of returns data for Training (train_returns_df): {train_returns_df.shape}"
            )
            print(f"\n--- Testing create_features_and_targets_from_real_data ---")
            X_real_test, Y_real_test = create_features_and_targets_from_real_data(
                train_returns_df,
                n_lags=n_lags_for_train,
                pred_horizon=pred_horizon_for_train,
            )
            if X_real_test.numel() > 0 and Y_real_test.numel() > 0:
                print(f"\nGenerated Features X_real_test shape: {X_real_test.shape}")
                print(f"Generated Targets Y_real_test shape: {Y_real_test.shape}")
            else:
                print(
                    "\nNo features/targets generated from the loaded real data for the test period."
                )
    else:
        print("\nNo data loaded, skipping further tests in real_data.py module.")

    print("\n--- Real Data Module Test Finished ---")
