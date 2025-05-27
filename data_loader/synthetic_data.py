# portfolio_optimizer_project/data_loader/synthetic_data.py
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Default values (can be overridden by Hydra config when called from main.py)
# These are here just for standalone testing of this module if needed.
N_SAMPLES_DEFAULT = 2000
N_ASSETS_DEFAULT = 5
N_FACTORS_DEFAULT = 2
FACTOR_AR_RHO_DEFAULT = 0.7
FACTOR_VOL_DEFAULT = 0.015
IDIOSYNCRATIC_VOL_DEFAULT = 0.02
AVG_BETA_DEFAULT = 0.8
N_LAGS_DEFAULT = 10
PRED_HORIZON_DEFAULT = 1


def generate_synthetic_market_data(
    n_samples: int = N_SAMPLES_DEFAULT,
    n_assets: int = N_ASSETS_DEFAULT,
    n_factors: int = N_FACTORS_DEFAULT,
    factor_ar_rho: float = FACTOR_AR_RHO_DEFAULT,
    factor_vol: float = FACTOR_VOL_DEFAULT,
    idiosyncratic_vol: float = IDIOSYNCRATIC_VOL_DEFAULT,
    avg_beta: float = AVG_BETA_DEFAULT,
    seed: int = None,  # Allow seed to be passed for reproducibility
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generates synthetic market data based on a factor model.
    - Factor returns follow an AR(1) process.
    - Asset returns are driven by factor exposures and idiosyncratic noise.
    - Calculates the true covariance matrix based on the model.
    """
    if seed is not None:
        np.random.seed(seed)
        # Note: torch.manual_seed should be set in the main script if using torch random ops here

    print(f"\n--- Generating Synthetic Market Data ---")
    print(
        f"Parameters: n_samples={n_samples}, n_assets={n_assets}, n_factors={n_factors}, seed={seed}"
    )

    # 1. Generate Factor Exposures (Betas)
    factor_betas = np.random.normal(loc=avg_beta, scale=0.3, size=(n_assets, n_factors))
    if n_factors > 1 and n_assets > 1:
        factor_betas[
            n_assets // 2 :, 0
        ] *= -0.5  # Make some exposures negative for diversity
    # print(f"Factor Betas (True Exposures) shape: {factor_betas.shape}")

    # 2. Generate Factor Returns (AR(1) process for each factor)
    factor_returns = np.zeros((n_samples, n_factors))
    for f_idx in range(n_factors):
        factor_innovations = np.random.normal(loc=0, scale=factor_vol, size=n_samples)
        factor_returns[0, f_idx] = factor_innovations[0]  # Initial value
        for t in range(1, n_samples):
            factor_returns[t, f_idx] = (
                factor_ar_rho * factor_returns[t - 1, f_idx] + factor_innovations[t]
            )
    factor_returns_df = pd.DataFrame(
        factor_returns, columns=[f"Factor_{j + 1}" for j in range(n_factors)]
    )
    # print(f"Factor Returns shape: {factor_returns_df.shape}")

    # 3. Generate Asset Returns
    factor_component_returns = factor_returns @ factor_betas.T
    idiosyncratic_noise = np.random.normal(
        loc=0, scale=idiosyncratic_vol, size=(n_samples, n_assets)
    )
    asset_returns = factor_component_returns + idiosyncratic_noise
    asset_returns_df = pd.DataFrame(
        asset_returns, columns=[f"Asset_{i + 1}" for i in range(n_assets)]
    )
    # print(f"Asset Returns shape: {asset_returns_df.shape}")

    # 4. Calculate True Covariance Matrix (Sigma_true) from the model
    # Var(F) for AR(1) F_t = rho*F_{t-1} + eps_t is Var(eps) / (1 - rho^2)
    if abs(factor_ar_rho) < 1.0:  # Ensure stationarity for variance formula
        var_factor = (factor_vol**2) / (1 - factor_ar_rho**2)
    else:  # If rho >= 1, process is non-stationary, variance is technically undefined or infinite. Use innovation variance.
        print(
            "Warning: factor_ar_rho is >= 1 or <= -1. Factor process is non-stationary. Using factor_vol^2 for var_factor."
        )
        var_factor = factor_vol**2

    Sigma_factor_diag = np.diag(
        [var_factor] * n_factors
    )  # Assumes factors are contemporaneously uncorrelated for simplicity
    Sigma_idiosyncratic_diag = np.diag([idiosyncratic_vol**2] * n_assets)

    true_asset_Sigma = (
        factor_betas @ Sigma_factor_diag @ factor_betas.T + Sigma_idiosyncratic_diag
    )
    true_asset_Sigma = true_asset_Sigma.astype(np.float64)  # For CVXPY
    # Small regularization for positive definiteness, just in case of numerical issues
    true_asset_Sigma += np.eye(n_assets) * 1e-9  # Slightly increased regularization
    # print(f"True Asset Covariance Matrix (Sigma_true) shape: {true_asset_Sigma.shape}")

    return asset_returns_df, factor_returns_df, factor_betas, true_asset_Sigma


def plot_generated_data_characteristics(
    asset_returns_df: pd.DataFrame,
    factor_returns_df: pd.DataFrame,
    factor_betas: np.ndarray,
    true_Sigma: np.ndarray,
    save_path_prefix: str = "results/plots/",  # Allow specifying save path
):
    """Plots characteristics of the generated synthetic data."""
    print("\n--- Plotting Generated Data Characteristics ---")
    n_assets = asset_returns_df.shape[1]
    n_factors = factor_returns_df.shape[1]
    n_assets_to_plot = min(n_assets, 3)
    n_factors_to_plot = min(n_factors, 3)

    # Time Series Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    if n_factors > 0:
        for i in range(n_factors_to_plot):
            axes[0].plot(factor_returns_df.iloc[:200, i], label=f"Factor {i + 1}")
        axes[0].set_title("Sample Factor Returns (First 200 Samples)")
        axes[0].legend()
        axes[0].grid(True)
    else:
        axes[0].set_title("No Factors Generated")

    if n_assets > 0:
        for i in range(n_assets_to_plot):
            axes[1].plot(asset_returns_df.iloc[:200, i], label=f"Asset {i + 1}")
        axes[1].set_title("Sample Asset Returns (First 200 Samples)")
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].set_title("No Assets Generated")

    plt.xlabel("Time Step")
    plt.tight_layout()
    if save_path_prefix:
        plt.savefig(f"{save_path_prefix}synthetic_data_series.png")
    plt.show()

    # Factor Betas Heatmap
    if n_factors > 0 and n_assets > 0:
        plt.figure(figsize=(max(6, n_factors * 1.5), max(5, n_assets * 0.6)))
        sns.heatmap(
            factor_betas,
            annot=True,
            cmap="viridis",
            fmt=".2f",
            yticklabels=[f"Asset {i + 1}" for i in range(n_assets)],
            xticklabels=[f"Factor {j + 1}" for j in range(n_factors)],
        )
        plt.title("True Factor Exposures (Betas)")
        plt.xlabel("Factors")
        plt.ylabel("Assets")
        plt.tight_layout()
        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}true_factor_betas.png")
        plt.show()

    # True Sigma Heatmap
    if n_assets > 0:
        plt.figure(figsize=(max(6, n_assets * 1.2), max(5, n_assets * 1.0)))
        sns.heatmap(
            true_Sigma,
            annot=True,
            cmap="coolwarm",
            fmt=".4f",
            square=True,
            xticklabels=[f"Asset {i + 1}" for i in range(n_assets)],
            yticklabels=[f"Asset {i + 1}" for i in range(n_assets)],
        )
        plt.title("True Asset Covariance Matrix (Sigma_true)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}true_sigma.png")
        plt.show()

        condition_number = np.linalg.cond(true_Sigma)
        print(f"Condition number of True Sigma: {condition_number:.4e}")
        if condition_number > 1e8:  # A somewhat arbitrary threshold
            print(
                "WARNING: True Sigma has a high condition number, which might cause solver instability."
            )


def create_features_and_targets_from_synthetic(
    asset_returns_df: pd.DataFrame,
    n_lags: int = N_LAGS_DEFAULT,
    pred_horizon: int = PRED_HORIZON_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Creates lagged asset returns as features (X) and future returns as targets (Y).
    """
    print(f"\n--- Creating Features and Targets from Synthetic Data ---")
    n_assets = asset_returns_df.shape[1]
    X_list, Y_list = [], []

    total_timesteps = len(asset_returns_df)

    if total_timesteps < n_lags + pred_horizon:
        print(
            f"Warning: Not enough data ({total_timesteps} timesteps) for n_lags ({n_lags}) + pred_horizon ({pred_horizon}). Returning empty tensors."
        )
        return torch.empty(0, n_assets * n_lags, dtype=torch.float32), torch.empty(
            0, n_assets, dtype=torch.float32
        )

    for t in range(n_lags, total_timesteps - pred_horizon + 1):
        feature_window = asset_returns_df.iloc[t - n_lags : t]
        features_t = feature_window.values.flatten()  # Shape: (n_assets * n_lags)

        target_t = asset_returns_df.iloc[
            t + pred_horizon - 1
        ].values  # Shape: (n_assets,)

        X_list.append(features_t)
        Y_list.append(target_t)

    if not X_list:
        print(
            "Warning: No features/targets generated despite sufficient initial timesteps. Check logic."
        )
        return torch.empty(0, n_assets * n_lags, dtype=torch.float32), torch.empty(
            0, n_assets, dtype=torch.float32
        )

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_list), dtype=torch.float32)
    print(f"Features (X_train) shape: {X_tensor.shape}")
    print(f"Targets (Y_train) shape: {Y_tensor.shape}")
    return X_tensor, Y_tensor


if __name__ == "__main__":
    # --- Example of using this module directly for testing ---
    print("Testing synthetic_data.py module...")

    # Use default parameters defined at the top of this file for the test
    asset_rets_test, factor_rets_test, betas_test, sigma_test = (
        generate_synthetic_market_data(seed=123)
    )

    # You need to create a results/plots directory if it doesn't exist for save_path_prefix to work
    import os

    results_plot_path = "../results/plots/"  # Relative path for testing
    if not os.path.exists(results_plot_path):
        os.makedirs(results_plot_path)
        print(f"Created directory for plots: {results_plot_path}")

    plot_generated_data_characteristics(
        asset_rets_test,
        factor_rets_test,
        betas_test,
        sigma_test,
        save_path_prefix=results_plot_path,  # Pass a valid path
    )

    X_test, Y_test = create_features_and_targets_from_synthetic(asset_rets_test)

    print("\n--- Module Test Finished ---")
    print(
        f"Sample X_test (first 2 features of first sample): {X_test[0, : min(2, X_test.shape[1])].numpy() if X_test.numel() > 0 else 'N/A'}"
    )
    print(
        f"Sample Y_test (first target vector): {Y_test[0].numpy() if Y_test.numel() > 0 else 'N/A'}"
    )
