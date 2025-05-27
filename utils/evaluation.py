# portfolio_optimizer_project/utils/evaluation.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List


def calculate_portfolio_returns(
    weights: torch.Tensor,  # Shape: (num_timesteps, num_assets)
    actual_returns: torch.Tensor,  # Shape: (num_timesteps, num_assets)
) -> torch.Tensor:
    """Calculates the time series of portfolio returns."""
    if weights.shape != actual_returns.shape:
        raise ValueError(
            f"Weights shape {weights.shape} and returns shape {actual_returns.shape} must match."
        )
    portfolio_returns = torch.sum(weights * actual_returns, dim=1)
    return portfolio_returns


def calculate_sharpe_ratio(
    portfolio_returns: torch.Tensor,  # Time series of portfolio returns
    risk_free_rate_daily: float = 0.0,  # Daily risk-free rate
    annualization_factor: int = 252,  # For daily returns
) -> float:
    """Calculates the annualized Sharpe Ratio."""
    if portfolio_returns.numel() < 2:  # Need at least 2 returns for std dev
        return np.nan

    excess_returns = portfolio_returns - risk_free_rate_daily
    mean_excess_return = torch.mean(excess_returns)
    std_dev_excess_return = torch.std(excess_returns)

    if std_dev_excess_return.item() == 0:  # Avoid division by zero
        return (
            np.nan if mean_excess_return.item() <= 0 else np.inf
        )  # Or some other indicator

    sharpe_ratio_daily = mean_excess_return / std_dev_excess_return
    annualized_sharpe_ratio = sharpe_ratio_daily * np.sqrt(annualization_factor)
    return annualized_sharpe_ratio.item()


def calculate_max_drawdown(portfolio_returns_series: pd.Series) -> float:
    """Calculates the Maximum Drawdown from a pandas Series of portfolio returns."""
    if portfolio_returns_series.empty:
        return np.nan
    cumulative_returns = (1 + portfolio_returns_series).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown


def calculate_cumulative_return(portfolio_returns_series: pd.Series) -> float:
    """Calculates the total cumulative return."""
    if portfolio_returns_series.empty:
        return np.nan
    total_return = (1 + portfolio_returns_series).prod() - 1
    return total_return


def calculate_annualized_return(
    portfolio_returns_series: pd.Series, annualization_factor: int = 252
) -> float:
    """Calculates the annualized return."""
    if portfolio_returns_series.empty or len(portfolio_returns_series) == 0:
        return np.nan
    num_years = len(portfolio_returns_series) / annualization_factor
    if num_years == 0:
        return np.nan
    total_return = calculate_cumulative_return(portfolio_returns_series)
    annualized_return = (1 + total_return) ** (1 / num_years) - 1
    return annualized_return


def calculate_annualized_volatility(
    portfolio_returns_series: pd.Series, annualization_factor: int = 252
) -> float:
    """Calculates the annualized volatility (standard deviation)."""
    if portfolio_returns_series.empty or len(portfolio_returns_series) < 2:
        return np.nan
    volatility = portfolio_returns_series.std() * np.sqrt(annualization_factor)
    return volatility


def evaluate_portfolio_performance(
    predicted_weights_timeseries: torch.Tensor,  # (num_test_points, num_assets)
    actual_returns_timeseries: torch.Tensor,  # (num_test_points, num_assets)
    asset_labels: List[str],
    risk_free_rate_annual: float = 0.01,  # Annual risk-free rate
    annualization_factor: int = 252,
):
    """
    Evaluates portfolio performance on test data and prints metrics.
    """
    print("\n--- Portfolio Performance Evaluation ---")
    if predicted_weights_timeseries.shape[0] == 0:
        print("No data to evaluate.")
        return {}

    # Ensure tensors are on CPU for pandas conversion
    predicted_weights_timeseries = predicted_weights_timeseries.cpu()
    actual_returns_timeseries = actual_returns_timeseries.cpu()

    portfolio_daily_returns_tensor = calculate_portfolio_returns(
        predicted_weights_timeseries, actual_returns_timeseries
    )
    portfolio_daily_returns_series = pd.Series(portfolio_daily_returns_tensor.numpy())

    # --- Calculate Metrics ---
    print(f"Evaluation period: {len(portfolio_daily_returns_series)} trading days.")

    total_return = calculate_cumulative_return(portfolio_daily_returns_series)
    annualized_return = calculate_annualized_return(
        portfolio_daily_returns_series, annualization_factor
    )
    annualized_vol = calculate_annualized_volatility(
        portfolio_daily_returns_series, annualization_factor
    )

    daily_rfr = (1 + risk_free_rate_annual) ** (1 / annualization_factor) - 1
    sharpe = calculate_sharpe_ratio(
        portfolio_daily_returns_tensor, daily_rfr, annualization_factor
    )

    max_dd = calculate_max_drawdown(portfolio_daily_returns_series)

    print(f"Cumulative Return: {total_return:.4%}")
    print(f"Annualized Return: {annualized_return:.4%}")
    print(f"Annualized Volatility: {annualized_vol:.4%}")
    print(f"Annualized Sharpe Ratio (RFR={risk_free_rate_annual:.2%}): {sharpe:.2f}")
    print(f"Maximum Drawdown: {max_dd:.4%}")

    # Average allocation
    avg_allocation = pd.Series(
        predicted_weights_timeseries.mean(dim=0).numpy(), index=asset_labels
    )
    print("\nAverage Allocation over Test Period:")
    print(avg_allocation.round(4))

    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    (1 + portfolio_daily_returns_series).cumprod().plot()
    plt.title("Portfolio Cumulative Returns (Out-of-Sample)")
    plt.xlabel("Time (Test Period Days)")
    plt.ylabel("Cumulative Growth")
    plt.grid(True)
    plt.tight_layout()
    # In main.py, you'd pass a save_path from Hydra's output_dir
    # plt.savefig("test_cumulative_returns.png")
    plt.show()

    return {
        "cumulative_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "average_allocation": avg_allocation,
    }


if __name__ == "__main__":
    print("--- Testing utils/evaluation.py module ---")
    num_test_samples = 252  # One year of daily data
    num_test_assets = 3
    asset_labs = [f"Asset {i + 1}" for i in range(num_test_assets)]

    # Dummy predicted weights (could be more varied)
    dummy_weights = torch.rand(num_test_samples, num_test_assets)
    dummy_weights = dummy_weights / dummy_weights.sum(
        dim=1, keepdim=True
    )  # Normalize to sum to 1

    # Dummy actual returns
    dummy_returns = (
        torch.randn(num_test_samples, num_test_assets) * 0.01
    )  # Mean 0, std 1% daily

    results = evaluate_portfolio_performance(
        predicted_weights_timeseries=dummy_weights,
        actual_returns_timeseries=dummy_returns,
        asset_labels=asset_labs,
        risk_free_rate_annual=0.02,
    )
    print("\nEvaluation results dictionary:\n", results)
