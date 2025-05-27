# portfolio_optimizer_project/utils/losses.py
import torch
import numpy as np  # For np.sqrt if used with tensors, or torch.sqrt

# For logging within this module if needed, though often main script handles logging
# import logging
# log = logging.getLogger(__name__)


def negative_mean_return_loss(
    predicted_weights: torch.Tensor,  # Shape: (batch_size, n_assets)
    actual_future_returns: torch.Tensor,  # Shape: (batch_size, n_assets)
) -> torch.Tensor:
    """
    Calculates the negative mean of portfolio returns.
    Aim: Maximize mean return.
    """
    if predicted_weights is None or predicted_weights.nelement() == 0:
        return torch.tensor(
            0.0, requires_grad=True, device=actual_future_returns.device
        )
    if torch.isnan(predicted_weights).any() or torch.isinf(predicted_weights).any():
        return torch.tensor(
            float("inf"), device=actual_future_returns.device
        )  # High penalty

    portfolio_returns_batch = torch.sum(
        predicted_weights * actual_future_returns, dim=1
    )  # (batch_size,)

    if (
        torch.isnan(portfolio_returns_batch).any()
        or torch.isinf(portfolio_returns_batch).any()
    ):
        return torch.tensor(float("inf"), device=actual_future_returns.device)

    loss_val = -torch.mean(portfolio_returns_batch)

    if torch.isnan(loss_val) or torch.isinf(loss_val):
        return torch.tensor(
            1e5, device=actual_future_returns.device
        )  # Large finite penalty
    return loss_val


def sharpe_ratio_loss(
    predicted_weights: torch.Tensor,  # Shape: (batch_size, n_assets)
    actual_future_returns: torch.Tensor,  # Shape: (batch_size, n_assets)
    risk_free_rate_daily: float = 0.0,  # Daily risk-free rate
    epsilon: float = 1e-8,  # Small value to avoid division by zero
) -> torch.Tensor:
    """
    Calculates the negative of the Sharpe Ratio (to be minimized).
    Sharpe Ratio = Mean(Excess Returns) / StdDev(Excess Returns).
    Assumes returns are for a single period (e.g., daily) for all items in batch.
    The annualization factor is implicitly handled if this loss is used per batch,
    and the overall effect is to prefer higher batch Sharpe.
    """
    if predicted_weights is None or predicted_weights.nelement() == 0:
        return torch.tensor(
            0.0, requires_grad=True, device=actual_future_returns.device
        )  # Or a penalty
    if torch.isnan(predicted_weights).any() or torch.isinf(predicted_weights).any():
        return torch.tensor(float("inf"), device=actual_future_returns.device)

    portfolio_returns_batch = torch.sum(
        predicted_weights * actual_future_returns, dim=1
    )  # (batch_size,)

    if (
        torch.isnan(portfolio_returns_batch).any()
        or torch.isinf(portfolio_returns_batch).any()
    ):
        return torch.tensor(float("inf"), device=actual_future_returns.device)

    if portfolio_returns_batch.numel() < 2:  # Need at least 2 samples for std dev
        # print("Warning: Sharpe ratio loss requires at least 2 samples in batch for std dev.")
        return -torch.mean(portfolio_returns_batch)  # Fallback to negative mean return

    excess_returns_batch = portfolio_returns_batch - risk_free_rate_daily

    mean_excess_return = torch.mean(excess_returns_batch)
    std_dev_excess_return = torch.std(excess_returns_batch)

    # Negative Sharpe Ratio (to be minimized)
    if std_dev_excess_return < epsilon:  # Avoid division by zero or very small std
        # If std is ~0, Sharpe is effectively inf if mean_er > 0, -inf if mean_er < 0, 0 if mean_er ~0
        # We want to maximize Sharpe, so minimizing -Sharpe.
        # If mean_er > 0 and std ~0, -Sharpe should be very negative (good).
        # If mean_er < 0 and std ~0, -Sharpe should be very positive (bad).
        if mean_excess_return > epsilon:  # Very good scenario, high positive sharpe
            return -mean_excess_return / epsilon  # Large negative number (good loss)
        elif mean_excess_return < -epsilon:  # Very bad scenario, high negative sharpe
            return -mean_excess_return / epsilon  # Large positive number (bad loss)
        else:  # mean_er is also ~0
            return torch.tensor(0.0, device=predicted_weights.device)  # Neutral loss

    sharpe_ratio = mean_excess_return / (
        std_dev_excess_return + epsilon
    )  # Add epsilon for stability

    loss_val = -sharpe_ratio  # Minimize negative Sharpe Ratio

    if torch.isnan(loss_val) or torch.isinf(loss_val):
        # This can happen if mean_excess_return is huge and std_dev_excess_return is tiny (but not < epsilon)
        # Or if something else went wrong.
        # print(f"Warning: Sharpe ratio loss became NaN/Inf. mean_er={mean_excess_return.item()}, std_er={std_dev_excess_return.item()}")
        # Fallback to ensure a real number is returned for the loss.
        # A large penalty if Sharpe was supposed to be good (mean_er > 0), or less penalty if bad.
        return torch.tensor(
            1e5 if mean_excess_return <= 0 else -1e5, device=predicted_weights.device
        )

    return loss_val


def mean_variance_loss(
    predicted_weights: torch.Tensor,  # Shape: (batch_size, n_assets)
    actual_future_returns: torch.Tensor,  # Shape: (batch_size, n_assets)
    covariance_matrix: torch.Tensor,  # Shape: (n_assets, n_assets) - The Sigma matrix
    lambda_risk_aversion: float = 1.0,  # Risk aversion parameter (lambda in this context)
) -> torch.Tensor:
    """
    Calculates a risk-averse loss: Lambda * PortfolioVariance - MeanPortfolioReturn.
    This is equivalent to minimizing -(MeanReturn - Lambda * Variance).
    Aim: Maximize risk-adjusted return.
    """
    if predicted_weights is None or predicted_weights.nelement() == 0:
        return torch.tensor(
            0.0, requires_grad=True, device=actual_future_returns.device
        )
    if torch.isnan(predicted_weights).any() or torch.isinf(predicted_weights).any():
        return torch.tensor(float("inf"), device=actual_future_returns.device)

    portfolio_returns_batch = torch.sum(
        predicted_weights * actual_future_returns, dim=1
    )  # (batch_size,)
    mean_portfolio_return = torch.mean(portfolio_returns_batch)

    # Calculate portfolio variance for each sample in the batch
    # portfolio_variance_batch = w^T * Sigma * w
    # predicted_weights: (batch_size, n_assets)
    # covariance_matrix: (n_assets, n_assets)
    # Need to batch matrix multiplication: (B,1,N) @ (N,N) @ (B,N,1) -> (B,1,1)

    # Ensure covariance_matrix is on the same device as weights
    sigma = covariance_matrix.to(predicted_weights.device)

    portfolio_variance_terms = []
    for i in range(predicted_weights.shape[0]):
        w_i = predicted_weights[i].unsqueeze(0)  # Shape (1, n_assets)
        variance_i = w_i @ sigma @ w_i.T  # (1,N)@(N,N)@(N,1) -> (1,1)
        portfolio_variance_terms.append(variance_i.squeeze())

    if (
        not portfolio_variance_terms
    ):  # Should not happen if predicted_weights has samples
        mean_portfolio_variance = torch.tensor(0.0, device=predicted_weights.device)
    else:
        portfolio_variance_batch = torch.stack(
            portfolio_variance_terms
        )  # (batch_size,)
        mean_portfolio_variance = torch.mean(portfolio_variance_batch)

    # Loss = Lambda * Variance - MeanReturn (to be minimized)
    loss_val = lambda_risk_aversion * mean_portfolio_variance - mean_portfolio_return

    if torch.isnan(loss_val) or torch.isinf(loss_val):
        # print(f"Warning: Mean-variance loss became NaN/Inf. mean_ret={mean_portfolio_return.item()}, mean_var={mean_portfolio_variance.item()}")
        return torch.tensor(1e5, device=predicted_weights.device)
    return loss_val


# You could add Sortino Ratio loss here as well.
# It's similar to Sharpe but uses downside deviation.

if __name__ == "__main__":
    print("--- Testing utils/losses.py module ---")
    batch_s = 64
    n_assets_test = 5
    device_test = torch.device("cpu")

    # Dummy data
    dummy_weights = torch.rand(batch_s, n_assets_test, device=device_test)
    dummy_weights = dummy_weights / dummy_weights.sum(dim=1, keepdim=True)
    dummy_returns = (
        torch.randn(batch_s, n_assets_test, device=device_test) * 0.02
    )  # Mean 0, std 2% daily

    # Test negative_mean_return_loss
    loss_neg_mean_ret = negative_mean_return_loss(dummy_weights, dummy_returns)
    print(f"Negative Mean Return Loss: {loss_neg_mean_ret.item():.6f}")
    assert not torch.isnan(loss_neg_mean_ret) and not torch.isinf(loss_neg_mean_ret)

    # Test sharpe_ratio_loss
    loss_sharpe = sharpe_ratio_loss(
        dummy_weights, dummy_returns, risk_free_rate_daily=0.0001 / 252
    )
    print(f"Sharpe Ratio Loss (Negative Sharpe): {loss_sharpe.item():.6f}")
    assert not torch.isnan(loss_sharpe) and not torch.isinf(loss_sharpe)

    # Test sharpe with zero std dev
    const_returns = torch.ones_like(dummy_returns) * 0.001
    loss_sharpe_zero_std_pos_mean = sharpe_ratio_loss(
        dummy_weights, const_returns, risk_free_rate_daily=0.0
    )
    print(
        f"Sharpe Loss (zero std, pos mean): {loss_sharpe_zero_std_pos_mean.item():.6f}"
    )  # Should be very negative
    loss_sharpe_zero_std_neg_mean = sharpe_ratio_loss(
        dummy_weights, -const_returns, risk_free_rate_daily=0.0
    )
    print(
        f"Sharpe Loss (zero std, neg mean): {loss_sharpe_zero_std_neg_mean.item():.6f}"
    )  # Should be very positive

    # Test mean_variance_loss
    # Dummy Sigma
    A_test = np.random.rand(n_assets_test, n_assets_test)
    sigma_test_np = (
        np.dot(A_test, A_test.transpose()) * 0.0001 + np.eye(n_assets_test) * 1e-6
    )  # Daily variance scale
    sigma_test_np = (sigma_test_np + sigma_test_np.T) / 2.0
    sigma_test_tensor = torch.tensor(
        sigma_test_np, dtype=torch.float32, device=device_test
    )

    loss_mean_var = mean_variance_loss(
        dummy_weights, dummy_returns, sigma_test_tensor, lambda_risk_aversion=1.0
    )
    print(f"Mean-Variance Loss (lambda=1.0): {loss_mean_var.item():.6f}")
    assert not torch.isnan(loss_mean_var) and not torch.isinf(loss_mean_var)

    loss_mean_var_high_lambda = mean_variance_loss(
        dummy_weights, dummy_returns, sigma_test_tensor, lambda_risk_aversion=10.0
    )
    print(f"Mean-Variance Loss (lambda=10.0): {loss_mean_var_high_lambda.item():.6f}")
    assert not torch.isnan(loss_mean_var_high_lambda) and not torch.isinf(
        loss_mean_var_high_lambda
    )
    # Expect loss_mean_var_high_lambda to be higher if variance is positive, as it penalizes variance more

    print("\n--- Losses Module Test Finished ---")
