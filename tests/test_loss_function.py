import unittest
import torch
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Assuming portfolio_objective_loss is in main.py or a utils file
# If it's in main.py, you might need to refactor it into a utility for easier testing
# For now, let's assume you've moved it or can import it.
# from main import portfolio_objective_loss
# If you move it to utils/losses.py:
# from utils.losses import portfolio_objective_loss


# Placeholder: if you copy the function here for testing
def portfolio_objective_loss(
    predicted_weights: torch.Tensor, actual_future_returns: torch.Tensor
) -> torch.Tensor:
    if predicted_weights is None or predicted_weights.nelement() == 0:
        return torch.tensor(
            0.0, requires_grad=True, device=actual_future_returns.device
        )
    if torch.isnan(predicted_weights).any() or torch.isinf(predicted_weights).any():
        return torch.tensor(float("inf"), device=actual_future_returns.device)
    portfolio_returns_batch = torch.sum(
        predicted_weights * actual_future_returns, dim=1
    )
    if (
        torch.isnan(portfolio_returns_batch).any()
        or torch.isinf(portfolio_returns_batch).any()
    ):
        return torch.tensor(float("inf"), device=actual_future_returns.device)
    loss_val = -torch.mean(portfolio_returns_batch)
    if torch.isnan(loss_val) or torch.isinf(loss_val):
        return torch.tensor(1e5, device=actual_future_returns.device)
    return loss_val


class TestLossFunctions(unittest.TestCase):
    def test_portfolio_objective_loss_basic(self):
        weights = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
        returns = torch.tensor([[0.1, 0.05], [-0.02, 0.03]])
        # Expected portfolio returns:
        # Batch 1: 0.5*0.1 + 0.5*0.05 = 0.05 + 0.025 = 0.075
        # Batch 2: 0.3*(-0.02) + 0.7*0.03 = -0.006 + 0.021 = 0.015
        # Mean portfolio return: (0.075 + 0.015) / 2 = 0.090 / 2 = 0.045
        # Expected loss: -0.045
        expected_loss = -0.045
        loss = portfolio_objective_loss(weights, returns)
        self.assertAlmostEqual(loss.item(), expected_loss, places=5)

    def test_portfolio_objective_loss_empty_weights(self):
        weights = torch.empty(0, 2)
        returns = torch.tensor([[0.1, 0.05]])
        loss = portfolio_objective_loss(weights, returns)
        self.assertEqual(loss.item(), 0.0)

        weights_none = None
        loss_none = portfolio_objective_loss(weights_none, returns)
        self.assertEqual(loss_none.item(), 0.0)

    def test_portfolio_objective_loss_nan_weights(self):
        weights = torch.tensor([[0.5, float("nan")]])
        returns = torch.tensor([[0.1, 0.05]])
        loss = portfolio_objective_loss(weights, returns)
        self.assertTrue(
            torch.isinf(loss) or loss.item() > 1e4
        )  # Handles both inf or large penalty

    def test_portfolio_objective_loss_nan_returns_in_batch(self):
        weights = torch.tensor([[0.5, 0.5]])
        returns = torch.tensor(
            [[0.1, float("nan")]]
        )  # NaN in actual returns used by sum()
        loss = portfolio_objective_loss(weights, returns)
        self.assertTrue(torch.isinf(loss) or loss.item() > 1e4)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
