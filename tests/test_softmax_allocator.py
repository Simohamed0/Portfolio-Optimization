import unittest
import torch
import numpy as np
import os

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.softmax_allocator import WeightPredictorNet, SoftmaxPortfolioAllocator


class TestSoftmaxAllocator(unittest.TestCase):
    def test_weight_predictor_net(self):
        input_dim = 50
        n_assets = 5
        batch_size = 4
        net = WeightPredictorNet(input_dim, n_assets, hidden_dims=[32, 16])
        dummy_input = torch.randn(batch_size, input_dim)
        output_logits = net(dummy_input)

        self.assertEqual(output_logits.shape, (batch_size, n_assets))

    def test_softmax_portfolio_allocator(self):
        input_dim = 60  # e.g., 6 assets * 10 lags
        n_assets = 6
        batch_size = 8

        allocator = SoftmaxPortfolioAllocator(
            input_dim,
            n_assets,
            weight_predictor_params={"hidden_dims": [64, 32], "dropout_rate": 0.1},
        )
        allocator.eval()  # Set to eval for consistent dropout if used

        dummy_features = torch.randn(batch_size, input_dim)
        portfolio_weights, logits = allocator(dummy_features)

        self.assertEqual(portfolio_weights.shape, (batch_size, n_assets))
        self.assertEqual(logits.shape, (batch_size, n_assets))

        # Check properties of softmax output
        for i in range(batch_size):
            self.assertTrue(
                torch.all(portfolio_weights[i] >= -1e-6)
            )  # Allow for small float errors
            self.assertAlmostEqual(
                torch.sum(portfolio_weights[i]).item(), 1.0, places=5
            )

    def test_model_to_device(self):
        input_dim = 10
        n_assets = 2
        allocator = SoftmaxPortfolioAllocator(input_dim, n_assets)

        try:
            allocator.to(torch.device("cpu"))
            # If CUDA available and you want to test:
            # if torch.cuda.is_available():
            #     allocator.to(torch.device("cuda"))
        except Exception as e:
            self.fail(f"Model to_device failed: {e}")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
