# portfolio_optimizer_project/models/softmax_allocator.py
import torch
import torch.nn as nn
import numpy as np


# --- Neural Network for Predicting Logits for Weights ---
class WeightPredictorNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.3,
    ):
        """
        Neural network to predict logits for portfolio weights.
        Args:
            input_dim (int): Dimensionality of the input features.
            n_assets (int): Number of assets, defines the output dimension.
            hidden_dims (list, optional): List of integers for hidden layer sizes.
                                          Defaults to [128, 64].
            dropout_rate (float, optional): Dropout rate. Defaults to 0.3.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]  # Default architecture

        print(f"\n--- Initializing WeightPredictorNet ---")
        print(
            f"Input dimension: {input_dim}, Hidden dimensions: {hidden_dims}, Output (n_assets logits): {n_assets}"
        )

        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, n_assets))  # Output raw logits

        self.network = nn.Sequential(*layers)
        print("WeightPredictorNet initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs logits for portfolio weights."""
        return self.network(x)


# --- End-to-End Portfolio Allocator with Softmax ---
class SoftmaxPortfolioAllocator(nn.Module):
    def __init__(
        self, input_dim: int, n_assets: int, weight_predictor_params: dict = None
    ):
        """
        Main model combining WeightPredictorNet and Softmax.
        Args:
            input_dim (int): Input dimension for WeightPredictorNet.
            n_assets (int): Number of assets.
            weight_predictor_params (dict, optional): Parameters for WeightPredictorNet
                                                     (e.g., {'hidden_dims': [256, 128], 'dropout_rate': 0.2}).
                                                     If None, uses defaults in WeightPredictorNet.
        """
        super().__init__()
        print(f"\n--- Initializing SoftmaxPortfolioAllocator ---")

        if weight_predictor_params is None:
            weight_predictor_params = {}  # Use defaults in WeightPredictorNet

        self.weight_predictor = WeightPredictorNet(
            input_dim=input_dim,
            n_assets=n_assets,
            **weight_predictor_params,  # Pass through any other params like hidden_dims, dropout_rate
        )
        self.softmax = nn.Softmax(
            dim=-1
        )  # Apply softmax along the last dimension (assets)
        print("SoftmaxPortfolioAllocator initialized.")

    def forward(self, x_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_features: Batch of input features (batch_size, input_dim)
        Returns:
            portfolio_weights (torch.Tensor): (batch_size, n_assets) - after softmax
            logits (torch.Tensor): (batch_size, n_assets) - raw output from weight_predictor
        """
        logits = self.weight_predictor(x_features)  # Shape: (batch_size, n_assets)
        portfolio_weights = self.softmax(logits)  # Shape: (batch_size, n_assets)

        return portfolio_weights, logits


if __name__ == "__main__":
    # --- Example of using this module directly for testing ---
    print("\nTesting models/softmax_allocator.py module...")
    test_input_dim = 50
    test_n_assets = 5
    batch_s = 4

    # Test WeightPredictorNet
    print("\n-- Testing WeightPredictorNet --")
    wp_params = {"hidden_dims": [32, 16], "dropout_rate": 0.1}
    predictor_net = WeightPredictorNet(test_input_dim, test_n_assets, **wp_params)
    dummy_input_features = torch.randn(batch_s, test_input_dim)
    logits_output = predictor_net(dummy_input_features)
    print(f"WeightPredictorNet input shape: {dummy_input_features.shape}")
    print(f"WeightPredictorNet output logits shape: {logits_output.shape}")
    assert logits_output.shape == (batch_s, test_n_assets)

    # Test SoftmaxPortfolioAllocator
    print("\n-- Testing SoftmaxPortfolioAllocator --")
    allocator_model = SoftmaxPortfolioAllocator(
        test_input_dim, test_n_assets, weight_predictor_params=wp_params
    )
    allocator_model.eval()  # Set to eval mode for consistent dropout behavior if testing MC later (not here though)

    portfolio_weights, raw_logits = allocator_model(dummy_input_features)
    print(f"SoftmaxPortfolioAllocator input shape: {dummy_input_features.shape}")
    print(
        f"SoftmaxPortfolioAllocator output portfolio_weights shape: {portfolio_weights.shape}"
    )
    print(f"SoftmaxPortfolioAllocator output raw_logits shape: {raw_logits.shape}")
    assert portfolio_weights.shape == (batch_s, test_n_assets)
    assert raw_logits.shape == (batch_s, test_n_assets)

    print(
        f"Sample predicted weights (1st in batch): {portfolio_weights[0].detach().numpy().round(4)}"
    )
    print(
        f"Sum of sample predicted weights (1st in batch): {portfolio_weights[0].sum().item():.4f}"
    )  # Should be 1.0
    assert np.isclose(portfolio_weights[0].sum().item(), 1.0)
    assert torch.all(portfolio_weights >= 0)

    print("\n--- Module Test Finished ---")
