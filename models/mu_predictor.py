# portfolio_optimizer_project/models/mu_predictor.py
import torch
import torch.nn as nn


class MuPredictionNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_assets: int,
        hidden_dims: list = None,
        dropout_rate: float = 0.3,
    ):
        """
        Neural network to predict expected asset returns (mu).
        Args:
            input_dim (int): Dimensionality of the input features.
            n_assets (int): Number of assets, defines the output dimension (mu for each asset).
            hidden_dims (list, optional): List of integers for hidden layer sizes. Defaults to [128, 64].
            dropout_rate (float, optional): Dropout rate. Defaults to 0.3.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        print(f"\n--- Initializing MuPredictionNet ---")
        print(
            f"Input dimension: {input_dim}, Hidden dimensions: {hidden_dims}, Output (n_assets mu): {n_assets}"
        )

        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, n_assets))  # Output raw predicted mu

        self.network = nn.Sequential(*layers)
        print("MuPredictionNet initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs predicted mu for each asset."""
        return self.network(x)


if __name__ == "__main__":
    print("\nTesting models/mu_predictor.py module...")
    test_input_dim = 50
    test_n_assets = 5
    batch_s = 4
    net = MuPredictionNet(test_input_dim, test_n_assets, hidden_dims=[32, 16])
    dummy_input = torch.randn(batch_s, test_input_dim)
    mu_output = net(dummy_input)
    print(f"Input shape: {dummy_input.shape}, Output mu_pred shape: {mu_output.shape}")
    assert mu_output.shape == (batch_s, test_n_assets)
    print("MuPredictionNet test finished.")
