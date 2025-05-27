# portfolio_optimizer_project/models/lstm_allocator.py
import torch
import torch.nn as nn
from typing import List, Optional, Dict
import numpy as np


class LSTMPredictorNet(nn.Module):
    def __init__(
        self,
        n_assets: int,  # Number of assets (becomes input_size per time step for LSTM)
        n_lags: int,  # Sequence length for LSTM
        lstm_hidden_size: int = 64,
        num_lstm_layers: int = 1,
        lstm_dropout: float = 0.0,  # Dropout between LSTM layers if num_lstm_layers > 1
        fc_hidden_dims: Optional[List[int]] = None,  # FC layers after LSTM
        fc_dropout_rate: float = 0.3,
    ):
        """
        LSTM-based network to predict logits for portfolio weights from sequences of asset returns.
        Args:
            n_assets (int): Number of assets. Each time step in the sequence will have n_assets features.
            n_lags (int): The sequence length (number of past time steps to consider).
            lstm_hidden_size (int): Number of features in the hidden state h of LSTM.
            num_lstm_layers (int): Number of recurrent LSTM layers.
            lstm_dropout (float): Dropout probability for LSTM layers (if num_layers > 1).
            fc_hidden_dims (list, optional): List of integers for fully connected hidden layer sizes after LSTM.
                                             If None or empty, LSTM output directly goes to final linear layer.
            fc_dropout_rate (float): Dropout rate for FC layers.
        """
        super().__init__()
        print(f"\n--- Initializing LSTMPredictorNet ---")
        print(
            f"n_assets (LSTM input_size per step): {n_assets}, n_lags (seq_len): {n_lags}"
        )
        print(
            f"LSTM: hidden_size={lstm_hidden_size}, num_layers={num_lstm_layers}, dropout={lstm_dropout}"
        )
        print(f"FC Head: hidden_dims={fc_hidden_dims}, dropout={fc_dropout_rate}")

        self.n_assets = n_assets
        self.n_lags = n_lags  # Will be needed if reshaping happens outside
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(
            input_size=n_assets,  # Each time step has features for all n_assets
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,  # Input tensor format: (batch_size, seq_len, input_size)
            dropout=lstm_dropout if num_lstm_layers > 1 else 0.0,
        )

        if fc_hidden_dims is None:
            fc_hidden_dims = []  # No intermediate FC hidden layers

        # Fully connected layers after LSTM
        fc_layers = []
        current_dim = lstm_hidden_size  # Output from LSTM (last hidden state)
        for h_dim in fc_hidden_dims:
            fc_layers.append(nn.Linear(current_dim, h_dim))
            # BatchNorm after LSTM and before FC head might be tricky,
            # consider nn.LayerNorm(current_dim) on LSTM output if needed.
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(fc_dropout_rate))
            current_dim = h_dim

        fc_layers.append(nn.Linear(current_dim, n_assets))  # Output logits for n_assets
        self.fc_head = nn.Sequential(*fc_layers)
        print("LSTMPredictorNet initialized.")

    def forward(self, x_sequential: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_sequential (torch.Tensor): Input features of shape (batch_size, sequence_length, n_assets).
                                         sequence_length should be n_lags.
                                         n_assets is the number of features per time step.
        Returns:
            torch.Tensor: Logits for portfolio weights (batch_size, n_assets).
        """
        if x_sequential.dim() != 3 or x_sequential.shape[2] != self.n_assets:
            # Attempt to reshape if input is flattened (batch_size, n_lags * n_assets)
            if (
                x_sequential.dim() == 2
                and x_sequential.shape[1] == self.n_lags * self.n_assets
            ):
                # print(f"Reshaping input from {x_sequential.shape} to (batch, seq_len, features_per_step)")
                x_sequential = x_sequential.view(-1, self.n_lags, self.n_assets)
            else:
                raise ValueError(
                    f"Expected input shape (batch, seq_len={self.n_lags}, features_per_step={self.n_assets}), "
                    f"but got {x_sequential.shape}"
                )

        # Initialize hidden and cell states for LSTM (for each batch)
        # (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(
            self.num_lstm_layers, x_sequential.size(0), self.lstm_hidden_size
        ).to(x_sequential.device)
        c0 = torch.zeros(
            self.num_lstm_layers, x_sequential.size(0), self.lstm_hidden_size
        ).to(x_sequential.device)

        # LSTM forward pass
        # lstm_out contains hidden states for all time steps: (batch_size, seq_len, lstm_hidden_size)
        # hn is the final hidden state: (num_layers, batch_size, lstm_hidden_size)
        # cn is the final cell state: (num_layers, batch_size, lstm_hidden_size)
        lstm_out, (hn, cn) = self.lstm(x_sequential, (h0, c0))

        # We typically use the hidden state of the last LSTM layer from the last time step.
        # hn is (num_layers, batch_size, hidden_size). We want the last layer's output.
        last_layer_hidden_state = hn[-1, :, :]  # Shape: (batch_size, lstm_hidden_size)
        # Alternatively, use the output of the last time step from lstm_out:
        # last_time_step_output = lstm_out[:, -1, :] # Also (batch_size, lstm_hidden_size)
        # They are usually the same if not using bidirectional LSTMs etc.

        logits = self.fc_head(last_layer_hidden_state)  # Pass to FC layers
        return logits


class SoftmaxPortfolioAllocatorLSTM(nn.Module):
    def __init__(
        self,
        # input_dim: int, # Not directly needed here, LSTMPredictorNet defines its input from n_assets, n_lags
        n_assets: int,
        n_lags: int,  # Needed by LSTMPredictorNet
        lstm_predictor_params: Optional[Dict] = None,
    ):
        """
        Main model combining LSTMPredictorNet and Softmax for portfolio allocation.
        Args:
            n_assets (int): Number of assets.
            n_lags (int): Sequence length for LSTM input.
            lstm_predictor_params (dict, optional): Parameters for LSTMPredictorNet
                                                     (e.g., lstm_hidden_size, num_lstm_layers, etc.).
        """
        super().__init__()
        print(f"\n--- Initializing SoftmaxPortfolioAllocatorLSTM ---")

        if lstm_predictor_params is None:
            lstm_predictor_params = {}  # Use defaults in LSTMPredictorNet

        self.weight_predictor = LSTMPredictorNet(
            n_assets=n_assets, n_lags=n_lags, **lstm_predictor_params
        )
        self.softmax = nn.Softmax(dim=-1)
        print("SoftmaxPortfolioAllocatorLSTM initialized.")

    def forward(self, x_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_features (torch.Tensor): Input features.
                                       Expected to be either (batch_size, sequence_length, n_assets)
                                       OR (batch_size, sequence_length * n_assets) which will be reshaped.
        Returns:
            portfolio_weights (torch.Tensor): (batch_size, n_assets) - after softmax
            logits (torch.Tensor): (batch_size, n_assets) - raw output from LSTMPredictorNet
        """
        # LSTMPredictorNet's forward method handles potential reshaping if input is flat
        logits = self.weight_predictor(x_features)
        portfolio_weights = self.softmax(logits)

        return portfolio_weights, logits
