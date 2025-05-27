# portfolio_optimizer_project/utils/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def plot_training_loss(
    losses_list: list,
    title: str = "Training Loss Over Epochs",
    xlabel: str = "Epoch",
    ylabel: str = "Average Loss",
    save_path: str = None,
):
    """Plots the training loss."""
    plt.figure(figsize=(10, 5))
    valid_losses = [l for l in losses_list if not (np.isnan(l) or np.isinf(l))]
    if valid_losses:
        plt.plot(valid_losses, label="Training Loss")
    else:
        plt.plot([], label="No valid losses to plot")  # Handle case of all NaN/Inf
        print("Warning: No valid training losses were provided for plotting.")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training loss plot saved to: {save_path}")
    plt.show()


def plot_portfolio_weights(
    weights_np: np.ndarray,
    labels: list,
    title: str = "Portfolio Weights",
    save_path: str = None,
):
    """Plots portfolio weights as a bar chart."""
    if weights_np is None or weights_np.size == 0:
        print("Warning: No weights provided for plotting.")
        return

    plt.figure(
        figsize=(max(8, len(labels) * 0.8), 5)
    )  # Adjust width based on number of labels
    sns.barplot(
        x=labels, y=weights_np, palette="viridis_r"
    )  # Changed palette for variety
    plt.title(title)
    plt.xlabel("Asset")
    plt.ylabel("Weight")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(
        0,
        max(
            0.01,
            np.nan_to_num(weights_np).max() * 1.15 if weights_np.size > 0 else 0.01,
        ),
    )
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Portfolio weights plot saved to: {save_path}")
    plt.show()


def plot_covariance_matrix(
    sigma_matrix_np: np.ndarray,
    labels: list,
    title: str = "Covariance Matrix (Σ)",
    fmt: str = ".3f",  # Format for annotations
    save_path: str = None,
):
    """Plots a heatmap of the covariance matrix."""
    if sigma_matrix_np is None or sigma_matrix_np.size == 0:
        print("Warning: No covariance matrix provided for plotting.")
        return

    plt.figure(
        figsize=(
            max(7, sigma_matrix_np.shape[0] * 0.7),
            max(6, sigma_matrix_np.shape[1] * 0.6),
        )
    )
    sns.heatmap(
        sigma_matrix_np,
        annot=True,
        cmap="coolwarm",
        fmt=fmt,
        square=True,
        cbar_kws={"shrink": 0.8},
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Asset")  # Assuming square matrix with same labels for x and y
    plt.ylabel("Asset")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Covariance matrix plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    # --- Example of using this module directly for testing ---
    print("Testing utils/plotting.py module...")

    # Test plot_training_loss
    dummy_losses = [-0.01, -0.005, -0.002, -0.001, np.nan, -0.0005]
    plot_training_loss(dummy_losses, title="Test Training Loss")

    # Test plot_portfolio_weights
    dummy_weights = np.array([0.1, 0.3, 0.05, 0.25, 0.3])
    dummy_labels = [f"Asset {i + 1}" for i in range(len(dummy_weights))]
    plot_portfolio_weights(
        dummy_weights, labels=dummy_labels, title="Test Portfolio Weights"
    )

    # Test plot_covariance_matrix
    dummy_sigma = np.array(
        [[0.010, 0.002, 0.001], [0.002, 0.008, 0.003], [0.001, 0.003, 0.005]]
    )
    sigma_labels = [f"S{i + 1}" for i in range(dummy_sigma.shape[0])]
    plot_covariance_matrix(
        dummy_sigma, labels=sigma_labels, title="Test Covariance Matrix"
    )

    print("\n--- Plotting Module Test Finished ---")
