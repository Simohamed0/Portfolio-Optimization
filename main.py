# portfolio_optimizer_project/main.py
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
import numpy as np
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
from typing import Optional, List

log = logging.getLogger(__name__)

# --- Import Project Modules ---
try:
    from data_loader.synthetic_data import (
        generate_synthetic_market_data as generate_synthetic,
        plot_generated_data_characteristics as plot_synthetic,  # Renamed in synthetic_data.py
        create_features_and_targets_from_synthetic,
    )
    from data_loader.real_data import (
        load_and_process_real_stock_data,
        calculate_covariance_from_real_returns,
        create_features_and_targets_from_real_data,
    )
    from models.softmax_allocator import SoftmaxPortfolioAllocator
    from utils.plotting import (
        plot_training_loss,
        plot_portfolio_weights,
        plot_covariance_matrix,
    )
    from utils.evaluation import evaluate_portfolio_performance  # ADDED IMPORT
except ImportError as e:
    print("Error importing project modules. Check paths and __init__.py files.")
    print(f"Details: {e}")  # Print more details of the import error
    # Ensure your PYTHONPATH is set up correctly if running from an IDE in a complex workspace
    # or that you are running 'python main.py' from the 'portfolio_optimizer_project' root.
    raise


# --- Portfolio Objective Loss Function (Unchanged) ---
def portfolio_objective_loss(
    predicted_weights: torch.Tensor, actual_future_returns: torch.Tensor
) -> torch.Tensor:
    # (Implementation as before - looks good)
    if predicted_weights is None or predicted_weights.nelement() == 0:
        log.warning(
            "Predicted weights are None or empty in loss function. Returning zero loss."
        )
        return torch.tensor(
            0.0, requires_grad=True, device=actual_future_returns.device
        )
    if torch.isnan(predicted_weights).any() or torch.isinf(predicted_weights).any():
        log.warning(
            "NaN/Inf in predicted_weights to portfolio_loss. Returning high penalty loss."
        )
        return torch.tensor(float("inf"), device=actual_future_returns.device)
    portfolio_returns_batch = torch.sum(
        predicted_weights * actual_future_returns, dim=1
    )
    if (
        torch.isnan(portfolio_returns_batch).any()
        or torch.isinf(portfolio_returns_batch).any()
    ):
        log.warning(
            "NaN/Inf in portfolio_returns_batch during loss calculation. Returning high penalty loss."
        )
        return torch.tensor(float("inf"), device=actual_future_returns.device)
    loss_val = -torch.mean(portfolio_returns_batch)
    if torch.isnan(loss_val) or torch.isinf(loss_val):
        log.warning("Loss became NaN/Inf. Returning large finite penalty value.")
        return torch.tensor(1e5, device=actual_future_returns.device)
    return loss_val


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def run_experiment(cfg: DictConfig) -> None:
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    log.info("--- Experiment Configuration ---")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    log.info(f"Output directory: {output_dir}")
    log.info("-------------------------------")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available() and cfg.device == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = (
        torch.device("cpu")
        if cfg.device == "cuda" and not torch.cuda.is_available()
        else torch.device(cfg.device)
    )
    if cfg.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA specified but not available. Using CPU.")
    log.info(f"Using device: {device}")

    # --- 1. Data Generation and Preprocessing ---
    log.info("\n--- Stage 1: Data Generation & Preprocessing ---")

    X_all_features: Optional[torch.Tensor] = None
    Y_all_targets: Optional[torch.Tensor] = None
    sigma_for_context: Optional[np.ndarray] = (
        None  # Covariance matrix for context/evaluation
    )
    asset_labels_for_plots: List[str] = []
    actual_n_assets: int = 0

    if cfg.data_gen.get("source_type") == "real_stooq":
        log.info("Using real Stooq data source.")
        # Path to data dir should be relative to the original CWD where `python main.py` is run
        data_dir_abs = os.path.join(
            hydra.utils.get_original_cwd(), cfg.data_gen.load_params.data_dir
        )

        all_daily_returns_df, loaded_tickers = load_and_process_real_stock_data(
            tickers=list(cfg.data_gen.load_params.tickers),
            data_dir=data_dir_abs,
            price_column=cfg.data_gen.load_params.price_column,
            start_date_str=cfg.data_gen.load_params.start_date_str,
            end_date_str=cfg.data_gen.load_params.end_date_str,
            seed=cfg.seed,
        )

        if all_daily_returns_df.empty:
            log.error("Failed to load real stock data.")
            raise SystemExit
        actual_n_assets = all_daily_returns_df.shape[1]
        asset_labels_for_plots = loaded_tickers
        log.info(f"Processed real data for {actual_n_assets} assets: {loaded_tickers}")

        sigma_calc_cfg = cfg.data_gen.sigma_calc_params
        sigma_period_returns = all_daily_returns_df.loc[
            sigma_calc_cfg.sigma_period_start_date : sigma_calc_cfg.sigma_period_end_date
        ]
        if (
            sigma_period_returns.empty
            or len(sigma_period_returns) < actual_n_assets * 2
        ):  # Check for sufficient data
            log.warning(
                f"Sigma period data insufficient ({len(sigma_period_returns)} days). Using all loaded data for Sigma."
            )
            sigma_period_returns = all_daily_returns_df
            if (
                sigma_period_returns.empty
                or len(sigma_period_returns) < actual_n_assets * 2
            ):
                log.error("Not enough data even in fallback for Sigma. Exiting.")
                raise SystemExit

        annualized_sigma_np = calculate_covariance_from_real_returns(
            sigma_period_returns,
            annualization_factor=sigma_calc_cfg.annualization_factor,
        )
        if annualized_sigma_np is None:
            log.error("Sigma calculation failed.")
            raise SystemExit
        sigma_for_context = annualized_sigma_np  # This is annualized

        # Features and targets from the *entire* specified feature_target period
        # Splitting into train/test will happen after this
        feature_target_cfg = cfg.data_gen.feature_target_params
        full_feature_period_returns = all_daily_returns_df.loc[
            feature_target_cfg.train_period_start_date : feature_target_cfg.train_period_end_date
        ]

        X_all_features, Y_all_targets = create_features_and_targets_from_real_data(
            full_feature_period_returns,
            n_lags=feature_target_cfg.n_lags,
            pred_horizon=feature_target_cfg.pred_horizon,
        )

    elif (
        cfg.data_gen._target_
        == "data_loader.synthetic_data.generate_synthetic_market_data"
    ):
        log.info("Using synthetic data source.")
        asset_returns_df, factor_returns_df, factor_betas_true, sigma_np_synthetic = (
            generate_synthetic(  # Using alias
                n_samples=cfg.data_gen.n_samples,
                n_assets=cfg.data_gen.n_assets,
                n_factors=cfg.data_gen.n_factors,
                factor_ar_rho=cfg.data_gen.factor_ar_rho,
                factor_vol=cfg.data_gen.factor_vol,
                idiosyncratic_vol=cfg.data_gen.idiosyncratic_vol,
                avg_beta=cfg.data_gen.avg_beta,
                seed=cfg.seed,
            )
        )
        sigma_for_context = sigma_np_synthetic  # This is the true Sigma
        plot_synthetic(
            asset_returns_df,
            factor_returns_df,
            factor_betas_true,
            sigma_for_context,
            save_path_prefix=os.path.join(output_dir, "synthetic_"),
        )
        X_all_features, Y_all_targets = create_features_and_targets_from_synthetic(
            asset_returns_df,
            n_lags=cfg.data_gen.n_lags,
            pred_horizon=cfg.data_gen.pred_horizon,
        )
        actual_n_assets = cfg.data_gen.n_assets
        asset_labels_for_plots = [f"Asset {i + 1}" for i in range(actual_n_assets)]
    else:
        log.error(f"Unknown data_gen: {cfg.data_gen}")
        raise ValueError("Invalid data config.")

    if X_all_features.numel() == 0 or Y_all_targets.numel() == 0:
        log.error("No data (X or Y) was generated/loaded. Exiting.")
        raise SystemExit

    # --- Chronological Train-Test Split ---
    # (Applicable to both real and synthetic as both are time series)
    test_split_ratio = cfg.evaluation.get(
        "test_split_ratio", 0.2
    )  # Get from config, default 0.2
    if not (0 < test_split_ratio < 1):
        log.warning(
            f"Invalid test_split_ratio ({test_split_ratio}). Defaulting to 0.2."
        )
        test_split_ratio = 0.2

    num_total_samples = X_all_features.shape[0]
    split_idx = int(num_total_samples * (1 - test_split_ratio))

    if split_idx < 1 or num_total_samples - split_idx < 1:
        log.error(
            f"Not enough samples ({num_total_samples}) to perform train/test split with ratio {test_split_ratio}. Need at least 2 samples for train and test each."
        )
        # Fallback: use all data for training, no proper test set for evaluation
        X_train_tensor = X_all_features
        Y_train_tensor = Y_all_targets
        X_test_tensor = (
            torch.empty(0, X_all_features.shape[1])
            if X_all_features.numel() > 0
            else torch.empty(0, 0)
        )
        Y_test_tensor = (
            torch.empty(0, Y_all_targets.shape[1])
            if Y_all_targets.numel() > 0
            else torch.empty(0, 0)
        )
        log.warning(
            "Using all available data for training. Out-of-sample evaluation will be limited."
        )
    else:
        X_train_tensor = X_all_features[:split_idx]
        Y_train_tensor = Y_all_targets[:split_idx]
        X_test_tensor = X_all_features[split_idx:]
        Y_test_tensor = Y_all_targets[split_idx:]

    log.info(
        f"Data split: Train X {X_train_tensor.shape}, Train Y {Y_train_tensor.shape}"
    )
    log.info(f"Data split: Test X {X_test_tensor.shape}, Test Y {Y_test_tensor.shape}")

    n_lags_used = (
        cfg.data_gen.feature_target_params.n_lags
        if cfg.data_gen.get("source_type") == "real_stooq"
        else cfg.data_gen.n_lags
    )
    actual_input_dim = actual_n_assets * n_lags_used
    if X_train_tensor.numel() > 0 and X_train_tensor.shape[1] != actual_input_dim:
        log.error(
            f"Mismatch X_train feat dim. Expected {actual_input_dim}, got {X_train_tensor.shape[1]}"
        )
        raise SystemExit

    # --- 2. Model Initialization ---
    log.info("\n--- Stage 2: Model Initialization ---")
    model_config_node = cfg.model
    wp_params = model_config_node.get("weight_predictor_params", {})
    if isinstance(wp_params.get("hidden_dims"), ListConfig):
        wp_params["hidden_dims"] = list(wp_params["hidden_dims"])
    model = hydra.utils.instantiate(
        model_config_node,
        input_dim=actual_input_dim,
        n_assets=actual_n_assets,
        weight_predictor_params=wp_params,
        _recursive_=False,
    ).to(device)
    try:
        optimizer = torch.optim.Adam(
            model.weight_predictor.parameters(), lr=cfg.training.learning_rate
        )
    except AttributeError:
        log.error("Model missing 'weight_predictor'.")
        raise
    log.info(f"Model '{model_config_node._target_}' and optimizer initialized.")

    # --- 3. Training Loop ---
    # (Training loop as before, using X_train_tensor, Y_train_tensor)
    log.info("\n--- Stage 3: Training ---")
    train_losses = []
    n_samples_train = X_train_tensor.shape[0]
    if n_samples_train == 0:
        log.error("No training samples available after split. Exiting.")
        raise SystemExit

    for epoch_idx in range(cfg.training.n_epochs):
        model.train()
        epoch_total_loss = 0.0
        permutation = torch.randperm(n_samples_train)
        for i in range(0, n_samples_train, cfg.training.batch_size):
            optimizer.zero_grad()
            indices = permutation[i : i + cfg.training.batch_size]
            X_batch = X_train_tensor[indices].to(device)
            Y_batch_actual_returns = Y_train_tensor[indices].to(device)
            if X_batch.shape[0] == 0:
                continue
            predicted_portfolio_weights, _ = model(X_batch)
            loss = portfolio_objective_loss(
                predicted_portfolio_weights, Y_batch_actual_returns
            )
            if (
                torch.isnan(loss) or torch.isinf(loss) or abs(loss.item()) > 1e4
            ):  # abs for positive or negative large loss
                log.warning(
                    f"Epoch {epoch_idx + 1}, Batch {i // cfg.training.batch_size + 1}: Unstable loss ({loss.item():.2e}). Skipping."
                )
                epoch_total_loss += (1e5) * X_batch.shape[0]
                continue
            loss.backward()
            if (
                cfg.training.clip_grad_norm is not None
                and cfg.training.clip_grad_norm > 0
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.weight_predictor.parameters(),
                    max_norm=cfg.training.clip_grad_norm,
                )
            optimizer.step()
            epoch_total_loss += loss.item() * X_batch.shape[0]
        avg_epoch_loss = (
            epoch_total_loss / n_samples_train if n_samples_train > 0 else float("nan")
        )
        train_losses.append(avg_epoch_loss)
        if (epoch_idx + 1) % 1 == 0 or epoch_idx == cfg.training.n_epochs - 1:
            log.info(
                f"Epoch [{epoch_idx + 1}/{cfg.training.n_epochs}]: Avg Loss = {avg_epoch_loss:.6f}"
            )
            if (
                (epoch_idx + 1) % 10 == 0
                or epoch_idx == cfg.training.n_epochs - 1
                or epoch_idx == 0
            ):
                try:
                    param_to_check = model.weight_predictor.network[0].weight
                    if param_to_check.grad is not None:
                        grad_sample = param_to_check.grad.abs()
                        log.info(
                            f"  Sample Grads (Net[0].weight): Mean={grad_sample.mean().item():.2e}, Max={grad_sample.max().item():.2e}"
                        )
                    else:
                        log.info(
                            f"  Sample Grads for Net[0].weight are None (Epoch {epoch_idx + 1})."
                        )
                except Exception as e:
                    log.warning(f"Could not get sample gradients: {e}")
    log.info("\n--- Training Finished ---")

    # --- 4. Plotting and Saving Training Results ---
    plot_training_loss(
        train_losses, save_path=os.path.join(output_dir, "training_loss.png")
    )

    # --- 5. Out-of-Sample Evaluation ---
    if (
        X_test_tensor is not None
        and Y_test_tensor is not None
        and X_test_tensor.numel() > 0
    ):
        log.info("\n--- Stage 4: Out-of-Sample Evaluation ---")
        model.eval()
        all_predicted_weights_test_list = []
        # Process test set in batches if it's large
        test_batch_size = (
            cfg.training.batch_size
        )  # Can use same batch size or a different one for eval
        for i in range(0, X_test_tensor.shape[0], test_batch_size):
            X_test_batch = X_test_tensor[i : i + test_batch_size].to(device)
            with torch.no_grad():
                predicted_weights_batch_test, _ = model(X_test_batch)
            all_predicted_weights_test_list.append(predicted_weights_batch_test.cpu())

        if all_predicted_weights_test_list:
            all_predicted_weights_test = torch.cat(
                all_predicted_weights_test_list, dim=0
            )

            eval_results = evaluate_portfolio_performance(
                predicted_weights_timeseries=all_predicted_weights_test,
                actual_returns_timeseries=Y_test_tensor.cpu(),  # Y_test_tensor is already on CPU from split
                asset_labels=asset_labels_for_plots,
                risk_free_rate_annual=cfg.evaluation.risk_free_rate_annual,
                annualization_factor=cfg.evaluation.annualization_factor,
                # Add plot_save_prefix to evaluate_portfolio_performance if you want it to save plots
            )
            log.info(f"Evaluation Metrics on Test Set: {eval_results}")
            try:  # Save evaluation metrics
                eval_metrics_df = pd.DataFrame.from_dict(
                    eval_results, orient="index", columns=["value"]
                )
                # For avg_allocation, which is a Series:
                if "average_allocation" in eval_results:
                    avg_alloc_series = eval_results["average_allocation"]
                    avg_alloc_df = avg_alloc_series.reset_index()
                    avg_alloc_df.columns = ["Asset", "Average Weight"]
                    avg_alloc_df.to_csv(
                        os.path.join(output_dir, "average_allocation_test.csv"),
                        index=False,
                    )
                    # plot the average allocation
                    plot_portfolio_weights(
                        avg_alloc_series.values,
                        labels=avg_alloc_series.index.tolist(),
                        title="Average Portfolio Allocation (Test Set)",
                        save_path=os.path.join(
                            output_dir, "average_allocation_test.png"
                        ),
                    )
                    # Remove from main df to avoid issues with to_json for mixed types
                    del eval_metrics_df.loc["average_allocation"]

                eval_metrics_df.to_json(
                    os.path.join(output_dir, "evaluation_metrics.json"), indent=4
                )

                # Plot the last predicted weights from the test set for an example
                plot_portfolio_weights(
                    all_predicted_weights_test[-1].numpy(),
                    labels=asset_labels_for_plots,
                    title="Predicted Portfolio Weights (Last Test Sample)",
                    save_path=os.path.join(
                        output_dir, "predicted_weights_last_test_sample.png"
                    ),
                )

            except Exception as e_save:
                log.error(f"Could not save evaluation metrics/plots: {e_save}")
        else:
            log.info("No predicted weights generated for the test set.")
    else:
        log.info("Skipping out-of-sample evaluation as no test data is available.")

    # Plot the primary covariance matrix used/generated
    if sigma_for_context is not None:
        title_suffix = (
            "Real Data (Annualized)"
            if cfg.data_gen.get("source_type") == "real_stooq"
            else "Synthetic Data"
        )
        plot_covariance_matrix(
            sigma_for_context,
            labels=asset_labels_for_plots,
            title=f"Primary Covariance Matrix ({title_suffix})",
            save_path=os.path.join(output_dir, "primary_covariance_matrix.png"),
        )

    final_config_path = os.path.join(output_dir, "run_config_final.yaml")
    with open(final_config_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    log.info(f"Final resolved configuration saved to {final_config_path}")
    log.info(f"All results saved to directory: {output_dir}"m )


if __name__ == "__main__":
    run_experiment()
