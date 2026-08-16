import gc

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import BATCH_SIZE, EPOCHS, INTERVAL_SECONDS, VALIDATION_DAYS
from dataloading import group_time_intervals, load_train_data
from dataset import StockDataset, get_valid_sequence_columns
from evaluation import (
    plot_loss_curves, plot_lr_schedule, plot_prediction_error_across_window,
    print_example_predictions, run_competition_test_evaluation,
)
from factor_models import (
    add_hmm_features, add_index_return_features, add_oof_index_return_features,
    add_rmt_features, fit_hmm_filter, fit_index_weights, fit_rmt_factors,
    get_rmt_feature_names,
)
from features import (
    add_auction_price_features, add_basic_features, add_causal_savgol_features,
    add_cross_sectional_features, add_ewma_features, add_hampel_feature,
    add_pairwise_triplet_features, add_realized_vol_features, add_sequence_features,
    add_volatility_features, compute_stock_stats, get_stock_agg_feature_names,
)
from markov_and_pca import add_markov_features, add_pca_features, fit_markov_model, fit_pca_model
from models import build_model
from train import log_result, results_log, run_feature_selection, train_lightgbm, train_model

BASE_FEATURES_RAW = [
    "seconds_in_bucket", "imbalance_size", "imbalance_buy_sell_flag", "reference_price",
    "matched_size", "bid_price", "bid_size", "ask_price", "ask_size", "wap",
    "bid_ask_spread", "mid_price", "imbalance_ratio", "wap_reference_diff",
    "wap_mid_diff", "size_ratio", "matched_imbalance_ratio", "wap_return",
]
BASIC_FEATURE_DUPES = {
    "bid_ask_spread", "mid_price", "imbalance_ratio", "wap_reference_diff",
    "wap_mid_diff", "size_ratio", "matched_imbalance_ratio",
}


# Main pipeline
def main():
    results_log.clear()

    train = load_train_data()
    train = group_time_intervals(train, INTERVAL_SECONDS)

    # Feature engineering
    train, basic_feature_names = add_basic_features(train)
    train, auction_feature_names = add_auction_price_features(train)
    train, pairwise_feature_names, triplet_feature_names = add_pairwise_triplet_features(train)
    train, rolling_feature_names = add_sequence_features(train)
    train, ewma_feature_names = add_ewma_features(train)
    train, savgol_feature_names = add_causal_savgol_features(train)
    train, volatility_feature_names = add_volatility_features(train)
    train, realized_vol_feature_names = add_realized_vol_features(train)
    train, hampel_feature_names = add_hampel_feature(train)
    train, cross_sectional_feature_names = add_cross_sectional_features(train)
    train[cross_sectional_feature_names] = train[cross_sectional_feature_names].fillna(0)

    stock_agg_feature_names = get_stock_agg_feature_names()

    # Base features
    base_features = BASE_FEATURES_RAW + [n for n in basic_feature_names if n not in BASIC_FEATURE_DUPES]

    features = (
        base_features + auction_feature_names + pairwise_feature_names + triplet_feature_names
        + rolling_feature_names + stock_agg_feature_names + ewma_feature_names + savgol_feature_names
        + volatility_feature_names + realized_vol_feature_names + hampel_feature_names
        + cross_sectional_feature_names
    )

    computed_so_far = [f for f in features if f not in stock_agg_feature_names]
    train[computed_so_far] = train[computed_so_far].replace([np.inf, -np.inf], np.nan)
    train[computed_so_far] = train[computed_so_far].fillna(0).astype(np.float32)

    # Train / validation split
    unique_dates = np.array(sorted(train["date_id"].unique()))
    if len(unique_dates) <= VALIDATION_DAYS:
        raise ValueError(f"Need more than {VALIDATION_DAYS} unique dates; found {len(unique_dates)}.")

    train_dates, val_dates = unique_dates[:-VALIDATION_DAYS], unique_dates[-VALIDATION_DAYS:]
    train_df = train[train["date_id"].isin(train_dates)].copy()
    val_df = train[train["date_id"].isin(val_dates)].copy()

    print(
        f"Date split: train {train_dates[0]}-{train_dates[-1]} ({len(train_dates)} dates), "
        f"validation {val_dates[0]}-{val_dates[-1]} ({len(val_dates)} dates)"
    )
    print(
        "NOTE: competition test.csv overlaps the tail of train.csv in this dataset, so its Test MAE "
        "is a replay/competition check, not an independent out-of-sample score."
    )

    del train
    gc.collect()

    # Stock statistics
    stock_stats = compute_stock_stats(train_df)
    train_df = train_df.merge(stock_stats, on="stock_id", how="left")
    val_df = val_df.merge(stock_stats, on="stock_id", how="left")
    for df in (train_df, val_df):
        df[stock_agg_feature_names] = df[stock_agg_feature_names].fillna(0)

    # Synthetic index
    raw_train_for_weights = train_df.copy()
    train_df = add_oof_index_return_features(train_df)
    stock_weights = fit_index_weights(raw_train_for_weights)
    val_df = add_index_return_features(val_df, stock_weights)
    del raw_train_for_weights

    features += ["index_return", "stock_minus_index_return"]

    # RMT
    rmt_state = fit_rmt_factors(train_df)
    train_df = add_rmt_features(train_df, rmt_state)
    val_df = add_rmt_features(val_df, rmt_state)
    features += get_rmt_feature_names(rmt_state["n_factors"])

    # Soft state model
    hmm_state = fit_hmm_filter(train_df)
    train_df, hmm_feature_names = add_hmm_features(train_df, hmm_state)
    val_df, _ = add_hmm_features(val_df, hmm_state)
    hmm_feature_names = [n for n in hmm_feature_names if n != "hmm_expected_target"]
    features += hmm_feature_names

    # PCA
    pca_state = fit_pca_model(train_df)
    train_df, pca_feature_names = add_pca_features(train_df, pca_state)
    val_df, _ = add_pca_features(val_df, pca_state)
    features += pca_feature_names

    # Markov baseline
    markov_state = fit_markov_model(train_df)
    train_df = add_markov_features(train_df, markov_state)
    val_df = add_markov_features(val_df, markov_state)

    markov_val = get_valid_sequence_columns(val_df, ["target", "markov_expected_target"])
    markov_val_mae_all = mean_absolute_error(
        markov_val["target"].reshape(-1), markov_val["markov_expected_target"].reshape(-1)
    )
    markov_val_mae_last = mean_absolute_error(
        markov_val["target"][:, -1], markov_val["markov_expected_target"][:, -1]
    )
    log_result("Markov state model", markov_val_mae_all, category="Baseline", evaluation="All steps")
    log_result("Markov state model", markov_val_mae_last, category="Baseline", evaluation="Final step")

    # Clean / scale
    for df in (train_df, val_df):
        df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    all_features = list(features)
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features]).astype(np.float32)
    val_df[features] = scaler.transform(val_df[features]).astype(np.float32)

    # Datasets
    train_dataset = StockDataset(train_df, features, verbose=False)
    val_dataset = StockDataset(val_df, features, verbose=False)
    del train_df, val_df
    gc.collect()

    # Feature selection
    X_train_all = train_dataset.X.numpy().reshape(-1, train_dataset.X.shape[-1])
    y_train_all = train_dataset.y.numpy().reshape(-1)
    X_val_all = val_dataset.X.numpy().reshape(-1, val_dataset.X.shape[-1])
    y_val_all = val_dataset.y.numpy().reshape(-1)

    kept_features, keep_idx = run_feature_selection(X_train_all, y_train_all, X_val_all, y_val_all, features)
    train_dataset.X = train_dataset.X[:, :, keep_idx]
    val_dataset.X = val_dataset.X[:, :, keep_idx]
    features = kept_features

    # Final arrays
    X_train_all = train_dataset.X.numpy().reshape(-1, train_dataset.X.shape[-1])
    y_train_all = train_dataset.y.numpy().reshape(-1)
    X_val_all = val_dataset.X.numpy().reshape(-1, val_dataset.X.shape[-1])
    y_val_all = val_dataset.y.numpy().reshape(-1)
    X_val_last = val_dataset.X[:, -1, :].numpy()
    y_val_last = val_dataset.y_last.numpy()

    # LightGBM
    lgbm_result = train_lightgbm(X_train_all, y_train_all, X_val_all, y_val_all, X_val_last, y_val_last)
    lgbm = lgbm_result["model"]
    log_result("LightGBM", lgbm_result["val_mae_all"], category="Baseline", evaluation="All steps")
    log_result("LightGBM", lgbm_result["val_mae_last"], category="Baseline", evaluation="Final step")

    # Neural loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Neural model
    model_builders = [("LSTM + Transformer", lambda: build_model(len(features), device, seed=42))]
    neural_results = {}

    for name, builder in model_builders:
        print(f"\nTraining: {name}")
        m = builder()
        result = train_model(m, name, device, train_loader, val_loader, epochs=EPOCHS)
        neural_results[name] = result

        print(f"  -> done (best epoch {result['best_epoch']}, validation final-step MAE {result['best_val_mae_last_step']:.4f})")
        log_result(name, result["best_val_mae"], category="Neural", evaluation="All steps")
        log_result(name, result["best_val_mae_last_step"], category="Neural", evaluation="Final step")

    primary = neural_results["LSTM + Transformer"]

    # Plots
    plot_loss_curves(neural_results)
    plot_lr_schedule(primary)
    plot_prediction_error_across_window(
        neural_results, lgbm_result["val_preds_all"], markov_val["markov_expected_target"],
        markov_val["target"], val_dataset,
    )
    print_example_predictions(val_dataset.y_last.numpy(), primary["best_preds_last"])

    # Competition test
    run_competition_test_evaluation(
        scaler=scaler, stock_stats=stock_stats, stock_weights=stock_weights, rmt_state=rmt_state,
        hmm_state=hmm_state, pca_state=pca_state, markov_state=markov_state, all_features=all_features,
        features=features, lgbm=lgbm, neural_results=neural_results, device=device,
    )

    # Results
    results_df = pd.DataFrame(results_log)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_colwidth", 40)

    for evaluation in ["Final step", "All steps"]:
        table = results_df[results_df["Evaluation"] == evaluation].copy()
        table = table.sort_values(["Test MAE", "Val MAE"], na_position="last").reset_index(drop=True)

        print("\n" + "=" * 98)
        print(f"{evaluation.upper()} COMPARISON  (LAST-81-DAY VALIDATION vs COMPETITION TEST; MAE in bps, lower is better)")
        print("=" * 98)

        display_table = table[["Model", "Category", "Val MAE", "Test MAE"]].rename(
            columns={"Val MAE": "Validation MAE", "Test MAE": "Competition Test MAE"}
        )
        print(display_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("NOTE: Competition Test MAE is not an independent out-of-sample score because the supplied test.csv overlaps the end of train.csv.")
        print("=" * 98)


if __name__ == "__main__":
    main()