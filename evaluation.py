import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader

from config import BATCH_SIZE, INTERVAL_SECONDS, REVEALED_TARGETS_PATH, SEQ_LEN, TEST_CSV_PATH
from dataloading import group_time_intervals
from dataset import ArrayDataset, StockDataset, get_valid_sequence_columns
from factor_models import add_hmm_features, add_index_return_features, add_rmt_features
from features import (
    add_auction_price_features, add_basic_features, add_causal_savgol_features,
    add_cross_sectional_features, add_ewma_features, add_hampel_feature,
    add_pairwise_triplet_features, add_realized_vol_features, add_sequence_features,
    add_volatility_features,
)
from markov_and_pca import add_markov_features, add_pca_features
from train import log_test_mae


# Predictions
def _torch_preds_loader(torch_model, X_tensor, device, batch_size):
    torch_model.eval()
    loader = DataLoader(ArrayDataset(X_tensor, torch.zeros(len(X_tensor))), batch_size=batch_size, shuffle=False)
    return loader


def get_torch_preds_last(torch_model, X_tensor, device, batch_size=BATCH_SIZE):
    loader = _torch_preds_loader(torch_model, X_tensor, device, batch_size)
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            out = torch_model(X_batch.to(device))
            preds.append(out[:, -1].cpu().numpy())
    return np.concatenate(preds)


def get_torch_preds_all(torch_model, X_tensor, device, batch_size=BATCH_SIZE):
    loader = _torch_preds_loader(torch_model, X_tensor, device, batch_size)
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            preds.append(torch_model(X_batch.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


# Competition test
def run_competition_test_evaluation(
    scaler, stock_stats, stock_weights, rmt_state, hmm_state, pca_state, markov_state,
    all_features, features, lgbm, neural_results, device,
):
    if not (os.path.exists(TEST_CSV_PATH) and os.path.exists(REVEALED_TARGETS_PATH)):
        print("Competition test evaluation skipped: test.csv or revealed_targets.csv not found.")
        return None

    try:
        test_raw = pd.read_csv(TEST_CSV_PATH)
        revealed_raw = pd.read_csv(REVEALED_TARGETS_PATH)

        required_revealed_cols = {"stock_id", "seconds_in_bucket", "revealed_target", "revealed_date_id"}
        missing = required_revealed_cols.difference(revealed_raw.columns)
        if missing:
            print(f"Competition test evaluation skipped: missing revealed-target columns: {sorted(missing)}")
            return None

        float_cols_test = test_raw.select_dtypes(include=["float64"]).columns
        test_raw[float_cols_test] = test_raw[float_cols_test].astype(np.float32)

        test_df = group_time_intervals(test_raw, INTERVAL_SECONDS)
        test_df, _ = add_basic_features(test_df)
        test_df, _ = add_auction_price_features(test_df)
        test_df, _, _ = add_pairwise_triplet_features(test_df)
        test_df, _ = add_sequence_features(test_df)
        test_df, _ = add_ewma_features(test_df)
        test_df, _ = add_causal_savgol_features(test_df)
        test_df, _ = add_volatility_features(test_df)
        test_df, _ = add_realized_vol_features(test_df)
        test_df, _ = add_hampel_feature(test_df)
        test_df, _ = add_cross_sectional_features(test_df)

        test_df = test_df.merge(stock_stats, on="stock_id", how="left")
        stock_agg_feature_names = [c for c in stock_stats.columns if c != "stock_id"]
        test_df[stock_agg_feature_names] = test_df[stock_agg_feature_names].fillna(0)

        test_df = add_index_return_features(test_df, stock_weights)
        test_df = add_rmt_features(test_df, rmt_state)
        test_df, _ = add_hmm_features(test_df, hmm_state)
        test_df, _ = add_pca_features(test_df, pca_state)
        test_df = add_markov_features(test_df, markov_state)

        revealed = revealed_raw.dropna(
            subset=["stock_id", "seconds_in_bucket", "revealed_target", "revealed_date_id"]
        ).copy()
        revealed["stock_id"] = revealed["stock_id"].astype(np.int64)
        revealed["revealed_date_id"] = revealed["revealed_date_id"].astype(np.int64)
        revealed["seconds_in_bucket"] = revealed["seconds_in_bucket"].astype(np.int64)
        revealed["time_group"] = revealed["seconds_in_bucket"] // INTERVAL_SECONDS * INTERVAL_SECONDS

        revealed = (
            revealed.sort_values(["stock_id", "revealed_date_id", "seconds_in_bucket"])
            .groupby(["stock_id", "revealed_date_id", "time_group"], as_index=False)
            .agg({"seconds_in_bucket": "last", "revealed_target": "last"})
        )

        duplicate_keys = revealed.duplicated(["stock_id", "revealed_date_id", "seconds_in_bucket"]).sum()
        if duplicate_keys:
            raise ValueError(f"Found {duplicate_keys} duplicate revealed-target keys.")

        test_df = test_df.merge(
            revealed[["stock_id", "revealed_date_id", "seconds_in_bucket", "revealed_target"]],
            left_on=["stock_id", "date_id", "seconds_in_bucket"],
            right_on=["stock_id", "revealed_date_id", "seconds_in_bucket"],
            how="left", validate="one_to_one",
        )
        test_df = test_df.rename(columns={"revealed_target": "target"})

        n_total_rows = len(test_df)
        n_labeled_rows = int(test_df["target"].notna().sum())
        labeled_dates = sorted(test_df.loc[test_df["target"].notna(), "date_id"].unique().tolist())
        unlabeled_dates = sorted(test_df.loc[test_df["target"].isna(), "date_id"].unique().tolist())

        print("\nCompetition test target alignment:")
        print(f"  matched labeled rows: {n_labeled_rows:,}/{n_total_rows:,}")
        print(f"  evaluated test dates: {labeled_dates}")
        if unlabeled_dates:
            print(f"  excluded unrevealed test dates: {unlabeled_dates}")

        if n_labeled_rows == 0:
            print("Competition test evaluation skipped: no revealed labels matched test rows.")
            return None

        test_df = test_df.dropna(subset=["target"]).copy()

        sizes = test_df.groupby(["stock_id", "date_id"]).size()
        valid_keys = sizes[sizes == SEQ_LEN].index
        row_idx = test_df.set_index(["stock_id", "date_id"]).index
        test_df = test_df[row_idx.isin(valid_keys)].copy()

        n_sequences = len(valid_keys)
        if n_sequences == 0:
            print("Competition test evaluation skipped: no complete revealed test sequences.")
            return None

        expected_rows = n_sequences * SEQ_LEN
        print(f"  complete evaluated stock-days: {n_sequences:,}")
        print(f"  all-step observations: {expected_rows:,}")
        print(f"  final-step observations: {n_sequences:,}")

        markov_test = get_valid_sequence_columns(test_df, ["target", "markov_expected_target"])
        markov_test_mae_all = mean_absolute_error(
            markov_test["target"].reshape(-1), markov_test["markov_expected_target"].reshape(-1)
        )
        markov_test_mae_last = mean_absolute_error(
            markov_test["target"][:, -1], markov_test["markov_expected_target"][:, -1]
        )

        test_df[all_features] = test_df[all_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        test_df[all_features] = scaler.transform(test_df[all_features]).astype(np.float32)

        test_dataset = StockDataset(test_df, features, verbose=False)
        X_test_seq = test_dataset.X
        y_test_all_2d = test_dataset.y.numpy()
        y_test_all = y_test_all_2d.reshape(-1)
        y_test_last = test_dataset.y_last.numpy()
        X_test_all = test_dataset.X.numpy().reshape(-1, test_dataset.X.shape[-1])
        X_test_last = test_dataset.X[:, -1, :].numpy()

        log_test_mae("LightGBM", "All steps", mean_absolute_error(y_test_all, lgbm.predict(X_test_all)))
        log_test_mae("LightGBM", "Final step", mean_absolute_error(y_test_last, lgbm.predict(X_test_last)))
        log_test_mae("Markov state model", "All steps", markov_test_mae_all)
        log_test_mae("Markov state model", "Final step", markov_test_mae_last)

        for name, result in neural_results.items():
            preds_all = get_torch_preds_all(result["model"], X_test_seq, device)
            preds_last = preds_all[:, -1]
            log_test_mae(name, "All steps", mean_absolute_error(y_test_all_2d.reshape(-1), preds_all.reshape(-1)))
            log_test_mae(name, "Final step", mean_absolute_error(y_test_last, preds_last))

        return test_dataset

    except Exception as e:
        print(f"Competition test evaluation failed: {type(e).__name__}: {e}")
        return None


# Training plots
def plot_loss_curves(neural_results):
    for name, result in neural_results.items():
        plt.figure(figsize=(8, 5))
        plt.plot(result["train_maes"], label="Eval-Train MAE")
        plt.plot(result["val_maes"], label="Validation MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE (bps)")
        plt.title(f"{name}: Eval-Train and Validation MAE")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_lr_schedule(primary_result):
    plt.figure(figsize=(8, 5))
    plt.plot(primary_result["learning_rates"], label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("OneCycle Learning Rate Schedule")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Auction error plot
def plot_prediction_error_across_window(
    neural_results, lgbm_preds_all_steps, markov_preds_all_steps, markov_labels_all_steps, val_dataset,
):
    time_axis = np.arange(SEQ_LEN) * INTERVAL_SECONDS
    labels = val_dataset.y.numpy()
    lgbm_preds = lgbm_preds_all_steps.reshape(-1, SEQ_LEN)
    lgbm_err = np.abs(lgbm_preds - labels)
    markov_err = np.abs(markov_preds_all_steps - markov_labels_all_steps)

    plt.figure(figsize=(8, 5))
    for name, result in neural_results.items():
        err = np.abs(result["best_preds_all"] - result["best_labels_all"])
        plt.plot(time_axis, err.mean(axis=0), marker="o", label=name)

    plt.plot(time_axis, lgbm_err.mean(axis=0), marker="o", label="LightGBM")
    plt.plot(time_axis, markov_err.mean(axis=0), marker="o", label="Markov state model")
    plt.xlabel("Seconds into auction window")
    plt.ylabel("Mean absolute error (bps)")
    plt.title("Prediction Error Across the Auction Window")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example predictions
def print_example_predictions(y_val_last, preds_last, n=10):
    print()
    print("Example Predictions (final timestep of window):")
    for i in range(min(n, len(y_val_last))):
        actual, predicted = y_val_last[i], preds_last[i]
        print(f"Example {i + 1} | Actual: {actual:.2f} bps | Predicted: {predicted:.2f} bps | Abs Error: {abs(actual - predicted):.2f} bps")