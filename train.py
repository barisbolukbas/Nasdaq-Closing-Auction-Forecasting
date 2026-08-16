import numpy as np
import torch

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from torch import nn

from config import EPS, MIN_FEATURE_IMPORTANCE_FRAC, PATIENCE, PEAK_LR, WEIGHT_DECAY

# Results
results_log = []


def log_result(name, val_mae, test_mae=None, category="", evaluation=""):
    results_log.append({
        "Model": name, "Evaluation": evaluation, "Category": category,
        "Val MAE": val_mae, "Test MAE": test_mae,
    })


def log_test_mae(name, evaluation, mae):
    for row in results_log:
        if row["Model"] == name and row["Evaluation"] == evaluation:
            row["Test MAE"] = mae
            return
    log_result(name, np.nan, test_mae=mae, category="Test-only", evaluation=evaluation)


# Feature selection
def run_feature_selection(X_train_all, y_train_all, X_val_all, y_val_all, features):
    feature_selector = LGBMRegressor(
        n_estimators=2000, num_leaves=63, max_depth=-1, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, objective="mae", verbosity=-1,
        importance_type="gain",
    )
    feature_selector.fit(
        X_train_all, y_train_all, eval_set=[(X_val_all, y_val_all)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    importances = feature_selector.feature_importances_.astype(np.float64)
    importance_frac = importances / (importances.sum() + EPS)
    keep_mask = importance_frac >= MIN_FEATURE_IMPORTANCE_FRAC
    kept_features = [f for f, keep in zip(features, keep_mask) if keep]

    print(f"Feature selection: keeping {len(kept_features)}/{len(features)} features")
    return kept_features, np.where(keep_mask)[0]


# LightGBM
def train_lightgbm(X_train_all, y_train_all, X_val_all, y_val_all, X_val_last, y_val_last):
    lgbm = LGBMRegressor(
        n_estimators=3000, num_leaves=63, max_depth=-1, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, objective="mae", verbosity=-1,
    )
    lgbm.fit(
        X_train_all, y_train_all, eval_set=[(X_val_all, y_val_all)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    preds_all = lgbm.predict(X_val_all)
    preds_last = lgbm.predict(X_val_last)

    return {
        "model": lgbm,
        "val_mae_all": mean_absolute_error(y_val_all, preds_all),
        "val_mae_last": mean_absolute_error(y_val_last, preds_last),
        "val_preds_all": preds_all,
        "val_preds_last": preds_last,
    }


# Neural training
loss_fn = nn.L1Loss()


def _evaluate_torch_loader(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds_all.append(outputs.cpu().numpy())
            labels_all.append(y_batch.cpu().numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)

    mae_all = mean_absolute_error(labels_all.reshape(-1), preds_all.reshape(-1))
    mae_last = mean_absolute_error(labels_all[:, -1], preds_all[:, -1])
    return mae_all, mae_last, preds_all, labels_all


def train_model(model, name, device, train_loader, val_loader, epochs, patience=PATIENCE, verbose=True, print_every=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=PEAK_LR, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy="cos",
    )

    train_maes, val_maes, learning_rates = [], [], []
    best_val_mae, best_model_state, best_epoch = float("inf"), None, 0
    best_preds_all = best_labels_all = best_preds_last = best_labels_last = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        train_mae, _, _, _ = _evaluate_torch_loader(model, train_loader, device)
        val_mae, val_mae_last, epoch_preds_all, epoch_labels_all = _evaluate_torch_loader(model, val_loader, device)

        current_lr = optimizer.param_groups[0]["lr"]
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        learning_rates.append(current_lr)

        is_last_epoch = epoch + 1 == epochs
        if verbose and (epoch == 0 or (epoch + 1) % print_every == 0 or is_last_epoch):
            print(
                f"    epoch {epoch + 1:>3}/{epochs} | eval-train MAE {train_mae:.4f} | "
                f"validation MAE {val_mae:.4f} | best so far {min(best_val_mae, val_mae):.4f}"
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            best_preds_all, best_labels_all = epoch_preds_all, epoch_labels_all
            best_preds_last, best_labels_last = epoch_preds_all[:, -1], epoch_labels_all[:, -1]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"    early stopping at epoch {epoch + 1} (best epoch {best_epoch}, validation MAE {best_val_mae:.4f})")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    return {
        "name": name,
        "model": model,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "best_val_mae_last_step": mean_absolute_error(best_labels_last, best_preds_last),
        "train_maes": train_maes,
        "val_maes": val_maes,
        "learning_rates": learning_rates,
        "best_preds_all": best_preds_all,
        "best_labels_all": best_labels_all,
        "best_preds_last": best_preds_last,
        "best_labels_last": best_labels_last,
    }