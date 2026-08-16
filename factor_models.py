import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import (
    EPS, HMM_INDICATOR_COL, HMM_N_STATES,
    INDEX_OOF_FOLDS, RMT_N_FACTORS,
)

_SORT_KEYS = ["stock_id", "date_id", "seconds_in_bucket"]


# Synthetic index
def fit_index_weights(train_df):
    fit_df = train_df.copy()

    future = fit_df[["stock_id", "date_id", "seconds_in_bucket", "wap"]].copy()
    future["seconds_in_bucket"] -= 60
    future = future.rename(columns={"wap": "_wap_plus_60"})

    fit_df = fit_df.merge(future, on=["stock_id", "date_id", "seconds_in_bucket"], how="left", validate="one_to_one")
    fit_df["_stock_future_return_60"] = fit_df["_wap_plus_60"] / (fit_df["wap"] + EPS) - 1.0
    fit_df["_implied_index_return_60"] = fit_df["_stock_future_return_60"] - fit_df["target"] / 10000.0
    fit_df = fit_df.dropna(subset=["_stock_future_return_60", "_implied_index_return_60"])

    return_pivot = fit_df.pivot_table(
        index=["date_id", "time_group"], columns="stock_id", values="_stock_future_return_60"
    ).fillna(0.0)
    implied_index_series = fit_df.pivot_table(
        index=["date_id", "time_group"], columns="stock_id", values="_implied_index_return_60"
    ).mean(axis=1)

    common_idx = return_pivot.index.intersection(implied_index_series.index)
    X_weights, y_weights = return_pivot.loc[common_idx].values, implied_index_series.loc[common_idx].values

    weight_reg = LinearRegression(fit_intercept=False).fit(X_weights, y_weights)
    return pd.Series(weight_reg.coef_, index=return_pivot.columns, name="index_weight")


def add_index_return_features(df, weights):
    df = df.copy()
    w = df["stock_id"].map(weights).fillna(0.0)
    contrib = df["wap_return"] * w
    df["index_return"] = contrib.groupby([df["date_id"], df["time_group"]]).transform("sum")
    df["stock_minus_index_return"] = df["wap_return"] - df["index_return"]
    return df


def add_oof_index_return_features(train_df, n_folds=INDEX_OOF_FOLDS):
    df = train_df.copy()
    dates = np.array(sorted(df["date_id"].unique()))
    if len(dates) == 0:
        return add_index_return_features(df, pd.Series(dtype=float))

    blocks = [b for b in np.array_split(dates, min(n_folds, len(dates))) if len(b)]
    pieces = []

    stock_ids = np.array(sorted(df["stock_id"].unique()))
    equal_weights = pd.Series(np.full(len(stock_ids), 1.0 / max(len(stock_ids), 1)), index=stock_ids, name="index_weight")

    for i, block_dates in enumerate(blocks):
        block = df[df["date_id"].isin(block_dates)].copy()
        if i == 0:
            weights = equal_weights
        else:
            past_dates = np.concatenate(blocks[:i])
            weights = fit_index_weights(df[df["date_id"].isin(past_dates)])
        pieces.append(add_index_return_features(block, weights))

    return pd.concat(pieces, axis=0).sort_index()


# RMT factors
def fit_rmt_factors(train_df, n_factors=RMT_N_FACTORS):
    return_pivot = train_df.pivot_table(index=["date_id", "time_group"], columns="stock_id", values="wap_return").fillna(0.0)
    stock_ids = return_pivot.columns.to_numpy()

    mean_ = return_pivot.mean(axis=0).values
    std_ = return_pivot.std(axis=0).replace(0, 1.0).values
    standardized = (return_pivot - return_pivot.mean(axis=0)) / return_pivot.std(axis=0).replace(0, 1.0)
    corr_matrix = np.corrcoef(standardized.values, rowvar=False)

    n_stocks, n_obs = corr_matrix.shape[0], len(standardized)
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    q_ratio = n_stocks / n_obs
    mp_upper = (1 + np.sqrt(q_ratio)) ** 2
    n_signal_factors = int(np.sum(eigvals > mp_upper))
    top_eigenvalue_frac = float(eigvals[0] / n_stocks)

    loadings = eigvecs[:, :n_factors]
    explained_variance_frac = np.sum(loadings ** 2 * eigvals[:n_factors][None, :], axis=1)
    idiosyncratic_frac = np.clip(1.0 - explained_variance_frac, 0.0, 1.0)

    stock_stats = pd.DataFrame({"stock_id": stock_ids})
    for k in range(n_factors):
        stock_stats[f"rmt_factor{k + 1}_loading"] = loadings[:, k]
    stock_stats["rmt_idiosyncratic_frac"] = idiosyncratic_frac
    stock_stats["rmt_wap_return_mean"] = mean_
    stock_stats["rmt_wap_return_std"] = std_

    return {
        "stock_stats": stock_stats, "n_factors": n_factors,
        "top_eigenvalue_frac": top_eigenvalue_frac, "n_signal_factors": n_signal_factors,
    }


def add_rmt_features(df, rmt_state):
    df = df.copy()
    stock_stats, n_factors = rmt_state["stock_stats"], rmt_state["n_factors"]
    df = df.merge(stock_stats, on="stock_id", how="left")

    loading_cols = [f"rmt_factor{k + 1}_loading" for k in range(n_factors)]
    fill_defaults = {c: 0.0 for c in loading_cols}
    fill_defaults.update(rmt_idiosyncratic_frac=1.0, rmt_wap_return_mean=0.0, rmt_wap_return_std=1.0)
    df = df.fillna(fill_defaults)

    standardized_return = (df["wap_return"] - df["rmt_wap_return_mean"]) / df["rmt_wap_return_std"]

    for k in range(n_factors):
        loading_col = f"rmt_factor{k + 1}_loading"
        factor_return_col = f"rmt_factor{k + 1}_return"
        residual_col = f"stock_minus_rmt_factor{k + 1}_return"

        contrib = standardized_return * df[loading_col]
        df[factor_return_col] = contrib.groupby([df["date_id"], df["time_group"]]).transform("sum")
        df[residual_col] = standardized_return - df[loading_col] * df[factor_return_col]

    df["rmt_top_eigenvalue_frac"] = rmt_state["top_eigenvalue_frac"]
    df["rmt_n_signal_factors"] = float(rmt_state["n_signal_factors"])
    return df


def get_rmt_feature_names(n_factors=RMT_N_FACTORS):
    names = []
    for k in range(n_factors):
        names += [f"rmt_factor{k + 1}_loading", f"rmt_factor{k + 1}_return", f"stock_minus_rmt_factor{k + 1}_return"]
    names += ["rmt_idiosyncratic_frac", "rmt_top_eigenvalue_frac", "rmt_n_signal_factors"]
    return names


# Soft state model
def _gaussian_likelihood(x, means, stds):
    return np.exp(-0.5 * ((x - means) / stds) ** 2) / (stds * np.sqrt(2 * np.pi))


def fit_hmm_filter(train_df, n_states=HMM_N_STATES, indicator_col=HMM_INDICATOR_COL):
    state_edges = np.quantile(train_df[indicator_col], np.linspace(0, 1, n_states + 1))
    state_edges[0] -= 1e-6
    state_edges[-1] += 1e-6

    def assign_state(df):
        return np.digitize(df[indicator_col].values, state_edges[1:-1])

    fit_df = train_df.copy()
    fit_df["hmm_hard_state"] = assign_state(fit_df)
    fit_df["hmm_prev_hard_state"] = fit_df.groupby(["stock_id", "date_id"])["hmm_hard_state"].shift(1)

    trans_counts = pd.crosstab(fit_df["hmm_prev_hard_state"], fit_df["hmm_hard_state"])
    trans_counts.index = trans_counts.index.astype(int)
    trans_counts = trans_counts.reindex(index=range(n_states), columns=range(n_states), fill_value=0)

    row_sums = trans_counts.sum(axis=1)
    transition_matrix = trans_counts.div(row_sums.replace(0, 1), axis=0).values
    unseen = (row_sums == 0).values
    if unseen.any():
        transition_matrix[unseen] = 1.0 / n_states

    state_mean = fit_df.groupby("hmm_hard_state")[indicator_col].mean().reindex(range(n_states)).values
    state_std = fit_df.groupby("hmm_hard_state")[indicator_col].std().reindex(range(n_states)).fillna(fit_df[indicator_col].std()).values
    state_std = np.where(state_std < 1e-6, 1e-6, state_std)

    state_target_mean = fit_df.groupby("hmm_hard_state")["target"].mean().reindex(range(n_states)).fillna(fit_df["target"].mean()).values

    return {
        "n_states": n_states, "indicator_col": indicator_col,
        "transition_matrix": transition_matrix, "state_mean": state_mean,
        "state_std": state_std, "state_target_mean": state_target_mean,
    }


def add_hmm_features(df, hmm_state):
    n_states = hmm_state["n_states"]
    indicator_col = hmm_state["indicator_col"]
    transition_matrix = hmm_state["transition_matrix"]
    state_mean, state_std = hmm_state["state_mean"], hmm_state["state_std"]
    state_target_mean = hmm_state["state_target_mean"]

    df = df.sort_values(_SORT_KEYS).copy()
    n = len(df)
    state_probs = np.full((n, n_states), 1.0 / n_states)
    idx = 0

    for _, g in df.groupby(["stock_id", "date_id"], sort=False):
        vals = g[indicator_col].to_numpy()
        T = len(vals)
        belief = np.full(n_states, 1.0 / n_states)

        for t in range(T):
            predict = belief @ transition_matrix
            state_probs[idx + t] = predict
            likelihood = _gaussian_likelihood(vals[t], state_mean, state_std)
            updated = predict * likelihood
            updated_sum = updated.sum()
            belief = updated / updated_sum if updated_sum > 1e-300 else predict

        idx += T

    names = []
    for k in range(n_states):
        col = f"hmm_state{k + 1}_prob"
        df[col] = state_probs[:, k]
        names.append(col)

    df["hmm_expected_target"] = state_probs @ state_target_mean
    names.append("hmm_expected_target")
    return df, names