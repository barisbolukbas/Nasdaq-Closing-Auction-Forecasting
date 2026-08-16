import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from config import (
    MARKOV_INDICATOR_COL, N_MARKOV_STATES,
    PCA_N_COMPONENTS, PCA_SOURCE_COLS,
)

_SORT_KEYS = ["stock_id", "date_id", "seconds_in_bucket"]


# Markov model
def fit_markov_model(train_df, n_states=N_MARKOV_STATES, indicator_col=MARKOV_INDICATOR_COL):
    state_edges = np.quantile(train_df[indicator_col], np.linspace(0, 1, n_states + 1))
    state_edges[0] -= 1e-6
    state_edges[-1] += 1e-6

    def assign_state(df):
        return np.digitize(df[indicator_col].values, state_edges[1:-1])

    train_df = train_df.sort_values(_SORT_KEYS).copy()
    train_df["state"] = assign_state(train_df)
    train_df["prev_state"] = train_df.groupby(["stock_id", "date_id"])["state"].shift(1)

    trans_counts = pd.crosstab(train_df["prev_state"], train_df["state"])
    trans_counts.index = trans_counts.index.astype(int)
    trans_counts = trans_counts.reindex(index=range(n_states), columns=range(n_states), fill_value=0)

    row_sums = trans_counts.sum(axis=1)
    transition_matrix = trans_counts.div(row_sums.replace(0, 1), axis=0).values
    unseen_rows = (row_sums == 0).values
    if unseen_rows.any():
        transition_matrix[unseen_rows] = 1.0 / n_states

    state_target_mean = train_df.groupby("state")["target"].mean().reindex(range(n_states)).fillna(train_df["target"].mean()).values
    global_target_mean = float(train_df["target"].mean())

    return {
        "n_states": n_states, "indicator_col": indicator_col, "state_edges": state_edges,
        "transition_matrix": transition_matrix, "state_target_mean": state_target_mean,
        "global_target_mean": global_target_mean,
    }


def assign_markov_state(df, markov_state):
    edges, indicator_col = markov_state["state_edges"], markov_state["indicator_col"]
    return np.digitize(df[indicator_col].values, edges[1:-1])


def markov_expected_target(prev_states, markov_state):
    transition_matrix = markov_state["transition_matrix"]
    state_target_mean = markov_state["state_target_mean"]
    global_target_mean = markov_state["global_target_mean"]

    out = np.full(len(prev_states), global_target_mean, dtype=np.float32)
    known = prev_states.notna()
    idx = prev_states[known].astype(int).values
    out[known.values] = transition_matrix[idx] @ state_target_mean
    return out


def add_markov_features(df, markov_state):
    df = df.sort_values(_SORT_KEYS).copy()
    df["state"] = assign_markov_state(df, markov_state)
    df["prev_state"] = df.groupby(["stock_id", "date_id"])["state"].shift(1)
    df["markov_expected_target"] = markov_expected_target(df["prev_state"], markov_state)
    return df


# PCA
def fit_pca_model(train_df, source_cols=PCA_SOURCE_COLS, n_components=PCA_N_COMPONENTS):
    pca_model = PCA(n_components=n_components).fit(train_df[source_cols].fillna(0))
    return {"model": pca_model, "source_cols": source_cols, "n_components": n_components}


def add_pca_features(df, pca_state):
    df = df.copy()
    pca_model, source_cols, n_components = pca_state["model"], pca_state["source_cols"], pca_state["n_components"]
    comps = pca_model.transform(df[source_cols].fillna(0))

    names = []
    for k in range(n_components):
        col = f"pca_component_{k + 1}"
        df[col] = comps[:, k]
        names.append(col)

    return df, names