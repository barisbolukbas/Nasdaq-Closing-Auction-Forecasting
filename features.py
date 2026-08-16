from itertools import combinations
import numpy as np

from config import (
    EPS, EWMA_SPANS, GARCH_ALPHA, GARCH_BETA, GARCH_OMEGA,
    HAMPEL_K, HAMPEL_WINDOW, INTERVAL_SECONDS, PRICE_COLS,
    REALIZED_VOL_WINDOW, ROLLING_WINDOWS, SAVGOL_POLYORDER,
    SAVGOL_WINDOW, STOCK_AGG_SOURCE_COLS, WAP_RETURN_EXTRA_HORIZONS,
    WINDOW_END_SECONDS,
)

_SORT_KEYS = ["stock_id", "date_id", "seconds_in_bucket"]


def _sorted(df):
    return df.sort_values(_SORT_KEYS).copy()


# Basic features
def add_basic_features(df):
    df = df.copy()
    df["bid_ask_spread"] = df["ask_price"] - df["bid_price"]
    df["mid_price"] = (df["ask_price"] + df["bid_price"]) / 2
    df["imbalance_ratio"] = df["imbalance_size"] / (df["matched_size"] + 1)
    df["wap_reference_diff"] = df["wap"] - df["reference_price"]
    df["wap_mid_diff"] = df["wap"] - df["mid_price"]
    df["size_ratio"] = df["bid_size"] / (df["ask_size"] + 1)
    df["matched_imbalance_ratio"] = df["matched_size"] / (df["matched_size"] + df["imbalance_size"] + EPS)
    df["book_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"] + EPS)
    df["signed_imbalance_size"] = df["imbalance_size"] * df["imbalance_buy_sell_flag"]
    df["signed_imbalance_ratio"] = df["imbalance_ratio"] * df["imbalance_buy_sell_flag"]
    df["microprice"] = (df["bid_price"] * df["ask_size"] + df["ask_price"] * df["bid_size"]) / (df["bid_size"] + df["ask_size"] + EPS)
    df["microprice_mid_diff"] = df["microprice"] - df["mid_price"]

    names = [
        "bid_ask_spread", "mid_price", "imbalance_ratio", "wap_reference_diff",
        "wap_mid_diff", "size_ratio", "matched_imbalance_ratio", "book_imbalance",
        "signed_imbalance_size", "signed_imbalance_ratio", "microprice", "microprice_mid_diff",
    ]

    if "far_price" in df.columns and "near_price" in df.columns:
        df["far_near_diff"] = df["far_price"] - df["near_price"]
        df["far_ref_diff"] = df["far_price"] - df["reference_price"]
        df["near_ref_diff"] = df["near_price"] - df["reference_price"]
        df["far_wap_diff"] = df["far_price"] - df["wap"]
        df["near_wap_diff"] = df["near_price"] - df["wap"]
        names += ["far_near_diff", "far_ref_diff", "near_ref_diff", "far_wap_diff", "near_wap_diff"]

    df["time_to_close"] = WINDOW_END_SECONDS - df["seconds_in_bucket"]
    df["time_to_close_frac"] = df["time_to_close"] / WINDOW_END_SECONDS
    df["time_to_close_x_imbalance_ratio"] = df["time_to_close_frac"] * df["imbalance_ratio"]
    df["time_to_close_x_book_imbalance"] = df["time_to_close_frac"] * df["book_imbalance"]
    df["time_to_close_x_spread"] = df["time_to_close_frac"] * df["bid_ask_spread"]
    names += [
        "time_to_close", "time_to_close_frac", "time_to_close_x_imbalance_ratio",
        "time_to_close_x_book_imbalance", "time_to_close_x_spread",
    ]
    return df, names


# Auction price features
def add_auction_price_features(df):
    df = _sorted(df)
    df["auction_progress"] = df["seconds_in_bucket"] / WINDOW_END_SECONDS
    df["time_left_frac"] = (WINDOW_END_SECONDS - df["seconds_in_bucket"]) / WINDOW_END_SECONDS
    names = ["auction_progress", "time_left_frac"]

    if "far_price" not in df.columns or "near_price" not in df.columns:
        return df, names

    df["far_available"] = df["far_price"].notna().astype(np.float32)
    df["near_available"] = df["near_price"].notna().astype(np.float32)
    names += ["far_available", "near_available"]

    df["far_wap_rel"] = (df["far_price"] - df["wap"]) / (df["wap"].abs() + EPS)
    df["near_wap_rel"] = (df["near_price"] - df["wap"]) / (df["wap"].abs() + EPS)
    df["far_ref_rel"] = (df["far_price"] - df["reference_price"]) / (df["reference_price"].abs() + EPS)
    df["near_ref_rel"] = (df["near_price"] - df["reference_price"]) / (df["reference_price"].abs() + EPS)
    df["far_near_rel"] = (df["far_price"] - df["near_price"]) / (df["near_price"].abs() + EPS)
    names += ["far_wap_rel", "near_wap_rel", "far_ref_rel", "near_ref_rel", "far_near_rel"]

    df["far_near_abs_gap"] = (df["far_price"] - df["near_price"]).abs()
    df["near_wap_abs_gap"] = (df["near_price"] - df["wap"]).abs()
    df["far_wap_abs_gap"] = (df["far_price"] - df["wap"]).abs()
    names += ["far_near_abs_gap", "near_wap_abs_gap", "far_wap_abs_gap"]

    grp = df.groupby(["stock_id", "date_id"], sort=False)
    for col in ["far_price", "near_price"]:
        prev = grp[col].shift(1)
        df[f"{col}_change"] = df[col] - prev
        denom = prev.where(prev.abs() >= 1e-6, np.nan)
        df[f"{col}_return"] = df[col] / denom - 1.0
        names += [f"{col}_change", f"{col}_return"]

    df["far_near_gap_change"] = grp["far_near_abs_gap"].diff()
    df["far_near_convergence_rate"] = -grp["far_near_abs_gap"].diff() / INTERVAL_SECONDS
    df["near_wap_convergence_rate"] = -grp["near_wap_abs_gap"].diff() / INTERVAL_SECONDS
    names += ["far_near_gap_change", "far_near_convergence_rate", "near_wap_convergence_rate"]

    df["near_wap_x_progress"] = df["near_wap_rel"] * df["auction_progress"]
    df["far_wap_x_progress"] = df["far_wap_rel"] * df["auction_progress"]
    df["far_near_x_progress"] = df["far_near_rel"] * df["auction_progress"]
    names += ["near_wap_x_progress", "far_wap_x_progress", "far_near_x_progress"]

    far_direction = np.sign(df["far_price"] - df["reference_price"])
    near_direction = np.sign(df["near_price"] - df["reference_price"])
    both_available = (df["far_available"] > 0) & (df["near_available"] > 0)
    df["far_near_same_direction"] = (both_available & (far_direction == near_direction)).astype(np.float32)
    df["near_above_wap"] = ((df["near_available"] > 0) & (df["near_price"] > df["wap"])).astype(np.float32)
    df["far_above_wap"] = ((df["far_available"] > 0) & (df["far_price"] > df["wap"])).astype(np.float32)
    names += ["far_near_same_direction", "near_above_wap", "far_above_wap"]

    df["near_wap_x_signed_imbalance"] = df["near_wap_rel"] * df["signed_imbalance_ratio"]
    df["far_wap_x_signed_imbalance"] = df["far_wap_rel"] * df["signed_imbalance_ratio"]
    df["near_wap_x_book_imbalance"] = df["near_wap_rel"] * df["book_imbalance"]
    df["auction_pressure"] = df["near_wap_rel"] * df["signed_imbalance_ratio"] * df["auction_progress"]
    names += ["near_wap_x_signed_imbalance", "far_wap_x_signed_imbalance", "near_wap_x_book_imbalance", "auction_pressure"]
    return df, names


# Price combinations
def add_pairwise_triplet_features(df):
    df = df.copy()
    pairwise_names, triplet_names = [], []

    for a, b in combinations(PRICE_COLS, 2):
        df[f"{a}_{b}_diff"] = df[a] - df[b]
        denom = df[b].where(df[b].abs() >= 1e-6, np.nan)
        df[f"{a}_{b}_ratio"] = (df[a] / denom).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-100, 100)
        pairwise_names += [f"{a}_{b}_diff", f"{a}_{b}_ratio"]

    for a, b, c in combinations(PRICE_COLS, 3):
        trio = df[[a, b, c]]
        max_, min_ = trio.max(axis=1), trio.min(axis=1)
        mid_ = trio.sum(axis=1) - max_ - min_
        denom = (mid_ - min_).clip(lower=1e-6)
        df[f"{a}_{b}_{c}_imb"] = ((max_ - mid_) / denom).clip(0, 100)
        triplet_names.append(f"{a}_{b}_{c}_imb")

    return df, pairwise_names, triplet_names


# Sequence features
def add_sequence_features(df):
    df = _sorted(df)
    grp = df.groupby(["stock_id", "date_id"])

    df["wap_lag1"] = grp["wap"].shift(1)
    df["wap_return"] = df["wap"] / (df["wap_lag1"].replace(0, np.nan) + EPS) - 1

    horizon_names = []
    for h in WAP_RETURN_EXTRA_HORIZONS:
        df[f"wap_lag{h}"] = grp["wap"].shift(h)
        df[f"wap_return_h{h}"] = df["wap"] / (df[f"wap_lag{h}"].replace(0, np.nan) + EPS) - 1
        horizon_names.append(f"wap_return_h{h}")

    roll_names = []
    for w in ROLLING_WINDOWS:
        df[f"imbalance_roll{w}"] = grp["imbalance_ratio"].transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
        df[f"spread_roll{w}"] = grp["bid_ask_spread"].transform(lambda x, w=w: x.rolling(w, min_periods=1).mean())
        df[f"wap_roll_std{w}"] = grp["wap"].transform(lambda x, w=w: x.rolling(w, min_periods=1).std())
        roll_names += [f"imbalance_roll{w}", f"spread_roll{w}", f"wap_roll_std{w}"]

    return df.fillna(0), roll_names + horizon_names


# EWMA features
def add_ewma_features(df):
    df = _sorted(df)
    grp = df.groupby(["stock_id", "date_id"])
    names = []
    for col in ["wap", "imbalance_ratio", "bid_ask_spread"]:
        for span_name, span in EWMA_SPANS.items():
            name = f"ewma_{col}_{span_name}"
            df[name] = grp[col].transform(lambda x, span=span: x.ewm(span=span, adjust=False).mean())
            names.append(name)
    return df, names


# Causal smoothing
def _causal_savgol_last(values):
    n = len(values)
    if n < 3:
        return values[-1]
    order = min(SAVGOL_POLYORDER, n - 1)
    x = np.arange(n)
    return np.polyval(np.polyfit(x, values, order), x[-1])


def add_causal_savgol_features(df):
    df = _sorted(df)
    grp = df.groupby(["stock_id", "date_id"])
    df["sg_wap"] = grp["wap"].transform(
        lambda x: x.rolling(SAVGOL_WINDOW, min_periods=1).apply(_causal_savgol_last, raw=True)
    )
    df["sg_wap_gap"] = df["wap"] - df["sg_wap"]
    return df, ["sg_wap", "sg_wap_gap"]


# Volatility
def _garch_recursion(returns):
    n = len(returns)
    var = np.zeros(n)
    if n == 0:
        return var
    var[0] = returns[0] ** 2 + GARCH_OMEGA
    for t in range(1, n):
        var[t] = GARCH_OMEGA + GARCH_ALPHA * returns[t - 1] ** 2 + GARCH_BETA * var[t - 1]
    return var


def add_volatility_features(df):
    df = _sorted(df)
    grp = df.groupby(["stock_id", "date_id"])

    df["ewma_volatility"] = grp["wap_return"].transform(lambda x: (x ** 2).ewm(span=10, adjust=False).mean()) ** 0.5

    garch_vol = np.zeros(len(df))
    idx = 0
    for _, g in df.groupby(["stock_id", "date_id"], sort=False):
        r = g["wap_return"].to_numpy()
        garch_vol[idx:idx + len(g)] = np.sqrt(_garch_recursion(r))
        idx += len(g)
    df["garch_volatility"] = garch_vol

    fast = grp["wap_return"].transform(lambda x: (x ** 2).ewm(span=3, adjust=False).mean()) ** 0.5
    slow = grp["wap_return"].transform(lambda x: (x ** 2).ewm(span=20, adjust=False).mean()) ** 0.5
    df["vol_regime_ratio"] = fast / (slow + EPS)

    return df, ["ewma_volatility", "garch_volatility", "vol_regime_ratio"]


def add_realized_vol_features(df):
    df = _sorted(df)
    grp = df.groupby(["stock_id", "date_id"])

    df["realized_vol"] = grp["wap_return"].transform(lambda x: (x ** 2).rolling(REALIZED_VOL_WINDOW, min_periods=1).sum())
    abs_ret_lag1 = grp["wap_return"].transform(lambda x: x.abs().shift(1)).fillna(0)
    bipower_term = df["wap_return"].abs() * abs_ret_lag1

    df["bipower_variation"] = bipower_term.groupby([df["stock_id"], df["date_id"]]).transform(
        lambda x: x.rolling(REALIZED_VOL_WINDOW, min_periods=1).sum()
    ) * (np.pi / 2)

    df["jump_component"] = (df["realized_vol"] - df["bipower_variation"]).clip(lower=0)
    return df, ["realized_vol", "bipower_variation", "jump_component"]


# Outlier feature
def add_hampel_feature(df):
    df = _sorted(df)
    grp = df.groupby(["stock_id", "date_id"])
    rolling_median = grp["wap"].transform(lambda x: x.rolling(HAMPEL_WINDOW, min_periods=1).median())
    rolling_mad = grp["wap"].transform(
        lambda x: (x - x.rolling(HAMPEL_WINDOW, min_periods=1).median()).abs().rolling(HAMPEL_WINDOW, min_periods=1).median()
    )
    df["is_wap_outlier"] = ((df["wap"] - rolling_median).abs() > HAMPEL_K * 1.4826 * rolling_mad + EPS).astype(np.float32)
    return df, ["is_wap_outlier"]


# Cross-sectional features
def add_cross_sectional_features(df):
    df = df.copy()
    group_cols = ["date_id", "time_group"]
    names = []
    for col in ["wap_return", "imbalance_ratio", "book_imbalance"]:
        grp = df.groupby(group_cols)[col]
        df[f"cross_sect_rank_{col}"] = grp.rank(pct=True)
        df[f"cross_sect_zscore_{col}"] = (df[col] - grp.transform("mean")) / (grp.transform("std") + EPS)
        names += [f"cross_sect_rank_{col}", f"cross_sect_zscore_{col}"]
    return df, names


# Stock statistics
def get_stock_agg_feature_names():
    return [f"stock_{col}_{stat}" for col in STOCK_AGG_SOURCE_COLS for stat in ["mean", "std"]]


def compute_stock_stats(train_df):
    stock_stats = train_df.groupby("stock_id")[STOCK_AGG_SOURCE_COLS].agg(["mean", "std"])
    stock_stats.columns = [f"stock_{col}_{stat}" for col, stat in stock_stats.columns]
    return stock_stats.reset_index()