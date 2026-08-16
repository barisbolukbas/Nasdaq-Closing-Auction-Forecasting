import numpy as np
import pandas as pd

from config import INTERVAL_SECONDS, TRAIN_CSV_PATH


#Data loading
def load_train_data(path=TRAIN_CSV_PATH):
    train = pd.read_csv(path)
    train = train.dropna(subset=["target"])

    float_cols = train.select_dtypes(include=["float64"]).columns
    train[float_cols] = train[float_cols].astype(np.float32)

    return train


def group_time_intervals(df, interval_seconds=INTERVAL_SECONDS):
    df = df.copy()
    df["time_group"] = (df["seconds_in_bucket"] // interval_seconds) * interval_seconds

    agg = {
        "seconds_in_bucket": "last",
        "imbalance_size": "mean",
        "imbalance_buy_sell_flag": "last",
        "reference_price": "last",
        "matched_size": "mean",
        "bid_price": "last",
        "bid_size": "mean",
        "ask_price": "last",
        "ask_size": "mean",
        "wap": "last",
    }

    if "far_price" in df.columns:
        agg["far_price"] = "last"

    if "near_price" in df.columns:
        agg["near_price"] = "last"

    if "target" in df.columns:
        agg["target"] = "last"

    return df.groupby(
        ["stock_id", "date_id", "time_group"],
        as_index=False,
    ).agg(agg)