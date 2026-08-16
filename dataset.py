import numpy as np
import torch
from torch.utils.data import Dataset

from config import SEQ_LEN

_KEYS = ["stock_id", "date_id"]


def _valid_subset(df):
    sizes = df.groupby(_KEYS).size()
    valid_keys = sizes[sizes == SEQ_LEN].index
    row_idx = df.set_index(_KEYS).index
    return df[row_idx.isin(valid_keys)].sort_values(_KEYS + ["seconds_in_bucket"])


def get_last_timestep_frame(df):
    return _valid_subset(df).groupby(_KEYS, sort=False).tail(1)


def get_valid_sequence_columns(df, columns):
    valid = _valid_subset(df)
    return {col: valid[col].to_numpy().reshape(-1, SEQ_LEN) for col in columns}


class StockDataset(Dataset):
    def __init__(self, df, features, verbose=True):
        grouped = df.groupby(_KEYS, sort=False)
        all_groups = list(grouped)
        valid_groups = [(key, g) for key, g in all_groups if len(g) == SEQ_LEN]

        n_total, n_valid = len(all_groups), len(valid_groups)
        n_dropped = n_total - n_valid
        if verbose:
            print(
                f"  StockDataset: kept {n_valid:,}/{n_total:,} stock-date groups "
                f"({n_dropped:,} dropped for not having exactly {SEQ_LEN} rows, "
                f"{100 * n_dropped / max(n_total, 1):.1f}%)"
            )

        n_samples, n_features = n_valid, len(features)
        self.X = np.empty((n_samples, SEQ_LEN, n_features), dtype=np.float32)
        self.y = np.empty((n_samples, SEQ_LEN), dtype=np.float32)
        self.y_last = np.empty(n_samples, dtype=np.float32)
        self.date_ids = np.empty(n_samples, dtype=np.int64)

        for i, (key, g) in enumerate(valid_groups):
            g = g.sort_values("seconds_in_bucket")
            self.X[i] = g[features].values.astype(np.float32)
            self.y[i] = g["target"].values.astype(np.float32)
            self.y_last[i] = g["target"].iloc[-1]
            self.date_ids[i] = key[1]

        del valid_groups, all_groups
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)
        self.y_last = torch.from_numpy(self.y_last)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ArrayDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]