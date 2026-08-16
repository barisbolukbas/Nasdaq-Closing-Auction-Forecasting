# Nasdaq Closing Auction Forecasting

Short-term price prediction during the Nasdaq closing auction, built on the [Optiver "Trading at the Close"](https://www.kaggle.com/competitions/optiver-trading-at-the-close) Kaggle problem.

The task: predict a stock's 60-second future WAP (weighted average price) movement **relative to a synthetic market index**, using order-book, auction, and market-wide features from the ten-minute closing-auction window. Full writeup, math, and results are in **[`Paper.pdf`](./Paper.pdf)**.

## Summary

Each stock-day is treated as a 55-step time sequence (10-second buckets over the 540-second auction window). Three models are compared:

1. **Markov state model** — a 5-state baseline built from stock-minus-index return transitions.
2. **LightGBM** — gradient-boosted trees, each timestep treated as an independent row.
3. **LSTM + Transformer** — a 2-layer LSTM feeding a causal Transformer encoder with a local-and-global attention mask (last 15 timesteps + global positions every 20 seconds), followed by a residual regression head (linear prediction + learned correction).

The working hypothesis was that keeping the full auction sequence would let the neural model outperform the row-based baselines. That hypothesis wasn't confirmed:

| Model | Category | Validation MAE (final step) | Test MAE (final step) | Validation MAE (all steps) | Test MAE (all steps) |
|---|---|---|---|---|---|
| **LightGBM** | Baseline | **5.4536** | **5.0960** | 5.8734 | 5.4215 |
| LSTM + Transformer | Neural | 5.8026 | 5.2561 | 5.8884 | 5.4212 |
| Markov state model | Baseline | 6.0431 | 5.3838 | 5.9647 | 5.4678 |

*(MAE in basis points, lower is better; validation = last 81 trading days held out by date.)*

LightGBM won on the final-step metric outright and was statistically tied with the LSTM–Transformer on the all-step metric. The takeaway: the engineered features (rolling windows, EWMA, lagged returns, realized/GARCH volatility, RMT/PCA/HMM factors) already encode most of the temporal signal a sequence model would otherwise have to learn implicitly — so careful feature engineering rivaled architectural complexity here. See §11 of the paper for the full discussion.

> **Note:** the competition's `test.csv` overlaps the tail of `train.csv`, so "Test MAE" above is a replay/competition check, not an independent out-of-sample score. The validation split (last 81 dates, held out by `date_id`) is the meaningful out-of-sample number.

## Repository layout

```
config.py           Central constants: paths, windowing, feature/model hyperparameters
dataloading.py       Loads train.csv and groups rows into fixed-width time intervals
dataset.py           PyTorch Dataset that turns per-stock-day rows into (55, n_features) sequences
features.py          Order-book, auction, microprice, rolling/EWMA, volatility, outlier,
                      cross-sectional, and stock-level aggregate feature engineering
factor_models.py     Synthetic index reconstruction (OOF weights), Random Matrix Theory
                      factors, and the soft (HMM-style) market-state filter
markov_and_pca.py    5-state Markov transition model and PCA factor features
models.py            LSTM -> causal local+global Transformer -> residual regression head
train.py             Training loops (neural + LightGBM), feature selection, result logging
evaluation.py        Validation/competition-test evaluation, plotting, prediction reports
main.py              End-to-end pipeline entry point
```

`data_loading.py` is an empty stub left over from an earlier pass — `dataloading.py` is the one actually used.

## Setup

```bash
pip install numpy pandas scikit-learn lightgbm torch matplotlib
```

Download the [Optiver "Trading at the Close"](https://www.kaggle.com/competitions/optiver-trading-at-the-close) dataset from Kaggle and place `train.csv`, `test.csv`, and `revealed_targets.csv` in the repo root (paths are configurable in `config.py`).

## Usage

```bash
python main.py
```

This runs the full pipeline: feature engineering → train/validation split by date → fits the synthetic index, RMT, HMM, PCA, and Markov models → LightGBM feature selection → trains LightGBM and the LSTM–Transformer → evaluates both on validation and the competition test rows → prints a final MAE comparison table and produces the loss-curve / prediction-error plots.

Key knobs live in `config.py`: `INTERVAL_SECONDS`, `SEQ_LEN`, `VALIDATION_DAYS`, `EPOCHS`, `BATCH_SIZE`, `PEAK_LR`, attention window sizes, and feature-engineering constants (rolling windows, EWMA spans, GARCH parameters, PCA/RMT/HMM/Markov state counts).

## References

See §12 of [`Paper.pdf`](./Paper.pdf) for the full bibliography (Optiver competition, LightGBM, LSTM, Transformer/"Attention Is All You Need", GARCH/ARCH, Marchenko–Pastur, PCA, HMM).
