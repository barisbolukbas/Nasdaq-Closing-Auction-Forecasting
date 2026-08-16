#Data paths
TRAIN_CSV_PATH = "train.csv"
TEST_CSV_PATH = "test.csv"
REVEALED_TARGETS_PATH = "revealed_targets.csv"


#Aggregation / windowing
INTERVAL_SECONDS = 10
WINDOW_END_SECONDS = 540  #seconds_in_bucket runs 0..540
SEQ_LEN = WINDOW_END_SECONDS // INTERVAL_SECONDS + 1


#Dataset size
FULL_DATASET_STOCKS = 200
FULL_DATASET_DATES = 481


#General feature-engineering constants
PRICE_COLS = ["reference_price", "bid_price", "ask_price", "wap", "mid_price"]
EPS = 1e-8

ROLLING_WINDOWS = [3, 5, 10]
WAP_RETURN_EXTRA_HORIZONS = [2, 3, 5]

EWMA_SPANS = {"fast": 3, "slow": 10}

SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2

GARCH_OMEGA = 1e-8
GARCH_ALPHA = 0.1
GARCH_BETA = 0.85

REALIZED_VOL_WINDOW = 6

HAMPEL_WINDOW = 7
HAMPEL_K = 3.0

STOCK_AGG_SOURCE_COLS = [
    "wap", "bid_ask_spread", "matched_size", "imbalance_ratio",
    "imbalance_size", "bid_size", "ask_size",
]


#Factor models
RMT_N_FACTORS = 3

HMM_N_STATES = 3
HMM_INDICATOR_COL = "stock_minus_index_return"


#Markov model
N_MARKOV_STATES = 5
MARKOV_INDICATOR_COL = "stock_minus_index_return"


#PCA
PCA_SOURCE_COLS = [
    "imbalance_ratio", "book_imbalance", "bid_ask_spread",
    "size_ratio", "matched_imbalance_ratio",
]
PCA_N_COMPONENTS = 3


#Feature selection
MIN_FEATURE_IMPORTANCE_FRAC = 0.004


#Training / model hyperparameters
EPOCHS = 40
BATCH_SIZE = 64
PEAK_LR = 0.0001
WEIGHT_DECAY = 1e-4
PATIENCE = 15


#Validation
VALIDATION_DAYS = 81


#Index reconstruction
INDEX_OOF_FOLDS = 5


#Transformer attention
LOCAL_ATTENTION_WINDOW = 15
GLOBAL_ATTENTION_INTERVAL_SECONDS = 20
GLOBAL_ATTENTION_STRIDE = max(
    1,
    GLOBAL_ATTENTION_INTERVAL_SECONDS // INTERVAL_SECONDS,
)