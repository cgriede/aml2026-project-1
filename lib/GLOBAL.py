from pathlib import Path

KAGGLE_SUBMIT = True
KAGGLE_COMPETITION = "eth-aml-2026-project-1"
NOTE = "shallower xgb depth 2 min_child 20"
# Global constants code
RANDOM_STATE = 42
XGB_SEEDS = (42, 43, 44)
DATA_DIR = Path("data")
CLEAN_ABSMAX = 1e10
DROP_INSANE_ABSMAX = False

METHODS_OUTLIERS = [ 
    #"svm",
    #"isoforest",
]
# Global constants data
VAL_RATIO = 0.2
CV_FOLDS = 5

# High correlation feature selection
FS_CORR_MAX = 0.95

# After the corr cull: "monte_carlo", "mi", or "corr"
from typing import Literal
FS_METHOD: Literal["monte_carlo", "mi", "corr", "rrelieff", "cluster", "mrmr"] = "rrelieff"
FS_K = 250

# Monte Carlo feature selection
FS_N_ROUNDS = 10
FS_TRACK = 10
FS_N_TRIALS = 40
FS_SUBSET_SIZE = 250
FS_TOP_FRAC = 0.2
FS_SCORE_SPLIT = 0.2

# Training. The full-fit tree count is the CV median best_iteration + 1, capped by XGB_N_ESTIMATORS.
RIDGE_ALPHA = 1

# XGBoost.
XGB_N_ESTIMATORS = 2000
XGB_LEARNING_RATE = 0.005
XGB_MAX_DEPTH = 5
XGB_SUBSAMPLE = 0.9
XGB_COLSAMPLE_BYTREE = 0.3
XGB_MIN_CHILD_WEIGHT = 5
XGB_EARLY_STOPPING_ROUNDS = 50