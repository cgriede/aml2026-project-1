from pathlib import Path

KAGGLE_SUBMIT = False
KAGGLE_COMPETITION = "eth-aml-2026-project-1"
NOTE = "I4 keep: 3-seed mean B-XGB"
# Global constants code
RANDOM_STATE = 42
XGB_SEEDS = (42, 43, 44)
DATA_DIR = Path("data")
CLEAN_ABSMAX = 1e10
DROP_OUTLIERS = False

# Global constants data
VAL_RATIO = 0.2
CV_FOLDS = 5

# High correlation feature selection
FS_CORR_MAX = 0.95

# After the corr cull: "monte_carlo", "mi", or "corr"
FS_METHOD = "corr"
FS_K = 150

# Monte Carlo feature selection
FS_N_ROUNDS = 10
FS_TRACK = 10
FS_N_TRIALS = 40
FS_SUBSET_SIZE = 250
FS_TOP_FRAC = 0.2
FS_SCORE_SPLIT = 0.2