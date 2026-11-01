from pathlib import Path

# Global constants code
RANDOM_STATE = 42
DATA_DIR = Path("data")

# Global constants data
VAL_RATIO = 0.2

# High correlation feature selection
FS_CORR_MAX = 0.95

# Monte Carlo feature selection
FS_N_ROUNDS = 10
FS_TRACK = 10
FS_N_TRIALS = 40
FS_SUBSET_SIZE = 100
FS_TOP_FRAC = 0.2
FS_SCORE_SPLIT = 0.2