---
experiment_id: 3
slug: model_selection
title: "Exp 3 — Model selection"
current_stage: closeout
overall_verdict: keep_bag
blocked_by: null
code_path: experiments/003/
stages:
  charter: { status: done, completed_utc: "2026-11-02T15:40:00Z", notes: "Broad model list plus two age ensembles. Plots required." }
  literature: { status: skipped, notes: "Operator sent this straight to a run." }
  design: { status: done, completed_utc: "2026-11-02T15:40:00Z" }
  implement: { status: done, completed_utc: "2026-11-02T15:42:00Z", notes: "run_models.py and plot_models.py. Smoke passed." }
  run: { status: done, completed_utc: "2026-11-02T15:43:14Z", arms: [same_features] }
  closeout: { status: done, completed_utc: "2026-11-02T15:45:00Z", notes: "Keep xgboost_bag. Age ensembles did not clear the bar." }
run_lock_holder: null
decision_ids: []
---

# Exp 3 — Model selection (`model_selection`)

**Code (when implemented):** `experiments/003/`  
**Prior:** exp 1 kept `|corr|`. Exp 2 kept k=150. The accepted production model is the 3-seed XGBoost bag (`lib/train/xgboost_bag.py`, iteration 4).

## Charter

### Question

Which model should the age pipeline keep when every model sees the same rows, the same columns, and the same folds?

The feature set is not a factor. `|corr|` top-150 is selected once per outer training fold, and that column list is passed to every model. Prep matches exp 2: drop the 7 defect columns once, then inside the fold median-fill, z-score, `|r| ≥ 0.95` cull, and `|corr|` top-150.

### What is already known

Exp 2 scored three models on this exact feature rule. Those numbers are the reference, not a new result:

| Model | R² at k=150 |
|-------|------------:|
| Ridge | 0.289 |
| XGBoost, one seed, early-stopped on the outer test fold | 0.543 |
| SVR | 0.424 |

The bag was not in that table. Random forest was not in that table. The one-seed XGBoost number above used the outer test labels for early stopping. Ridge, SVR, and random forest never see those labels. This experiment removes that difference.

### Arm `same_features`

Outer split: `KFold(n_splits=5, shuffle=True, random_state=42)`, the same splits as exp 1 and exp 2.

For each fold:

1. Build the 150 columns on the training rows only.
2. Fit every model on those training rows only. The test fold is used for R² and RMSE, not for fitting or early stopping.
3. XGBoost picks its tree count on a 20% slice of the outer training fold, then refits on the whole training fold. The bag does the same for seeds `(42, 43, 44)`. The test fold is score-only.

No per-model search. Ages in the training file run from 42 to 97, mean 70.

| Model | Frozen setting |
|-------|----------------|
| `ridge` | `alpha=1`, `solver=svd` |
| `elastic_net` | `ElasticNetCV`, `l1_ratio=0.5`, 3-fold, inside the training fold |
| `knn` | 25 neighbors, distance weights |
| `svr` | RBF, `C=1`, `epsilon=0.1` |
| `random_forest` | 100 trees, `random_state=42` |
| `extra_trees` | 200 trees, `min_samples_leaf=2` |
| `hist_gbr` | histogram gradient boosting, 300 iterations, `lr=0.05` |
| `mlp` | layers `(64, 32)`, early stopping on an inner slice |
| `xgboost` | current `lib/train/xgboost.py` settings, one seed |
| `xgboost_bag` | incumbent. Mean of the three seeds |
| `age_mixture` | three XGBoost specialists on age tertiles of the training fold, mixed by a logistic gate |
| `age_residual` | the one-seed XGBoost plus a ridge residual model per tertile, mixed by the same kind of gate |

The two age models are the experimental arm. Tertile edges come from the training fold only. The gate sees features, not the test age. Out-of-fold predictions are kept so a plot can compare each model's response to true age, and MAE inside target quintiles.

### Success

The incumbent is `xgboost_bag`. Another model replaces it only when its mean paired XGBoost-bag ΔR² is at least 0.01 and larger than the standard error of the five fold deltas. That is the same floor as exp 1 and exp 2.

Record R² and RMSE for every model. The keep rule uses R².

### Gates

- `lib/` stays read-only. The runner lives under `experiments/003/`.
- One column list per fold. Models do not select their own features.
- `data/X_test.csv` and Kaggle are not used to pick the winner.
- One training job at a time, using `experiments/.pipeline_run.lock`.

### Parked

| Later arm | Why it waits |
|-----------|----------------|
| Per-model hyperparameter search | A tuned forest against an untuned forest is a different question. |
| A second feature set | Exp 1 and exp 2 already fixed the columns. |

### Charter checklist

- [x] One question: which model to keep on a shared feature set
- [x] Arm `same_features` named. Incumbent is `xgboost_bag`. Floor is 0.01 paired R²
- [x] `blocked_by` is null. Exp 2's k=150 result is an input
- [x] README index row exists

## Run

```bash
uv run python experiments/003/run_models.py
uv run python experiments/003/plot_models.py
```

Finished 2026-11-02T15:43:14Z. Five outer folds, `|corr|` top-150, test fold used only for scoring. XGBoost tree count comes from an inner slice, then the model refits on the whole training fold. That is why the bag is 0.535 here and 0.543 in exp 2, where early stopping saw the test fold.

| Model | R² | RMSE | vs bag |
|-------|---:|-----:|--------|
| xgboost_bag | **0.535** | 6.61 | |
| xgboost | 0.529 | 6.65 | −0.006 |
| hist_gbr | 0.513 | 6.76 | −0.022 |
| extra_trees | 0.509 | 6.79 | −0.026 |
| age_residual | 0.498 | 6.86 | −0.036 |
| random_forest | 0.493 | 6.90 | −0.042 |
| svr | 0.424 | 7.36 | −0.110 |
| knn | 0.405 | 7.48 | −0.130 |
| elastic_net | 0.362 | 7.73 | −0.173 |
| age_mixture | 0.350 | 7.79 | −0.185 |
| ridge | 0.289 | 8.17 | −0.246 |
| mlp | −0.833 | 13.13 | −1.368 |

No challenger clears +0.01 against the bag.

MAE in years by true-age quintile. The best cell in each range is marked in the plot.

| Age | n | Best MAE | Bag MAE | Bag bias |
|-----|--:|----------|--------:|---------:|
| 42-62 | 256 | age residual 6.86 | 6.94 | +6.5 |
| 63-68 | 260 | elastic net 3.12 | 3.94 | +1.2 |
| 69-73 | 261 | elastic net 2.90 | 3.59 | −0.4 |
| 74-78 | 202 | extra trees 3.30 | 3.35 | −1.8 |
| 79-97 | 233 | xgboost bag 6.95 | 6.95 | −6.8 |

## Closeout

**Keep `xgboost_bag`.** Tree models occupy the top of the table. The linear models and the MLP do not.

The age plots show one shared shape. Predictions shrink toward about 70. People aged 42-62 are predicted about 6 years too old. People aged 79-97 are predicted about 7 years too young. The middle of the range, 63-78, sits within about 2-4 years. Elastic net is the most accurate in that crowded middle and the least accurate in both tails, which is why its R² is 0.36.

`age_mixture` (a specialist per tertile) is noisier than one XGBoost and scores 0.350. `age_residual` is the best model on the youngest quintile by 0.08 years of MAE and does not beat the bag overall.

`lib/` was not changed.

## Artifacts

| Path | Role |
|------|------|
| `experiments/003/run_models.py` | Runner |
| `experiments/003/plot_models.py` | Response and age-band plots |
| `experiments/003/results/models.json` | Scores and out-of-fold predictions |
| `experiments/003/results/response_vs_age.png` | Predicted age against true age |
| `experiments/003/results/mae_by_age.png` | MAE by age quintile |
| `docs/experiments/pipeline/02-corr-k-grid.md` | Feature set this experiment freezes |

## Pipeline next step

None. The bag stays the production model. A later arm would have to attack the tail shrinkage directly. The tertile mixture did not.
