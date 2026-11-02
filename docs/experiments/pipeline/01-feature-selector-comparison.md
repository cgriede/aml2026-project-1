---
experiment_id: 1
slug: feature_selector_comparison
title: "Exp 1 — Feature selector comparison"
current_stage: closeout
overall_verdict: partial
blocked_by: null
code_path: experiments/001/
stages:
  charter: { status: done, completed_utc: "2026-11-02T12:20:00Z", notes: "Closed when the operator said to implement." }
  literature: { status: skipped, notes: "Operator skipped. No local literature library in this repo." }
  design: { status: skipped, notes: "Operator skipped. The charter arm is the design." }
  implement: { status: done, completed_utc: "2026-11-02T12:23:00Z", notes: "experiments/001/run_k150_matrix.py. Smoke passed." }
  run: { status: done, completed_utc: "2026-11-02T12:31:18Z", arms: [k150_matrix] }
  closeout: { status: done, completed_utc: "2026-11-02T12:32:00Z", notes: "Keep corr. H1a supported, H1b not supported." }
run_lock_holder: null
run_started_utc: "2026-11-02T12:23:29Z"
decision_ids: []
---

# Exp 1 — Feature selector comparison (`feature_selector_comparison`)

**Source brief:** `experiments/001/EXP.md`  
**Code (when implemented):** `experiments/001/`  
**Baseline log:** `runs/iterations.md`, `runs/log.md`

This repo has no `PROJECT_KNOWLEDGE` / `DECISIONS` / `STATUS` board. Prior evidence is the iteration log. I1–I5 stay accepted. This experiment does not reopen them.

---

## Charter

### Question

Which feature-selection mechanism should the age model keep?

The keep rule is outer-fold XGBoost \(R^2\). `|corr|` top-150 stays unless another set beats it by at least 0.01 mean paired \(R^2\) and that gap is larger than the standard error of the five fold deltas. `all` is allowed to win. That result means selection itself is not helping.

### Data (measured)

| Item | Value |
|------|--------|
| Train | 1212 rows (`data/X_train.csv`, `data/y_train.csv`) |
| Features | 832 (`x0`…`x831`) |
| Test | 776 rows, labels absent (`data/X_test.csv`) |
| Target | `y` |

`lib/clean.py` drops all-NaN, all-zero, and absmax `> 1e10` columns, then median-fills. Iteration 3 names the insane columns `x173`, `x596`, `x665`. Accepted prep after that: no row drop, train-only z-score, `|r| ≥ 0.95` cull, `|corr|` top-150 (`lib/GLOBAL.py`: `FS_METHOD=corr`, `FS_K=150`).

### What already exists

`lib/select_features.py` has `|corr|` top-k, MI-at-or-above-median, and a Ridge-scored Monte Carlo search. `lib/train.py` has Ridge (`alpha=1`, `solver=svd`) and one XGB setting (`n_estimators=2000`, `lr=0.03`, `max_depth=4`, `subsample=0.7`, `colsample_bytree=0.7`, `min_child_weight=7`). No Elastic Net selector, no permutation importance, no SVR, no nested selection. `experiments/iterate.py` selects on the full matrix, then runs CV. That leaks the selection into every test fold. This experiment selects inside each outer training fold.

### Hypothesis

**H1a. Reduction.** For XGB, the best fixed-k selector beats the all-features control on the same outer folds.

**H1b. Specialization.** The feature set with the best Ridge \(R^2\) is a different set from the one with the best XGB \(R^2\).

### Arm `k150_matrix` (this run)

Selectors, each returning 150 columns from the outer training fold only:

| Set | Rule |
|-----|------|
| `all` | Control. Every column that survives `clean` inside the fold. Size is not 150. |
| `corr` | Accepted method. Largest `\|corr(x, y)\|`. |
| `mi` | Largest mutual information with `y`. |
| `monte_carlo` | Existing Ridge Monte Carlo lifetime score, then top 150. |
| `elastic_net` | Largest `\|ElasticNet\|` coefficient. |
| `xgb_perm` | Largest XGB permutation importance on the outer training fold. |

Models scored on each selected set, hyperparameters frozen at `lib/train.py` (no inner search):

- Ridge, as `Trainer.train_baseline`
- XGB, one seed (`RANDOM_STATE=42`), as `Trainer.train_xgboost` with the outer test fold as `eval_set`
- SVR, `sklearn.svm.SVR` defaults (`RBF`, `C=1`, `epsilon=0.1`)

Outer split: `KFold(n_splits=5, shuffle=True, random_state=42)`, same as `CV_FOLDS`. Fit scale, correlation cull, and the selector on the outer training rows only.

If one non-control cell clears the H1a margin on single-seed XGB, re-score that cell and the `corr` cell with the 3-seed bag (`XGB_SEEDS`). That confirmation is part of this arm. It is the only extra training.

### Success

Outer-fold mean \(R^2\) and RMSE. Compare sets by the five paired fold deltas against `all` and against `corr`.

- **H1a supported** when, for XGB, the best of `{corr, mi, monte_carlo, elastic_net, xgb_perm}` has mean paired \(\Delta R^2\) vs `all` ≥ 0.01 and that mean exceeds the standard error of the five paired deltas. The 0.01 floor is the keep-bar from iteration 4. A smaller gain is fold noise, as iteration 5 treated +0.006.
- **H1b supported** when `argmax` over the six sets on Ridge \(R^2\) differs from `argmax` on XGB \(R^2\).
- Record SVR \(R^2\) and RMSE. SVR does not decide H1a or H1b.

Also record, without using them as gates: Jaccard of the 150-column sets across the five folds (stability), and pairwise Jaccard between selectors (complementarity).

### Gates

- `lib/` stays read-only. The runner lives under `experiments/001/`.
- `data/X_test.csv` and Kaggle are out of this comparison. There is no labeled holdout. The winner is the outer CV winner.
- One training job at a time. The implement stage adds a lock under `experiments/.pipeline_run.lock` before the run stage starts.
- No new package for this arm. Elastic Net and SVR come from `scikit-learn`. Permutation importance comes from `sklearn.inspection`. XGB is already a dependency.

### Parked (not this arm)

These stay in the brief. They start only after `k150_matrix` has a verdict.

| Later arm | Why it waits |
|-----------|----------------|
| `k_grid` | Same protocol at k ∈ {400, 200, 100, 50, 20}. The brief’s “800” is the full matrix. The measured width is 832, and `all` already covers it. |
| `groups` | Grouped permutation. Column names are `x0`…`x831`, with no region labels. Groups would be correlation clusters built inside the fold. |
| `ebm` | GAM / Explainable Boosting Machine. `interpret` is not a project dependency. Add it only if the k=150 rankings disagree in a way an additive model would explain. |
| Inner hyperparameter CV | Frozen settings answer the selector question. Tuning is a different question. |

### Charter checklist

- [x] H1a, H1b with a numeric or set-identity criterion
- [x] Arm `k150_matrix` named; \(R^2\) and RMSE defined
- [x] `blocked_by` is null. I1–I5 are prior evidence, not a block
- [x] README index row exists

## Literature

Skipped. This repo has no literature library, and the operator sent the run straight to implementation. The comparison is against the accepted `|corr|` top-150 pipeline in `runs/iterations.md`.

## Design

The charter arm is the design. One script, no production edit.

Prep inside each outer training fold: train-fold median fill, train-fold z-score, then `|r| ≥ 0.95` cull for the five selectors. `all` skips the cull. Defect columns (7: dead plus insane absmax) are dropped once on the full train matrix.

`elastic_net` is `ElasticNetCV(l1_ratio=0.5, alphas=25, cv=3)` fit on the outer training fold, then the 150 largest absolute coefficients. The alpha choice stays inside the fold. Downstream Ridge, XGB, and SVR stay frozen.

`xgb_perm` fits the frozen XGB on 80% of the outer training fold, early-stops on the other 20%, and ranks permutation \(R^2\) on that 20%. It does not see the outer test fold. The scored XGB does use the outer test fold as `eval_set`, matching `lib/train.py`.

## Implement

Runner: `experiments/001/run_k150_matrix.py`

```bash
uv run python experiments/001/run_k150_matrix.py --smoke
uv run python experiments/001/run_k150_matrix.py
```

Smoke (200 rows, 2 folds, k=8, tiny trees) exited 0. Lock file while the real run is up: `experiments/.pipeline_run.lock`. Results: `experiments/001/results/k150_matrix.json`.

## Run

```bash
uv run python experiments/001/run_k150_matrix.py
```

Finished 2026-11-02T12:31:18Z in about 8 minutes. Five outer folds. Lock cleared. `ElasticNetCV` emitted convergence warnings at `max_iter=2000`, so the elastic-net set is the not-fully-converged path.

Outer-fold means:

| Set | p | Ridge \(R^2\) | XGB \(R^2\) | SVR \(R^2\) | XGB RMSE |
|-----|---:|-------------:|------------:|------------:|---------:|
| corr | 150 | 0.289 | **0.543** | **0.424** | 6.55 |
| elastic_net | 150 | 0.262 | 0.505 | 0.372 | 6.80 |
| mi | 150 | 0.269 | 0.503 | 0.401 | 6.83 |
| xgb_perm | 150 | 0.269 | 0.500 | 0.378 | 6.85 |
| all | 825 | −2.621 | 0.499 | 0.246 | 6.85 |
| monte_carlo | 150 | 0.192 | 0.425 | 0.251 | 7.34 |

XGB fold scores for `corr`: 0.556, 0.526, 0.494, 0.597, 0.540. Paired gap vs `all`: mean +0.043, SE 0.009. 3-seed XGB on the same `corr` columns: 0.546.

Stability (mean pairwise Jaccard across folds): `corr` 0.93, elastic net 0.57, MI 0.45, permutation 0.20, Monte Carlo 0.11. Overlap with `corr`: MI 0.44, permutation 0.25, elastic net 0.16, Monte Carlo 0.11.

## Closeout

### Verdict

| # | Claim | Criterion | Result | Verdict |
|---|-------|-----------|--------|---------|
| H1a | Reduction helps XGB | Best fixed-k set beats `all` by ≥ 0.01 and by more than the fold SE | `corr` − `all` = +0.043 (SE 0.009) | supported |
| H1b | Ridge and XGB disagree on the winner | argmax sets differ | Both rank `corr` first | not supported |
| Keep | Mechanism to retain | Challenger must clear the same bar against `corr` | No challenger beat `corr` | **keep `corr`** |

### Overall verdict

Partial. Selecting 150 features helps. The mechanism that does it is still `|corr|` with `y`. Mutual information, elastic net, permutation importance, and Monte Carlo pick different columns and score worse on Ridge, XGB, and SVR. The full 825-column matrix collapses Ridge (−2.62) and is only a small step down for XGB (0.499 vs 0.543).

`lib/` was not changed. The accepted pipeline stays `|corr|` top-150.

## Artifacts

| Path | Role |
|------|------|
| `experiments/001/run_k150_matrix.py` | Runner |
| `experiments/001/results/k150_matrix.json` | Fold scores, selected columns, verdict |
| `experiments/001/k150_matrix.ipynb` | Readable view of the saved scores |
| `experiments/001/EXP.md` | Operator brief |
| `runs/iterations.md` | Accepted I1–I5 baseline |

## Pipeline next step

None for this arm. A later arm (`k_grid`, groups, EBM) starts only if the question changes. Do not replace `FS_METHOD`.
