---
experiment_id: 9
slug: feature_union
title: "Exp 9 — Two specialists, one blend"
current_stage: implement
overall_verdict: null
blocked_by: null
code_path: experiments/009/
stages:
  charter: { status: done, completed_utc: "2026-11-04T16:27:00Z", notes: "Corr XGB plus RReliefF SVR. Mean and non-negative blend. Four diagnostic plots." }
  literature: { status: skipped, notes: "Operator brief already named the random-subspace / stacking pattern." }
  design: { status: done, completed_utc: "2026-11-04T16:27:00Z", notes: "Plan: feature_union_ensemble. Incumbent corr_xgb." }
  implement: { status: done, completed_utc: "2026-11-04T16:35:00Z", notes: "run_union.py and plot_union.py. Smoke next." }
  run: { status: pending }
  closeout: { status: pending }
run_lock_holder: null
decision_ids: []
---

# Exp 9 — Two specialists, one blend (`feature_union`)

**Code:** `experiments/009/`  
**Prior:** Exp 1 kept `|corr|` top-150 (XGB 0.543, SVR 0.424, Ridge 0.289). Exp 3 kept `xgboost_bag` on that same set. Ridge on corr columns is weak. Mutual information overlapped `|corr|` at Jaccard 0.44 and lost on every model. RReliefF is the other selector in play and has no outer-CV score yet.

## Question

On the same five outer folds, does blending a correlation XGBoost with an RReliefF RBF-SVR beat the correlation XGBoost by at least 0.01 mean paired R², and by more than the standard error of the five fold deltas?

The plots answer the second part: which specialist is closer on which ages.

## Specialists

Prep matches exp 3. Drop dead and insane columns once. `KFold(5, shuffle=True, random_state=42)`. Inside each training fold only: median fill, z-score, `|r| ≥ 0.95` cull.

| Arm | Features | Model |
|-----|----------|-------|
| `corr_xgb` | `|corr|` top-150 | XGBoost. Tree count from a 20% slice of the training fold, then refit on the whole training fold. Test fold is score-only. |
| `rrelieff_svr` | RReliefF top-150 | RBF SVR (`C=1`, `epsilon=0.1`) |

k stays 150. Record the Jaccard of the two column lists per fold.

## Blends

Both blends are scored on the outer test fold. Neither sees that fold while fitting.

| Arm | Rule |
|-----|------|
| `mean` | Average of the two test predictions. No fitted weights. |
| `nnls` | Non-negative weights and an intercept. Inside the outer training fold, a 3-fold split refits both specialists on the columns already chosen for that outer fold. `LinearRegression(positive=True, fit_intercept=True)` learns the blend on those inner out-of-fold predictions, then applies it to the outer test predictions. |

An age-dependent stacker waits. Build it only if the binned-MAE lines cross.

## Success

The incumbent is `corr_xgb`. `mean` or `nnls` replaces it only when its mean paired ΔR² is at least 0.01 and larger than the standard error of the five fold deltas. Also report both specialists' R². A weak RReliefF SVR that the non-negative blend then ignores is a valid negative result.

## Plots

Runner writes `experiments/009/results/union.json`. Plot script reads that file only.

1. Residual (`pred − age`) against true age. One panel per series, with a 5-year binned mean.
2. Mean absolute error by those same 5-year bins, one line per series, sample count as bars.
3. Scatter of the two specialists' residuals, plus prediction and residual correlations.
4. Worst 5% by mean absolute error of the two specialists: age, NaN count, and each residual.

Bonus: color the specialist residual scatters by raw NaN count.

## Gates

- `lib/` stays read-only. The runner lives under `experiments/009/`.
- `data/X_test.csv` and Kaggle are not used to pick the winner.
- One training job at a time, using `experiments/.pipeline_run.lock`.

## Implement

```bash
uv run python experiments/009/run_union.py --smoke
uv run python experiments/009/run_union.py
uv run python experiments/009/plot_union.py
```

## Artifacts

| Path | Role |
|------|------|
| `experiments/009/run_union.py` | Runner |
| `experiments/009/plot_union.py` | Residual, binned MAE, residual scatter, hard samples |
| `experiments/009/results/union.json` | OOF predictions, fold scores, verdict |
