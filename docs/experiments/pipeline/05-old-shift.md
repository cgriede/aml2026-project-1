---
experiment_id: 5
slug: old_shift
title: "Exp 5 — Old detector plus one constant"
current_stage: closeout
overall_verdict: keep_bag
blocked_by: null
code_path: experiments/005/
stages:
  charter: { status: done, completed_utc: "2026-11-02T17:02:00Z", notes: "Old tail has no within-group ranking. Test a detector and one shift." }
  literature: { status: skipped, notes: "Same folds and bag as exp 3." }
  design: { status: done, completed_utc: "2026-11-02T17:02:00Z" }
  implement: { status: done, completed_utc: "2026-11-02T17:02:00Z", notes: "experiments/005/run_old_shift.py" }
  run: { status: done, completed_utc: "2026-11-02T17:04:10Z", arms: [hard, soft] }
  closeout: { status: done, completed_utc: "2026-11-02T17:06:00Z", notes: "Hard and soft both lose R². Keep the bag." }
run_lock_holder: null
decision_ids: []
---

# Exp 5 — Old detector plus one constant (`old_shift`)

**Code:** `experiments/005/run_old_shift.py`  
**Prior:** exp 3 kept `xgboost_bag`. On its out-of-fold predictions, true age and predicted age correlate −0.03 above 79. Those predictions sit near 77 while the true ages sit near 84. A second age model on that slice has no order to fit.

## Question

Does flagging the old fifth and adding one constant to the bag beat the bag by at least 0.01 mean paired R², and by more than the standard error of the five fold deltas?

The age model is the exp 3 bag, reused on the same rows and the same five folds. It is not refit. The test fold is score-only.

Old, on a training fold, means age at or above that fold's 80th percentile. The detector is a logistic regression on the same `|corr|` top-150 columns. It does not see age at score time. Its cut is the training-fold probability quantile that flags the same fraction as the training old rate.

The constant is one number per fold. A 3-fold bag inside the training fold predicts the rows it did not fit. The shift is the mean of (true age − that prediction) on the old rows of those inner tests. Hard adds the whole shift to flagged people. Soft adds probability times the shift. Neither arm fits a second age.

## Success

The incumbent is the bag. Hard or soft replaces it only when its mean paired ΔR² is at least 0.01 and larger than the standard error of the five fold deltas.

## Gates

- `lib/` stays read-only.
- The bag predictions come from `experiments/003/results/models.json`.
- `data/X_test.csv` and Kaggle are not used.
- One training job at a time, using `experiments/.pipeline_run.lock`.

## Run

```bash
uv run python experiments/005/run_old_shift.py
```

Finished 2026-11-02T17:04:10Z. Five outer folds. The bag is the exp 3 out-of-fold prediction. The shift is fit on inner folds of the training rows only.

The training edge is 78.0 or 78.4. The shift is +6.3 to +7.1 years. The detector's test AUC is 0.73 to 0.85. It flags 38 to 65 of about 243 test rows.

| Arm | R² | RMSE | MAE | vs bag |
|-----|---:|-----:|----:|-------:|
| bag | **0.535** | 6.61 | 4.98 | |
| soft | 0.495 | 6.88 | 5.12 | −0.039 |
| hard | 0.466 | 7.08 | 5.32 | −0.069 |

Paired gap against the bag: hard −0.069 (SE 0.014), soft −0.039 (SE 0.012). Neither clears +0.01.

MAE by true-age quintile. Bias is predicted minus true.

| Age | n | Bag MAE | Hard MAE | Soft MAE | Bag bias | Hard bias |
|-----|--:|--------:|---------:|---------:|---------:|----------:|
| 42-62 | 256 | 6.94 | 7.25 | 7.42 | +6.5 | +6.8 |
| 63-68 | 260 | 3.94 | 4.34 | 4.35 | +1.2 | +1.7 |
| 69-73 | 261 | 3.59 | 4.52 | 4.24 | −0.4 | +0.8 |
| 74-78 | 202 | 3.35 | 4.62 | 3.90 | −1.8 | +0.2 |
| 79-97 | 233 | 6.95 | 5.79 | 5.50 | −6.8 | −3.1 |

## Closeout

**Keep the bag.** The constant has the right size, about +6.5 years, and the oldest quintile's MAE falls from 6.95 to 5.79 (hard) and 5.50 (soft). The detector also flags people the bag already had close. Those rows pick up years they did not need, and overall R² falls. `lib/` was not changed.

## Artifacts

| Path | Role |
|------|------|
| `experiments/005/run_old_shift.py` | Runner |
| `experiments/005/results/old_shift.json` | Fold scores, shift, detector AUC |
| `experiments/005/results/old_shift.png` | Bag against the hard shift, and MAE by quintile |
