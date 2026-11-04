---
experiment_id: 10
slug: inverse_density
title: "Exp 10 — Inverse-density weights"
current_stage: closeout
overall_verdict: null
blocked_by: null
code_path: experiments/010/
stages:
  charter: { status: done, completed_utc: "2026-11-04T16:50:00Z", notes: "Weight rare ages. Clip and mean-normalize. Score unweighted." }
  literature: { status: skipped, notes: "Yang et al. 2021 LDS, operator brief." }
  design: { status: done, completed_utc: "2026-11-04T16:50:00Z", notes: "5-year bins and Gaussian LDS against an unweighted XGB." }
  implement: { status: done, completed_utc: "2026-11-04T16:50:00Z", notes: "run_density_weights.py. Smoke ok." }
  run: { status: done, completed_utc: "2026-11-04T17:05:28Z", notes: "Grid plus extreme α=1. Neither beats none." }
  closeout: { status: done, completed_utc: "2026-11-04T17:06:00Z", notes: "Keep none. Raising the cap loses R²." }
run_lock_holder: null
decision_ids: []
---

# Exp 10 — Inverse-density weights (`inverse_density`)

**Code:** `experiments/010/run_density_weights.py`  
**Prior:** Exp 3 kept `xgboost_bag` on `|corr|` top-150. Exp 5 found the old tail is biased, and a constant shift fixed that tail while hurting everyone else. This experiment changes the training loss, not the prediction rule.

## Question

On the same five outer folds, does reweighting the training loss by the inverse age density beat an unweighted correlation XGBoost by at least 0.01 mean paired R², and by more than the standard error of the five fold deltas?

## Weights

Prep matches exp 9 `corr_xgb`. Drop dead and insane columns once. `KFold(5, shuffle=True, random_state=42)`. Inside each training fold only: median fill, z-score, `|r| ≥ 0.95` cull, `|corr|` top-150. The test fold is score-only.

The density is estimated from that training fold's ages. It is estimated again on the early-stopping fit slice, so the slice used to pick the tree count does not contribute its ages to its own weights. The early-stopping score and the outer R² are unweighted.

| Family | Density | Then |
|--------|---------|------|
| `bin_a03` `bin_a05` `bin_a07` | Counts in 5-year bins | w ∝ p(y)^(−α), α in {0.3, 0.5, 0.7} |
| `lds_s2_*` `lds_s3_*` | 1-year counts convolved with a Gaussian of σ = 2 or 3 years (Yang et al. 2021), same three α | same inverse |

Clip each unnormalized weight at 8× the median, then divide by the mean so the training weights average to 1. `none` is the unweighted model.

No copied rows and no synthetic neighbours. A duplicated rare row is the same person counted twice. A neighbour in feature space is not a neighbour in age.

## Success

The incumbent is `none`. A weighted arm replaces it only when its mean paired ΔR² is at least 0.01 and larger than the standard error of the five fold deltas. Also report MAE on the youngest and oldest deciles. A tail gain that loses overall R² does not replace `none`.

## Gates

- `lib/` stays read-only. The runner lives under `experiments/010/`.
- `data/X_test.csv` and Kaggle are not used.
- One training job at a time, using `experiments/.pipeline_run.lock`.

## Implement

```bash
uv run python experiments/010/run_density_weights.py --smoke
uv run python experiments/010/run_density_weights.py
```

## Closeout

**Keep `none`.** Finished 2026-11-04T17:05:28Z. The 8× cap was binding: α = 0.7 still peaked near 6×, same as α = 0.5. Lifting it to a pure inverse (α = 1, cap 20, cap 64, or no cap) pushes the rarest training rows to 11–40× and loses R² on every fold.

| Arm | R² | vs none | Young MAE | Mid MAE | Old MAE |
|-----|---:|--------:|----------:|--------:|--------:|
| none | **0.530** | | 9.37 | 3.84 | 9.17 |
| bin_a10_c20 | 0.486 | −0.044 (SE 0.012) | 9.40 | 4.27 | 8.71 |
| bin_a10_c64 | 0.491 | −0.039 (SE 0.010) | 9.26 | 4.25 | 8.60 |
| bin_a10 | 0.491 | −0.039 (SE 0.010) | 9.25 | 4.25 | 8.62 |

The old tail's MAE drops by about 0.5 year. The middle gives that back, and more. `lib/` was not changed.

## Artifacts

| Path | Role |
|------|------|
| `experiments/010/run_density_weights.py` | Runner and plot |
| `experiments/010/results/density_weights.json` | Fold scores, weight stats, verdict |
| `experiments/010/results/density_weights.png` | MAE by age, fold-0 weight curve |
