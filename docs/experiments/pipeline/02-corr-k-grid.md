---
experiment_id: 2
slug: corr_k_grid
title: "Exp 2 — |corr| prefix size"
current_stage: closeout
overall_verdict: keep_150
blocked_by: null
code_path: experiments/002/
stages:
  charter: { status: done, completed_utc: "2026-11-02T14:08:00Z", notes: "Operator set k in {25, 50, 75, 100, 150, 200, 300, 400, 600, 825}." }
  literature: { status: skipped, notes: "Same protocol as exp 1. No new literature." }
  design: { status: done, completed_utc: "2026-11-02T14:08:00Z", notes: "Nested |corr| prefixes. Same outer folds and frozen models as exp 1." }
  implement: { status: done, completed_utc: "2026-11-02T14:08:40Z", notes: "experiments/002/run_k_grid.py. Smoke passed." }
  run: { status: done, completed_utc: "2026-11-02T14:10:29Z", arms: [k_grid] }
  closeout: { status: done, completed_utc: "2026-11-02T14:11:00Z", notes: "No prefix cleared the bar against k=150." }
run_lock_holder: null
run_started_utc: "2026-11-02T14:08:47Z"
decision_ids: []
---

# Exp 2 — `|corr|` prefix size (`corr_k_grid`)

**Code:** `experiments/002/run_k_grid.py`  
**Prior:** exp 1 kept `|corr|` top-150. This arm only changes k.

## Question

On the same five outer folds, does another `|corr|` prefix beat k=150 by at least 0.01 mean paired XGBoost R² and by more than the standard error of the five fold deltas?

Prep matches exp 1: defect columns dropped once, then inside each training fold a median fill, z-score, and `|r| ≥ 0.95` cull. The ranking is `|corr(x, y)|` on the culled training fold. Prefixes are nested. `all` is the uncullled matrix, scored once per fold. Requested k=825 is larger than the culled width (821–823), so that row is every surviving column.

Models stay frozen: Ridge, one-seed XGB with the outer test fold as `eval_set`, SVR defaults. The keep rule uses XGB only.

## Run

```bash
uv run python experiments/002/run_k_grid.py
```

Finished 2026-11-02T14:10:29Z in about 2 minutes.

| k | p | Ridge R² | XGB R² | SVR R² | XGB vs 150 |
|---|--:|--------:|-------:|-------:|------------|
| 25 | 25 | 0.282 | 0.454 | 0.416 | −0.089 (SE 0.004) |
| 50 | 50 | 0.310 | 0.493 | 0.431 | −0.049 (SE 0.006) |
| 75 | 75 | 0.329 | 0.523 | 0.444 | −0.020 (SE 0.010) |
| 100 | 100 | 0.324 | 0.531 | 0.438 | −0.012 (SE 0.004) |
| 150 | 150 | 0.289 | 0.543 | 0.424 | |
| 200 | 200 | 0.225 | 0.545 | 0.421 | +0.003 (SE 0.007) |
| 300 | 300 | 0.099 | 0.520 | 0.373 | −0.023 (SE 0.009) |
| 400 | 400 | −0.019 | 0.493 | 0.335 | −0.050 (SE 0.006) |
| 600 | 600 | −0.477 | 0.501 | 0.286 | −0.041 (SE 0.005) |
| 825 | 822 | −2.649 | 0.505 | 0.246 | −0.037 (SE 0.010) |
| all | 825 | −2.621 | 0.499 | 0.246 | |

XGB at 150: 0.556, 0.526, 0.494, 0.597, 0.540. Same folds as exp 1.

## Closeout

k=200 is the highest XGB mean (0.545 vs 0.543). The paired gap is +0.003, SE 0.007, under the 0.01 floor. No other prefix is higher. **Keep k=150.**

Ridge and SVR peak at k=75 (0.329 and 0.444). That does not change the XGB keep rule. `FS_K` stays 150.
