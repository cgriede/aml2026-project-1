---
experiment_id: 4
slug: corr_cull
title: "Exp 4 — Keep near-duplicate features"
current_stage: closeout
overall_verdict: keep_cull
blocked_by: null
code_path: experiments/004/
stages:
  charter: { status: done, completed_utc: "2026-11-02T15:46:00Z", notes: "Operator asked whether |r| >= 0.95 pairs should both be kept." }
  literature: { status: skipped, notes: "Same protocol as exp 3. No new literature." }
  design: { status: done, completed_utc: "2026-11-02T15:46:00Z", notes: "Two arms, same k and folds. Cull is the incumbent." }
  implement: { status: done, completed_utc: "2026-11-02T15:46:00Z", notes: "experiments/004/run_corr_cull.py" }
  run: { status: done, completed_utc: "2026-11-02T15:47:15Z", arms: [cull, nocull] }
  closeout: { status: done, completed_utc: "2026-11-02T15:48:00Z", notes: "No-cull did not clear the bar. Keep the cull." }
run_lock_holder: null
decision_ids: []
---

# Exp 4 — Keep near-duplicate features (`corr_cull`)

**Code:** `experiments/004/run_corr_cull.py`  
**Prior:** exp 2 kept k=150. Exp 3 kept `xgboost_bag`. Both ran the `|r| ≥ 0.95` cull. Exp 2's `all` row is the full uncullled matrix, not a top-150 that still contains near-duplicates.

## Question

On the same five outer folds, does `|corr|` top-150 without the pairwise cull beat the culled top-150 by at least 0.01 mean paired bag R², and by more than the standard error of the five fold deltas?

Prep matches exp 3: defect columns dropped once, then inside each training fold a median fill and a z-score. `cull` then drops one feature from each pair with `|r| ≥ 0.95` and ranks the rest by `|corr(x, y)|`. `nocull` ranks the full scaled fold and still keeps 150, so a pair at −0.95 or +0.95 can both stay. k is 150 in both arms.

Models match exp 3: Ridge, and the 3-seed XGBoost bag with tree count chosen on an inner slice of the training fold. The test fold is score-only. The keep rule uses the bag. Ridge is reported because collinear columns are the reason the cull exists.

Also recorded per fold: Jaccard overlap of the two column lists, and how many pairs inside each list still have `|r| ≥ 0.95`.

## Run

```bash
uv run python experiments/004/run_corr_cull.py
```

Finished 2026-11-02T15:47:15Z. Five outer folds, k=150, test fold used only for scoring.

The cull removes 2 to 4 columns out of 825. On three folds those columns are outside the top 150, so both arms train on the same columns. Mean Jaccard of the two lists is 0.992. Inside the no-cull top 150, a fold holds 0, 1, or 2 pairs with `|r| ≥ 0.95`.

| Arm | Bag R² | Ridge R² | Pairs in the 150 |
|-----|-------:|---------:|-----------------:|
| cull | **0.535** | 0.289 | 0 |
| nocull | 0.531 | 0.283 | 0.6 |

Bag folds, cull: 0.541, 0.518, 0.485, 0.592, 0.538. No-cull: 0.518, 0.518, 0.485, 0.592, 0.541. The only fold where the lists differ and the bag moves is fold 0, 0.541 down to 0.518. Fold 4 moves the bag from 0.538 to 0.541 and the ridge from 0.297 to 0.267.

Paired gap, no-cull minus cull: bag −0.004 (SE 0.005), ridge −0.006 (SE 0.006). Neither clears +0.01.

## Closeout

**Keep the cull.** Leaving a `|r| ≥ 0.95` twin in the top 150 does not help the bag, and the cull almost never changes which 150 columns are used. `FS_CORR_MAX` stays 0.95. `lib/` was not changed.

## Artifacts

| Path | Role |
|------|------|
| `experiments/004/run_corr_cull.py` | Runner |
| `experiments/004/results/corr_cull.json` | Fold scores and the keep decision |
