# Experiment pipeline

One experiment, one file. This index is the stage board only.

| # | Slug | Doc | Code | Stage | Verdict | Blocked by |
|---|------|-----|------|-------|---------|------------|
| 1 | feature_selector_comparison | [01-feature-selector-comparison.md](01-feature-selector-comparison.md) | `experiments/001/` | closeout | partial (keep corr) | — |
| 2 | corr_k_grid | [02-corr-k-grid.md](02-corr-k-grid.md) | `experiments/002/` | closeout | keep k=150 | — |
| 3 | model_selection | [03-model-selection.md](03-model-selection.md) | `experiments/003/` | closeout | keep xgboost_bag | — |
| 4 | corr_cull | [04-corr-cull.md](04-corr-cull.md) | `experiments/004/` | closeout | keep cull | — |
| 5 | old_shift | [05-old-shift.md](05-old-shift.md) | `experiments/005/` | closeout | keep bag | — |
| 9 | feature_union | [09-feature-union.md](09-feature-union.md) | `experiments/009/` | implement (smoke ok) | — | — |
| 10 | inverse_density | [10-inverse-density.md](10-inverse-density.md) | `experiments/010/` | closeout | keep none | — |

Prior accepted pipeline (not a pipeline-doc experiment): `runs/iterations.md` I1–I5. Baseline to beat is `|corr|` top-150, Ridge CV 0.28, single XGB 0.542, 3-seed XGB 0.552. Those scores selected features on the full train matrix before CV.
