# Outlier removal after scaling

Isolation Forest and OneClassSVM score distance, so they run after the train z-score. `lib/main.py` fits `StandardScaler` on the cleaned train matrix, transforms train, then calls `remove_outliers`. The test matrix is scaled with that same scaler and is not row-filtered.

Detectors, on that scaled train matrix:

- `isoforest`: `IsolationForest(contamination=0.05)`
- `svm`: `OneClassSVM(nu=0.01, kernel="rbf")`, which fits a second `StandardScaler` on the already scaled matrix

Four `python lib` runs on 2026-11-04. Same path after the row filter: `|r| >= 0.95` cull, `|corr|` top-150, 5-fold single XGBoost (tree count = median best iteration + 1) and Ridge. No arm is the 3-seed bag.

Dropping rows changes the cull and the top-150, so the gap is the whole prep, not the same 150 columns with fewer rows. Folds are re-split on the surviving rows.

| arm | dropped | n | `|r|>=0.95` dropped | Ridge CV | XGB CV | trees |
|---|---:|---:|---|---:|---:|---:|
| none | 0 | 1212 | x317, x579 | 0.278 ± 0.057 | **0.542 ± 0.034** | 667 |
| isoforest | 61 | 1151 | x69, x790 | 0.269 ± 0.146 | 0.502 ± 0.073 | 194 |
| svm | 73 | 1139 | x317, x442, x579 | 0.283 ± 0.062 | 0.489 ± 0.037 | 752 |
| both (union) | 113 | 1099 | x442, x579, x69, x790 | 0.242 ± 0.120 | 0.497 ± 0.053 | 472 |

`both` flagged 61 (isoforest) then 52 new (svm).

## Verdict

Keep no row drop. Every detector loses about 0.04 to 0.05 XGB R² against the unfiltered run (0.542). That gap is larger than the unfiltered fold scatter (0.034). SVM's Ridge mean is 0.006 higher, inside that arm's fold scatter (0.062).

Same call as iteration 3. That run used an unscaled IF∩SVM and a smaller z-scored IF. This run is the distance detectors on the z-scored matrix, alone and as a union. Still delete useful rows.

## Who each detector flags

Both detectors were fit on the same cleaned, z-scored train matrix, then the flagged ids were compared.

| group | n |
|---|---:|
| shared | 21 |
| isolation forest only | 40 |
| SVM only | 52 |

Isolation forest flags 61, the SVM flags 73, and 21 of those are the same rows. The union is 113. Adding 61 and 73 counts the shared rows twice.

Age in every group matches the bulk (median 70). Shared median 70, isolation-forest-only 71.5, SVM-only 67.5.

On the two strongest age columns, `x458` (|r| 0.444) and `x159` (|r| 0.438), the SVM-only rows sit in the cloud: 13% are beyond 2 bulk standard deviations on either column, against 9% of the bulk. Isolation-forest-only and the shared rows have a longer right tail (23% and 29% beyond 2 sd). Their means move about half a standard deviation, and the shared median on `x458` stays next to the bulk.

The plot is `experiments/007/overlap.png`. The unique rows, and most of the shared rows, are ordinary subjects in a sparse part of the scaled space. The two detectors disagree on 92 of the 113 flagged rows. Deleting those rows removes real training data, which is the CV drop above.

## SVM `nu` grid

Same path, SVM only. Kernel `rbf`, `gamma="scale"`. `nu=0.01` is the svm row in the table above. The unfiltered run stays the comparison (XGB 0.542 ± 0.034, Ridge 0.278 ± 0.057).

| nu | dropped | n | `|r|>=0.95` dropped | Ridge CV | XGB CV | trees |
|---:|---:|---:|---|---:|---:|---:|
| 0.005 | 68 | 1144 | x317, x579, x69, x790 | 0.168 ± 0.088 | 0.530 ± 0.062 | 566 |
| 0.01 | 73 | 1139 | x317, x442, x579 | 0.283 ± 0.062 | 0.489 ± 0.037 | 752 |
| 0.05 | 93 | 1119 | x317, x442, x579, x69, x790 | 0.297 ± 0.093 | 0.539 ± 0.073 | 754 |
| 0.1 | 137 | 1075 | x317, x442, x69, x790 | 0.293 ± 0.064 | 0.490 ± 0.062 | 169 |

No `nu` beats the unfiltered XGB. `nu=0.05` is 0.003 below it, with a wider fold scatter (0.073 against 0.034). Its Ridge mean is 0.019 higher, inside that arm's fold standard error (0.041). `nu=0.005` cuts Ridge to 0.168. Keep no row drop.

## Isolation forest `contamination` grid

Same path, isolation forest only. `contamination=0.05` is the isoforest row in the first table. The unfiltered run stays the comparison (XGB 0.542 ± 0.034, Ridge 0.278 ± 0.057).

| contamination | dropped | n | `|r|>=0.95` dropped | Ridge CV | XGB CV | trees |
|---:|---:|---:|---|---:|---:|---:|
| 0.005 | 7 | 1205 | x317, x579 | 0.250 ± 0.104 | 0.540 ± 0.016 | 543 |
| 0.01 | 13 | 1199 | x317, x442, x579 | 0.296 ± 0.089 | 0.536 ± 0.026 | 529 |
| 0.05 | 61 | 1151 | x69, x790 | 0.269 ± 0.146 | 0.502 ± 0.073 | 194 |
| 0.1 | 122 | 1090 | x790 | 0.265 ± 0.137 | 0.544 ± 0.054 | 836 |

No setting beats the unfiltered XGB by 0.01. `contamination=0.1` is 0.002 above it, inside that arm's fold standard error (0.024), and the fold scatter is wider (0.054 against 0.034). `contamination=0.005` is 0.002 below, with a tight scatter, and Ridge falls to 0.250. Keep no row drop.
