# Isolated iterations

Accepted baseline at start: dead+insane drop, IF∩SVM (unscaled IF), train-only z-score, MI median, B-XGB.
Judge = Ridge CV. Goal = XGB CV.

| I | hyp | keep? | Ridge | XGB | n×p | note |
|---|---|---|---|---|---|---|
| 1 | `|corr|` top-k vs MI half | **yes, k=150** | 0.30 vs 0.07 | **0.519** vs 0.487 | 1192×150 | k=50 Ridge 0.32 / XGB 0.47; k=100 Ridge 0.32 / XGB 0.50. MI-half was a bad set. |
| 2 | wider k 200/250/300 | **no, stay 150** | 0.27 / 0.26 / 0.23 | 0.508 / 0.508 / 0.512 | 1192×200+ | XGB and Ridge both peak at 150. Extra weak cols hurt. |
| 3 | skip IF / z-score then IF | **yes, no row drop** | 0.28 vs 0.30 | **0.542** vs 0.519 | 1212×150 | IF∩SVM deleted useful rows. z-score IF still drops 21, XGB 0.537. |
| 4 | 3-seed mean / XGB+Ridge | **yes, 3-seed mean** | 0.28 | **0.552** vs 0.542 | 1212×150 | +0.010. XGB+Ridge 0.485, Ridge still too weak to average in. |
| 5 | missingness features | **no** | 0.28–0.30 | 0.542–0.548 | 1212×151–161 | `nan_count` \|r\| with age = 0.05. Best XGB +0.006, inside fold noise. |

## Accepted pipeline

dead + insane drop (`x173`,`x596`,`x665`) → **no row drop** → train-only z-score → corr-cull 0.95 → **`|corr|` top-150** → Ridge judge + **B-XGB 3-seed mean**.

Local: Ridge CV **0.28**, single XGB **0.542**, 3-seed **0.552**. Start of this loop was Ridge −0.06 / XGB 0.49.
