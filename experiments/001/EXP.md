# Exp 001 brief

Cycle: [01-feature-selector-comparison.md](../../docs/experiments/pipeline/01-feature-selector-comparison.md)

Notebook: [k150_matrix.ipynb](k150_matrix.ipynb)

Stage: **closeout**. Arm `k150_matrix` kept `|corr|` top-150. XGB 0.543 vs 0.499 for all 825 columns. MI, elastic net, permutation, and Monte Carlo were worse.

---

<span style="color: rgb(0, 0, 0);">Here are the condensed, actionable execution items based on the analysis:</span>

<span style="color: rgb(0, 0, 0);">**1. Group Correlated MRI Features**</span>

<span style="color: rgb(0, 0, 0);">**Create feature groups** based on anatomical regions, feature types, or correlation clusters.</span>

<span style="color: rgb(0, 0, 0);">**Prepare for grouped permutation importance** to handle redundant/correlated MRI measurements effectively.</span>

<span style="color: rgb(0, 0, 0);">**2. Implement the Core Feature Selectors (Phase 1)**</span>

<span style="color: rgb(0, 0, 0);">Build or configure four distinct feature selection methods:</span>

<span style="color: rgb(0, 0, 0);">\[ \] **Mutual Information:** To capture individual nonlinear relationships.</span>

<span style="color: rgb(0, 0, 0);">\[ \] **Elastic Net / Ridge:** To capture linear relationships and enforce sparsity.</span>

<span style="color: rgb(0, 0, 0);">\[ \] **XGBoost (Importance / Permutation):** To capture tree-exploitable relationships and feature interactions.</span>

<span style="color: rgb(0, 0, 0);">\[ \] **GAM / EBM (Explainable Boosting Machine):** To model nonlinear additive feature effects.</span>

<span style="color: rgb(0, 0, 0);">**3. Establish the Evaluation Framework**</span>

<span style="color: rgb(0, 0, 0);">**Set up Nested Cross-Validation:** Ensure feature selection is performed strictly within the outer training folds to prevent data leakage. Use inner CV for model hyperparameter tuning.</span>

<span style="color: rgb(0, 0, 0);">**Define Subset Sizes:** Test performance across multiple feature thresholds: **All 800, 400, 200, 100, 50, and 20 features**.</span>

<span style="color: rgb(0, 0, 0);">**Track Key Metrics:** Measure out-of-sample $R^2$, RMSE, feature selection stability across folds, and complementarity.</span>

<span style="color: rgb(0, 0, 0);">**Protect Test Data:** Retain a final untouched test set if you plan to compare and select a winning feature selection method.</span>

<span style="color: rgb(0, 0, 0);">**4. Run the Immediate Next Experiment**</span>

<span style="color: rgb(0, 0, 0);">Execute a baseline comparison to evaluate whether dimensionality reduction is helping and if current methods are overly specialized to Ridge.</span>

<span style="color: rgb(0, 0, 0);">**Models to test:** Ridge, XGBoost, and SVR.</span>

<span style="color: rgb(0, 0, 0);">**Feature sets to compare:**</span>

<span style="color: rgb(0, 0, 0);"></span>

<span style="color: rgb(0, 0, 0);">All 800 features</span>

<span style="color: rgb(0, 0, 0);">Current Monte Carlo-selected features</span>

<span style="color: rgb(0, 0, 0);">Mutual-information-selected features</span>

<span style="color: rgb(0, 0, 0);">Elastic-Net-selected features</span>

<span style="color: rgb(0, 0, 0);">XGBoost-permutation-selected features</span>

<span style="color: rgb(0, 0, 0);">**Execution:** Run all combinations using identical nested CV splits and compare predictive performance ($R^2$ / RMSE).</span>