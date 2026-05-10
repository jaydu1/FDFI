# Changelog

## [0.0.5] - 2026-04-28
### Added
- **Multiple testing correction**: `conf_int()` and `summary()` now support a `multitest_method` parameter (mimicking `statsmodels`) for controlling FDR/FWER. Adjusted p-values are returned as `pvalue_adj`.
- **Unified Group Importance**: `conf_int()` now supports a `groups` argument to compute group-level feature importance with uncertainty, replacing the separate `group_importance()` logic (now deprecated).
  - Accepts groups as a `dict` of index lists, a 1-D label array, or a binary `pandas.DataFrame` indicator matrix (features may belong to multiple groups).
  - Optional null-feature thresholding (`threshold_null=True`) zeros out per-feature UEIFs with negative mean before aggregation.
- **Improved API Consistency**: Renamed `phi_hat` to `score` in `conf_int()` output for better clarity.
- Per-sample UEIFs (`ueifs_X`, `ueifs_Z`) are now stored as instance attributes after calling `OTExplainer`, `EOTExplainer`, and `FlowExplainer`, enabling downstream group aggregation.

## [0.0.4] - 2026-04-01
### Added
- **Crossfitting**: new cross-fitted DFI explainer for valid inference at small sample sizes. Wraps any explainer class (`OTExplainer`, `EOTExplainer`, `FlowExplainer`) and performs K-fold cross-fitting so that the disentanglement map is never evaluated on its own training data.
- Flexible `cv` parameter accepts an `int` (shorthand for `KFold`) or any scikit-learn splitter instance (`StratifiedKFold`, `ShuffleSplit`, `RepeatedKFold`, `GroupKFold`, custom, etc.).
- Optional `y` and `groups` parameters for stratified and group-aware splitters.
- Overlapping test set handling: splitters like `ShuffleSplit` and `RepeatedKFold` that assign samples to multiple test sets are handled by per-sample UEIF averaging.
- Ensemble prediction on new data: `cf(X_new)` averages importance from all fold explainers.
- `Crossfitting` inherits `conf_int()` and `summary()` from the base `Explainer` class.
- `Crossfitting` exported from `fdfi` top-level package.
- 17 new tests covering init, OT/EOT/Flow cross-fitting, all splitter types, conf_int, summary, and ensemble prediction.

## [0.0.3] - 2026-03-19
