# Changelog

## [0.0.7] - 2026-05-26
### Added
- Working `fdfi.plots` visualizations for summary, waterfall, force, dependence, correlation heatmap, confidence interval, and diagnostics views.
- `summary_bar()` for sorted global FDFI bars with sanitized standard errors, optional group colors, and a returned sorted table.
- `correlation_heatmap()` with Pearson correlations, absolute-correlation hierarchical clustering, reordered feature names, and small-background warnings.
- Visualization tutorial and documentation examples covering `results["phi_X"]`, `results["se_X"]`, per-sample `ueifs_X`, `conf_int()` output, and explainer diagnostics.

### Changed
- Replaced placeholder plot tests with non-interactive Matplotlib `Agg` tests.
- Updated visualization documentation to remove stale "coming soon" language.

## [0.0.6] - 2026-05-17
### Added
- **`zscore` and `ranking` output fields**: `conf_int()` now always returns `zscore` (signed z-statistic `(score − margin) / se`) and `ranking` (integer rank by descending z-score, 1 = most important) for all targets and group modes.
- **`summary()` docstring**: full NumPy-style docstring added, documenting all forwarded `conf_int()` parameters including 0.0.5-introduced `groups` and `multitest_method`.
- **`Crossfitting` enhancements**: new `cv_kwargs` and `y_test` parameters; improved sklearn estimator detection; null-threshold logic in fold aggregation.
- **`FlowMatchingModel` improvements**: `dequantize_noise` parameter for binary data augmentation; `Jacobi_Batch` now uses a simpler sequential loop (one sample at a time via `Jacobi_N`).
- **Case study docs**: EOT FDFI case study notebook (`eot_case_study_sens50`) added to documentation under a new Case Studies section.
- **`docs/user_guide/concepts.rst`**: `conf_int()` return keys documented as a reference table.

### Fixed
- `Explainer` class docstring: corrected `from dfi import` → `from fdfi import`.
- `group_importance()` deprecation version tag corrected from `0.2.0` to `0.0.5`.
- `_format_summary()`: z-score now computed as `(score − margin) / se` (was `score / se`).
- `docs/conf.py` `release` field was stale at `0.0.4`; updated to match package version.
- `Crossfitting` docstring: RST bullet list in parameter description replaced with prose to fix Sphinx rendering error.
- `docs/tutorials/confidence_intervals.ipynb`: repaired two broken JSON cell boundaries.
- `MANIFEST.in`: exclude `docs/case_studies/data` and `docs/case_studies/results` from sdist.

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
