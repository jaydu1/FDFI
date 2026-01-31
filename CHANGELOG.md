# Changelog

## [0.0.1] - 2026-01-31
### Added
- OTExplainer (Gaussian optimal-transport DFI without cross-fitting).
- EOTExplainer (entropic optimal-transport DFI without cross-fitting).
- Exp3-based unit tests validating mean/variance aggregation for OT/EOT.
- Standard error outputs (`se_X`, `se_Z`) for explainer uncertainty.
- `conf_int` and `summary` methods for confidence intervals with variance-floor and margin adjustments.
- EOT options for adaptive epsilon, stochastic transport sampling, and Gaussian/empirical targets.

### Changed
- EOT/OT explainers now cache results for post-hoc CI computation via `conf_int`.
- Exp3 example now uses `conf_int` for CI bands and supports `FDFI_USE_EOT` toggle.
- `environment.yml` now includes plotting + sklearn deps for running examples.
- Removed legacy `setup.py` and `requirements*.txt` in favor of `pyproject.toml` extras.

### Changed
- DFIExplainer is renamed to OTExplainer (DFIExplainer remains as an alias).
- Explainer can skip flow training via `fit_flow=False`.
