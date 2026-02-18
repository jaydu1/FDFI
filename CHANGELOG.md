# Changelog

## [0.0.2] - 2026-02-17
### Added
- Shared diagnostics now emit qualitative labels (GOOD/MODERATE/POOR) with unified `[FDFI][DIAG]` logging for OT/EOT/Flow explainers.
- Utility functions `compute_latent_independence` and `compute_mmd` are promoted and documented in the public API.
- Notebook investigations now include SE calibration checks (formula vs bootstrap) and reconstruction-fidelity checks for FlowExplainer.

### Changed
- `conf_int()` now defaults to mixture-based variance floor and mixture-based margin for all explainers.
- Diagnostics implementation is generalized in the base explainer API (`diagnose` / `diagnostics`) for OT/EOT/Flow, and Flow-specific legacy diagnostics are removed.
- Flow diagnostics reconstruction now uses high-precision ODE tolerances to measure model fidelity instead of solver drift.
- Flow solver tolerances are configurable (`flow_solver_rtol/atol`, `diagnostics_solver_rtol/atol`) and flow training can be seeded via `flow_training_seed`.
- `compute_latent_independence` and `compute_mmd` are optimized for better computational efficiency.
- Package version is synchronized to `0.0.2` across package metadata, docs configuration, and tutorial notebook outputs.

## [0.0.1] - 2026-01-31
### Added
- **OTExplainer**: Gaussian optimal-transport DFI for feature importance computation
- **EOTExplainer**: Entropic optimal-transport DFI with adaptive epsilon, stochastic transport sampling, and Gaussian/empirical targets
- **FlowExplainer**: Flow-based DFI using normalizing flows for non-Gaussian data
  - CPI (Conditional Permutation Importance) method
  - SCPI (Sobol-CPI) method
  - Multiple sampling methods: `resample`, `permutation`, `normal`, `condperm`
- **TreeExplainer**, **LinearExplainer**, **KernelExplainer**: Classic explainer interfaces
- **Visualization functions**: `summary_plot`, `waterfall_plot`, `force_plot`, `dependence_plot`
- **Confidence intervals**: `conf_int` method for statistical inference with variance-floor and margin adjustments
- Standard error outputs (`se_X`, `se_Z`) for explainer uncertainty
- Comprehensive test suite with 50 tests
- Tutorial notebooks: quickstart, OT, EOT, Flow explainers, confidence intervals
- GitHub Actions workflows for CI/CD and PyPI publishing

### Changed
- EOT/OT explainers now cache results for post-hoc CI computation via `conf_int`.
- Exp3 example now uses `conf_int` for CI bands and supports `FDFI_USE_EOT` toggle.
- `environment.yml` now includes plotting + sklearn deps for running examples.
- Removed legacy `setup.py` and `requirements*.txt` in favor of `pyproject.toml` extras.
- DFIExplainer is renamed to OTExplainer (DFIExplainer remains as an alias).
- Explainer can skip flow training via `fit_flow=False`.
