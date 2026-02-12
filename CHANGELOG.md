# Changelog

## [0.0.1] - 2026-02-12

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
- Package renamed from `dfi` to `fdfi` for PyPI availability
- DFIExplainer renamed to OTExplainer (DFIExplainer remains as alias)
- Removed legacy `setup.py` and `requirements*.txt` in favor of `pyproject.toml` extras

### Dependencies
- Core: numpy, scipy, scikit-learn, matplotlib, seaborn
- Optional `[flow]`: torch, torchdiffeq (for FlowExplainer)
