---
orphan: true
---

# FDFI Documentation

This is the documentation source for FDFI (Flow-Disentangled Feature Importance).
Current documented release: `0.0.8`.

For the full documentation, see the main README in the project root or build the docs:

```bash
cd docs
python -m sphinx -b html . _build/html
open _build/html/index.html
```

## Documentation Structure

- **User Guide**: Core concepts, choosing explainers, installation, interpreting results, statistical inference
- **Tutorials**: Jupyter notebooks with worked examples
- **API Reference**: Full API documentation
- **Visualization**: Plotting helpers for FDFI scores, confidence intervals,
  diagnostics, and feature-correlation structure

## Quick Links

- [Concepts](user_guide/concepts.rst): Theory behind DFI and Flow-DFI
- [Choosing an Explainer](user_guide/choosing_explainer.rst): Which explainer to use
- [Interpreting Results](user_guide/interpreting_results.rst): What each output key means
- [Statistical Inference](user_guide/statistical_inference.rst): Confidence intervals, one-sided tests, FDR
- [Tutorials](tutorials/index.rst): Hands-on notebooks

## Diagnostics

All disentangled explainers (`OTExplainer`, `EOTExplainer`, and `FlowExplainer`)
expose a shared `diagnostics` dictionary with latent independence (dCor) and
distribution fidelity (MMD) metrics plus qualitative labels.

## Visualization

The `fdfi.plots` module includes working Matplotlib helpers:

- `summary_bar` and `summary_plot` for global and per-sample attributions.
- `waterfall_plot` and `force_plot` for single-explanation views.
- `dependence_plot` for feature-value relationships.
- `correlation_heatmap` for background correlation structure.
- `confidence_interval_plot` (supports one-sided CIs since 0.0.8) and `diagnostics_plot` for inference and quality checks.
