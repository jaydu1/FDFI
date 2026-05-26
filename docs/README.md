---
orphan: true
---

# FDFI Documentation

This is the documentation source for FDFI (Flow-Disentangled Feature Importance).
Current documented release: `0.0.7`.

For the full documentation, see the main README in the project root or build the docs:

```bash
cd docs
python -m sphinx -b html . _build/html
open _build/html/index.html
```

## Documentation Structure

- **User Guide**: Core concepts, choosing explainers, installation
- **Tutorials**: Jupyter notebooks with worked examples
- **API Reference**: Full API documentation
- **Visualization**: Plotting helpers for FDFI scores, confidence intervals,
  diagnostics, and feature-correlation structure

## Quick Links

- [Concepts](user_guide/concepts.rst): Theory behind DFI and Flow-DFI
- [Choosing an Explainer](user_guide/choosing_explainer.rst): Which explainer to use
- [Tutorials](tutorials/index.rst): Hands-on notebooks

## Diagnostics

All disentangled explainers (`OTExplainer`, `EOTExplainer`, and `FlowExplainer`)
expose a shared `diagnostics` dictionary with latent independence (dCor) and
distribution fidelity (MMD) metrics plus qualitative labels.

Confidence intervals (`conf_int`) use mixture defaults in v0.0.2 for both
variance floor and practical margin.

## Visualization

The `fdfi.plots` module includes working Matplotlib helpers:

- `summary_bar` and `summary_plot` for global and per-sample attributions.
- `waterfall_plot` and `force_plot` for single-explanation views.
- `dependence_plot` for feature-value relationships.
- `correlation_heatmap` for background correlation structure.
- `confidence_interval_plot` and `diagnostics_plot` for inference and quality checks.
