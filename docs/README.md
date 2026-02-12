# dfi Documentation

Welcome to the dfi (Flow-Disentangled Feature Importance) documentation.

## Overview

dfi is a Python library for computing feature importance using flow-disentangled methods, inspired by SHAP (SHapley Additive exPlanations).

## Installation

### From Source

```bash
git clone https://github.com/jaydu1/dfi.git
cd dfi
pip install -e .
```

### Dependencies

Install extras via `pyproject.toml`:

```bash
pip install -e ".[dev]"
pip install -e ".[plots]"
pip install -e ".[flow]"
```

## Quick Start

```python
import numpy as np
from dfi.explainers import Explainer

# Define your model
def model(X):
    return X.sum(axis=1)

# Create background data
X_background = np.random.randn(100, 10)

# Create an explainer
explainer = Explainer(model, data=X_background)

# Explain test instances
X_test = np.random.randn(10, 10)
# shap_values = explainer(X_test)  # Coming soon!
```

## API Reference

### Explainers

- `Explainer`: Base class for all explainers
- `TreeExplainer`: Optimized for tree-based models
- `LinearExplainer`: Optimized for linear models
- `KernelExplainer`: Model-agnostic explainer
- `OTExplainer`: Gaussian optimal-transport DFI (no cross-fitting)
- `EOTExplainer`: Entropic optimal-transport DFI (no cross-fitting)

### Confidence Intervals

All explainers return point estimates and can compute CIs post-hoc:

```python
results = explainer(X_test)
ci = explainer.conf_int(alpha=0.05, target="X", alternative="two-sided")
```

`conf_int` supports variance floors and practical margins:
```python
ci = explainer.conf_int(
    alpha=0.05,
    var_floor_method="fixed",
    var_floor_c=0.1,
    margin_method="mixture",
    alternative="two-sided",
)
```

### EOT Options

`EOTExplainer` supports adaptive epsilon, stochastic transport, and target choices:

```python
explainer = EOTExplainer(
    model.predict,
    X_background,
    auto_epsilon=True,
    stochastic_transport=True,
    n_transport_samples=10,
    target="gaussian",  # or "empirical"
)
```

### Plotting Functions

- `summary_plot`: Visualize feature importance across samples
- `waterfall_plot`: Show feature contributions for a single prediction
- `force_plot`: Interactive visualization of feature effects
- `dependence_plot`: Show feature value vs. importance relationship

### Utilities

- `validate_input`: Input validation and conversion
- `sample_background`: Sample background data
- `get_feature_names`: Generate or validate feature names
- `convert_to_link`: Apply link functions to predictions
- `check_additivity`: Verify SHAP additivity property

## Contributing

Contributions are welcome! Please see the repository for contribution guidelines.

## License

MIT License - see LICENSE file for details.

## References

<<<<<<< HEAD
dfi is inspired by:
=======
DFI is inspired by:
>>>>>>> 914762b723966c3192bac9ea445a716d8760dd38
- SHAP: https://github.com/slundberg/shap
- Disentangled feature importance methodology

## Contact

For questions and issues, please use the GitHub issue tracker:
https://github.com/jaydu1/dfi/issues
