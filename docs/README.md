# FDFI Documentation

Welcome to the FDFI (Flow-Disentangled Feature Importance) documentation.

## Overview

FDFI is a Python library for computing feature importance using flow-disentangled methods, inspired by SHAP (SHapley Additive exPlanations).

## Installation

### From Source

```bash
git clone https://github.com/jaydu1/FDFI.git
cd FDFI
pip install -e .
```

### Dependencies

Core dependencies:
- numpy >= 1.20.0
- scipy >= 1.7.0

Optional dependencies:
- matplotlib >= 3.5.0 (for plotting)
- seaborn >= 0.12.0 (for advanced plots)

Development dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from fdfi.explainers import Explainer

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

FDFI is inspired by:
- SHAP: https://github.com/slundberg/shap
- Flow-disentangled feature importance methodology

## Contact

For questions and issues, please use the GitHub issue tracker:
https://github.com/jaydu1/FDFI/issues
