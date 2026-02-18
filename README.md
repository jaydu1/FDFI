# FDFI - Flow-Disentangled Feature Importance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python library for computing feature importance using disentangled methods, inspired by SHAP.

Current release: `0.0.2`

## Overview

FDFI (Flow-Disentangled Feature Importance) is a Python module that provides interpretable machine learning explanations through disentangled feature importance methods. This package implements both DFI (Disentangled Feature Importance) and FDFI (Flow-DFI) methods. Similar to SHAP, FDFI helps you understand which features are driving your model's predictions.

## Features

- 🎯 **Multiple Explainer Types**: Tree, Linear, and Kernel explainers for different model types
- 🧭 **OT-Based DFI**: Gaussian OT (OTExplainer) and Entropic OT (EOTExplainer)
- 🌊 **Flow-DFI**: FlowExplainer with CPI and SCPI methods for non-Gaussian data
- 📊 **Rich Visualizations**: Summary, waterfall, force, and dependence plots
- 🔧 **Easy to Use**: Simple API similar to SHAP
- 🚀 **Extensible**: Built with modularity in mind for future enhancements

## Installation

### From Source

```bash
git clone https://github.com/jaydu1/FDFI.git
cd FDFI
pip install -e .
```

### Dependencies

Use `pyproject.toml` extras:

```bash
pip install -e ".[dev]"
pip install -e ".[plots]"
pip install -e ".[flow]"
```

## Quick Start

```python
import numpy as np
from fdfi.explainers import OTExplainer

# Define your model
def model(X):
    return X.sum(axis=1)

# Create background data
X_background = np.random.randn(100, 10)

# Create an explainer
explainer = OTExplainer(model, data=X_background, nsamples=50)

# Explain test instances
X_test = np.random.randn(10, 10)
results = explainer(X_test)

# Confidence intervals (post-hoc)
ci = explainer.conf_int(alpha=0.05, target="X", alternative="two-sided")
```

### CI Defaults in v0.0.2

By default, `conf_int()` now uses:

- `var_floor_method="mixture"`
- `margin_method="mixture"`

This improves stability for weak effects and avoids ad hoc thresholding in many use cases.
You can still override both methods explicitly if needed.

## EOT Options (Entropic OT)

`EOTExplainer` supports adaptive epsilon, stochastic transport sampling, and
Gaussian/empirical targets:

```python
from fdfi.explainers import EOTExplainer

explainer = EOTExplainer(
    model.predict,
    X_background,
    auto_epsilon=True,
    stochastic_transport=True,
    n_transport_samples=10,
    target="gaussian",  # or "empirical"
)
results = explainer(X_test)
```

## Flow-DFI with FlowExplainer

`FlowExplainer` uses normalizing flows for non-Gaussian data, supporting both CPI (Conditional Permutation Importance) and SCPI (Sobol-CPI):

- **CPI**: Average predictions first, then squared difference: $(Y - E[f(\tilde{X})])^2$
- **SCPI**: Squared differences first, then average: $E[(Y - f(\tilde{X}_b))^2]$

```python
from fdfi.explainers import FlowExplainer

# Create explainer with CPI (default)
explainer = FlowExplainer(
    model.predict,
    X_background,
    fit_flow=True,
    method='cpi',     # 'cpi', 'scpi', or 'both'
    num_steps=200,    # flow training steps
    nsamples=50,      # counterfactual samples
    sampling_method='resample',  # 'resample', 'permutation', 'normal', 'condperm'
)

results = explainer(X_test)
# results['phi_Z']: Z-space importance
# results['phi_X']: same as phi_Z (Z-space methods)

# Confidence intervals
ci = explainer.conf_int(alpha=0.05, target="Z", alternative="two-sided")
```

### Explainer diagnostics (new in v0.0.2)

Disentangled explainers (`OTExplainer`, `EOTExplainer`, and `FlowExplainer`) report two diagnostics with qualitative labels (GOOD / MODERATE / POOR) using consistent `[FDFI][DIAG]` logging:

- **Latent independence (median dCor)** — lower is better (thresholds: <0.10 good, <0.25 moderate).
- **Distribution fidelity (MMD)** — lower is better (thresholds: <0.05 good, <0.15 moderate).

Example log:

```
[FDFI][DIAG] Flow Model Diagnostics
[FDFI][DIAG] Latent independence (median dCor): 0.0421 [GOOD]  → lower is better
[FDFI][DIAG] Distribution fidelity (MMD):       0.0187 [GOOD]  → lower is better
```

Access diagnostics directly:

```python
diag = explainer.diagnostics
print(diag["latent_independence_median"], diag["latent_independence_label"])
print(diag["distribution_fidelity_mmd"], diag["distribution_fidelity_label"])
```

For advanced users, flow models can be trained separately:

```python
from fdfi.models import FlowMatchingModel

# Train flow model externally
flow_model = FlowMatchingModel(X_background, dim=X_background.shape[1])
flow_model.fit(num_steps=500, verbose='final')

# Set pre-trained flow
explainer = FlowExplainer(model.predict, X_background, fit_flow=False)
explainer.set_flow(flow_model)
```

## Project Structure

```
FDFI/
├── fdfi/                  # Main package directory
│   ├── __init__.py       # Package initialization
│   ├── explainers.py     # Explainer classes
│   ├── plots.py          # Visualization functions
│   └── utils.py          # Utility functions
├── tests/                 # Test suite
│   ├── test_explainers.py
│   ├── test_plots.py
│   └── test_utils.py
├── docs/                  # Documentation & tutorials
│   └── tutorials/        # Jupyter notebook tutorials
├── pyproject.toml        # Package configuration
└── README.md            # This file
```

## Development Status

🚧 **This is starter code for DFI development.** The core structure and API are in place, but full implementations are coming soon.

Current status:
- ✅ Package structure established
- ✅ Base classes and interfaces defined
- ✅ Testing framework set up
- ✅ Documentation structure created
- 🚧 Core algorithms (in development)
- 🚧 Visualization functions (in development)

## Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=fdfi --cov-report=html
```

## Documentation

Full documentation and tutorials are available in the `docs/` directory:
- [Quickstart Tutorial](docs/tutorials/quickstart.ipynb)
- [OT Explainer Tutorial](docs/tutorials/ot_explainer.ipynb)
- [EOT Explainer Tutorial](docs/tutorials/eot_explainer.ipynb)
- [Flow Explainer Tutorial](docs/tutorials/flow_explainer.ipynb)
- [Confidence Intervals](docs/tutorials/confidence_intervals.ipynb)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

FDFI is based on:
- Du, J.-H., Roeder, K., & Wasserman, L. (2025). Disentangled Feature Importance. *arXiv preprint arXiv:2507.00260*.
- Chen, X., Guo, Y., & Du, J.-H. (2026). Flow-Disentangled Feature Importance. In *The Thirteenth International Conference on Learning Representations (ICLR)*.

Related work:
- [SHAP](https://github.com/slundberg/shap): A game theoretic approach to explain machine learning models

## Citation

If you use DFI in your research, please cite:

```bibtex
@software{dfi2026,
  title={DFI: Python Library for Disentangled Feature Importance},
  author={DFI Team},
  year={2026},
  url={https://github.com/jaydu1/FDFI}
}

@article{du2025disentangled,
  title={Disentangled Feature Importance},
  author={Du, Jin-Hong and Roeder, Kathryn and Wasserman, Larry},
  journal={arXiv preprint arXiv:2507.00260},
  year={2025}
}

@inproceedings{chen2026flow,
  title={Flow-Disentangled Feature Importance},
  author={Chen, Xin and Guo, Yifan and Du, Jin-Hong},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2026}
}
```

## Contact

For questions and issues, please use the [GitHub issue tracker](https://github.com/jaydu1/FDFI/issues).
