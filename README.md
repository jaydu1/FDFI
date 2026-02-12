# dfi - Flow-Disentangled Feature Importance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python library for computing feature importance using disentangled methods, inspired by SHAP.

## Overview

dfi (Flow-Disentangled Feature Importance) is a Python module that provides interpretable machine learning explanations through flow-disentangled feature importance methods. Similar to SHAP, dfi helps you understand which features are driving your model's predictions.

## Features

- 🎯 **Multiple Explainer Types**: Tree, Linear, and Kernel explainers for different model types
- 🧭 **OT-Based DFI**: Gaussian OT (OTExplainer) and Entropic OT (EOTExplainer)
- 📊 **Rich Visualizations**: Summary, waterfall, force, and dependence plots
- 🔧 **Easy to Use**: Simple API similar to SHAP
- 🚀 **Extensible**: Built with modularity in mind for future enhancements

## Installation

### From Source

```bash
git clone https://github.com/jaydu1/dfi.git
cd dfi
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
from dfi.explainers import OTExplainer

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

## EOT Options (Entropic OT)

`EOTExplainer` supports adaptive epsilon, stochastic transport sampling, and
Gaussian/empirical targets:

```python
from dfi.explainers import EOTExplainer

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

## Project Structure

```
dfi/
├── dfi/                  # Main package directory
│   ├── __init__.py       # Package initialization
│   ├── explainers.py     # Explainer classes
│   ├── plots.py          # Visualization functions
│   └── utils.py          # Utility functions
├── tests/                 # Test suite
│   ├── test_explainers.py
│   ├── test_plots.py
│   └── test_utils.py
├── examples/              # Example scripts
│   ├── basic_example.py
│   └── tree_example.py
├── docs/                  # Documentation
│   ├── README.md
│   └── getting_started.md
├── pyproject.toml        # Package configuration
└── README.md            # This file
```

## Development Status

🚧 **This is starter code for dfi development.** The core structure and API are in place, but full implementations are coming soon.

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
pytest --cov=dfi --cov-report=html
```

## Examples

See the `examples/` directory for usage examples:

```bash
python examples/basic_example.py
python examples/tree_example.py
```

## Documentation

Full documentation is available in the `docs/` directory:
- [Getting Started](docs/getting_started.md)
- [API Reference](docs/README.md)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

dfi is inspired by:
- [SHAP](https://github.com/slundberg/shap): A game theoretic approach to explain machine learning models

## Citation

If you use dfi in your research, please cite:

```bibtex
@software{dfi2024,
  title={dfi: Flow-Disentangled Feature Importance},
  author={dfi Team},
  year={2024},
  url={https://github.com/jaydu1/dfi}
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

For questions and issues, please use the [GitHub issue tracker](https://github.com/jaydu1/dfi/issues).
