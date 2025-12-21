# FDFI - Flow-Disentangled Feature Importance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python library for computing feature importance using flow-disentangled methods, inspired by SHAP.

## Overview

FDFI (Flow-Disentangled Feature Importance) is a Python module that provides interpretable machine learning explanations through flow-disentangled feature importance methods. Similar to SHAP, FDFI helps you understand which features are driving your model's predictions.

## Features

- 🎯 **Multiple Explainer Types**: Tree, Linear, and Kernel explainers for different model types
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

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
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

# Explain test instances (full implementation coming soon!)
X_test = np.random.randn(10, 10)
# shap_values = explainer(X_test)
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
├── examples/              # Example scripts
│   ├── basic_example.py
│   └── tree_example.py
├── docs/                  # Documentation
│   ├── README.md
│   └── getting_started.md
├── pyproject.toml        # Package configuration
├── requirements.txt      # Core dependencies
└── README.md            # This file
```

## Development Status

🚧 **This is starter code for FDFI development.** The core structure and API are in place, but full implementations are coming soon.

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

FDFI is inspired by:
- [SHAP](https://github.com/slundberg/shap): A game theoretic approach to explain machine learning models
- Flow-disentangled feature importance methodology

## Citation

If you use FDFI in your research, please cite:

```bibtex
@software{fdfi2024,
  title={FDFI: Flow-Disentangled Feature Importance},
  author={FDFI Team},
  year={2024},
  url={https://github.com/jaydu1/FDFI}
}
```

## Contact

For questions and issues, please use the [GitHub issue tracker](https://github.com/jaydu1/FDFI/issues).
