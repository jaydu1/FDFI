# FDFI Examples

This directory contains example scripts demonstrating how to use FDFI.

## Available Examples

### basic_example.py
Basic usage of FDFI explainers with a simple linear model.

```bash
python basic_example.py
```

### tree_example.py
Template for using FDFI with tree-based models (Random Forests, XGBoost, etc.).

```bash
python tree_example.py
```

## Running Examples

Make sure FDFI is installed before running examples:

```bash
# Install in development mode
pip install -e .

# Run an example
python examples/basic_example.py
```

## Note

These examples use placeholder implementations. Full feature implementations are coming soon based on the flow-disentangled feature importance methodology.
