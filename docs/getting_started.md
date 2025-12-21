# Getting Started with FDFI

## Installation

```bash
pip install -e .
```

## Basic Usage

### 1. Create an Explainer

```python
from fdfi.explainers import Explainer

def my_model(X):
    # Your model's prediction function
    return predictions

explainer = Explainer(my_model)
```

### 2. Compute Feature Importance

```python
import numpy as np

X_test = np.random.randn(10, 5)  # Test data
# shap_values = explainer(X_test)  # Coming soon!
```

### 3. Visualize Results

```python
from fdfi.plots import summary_plot

# summary_plot(shap_values, X_test)  # Coming soon!
```

## Explainer Types

### TreeExplainer

For tree-based models (Random Forest, XGBoost, LightGBM):

```python
from fdfi.explainers import TreeExplainer
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

explainer = TreeExplainer(model)
```

### LinearExplainer

For linear models:

```python
from fdfi.explainers import LinearExplainer
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

explainer = LinearExplainer(model)
```

### KernelExplainer

Model-agnostic explainer (works with any model):

```python
from fdfi.explainers import KernelExplainer

explainer = KernelExplainer(model.predict, X_background)
```

## Next Steps

- See `examples/` directory for complete examples
- Read the API documentation for detailed usage
- Check out the methodology documentation for theoretical background
