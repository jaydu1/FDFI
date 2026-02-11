# Getting Started with dfi

## Installation

```bash
pip install -e .
```

## Basic Usage

### 1. Create an Explainer

```python
from dfi.explainers import Explainer

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
from dfi.plots import summary_plot

# summary_plot(shap_values, X_test)  # Coming soon!
```

## Explainer Types

### TreeExplainer

For tree-based models (Random Forest, XGBoost, LightGBM):

```python
from dfi.explainers import TreeExplainer
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

explainer = TreeExplainer(model)
```

### LinearExplainer

For linear models:

```python
from dfi.explainers import LinearExplainer
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

explainer = LinearExplainer(model)
```

### KernelExplainer

Model-agnostic explainer (works with any model):

```python
from dfi.explainers import KernelExplainer

explainer = KernelExplainer(model.predict, X_background)
```

### OTExplainer (Gaussian OT)

Gaussian optimal-transport DFI (no cross-fitting):

```python
from dfi.explainers import OTExplainer

explainer = OTExplainer(model.predict, X_background, nsamples=50)
results = explainer(X_test)
```

### EOTExplainer (Entropic OT)

Entropic OT DFI using a learned transport kernel:

```python
from dfi.explainers import EOTExplainer

explainer = EOTExplainer(model.predict, X_background, epsilon=0.1, nsamples=50)
results = explainer(X_test)
```

#### EOT Options

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

### Confidence Intervals

All explainers support post-hoc CIs via `conf_int`:

```python
results = explainer(X_test)
ci = explainer.conf_int(alpha=0.05, target="X", alternative="two-sided")
```

## Next Steps

- See `examples/` directory for complete examples
- Read the API documentation for detailed usage
- Check out the methodology documentation for theoretical background
