"""
Basic Example: Using dfi Explainers

This example demonstrates how to use dfi explainers with a simple model.
"""

import numpy as np
from dfi.explainers import Explainer


def main():
    """Run a basic example."""
    # Create some dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = X_train[:, 0] + 2 * X_train[:, 1] + np.random.randn(100) * 0.1
    
    # Define a simple linear model
    def model(X):
        """Simple linear model: y = x0 + 2*x1"""
        return X[:, 0] + 2 * X[:, 1]
    
    # Create test data
    X_test = np.random.randn(10, 10)
    
    # Create an explainer
    explainer = Explainer(model, data=X_train)
    
    print("dfi Basic Example")
    print("=" * 50)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Model predictions: {model(X_test)}")
    print("\nNote: Full explainer implementations are coming soon!")
    print("This starter code provides the structure for dfi development.")
    
    # When explainers are fully implemented, you would use:
    # shap_values = explainer(X_test)
    # print(f"SHAP values shape: {shap_values.shape}")


if __name__ == "__main__":
    main()
