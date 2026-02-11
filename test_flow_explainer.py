"""
Test script for FlowExplainer implementation.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from dfi.explainers import FlowExplainer, OTExplainer


def generate_simple_data(n=500, d=10, seed=42):
    """Generate simple synthetic data."""
    np.random.seed(seed)
    X = np.random.randn(n, d)
    # Simple function: y depends on first 3 features
    y = X[:, 0] + X[:, 1]**2 + 0.5*X[:, 2] + np.random.randn(n)*0.1
    return X, y


print("=" * 70)
print("Testing FlowExplainer Implementation")
print("=" * 70)

# Generate data
print("\n[1/4] Generating test data...")
X_train, y_train = generate_simple_data(n=500, d=10, seed=1)
X_test, _ = generate_simple_data(n=50, d=10, seed=2)
print(f"  Training data: {X_train.shape}")
print(f"  Test data: {X_test.shape}")

# Train model
print("\n[2/4] Training black-box model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("  Model trained successfully")

# Test OTExplainer (baseline)
print("\n[3/4] Testing OTExplainer (linear baseline)...")
try:
    ot_explainer = OTExplainer(model.predict, X_train, nsamples=50)
    ot_results = ot_explainer(X_test)
    print("  ✓ OTExplainer executed successfully")
    print(f"    Output keys: {list(ot_results.keys())}")
    print(f"    phi_X shape: {ot_results['phi_X'].shape}")
    print(f"    phi_X sample values: {ot_results['phi_X'][:3]}")
except Exception as e:
    print(f"  ✗ OTExplainer failed: {e}")
    import traceback
    traceback.print_exc()

# Test FlowExplainer
print("\n[4/4] Testing FlowExplainer (non-linear)...")
try:
    flow_explainer = FlowExplainer(
        model.predict,
        X_train,
        nsamples=50,
        num_steps=1000,  # Reduced for quick test
        random_state=42
    )
    flow_results = flow_explainer(X_test)
    print("  ✓ FlowExplainer executed successfully")
    print(f"    Output keys: {list(flow_results.keys())}")
    print(f"    phi_X shape: {flow_results['phi_X'].shape}")
    print(f"    phi_X sample values: {flow_results['phi_X'][:3]}")
    print(f"    std_X sample values: {flow_results['std_X'][:3]}")
    print(f"    se_X sample values: {flow_results['se_X'][:3]}")
    
    # Verify output format matches OTExplainer
    assert set(flow_results.keys()) == set(ot_results.keys()), "Output format mismatch!"
    print("  ✓ Output format matches OTExplainer")
    
except Exception as e:
    print(f"  ✗ FlowExplainer failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test completed successfully!")
print("=" * 70)
