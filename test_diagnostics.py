"""
Test script for Flow Model Diagnostics.

This script demonstrates the diagnostic suite for evaluating:
1. Latent independence (Distance Correlation)
2. Distribution fidelity (MMD)
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from dfi.explainers import FlowExplainer, OTExplainer


def main():
    print("\n" + "=" * 70)
    print("Testing Flow Model Diagnostics Suite")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(42)
    n_train = 200
    n_test = 50
    n_features = 8
    
    X_train = np.random.randn(n_train, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] + 0.5 * X_train[:, 2] + 
               0.1 * np.sum(X_train[:, 3:], axis=1) + 0.1 * np.random.randn(n_train))
    
    X_test = np.random.randn(n_test, n_features)
    
    # Train model
    print("\n[Step 1] Training black-box model...")
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    # Wrap model in a callable class (required by explainers)
    class CallableModel:
        def __init__(self, model):
            self.model = model
        def __call__(self, X):
            return self.model.predict(X)
    model = CallableModel(rf_model)
    print("✓ Model trained")
    
    # Initialize FlowExplainer (will compute diagnostics)
    print("\n[Step 2] Initializing FlowExplainer with diagnostics...")
    print("-" * 70)
    flow_explainer = FlowExplainer(
        model=model,
        data=X_train,
        nsamples=50,
        num_steps=500,  # Reduced for faster testing
        random_state=42
    )
    print("-" * 70)
    print("✓ FlowExplainer initialized")
    
    # Access diagnostics
    print("\n[Step 3] Diagnostic Metrics:")
    print("-" * 70)
    diagnostics = flow_explainer.diagnostics
    print(f"Latent Independence (Median dCor):  {diagnostics['latent_independence_median']:.6f}")
    print(f"  Full correlation matrix shape:    {diagnostics['latent_independence_dcor'].shape}")
    print(f"Distribution Fidelity (MMD):       {diagnostics['distribution_fidelity_mmd']:.6f}")
    print("-" * 70)
    
    # Interpretation
    print("\n[Step 4] Diagnostic Interpretation:")
    print("-" * 70)
    median_dcor = diagnostics['latent_independence_median']
    mmd = diagnostics['distribution_fidelity_mmd']
    
    if median_dcor < 0.1:
        print(f"✓ Latent Independence: EXCELLENT (dCor = {median_dcor:.6f})")
        print("  → Latent dimensions are statistically independent")
    elif median_dcor < 0.3:
        print(f"◐ Latent Independence: GOOD (dCor = {median_dcor:.6f})")
        print("  → Latent dimensions show moderate independence")
    else:
        print(f"✗ Latent Independence: POOR (dCor = {median_dcor:.6f})")
        print("  → Latent dimensions are coupled")
    
    print()
    if mmd < 0.1:
        print(f"✓ Distribution Fidelity: EXCELLENT (MMD = {mmd:.6f})")
        print("  → Flow model accurately captures data distribution")
    elif mmd < 0.3:
        print(f"◐ Distribution Fidelity: GOOD (MMD = {mmd:.6f})")
        print("  → Flow model captures distribution reasonably well")
    else:
        print(f"✗ Distribution Fidelity: POOR (MMD = {mmd:.6f})")
        print("  → Flow model may need more training steps")
    print("-" * 70)
    
    # Compare with OTExplainer
    print("\n[Step 5] Running OTExplainer for baseline comparison...")
    ot_explainer = OTExplainer(model=model, data=X_train, random_state=42)
    print("✓ OTExplainer initialized (no diagnostics - uses linear Gaussian)")
    
    # Get explanations
    print("\n[Step 6] Computing feature importance on test set...")
    flow_results = flow_explainer(X_test)
    ot_results = ot_explainer(X_test)
    
    print(f"\nFlowExplainer phi_X shape: {flow_results['phi_X'].shape}")
    print(f"OTExplainer phi_X shape:   {ot_results['phi_X'].shape}")
    
    # Show top features
    top_k = 3
    flow_top = np.argsort(-flow_results['phi_X'])[:top_k]
    ot_top = np.argsort(-ot_results['phi_X'])[:top_k]
    
    print(f"\nTop {top_k} features (Flow): {flow_top}")
    print(f"Top {top_k} features (OT):   {ot_top}")
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
