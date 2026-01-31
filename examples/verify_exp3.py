import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from fdfi.explainers import OTExplainer, EOTExplainer


def generate_exp3_data(n=1000, rho=0.8, seed=42):
    np.random.seed(seed)
    d = 50
    
    mean = np.zeros(d)
    cov = np.eye(d)
    cov[:10, :10] = rho  
    np.fill_diagonal(cov[:10, :10], 1.0)
    
    X = np.random.multivariate_normal(mean, cov, size=n)
    

    y = (np.arctan(X[:, 0] + X[:, 1]) * (X[:, 2] > 0) + 
         np.sin(X[:, 3] * X[:, 4]) * (X[:, 2] <= 0))
    y += np.random.normal(0, 0.1, size=n)
    return X, y


USE_EOT = os.getenv("FDFI_USE_EOT", "0").lower() in ("1", "true", "yes")

CI_ALPHA = 0.05
CI_TARGET = "X"
CI_ALTERNATIVE = "two-sided"
CI_VAR_FLOOR_METHOD = "fixed"  # or "mixture"
CI_VAR_FLOOR_C = 0.0
CI_MARGIN_METHOD = "fixed"  # or "mixture"

# Use default EOT settings via:
# FDFI_USE_EOT=1 python examples/verify_exp3.py

print("Step 1: Generating Exp3 simulation data...")
X_train, y_train = generate_exp3_data(n=2000, rho=0.8, seed=1)
X_test, _ = generate_exp3_data(n=1000, rho=0.8, seed=2)

print("Step 2: Fitting Black-box Model (Random Forest)...")
model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

explainer_name = "EOTExplainer" if USE_EOT else "OTExplainer"
print(f"Step 3: Running {explainer_name}...")

if USE_EOT:
    explainer = EOTExplainer(model.predict, X_train)
else:
    explainer = OTExplainer(model.predict, X_train, nsamples=50)

results = explainer(X_test)
phi_values = results['phi_X']
ci = explainer.conf_int(
    alpha=CI_ALPHA,
    target=CI_TARGET,
    alternative=CI_ALTERNATIVE,
    var_floor_method=CI_VAR_FLOOR_METHOD,
    var_floor_c=CI_VAR_FLOOR_C,
    margin_method=CI_MARGIN_METHOD,
)
ci_lower = ci["ci_lower"]
ci_upper = ci["ci_upper"]
ci_values = np.maximum(phi_values - ci_lower, ci_upper - phi_values)

plt.figure(figsize=(14, 8))
colors = ['#E45756']*5 + ['#1f77b4']*5 + ['#7f7f7f']*40

# Plot bars with styled error bars (thinner, more subtle)
bars = plt.bar(range(50), phi_values, yerr=ci_values, color=colors, capsize=2, 
               error_kw={'ecolor': 'black', 'elinewidth': 0.5, 'alpha': 0.5})

# Add reference lines
plt.axhline(0, color='black', lw=0.8, alpha=0.8)
plt.axvline(x=4.5, color='black', linestyle='--', alpha=0.5)
plt.axvline(x=9.5, color='black', linestyle='--', alpha=0.5)

# Set professional title with padding
plt.title("Exp3 Validation: Feature Importance with Uncertainty Quantification", 
          fontsize=15, fontweight='bold', pad=20)
plt.xlabel("Feature Index", fontsize=12)
plt.ylabel("Importance Score", fontsize=12)

# Set Y-axis limits with breathing room for labels and legend
max_y = np.max(phi_values + ci_values)
plt.ylim(-0.02, max_y * 1.45)

# Calculate label position (centered within each group)
label_y_pos = max_y * 1.25

# Add feature group labels with correct centering
plt.text(2, label_y_pos, "Active\n(0-4)", color='#E45756', fontweight='bold', ha='center', fontsize=11)
plt.text(7, label_y_pos, "Correlated Nulls\n(5-9)", color='#1f77b4', fontweight='bold', ha='center', fontsize=11)
plt.text(29.5, label_y_pos, "Independent Nulls\n(10-49)", color='#7f7f7f', fontweight='bold', ha='center', fontsize=11)

# Add grid for readability
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Create custom legend for feature groups and uncertainty
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ci_label = f"Uncertainty ({int((1 - CI_ALPHA) * 100)}% CI)"
legend_elements = [
    Patch(facecolor='#E45756', label='Active Features'),
    Patch(facecolor='#1f77b4', label='Correlated Nulls (Spurious)'),
    Patch(facecolor='#7f7f7f', label='Independent Nulls (Noise)'),
    Line2D([0], [0], color='black', lw=1, label=ci_label)
]

plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

# Adjust layout to prevent label clipping
plt.tight_layout()

print("\nResults Summary:")
print(f"Active Features Avg Score: {np.mean(phi_values[:5]):.4f}")
print(f"Correlated Nulls Avg Score: {np.mean(phi_values[5:10]):.4f}")
lower_ci = ci_lower[:10]
ci_pct = int((1 - CI_ALPHA) * 100)
print(f"First 10 features lower bound ({ci_pct}% CI) min: {np.min(lower_ci):.4f}")
print(f"First 10 features lower bound ({ci_pct}% CI) > 0: {np.all(lower_ci > 0)}")

plt.show()
