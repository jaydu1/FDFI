import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from fdfi.explainers import DFIExplainer  


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


print("Step 1: Generating Exp3 simulation data...")
X_train, y_train = generate_exp3_data(n=2000, rho=0.8, seed=1)
X_test, _ = generate_exp3_data(n=1000, rho=0.8, seed=2)

print("Step 2: Fitting Black-box Model (Random Forest)...")
model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

print("Step 3: Running your DFIExplainer...")

explainer = DFIExplainer(model.predict, X_train, nsamples=50)

results = explainer(X_test)
phi_values = results['phi_X']
std_values = results['std_X']

plt.figure(figsize=(14, 8))
colors = ['#E45756']*5 + ['#1f77b4']*5 + ['#7f7f7f']*40

# Plot bars with styled error bars (thinner, more subtle)
bars = plt.bar(range(50), phi_values, yerr=std_values, color=colors, capsize=2, 
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
max_y = np.max(phi_values + std_values)
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

legend_elements = [
    Patch(facecolor='#E45756', label='Active Features'),
    Patch(facecolor='#1f77b4', label='Correlated Nulls (Spurious)'),
    Patch(facecolor='#7f7f7f', label='Independent Nulls (Noise)'),
    Line2D([0], [0], color='black', lw=1, label='Uncertainty (±1 SD)')
]

plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.95)

# Adjust layout to prevent label clipping
plt.tight_layout()

print("\nResults Summary:")
print(f"Active Features Avg Score: {np.mean(phi_values[:5]):.4f}")
print(f"Correlated Nulls Avg Score: {np.mean(phi_values[5:10]):.4f}")

plt.show()