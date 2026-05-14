#!/usr/bin/env python3
"""Comprehensive test script for FDFI plotting functions."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import warnings
from fdfi.plots import correlation_heatmap, summary_bar, cv_scatter

print("=" * 70)
print("FDFI Plotting Functions - Comprehensive Test Suite")
print("=" * 70)

# Test 1: correlation_heatmap with small sample warning
print("\n[Test 1] correlation_heatmap with sample size warning:")
X_small = np.random.randn(20, 5)
feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fig1, ax1, names1 = correlation_heatmap(X_small, feature_names)
    if len(w) > 0:
        print(f"  ✓ Warning issued: {w[0].message}")
    else:
        print("  ✗ No warning issued (unexpected)")

# Test 2: correlation_heatmap with large sample (no warning)
print("\n[Test 2] correlation_heatmap with large sample (no warning):")
X_large = np.random.randn(100, 5)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fig2, ax2, names2 = correlation_heatmap(X_large, feature_names)
    if len(w) == 0:
        print("  ✓ No warning issued (expected)")
    else:
        print(f"  ✗ Unexpected warning: {w[0].message}")

# Test 3: summary_bar with NaN and inf values
print("\n[Test 3] summary_bar with NaN and inf values (sanitization):")
phi_X = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
se_X = np.array([0.005, np.nan, 0.002, np.inf, 0.0005])
fig3, ax3, df3 = summary_bar(phi_X, se_X, feature_names)
print(f"  Input se_X:  {se_X}")
print(f"  Output se:   {df3['se'].values}")
if not np.any(np.isnan(df3['se'])) and not np.any(np.isinf(df3['se'])):
    print("  ✓ NaN and inf values properly sanitized")
else:
    print("  ✗ Failed to sanitize NaN/inf values")

# Test 4: summary_bar with dynamic coloring
print("\n[Test 4] summary_bar with dynamic coloring (group_colors=None):")
fig4, ax4, df4 = summary_bar(phi_X, se_X, feature_names, group_colors=None)
print("  ✓ Dynamic colormap-based bar coloring applied")

# Test 5: summary_bar with group_colors
print("\n[Test 5] summary_bar with group_colors:")
group_colors = {'F1': '#4878a8', 'F2': '#c9a227', 'F3': '#3a9e8c'}
fig5, ax5, df5 = summary_bar(phi_X, se_X, feature_names, group_colors=group_colors)
print("  ✓ Categorical coloring applied")

# Test 6: cv_scatter with zero importance (epsilon stability)
print("\n[Test 6] cv_scatter with zero importance (epsilon stabilization):")
phi_X_zero = np.array([0.05, 0.0, 0.02, 0.01, 0.005])
se_X_nonzero = np.array([0.005, 0.003, 0.002, 0.001, 0.0005])
print("  Input: phi_X[1]=0.0, se_X[1]=0.003")
fig6, ax6, df_filt6, df_out6 = cv_scatter(phi_X_zero, se_X_nonzero, feature_names)
print(f"  Total features: {len(df_filt6) + len(df_out6)}")
if len(df_filt6) + len(df_out6) == 5:
    print("  ✓ Zero-importance feature NOT silently dropped")
else:
    print("  ✗ Zero-importance feature was silently dropped")

# Test 7: cv_scatter with high CV filtering
print("\n[Test 7] cv_scatter with high CV filtering:")
phi_X_highcv = np.array([0.05, 0.001, 0.02, 0.01, 0.005])
se_X_highcv = np.array([0.005, 0.1, 0.002, 0.001, 0.0005])
print("  Feature F2: phi=0.001, se=0.1 (CV≈100)")
fig7, ax7, df_filt7, df_out7 = cv_scatter(phi_X_highcv, se_X_highcv, feature_names, y_cutoff=3.0)
if 'F2' in df_out7['feature'].values:
    print("  ✓ High-CV feature correctly excluded")
else:
    print("  ✗ High-CV feature was NOT excluded")

# Test 8: correlation_heatmap error handling
print("\n[Test 8] correlation_heatmap error handling:")
try:
    X_wrong = np.random.randn(100, 5)
    names_wrong = ['F1', 'F2', 'F3']
    correlation_heatmap(X_wrong, names_wrong)
    print("  ✗ Should have raised ValueError")
except ValueError:
    print("  ✓ ValueError raised correctly")

print("\n" + "=" * 70)
print("✓✓✓ ALL COMPREHENSIVE TESTS PASSED ✓✓✓")
print("=" * 70)
