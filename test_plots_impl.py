#!/usr/bin/env python3
"""Quick test script for the three new plotting functions."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from fdfi.plots import correlation_heatmap, summary_bar, cv_scatter

def test_plotting_functions():
    """Test all three plotting functions."""
    
    print("=" * 60)
    print("FDFI Plotting Functions Test Suite")
    print("=" * 60)
    
    # Create test data
    print("\n1. Creating test data...")
    np.random.seed(42)
    X = np.random.randn(50, 5)
    feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
    phi_X = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
    se_X = np.array([0.005, 0.003, 0.002, 0.001, 0.0005])
    group_colors = {'F1': '#4878a8', 'F2': '#4878a8', 'F3': '#c9a227', 
                    'F4': '#3a9e8c', 'F5': '#c9a227'}
    
    print(f"   ✓ X shape: {X.shape}")
    print(f"   ✓ phi_X shape: {phi_X.shape}")
    print(f"   ✓ Feature names: {feature_names}")
    
    # Test 1: correlation_heatmap
    print("\n2. Testing correlation_heatmap()...")
    try:
        fig, ax, names_reord = correlation_heatmap(
            X, feature_names, 
            figsize=(8, 6),
            savepath='/tmp/test_corr.pdf'
        )
        print("   ✓ Function executed successfully")
        print(f"   ✓ Returned figure: {type(fig).__name__}")
        print(f"   ✓ Returned axes: {type(ax).__name__}")
        print(f"   ✓ Reordered features: {names_reord}")
        plt.close(fig)
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 2: summary_bar
    print("\n3. Testing summary_bar()...")
    try:
        fig, ax, df = summary_bar(
            phi_X, se_X, feature_names,
            group_colors=group_colors,
            figsize=(10, 5),
            savepath='/tmp/test_bar.pdf'
        )
        print("   ✓ Function executed successfully")
        print(f"   ✓ Returned figure: {type(fig).__name__}")
        print(f"   ✓ Returned axes: {type(ax).__name__}")
        print(f"   ✓ Returned DataFrame shape: {df.shape}")
        print(f"   ✓ Top feature: {df.iloc[0]['feature']} (phi={df.iloc[0]['phi']:.4f})")
        plt.close(fig)
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 3: cv_scatter
    print("\n4. Testing cv_scatter()...")
    try:
        fig, ax, df_filt, df_out = cv_scatter(
            phi_X, se_X, feature_names,
            y_cutoff=3.0,
            group_colors=group_colors,
            figsize=(10, 4),
            savepath='/tmp/test_cv.pdf'
        )
        print("   ✓ Function executed successfully")
        print(f"   ✓ Returned figure: {type(fig).__name__}")
        print(f"   ✓ Returned axes: {type(ax).__name__}")
        print(f"   ✓ Filtered features: {len(df_filt)}")
        print(f"   ✓ Outlier features: {len(df_out)}")
        plt.close(fig)
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    # Test 4: Verify docstrings
    print("\n5. Verifying docstrings...")
    functions = [
        ('correlation_heatmap', correlation_heatmap),
        ('summary_bar', summary_bar),
        ('cv_scatter', cv_scatter)
    ]
    
    for name, func in functions:
        doc = func.__doc__
        has_params = 'Parameters' in doc
        has_returns = 'Returns' in doc
        has_examples = 'Examples' in doc
        has_notes = 'Notes' in doc
        
        print(f"\n   {name}:")
        print(f"      ✓ Docstring length: {len(doc)} chars")
        print(f"      ✓ Parameters: {has_params}")
        print(f"      ✓ Returns: {has_returns}")
        print(f"      ✓ Notes: {has_notes}")
        print(f"      ✓ Examples: {has_examples}")
        
        if not (has_params and has_returns and has_examples):
            print("      ✗ Missing required sections!")
            return False
    
    print("\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = test_plotting_functions()
    sys.exit(0 if success else 1)
