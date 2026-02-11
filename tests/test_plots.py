"""
Tests for dfi plotting functions.
"""

import pytest
import numpy as np
from dfi.plots import (
    summary_plot,
    waterfall_plot,
    force_plot,
    dependence_plot
)


class TestPlots:
    """Test plotting functions."""
    
    def test_summary_plot_not_implemented(self):
        """Test that summary_plot raises NotImplementedError."""
        shap_values = np.random.randn(100, 10)
        with pytest.raises(NotImplementedError):
            summary_plot(shap_values)
    
    def test_waterfall_plot_not_implemented(self):
        """Test that waterfall_plot raises NotImplementedError."""
        shap_values = np.random.randn(10)
        with pytest.raises(NotImplementedError):
            waterfall_plot(shap_values)
    
    def test_force_plot_not_implemented(self):
        """Test that force_plot raises NotImplementedError."""
        shap_values = np.random.randn(10)
        with pytest.raises(NotImplementedError):
            force_plot(0.5, shap_values)
    
    def test_dependence_plot_not_implemented(self):
        """Test that dependence_plot raises NotImplementedError."""
        shap_values = np.random.randn(100, 10)
        features = np.random.randn(100, 10)
        with pytest.raises(NotImplementedError):
            dependence_plot(0, shap_values, features)
