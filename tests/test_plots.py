"""
Tests for dfi plotting functions.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from fdfi.plots import (
    correlation_heatmap,
    summary_bar,
    cv_scatter
)


class TestCorrelationHeatmap:
    """Test correlation_heatmap function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with valid inputs."""
        np.random.seed(42)
        X_background = np.random.randn(100, 5)
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        
        fig, ax, names_reord = correlation_heatmap(X_background, feature_names)
        
        assert fig is not None
        assert ax is not None
        assert len(names_reord) == 5
        assert set(names_reord) == set(feature_names)
        plt.close(fig)
    
    def test_sample_size_warning(self):
        """Test that warning is issued for small sample sizes."""
        X_background = np.random.randn(20, 5)
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax, names_reord = correlation_heatmap(X_background, feature_names)
            
            assert len(w) == 1
            assert "sample size" in str(w[0].message).lower()
            plt.close(fig)
    
    def test_no_warning_for_large_sample(self):
        """Test that no warning is issued for large sample sizes."""
        X_background = np.random.randn(100, 5)
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fig, ax, names_reord = correlation_heatmap(X_background, feature_names)
            
            assert len(w) == 0
            plt.close(fig)
    
    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises ValueError."""
        X_background = np.random.randn(100, 5)
        feature_names = ['F1', 'F2', 'F3']  # Wrong length
        
        with pytest.raises(ValueError):
            correlation_heatmap(X_background, feature_names)
    
    def test_invalid_shape_error(self):
        """Test that 1D array raises ValueError."""
        X_background = np.random.randn(100)
        feature_names = ['F1']
        
        with pytest.raises(ValueError):
            correlation_heatmap(X_background, feature_names)
    
    def test_savepath_creates_file(self, tmp_path):
        """Test that savepath parameter saves the figure."""
        X_background = np.random.randn(100, 5)
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        savefile = tmp_path / "test_corr.pdf"
        
        fig, ax, names_reord = correlation_heatmap(
            X_background, feature_names, savepath=str(savefile)
        )
        
        assert savefile.exists()
        plt.close(fig)


class TestSummaryBar:
    """Test summary_bar function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with valid inputs."""
        np.random.seed(42)
        phi_X = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
        se_X = np.array([0.005, 0.003, 0.002, 0.001, 0.0005])
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        
        fig, ax, df = summary_bar(phi_X, se_X, feature_names)
        
        assert fig is not None
        assert ax is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ['feature', 'phi', 'se']
        assert df['phi'].iloc[0] >= df['phi'].iloc[-1]  # Sorted descending
        plt.close(fig)
    
    def test_group_colors(self):
        """Test with group_colors parameter."""
        phi_X = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
        se_X = np.array([0.005, 0.003, 0.002, 0.001, 0.0005])
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        group_colors = {'F1': '#4878a8', 'F2': '#c9a227', 'F3': '#3a9e8c'}
        
        fig, ax, df = summary_bar(phi_X, se_X, feature_names, group_colors=group_colors)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_nan_handling(self):
        """Test that NaN values in se_X are handled gracefully."""
        phi_X = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
        se_X = np.array([0.005, np.nan, 0.002, np.inf, 0.0005])
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        
        fig, ax, df = summary_bar(phi_X, se_X, feature_names)
        
        assert fig is not None
        assert ax is not None
        assert not np.any(np.isnan(df['se']))
        assert not np.any(np.isinf(df['se']))
        plt.close(fig)
    
    def test_dynamic_coloring(self):
        """Test dynamic coloring when group_colors is None."""
        phi_X = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
        se_X = np.array([0.005, 0.003, 0.002, 0.001, 0.0005])
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        
        fig, ax, df = summary_bar(phi_X, se_X, feature_names, group_colors=None)
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_savepath_creates_file(self, tmp_path):
        """Test that savepath parameter saves the figure."""
        phi_X = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
        se_X = np.array([0.005, 0.003, 0.002, 0.001, 0.0005])
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        savefile = tmp_path / "test_bar.pdf"
        
        fig, ax, df = summary_bar(phi_X, se_X, feature_names, savepath=str(savefile))
        
        assert savefile.exists()
        plt.close(fig)


class TestCVScatter:
    """Test cv_scatter function."""
    
    def test_basic_functionality(self):
        """Test basic functionality with valid inputs."""
        np.random.seed(42)
        phi_X = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
        se_X = np.array([0.005, 0.003, 0.002, 0.001, 0.0005])
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        
        fig, ax, df_filt, df_out = cv_scatter(phi_X, se_X, feature_names)
        
        assert fig is not None
        assert ax is not None
        assert isinstance(df_filt, pd.DataFrame)
        assert isinstance(df_out, pd.DataFrame)
        assert 'cv' in df_filt.columns
        assert len(df_filt) + len(df_out) == 5
        plt.close(fig)
    
    def test_zero_importance_handling(self):
        """Test that zero-importance features are handled correctly."""
        phi_X = np.array([0.05, 0.0, 0.02, 0.01, 0.005])
        se_X = np.array([0.005, 0.003, 0.002, 0.001, 0.0005])
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        
        fig, ax, df_filt, df_out = cv_scatter(phi_X, se_X, feature_names)
        
        # Zero-importance feature should not be silently dropped
        assert len(df_filt) + len(df_out) == 5
        plt.close(fig)
    
    def test_high_cv_filtering(self):
        """Test that high-CV features are filtered correctly."""
        phi_X = np.array([0.05, 0.001, 0.02, 0.01, 0.005])
        se_X = np.array([0.005, 0.1, 0.002, 0.001, 0.0005])  # F2 has high CV
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        
        fig, ax, df_filt, df_out = cv_scatter(phi_X, se_X, feature_names, y_cutoff=3.0)
        
        # F2 should be excluded (high CV)
        assert 'F2' not in df_filt['feature'].values
        assert 'F2' in df_out['feature'].values
        plt.close(fig)
    
    def test_group_colors(self):
        """Test with group_colors parameter."""
        phi_X = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
        se_X = np.array([0.005, 0.003, 0.002, 0.001, 0.0005])
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        group_colors = {'F1': '#4878a8', 'F2': '#c9a227'}
        
        fig, ax, df_filt, df_out = cv_scatter(
            phi_X, se_X, feature_names, group_colors=group_colors
        )
        
        assert fig is not None
        assert ax is not None
        plt.close(fig)
    
    def test_savepath_creates_file(self, tmp_path):
        """Test that savepath parameter saves the figure."""
        phi_X = np.array([0.05, 0.03, 0.02, 0.01, 0.005])
        se_X = np.array([0.005, 0.003, 0.002, 0.001, 0.0005])
        feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
        savefile = tmp_path / "test_cv.pdf"
        
        fig, ax, df_filt, df_out = cv_scatter(
            phi_X, se_X, feature_names, savepath=str(savefile)
        )
        
        assert savefile.exists()
        plt.close(fig)
