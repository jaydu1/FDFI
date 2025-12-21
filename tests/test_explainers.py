"""
Tests for FDFI explainers.
"""

import pytest
import numpy as np
from fdfi.explainers import Explainer, TreeExplainer, LinearExplainer, KernelExplainer


class TestExplainer:
    """Test the base Explainer class."""
    
    def test_init(self):
        """Test explainer initialization."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        explainer = Explainer(dummy_model)
        assert explainer.model == dummy_model
        assert explainer.data is None
    
    def test_init_with_data(self):
        """Test explainer initialization with background data."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        data = np.random.randn(100, 10)
        explainer = Explainer(dummy_model, data=data)
        assert explainer.model == dummy_model
        assert np.array_equal(explainer.data, data)
    
    def test_call_not_implemented(self):
        """Test that calling base explainer raises NotImplementedError."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        explainer = Explainer(dummy_model)
        X = np.random.randn(10, 5)
        
        with pytest.raises(NotImplementedError):
            explainer(X)
    
    def test_shap_values_calls_call(self):
        """Test that shap_values is an alias for __call__."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        explainer = Explainer(dummy_model)
        X = np.random.randn(10, 5)
        
        # Both should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            explainer.shap_values(X)


class TestTreeExplainer:
    """Test the TreeExplainer class."""
    
    def test_init(self):
        """Test TreeExplainer initialization."""
        # Mock tree model
        model = object()
        explainer = TreeExplainer(model)
        assert explainer.model == model
    
    def test_call_not_implemented(self):
        """Test that TreeExplainer raises NotImplementedError."""
        model = object()
        explainer = TreeExplainer(model)
        X = np.random.randn(10, 5)
        
        with pytest.raises(NotImplementedError):
            explainer(X)


class TestLinearExplainer:
    """Test the LinearExplainer class."""
    
    def test_init(self):
        """Test LinearExplainer initialization."""
        model = object()
        explainer = LinearExplainer(model)
        assert explainer.model == model
    
    def test_call_not_implemented(self):
        """Test that LinearExplainer raises NotImplementedError."""
        model = object()
        explainer = LinearExplainer(model)
        X = np.random.randn(10, 5)
        
        with pytest.raises(NotImplementedError):
            explainer(X)


class TestKernelExplainer:
    """Test the KernelExplainer class."""
    
    def test_init_requires_data(self):
        """Test that KernelExplainer requires background data."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        with pytest.raises(ValueError, match="requires background data"):
            KernelExplainer(dummy_model, None)
    
    def test_init_with_data(self):
        """Test KernelExplainer initialization with data."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        data = np.random.randn(100, 10)
        explainer = KernelExplainer(dummy_model, data)
        assert explainer.model == dummy_model
        assert np.array_equal(explainer.data, data)
    
    def test_call_not_implemented(self):
        """Test that KernelExplainer raises NotImplementedError."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        data = np.random.randn(100, 10)
        explainer = KernelExplainer(dummy_model, data)
        X = np.random.randn(10, 10)
        
        with pytest.raises(NotImplementedError):
            explainer(X)
