"""
Tests for dfi explainers.
"""

import pytest
import numpy as np
from dfi.explainers import (
    Explainer,
    TreeExplainer,
    LinearExplainer,
    KernelExplainer,
    OTExplainer,
    EOTExplainer,
)


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
        explainer = Explainer(dummy_model, data=data, fit_flow=False)
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
        explainer = TreeExplainer(model, fit_flow=False)
        assert explainer.model == model
    
    def test_call_not_implemented(self):
        """Test that TreeExplainer raises NotImplementedError."""
        model = object()
        explainer = TreeExplainer(model, fit_flow=False)
        X = np.random.randn(10, 5)
        
        with pytest.raises(NotImplementedError):
            explainer(X)


class TestLinearExplainer:
    """Test the LinearExplainer class."""
    
    def test_init(self):
        """Test LinearExplainer initialization."""
        model = object()
        explainer = LinearExplainer(model, fit_flow=False)
        assert explainer.model == model
    
    def test_call_not_implemented(self):
        """Test that LinearExplainer raises NotImplementedError."""
        model = object()
        explainer = LinearExplainer(model, fit_flow=False)
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
        explainer = KernelExplainer(dummy_model, data, fit_flow=False)
        assert explainer.model == dummy_model
        assert np.array_equal(explainer.data, data)
    
    def test_call_not_implemented(self):
        """Test that KernelExplainer raises NotImplementedError."""
        def dummy_model(x):
            return x.sum(axis=1)
        
        data = np.random.randn(100, 10)
        explainer = KernelExplainer(dummy_model, data, fit_flow=False)
        X = np.random.randn(10, 10)
        
        with pytest.raises(NotImplementedError):
            explainer(X)


def generate_exp3_data(n=500, rho=0.8, seed=42):
    rng = np.random.default_rng(seed)
    d = 50
    mean = np.zeros(d)
    cov = np.eye(d)
    cov[:10, :10] = rho
    np.fill_diagonal(cov[:10, :10], 1.0)
    X = rng.multivariate_normal(mean, cov, size=n)
    y = (
        np.arctan(X[:, 0] + X[:, 1]) * (X[:, 2] > 0)
        + np.sin(X[:, 3] * X[:, 4]) * (X[:, 2] <= 0)
    )
    y += rng.normal(0, 0.1, size=n)
    return X, y


def exp3_model(X):
    return (
        np.arctan(X[:, 0] + X[:, 1]) * (X[:, 2] > 0)
        + np.sin(X[:, 3] * X[:, 4]) * (X[:, 2] <= 0)
    )


class TestOTExplainer:
    def test_exp3_mean_variance_consistency(self):
        X_train, _ = generate_exp3_data(n=300, seed=1)
        X_test, _ = generate_exp3_data(n=150, seed=2)

        explainer = OTExplainer(exp3_model, X_train, nsamples=15, random_state=0)
        results = explainer(X_test)

        assert results["phi_X"].shape == (50,)
        assert results["std_X"].shape == (50,)
        assert results["se_X"].shape == (50,)
        assert np.all(np.isfinite(results["phi_X"]))
        assert np.all(results["std_X"] >= 0)

        Z = (X_test - explainer.mean) @ explainer.L_inv
        y_pred = exp3_model(X_test)
        ueifs_Z = explainer._phi_Z(Z, y_pred)

        ueifs_X = ueifs_Z @ (explainer.L ** 2).T
        phi_X = np.mean(ueifs_X, axis=0)
        std_X = np.std(ueifs_X, axis=0)
        n = ueifs_X.shape[0]
        ddof = 1 if n > 1 else 0
        se_X = np.std(ueifs_X, axis=0, ddof=ddof) / np.sqrt(n)
        phi_Z = np.mean(ueifs_Z, axis=0)
        std_Z = np.std(ueifs_Z, axis=0)
        se_Z = np.std(ueifs_Z, axis=0, ddof=ddof) / np.sqrt(n)

        assert np.allclose(phi_X, results["phi_X"])
        assert np.allclose(std_X, results["std_X"])
        assert np.allclose(se_X, results["se_X"])
        assert np.allclose(phi_Z, results["phi_Z"])
        assert np.allclose(std_Z, results["std_Z"])
        assert np.allclose(se_Z, results["se_Z"])

        active_mean = phi_X[:5].mean()
        null_mean = phi_X[10:].mean()
        assert active_mean > null_mean
        ci = explainer.conf_int(
            alpha=0.05, target="X", alternative="two-sided", var_floor_c=0.0
        )
        assert np.all(ci["ci_lower"][:10] > 0)


class TestEOTExplainer:
    def test_exp3_mean_variance_consistency(self):
        X_train, _ = generate_exp3_data(n=80, seed=3)
        X_test, _ = generate_exp3_data(n=40, seed=4)

        explainer = EOTExplainer(exp3_model, X_train)
        results = explainer(X_test)

        assert results["phi_X"].shape == (50,)
        assert results["std_X"].shape == (50,)
        assert results["se_X"].shape == (50,)
        assert np.all(np.isfinite(results["phi_X"]))
        assert np.all(results["std_X"] >= 0)

        Z = (X_test - explainer.mean) @ explainer.L_inv
        y_pred = exp3_model(X_test)
        ueifs_Z = explainer._phi_Z(Z, y_pred)

        ueifs_X = ueifs_Z @ (explainer.L ** 2).T
        phi_X = np.mean(ueifs_X, axis=0)
        std_X = np.std(ueifs_X, axis=0)
        n = ueifs_X.shape[0]
        ddof = 1 if n > 1 else 0
        se_X = np.std(ueifs_X, axis=0, ddof=ddof) / np.sqrt(n)
        phi_Z = np.mean(ueifs_Z, axis=0)
        std_Z = np.std(ueifs_Z, axis=0)
        se_Z = np.std(ueifs_Z, axis=0, ddof=ddof) / np.sqrt(n)

        assert np.allclose(phi_X, results["phi_X"])
        assert np.allclose(std_X, results["std_X"])
        assert np.allclose(se_X, results["se_X"])
        assert np.allclose(phi_Z, results["phi_Z"])
        assert np.allclose(std_Z, results["std_Z"])
        assert np.allclose(se_Z, results["se_Z"])

        active_mean = phi_X[:5].mean()
        null_mean = phi_X[10:].mean()
        assert active_mean > null_mean
        ci = explainer.conf_int(
            alpha=0.05, target="X", alternative="two-sided", var_floor_c=0.0
        )
        assert np.all(ci["ci_lower"][:10] > 0)
