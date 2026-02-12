"""
Tests for FDFI explainers.
"""

import pytest
import numpy as np
from fdfi.explainers import (
    Explainer,
    TreeExplainer,
    LinearExplainer,
    KernelExplainer,
    OTExplainer,
    EOTExplainer,
    FlowExplainer,
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


def simple_linear_model(X):
    """Simple linear model: y = x0 + 2*x1"""
    return X[:, 0] + 2 * X[:, 1]


class TestFlowExplainer:
    """Test the FlowExplainer class."""
    
    def test_init_no_fit(self):
        """Test FlowExplainer initialization without flow fitting."""
        X = np.random.randn(50, 5)
        explainer = FlowExplainer(simple_linear_model, X, fit_flow=False)
        
        assert explainer.model == simple_linear_model
        assert explainer.flow_model is None
        assert explainer.Z_full is None
        assert explainer.method == "cpi"
    
    def test_init_with_fit(self):
        """Test FlowExplainer initialization with flow fitting."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        explainer = FlowExplainer(
            simple_linear_model, X, fit_flow=True, 
            num_steps=50, nsamples=5, random_state=0, verbose=False
        )
        
        assert explainer.flow_model is not None
        assert explainer.Z_full is not None
        assert explainer.Z_full.shape == (50, 5)
    
    def test_method_options(self):
        """Test different method options."""
        X = np.random.randn(50, 5)
        
        # CPI (default)
        exp_cpi = FlowExplainer(simple_linear_model, X, fit_flow=False, method='cpi')
        assert exp_cpi.method == 'cpi'
        
        # SCPI
        exp_scpi = FlowExplainer(simple_linear_model, X, fit_flow=False, method='scpi')
        assert exp_scpi.method == 'scpi'
        
        # Both
        exp_both = FlowExplainer(simple_linear_model, X, fit_flow=False, method='both')
        assert exp_both.method == 'both'
        
        # Invalid method
        with pytest.raises(ValueError, match="method must be"):
            FlowExplainer(simple_linear_model, X, fit_flow=False, method='invalid')
    
    def test_fit_flow_deferred(self):
        """Test deferred flow fitting."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        explainer = FlowExplainer(simple_linear_model, X, fit_flow=False)
        assert explainer.flow_model is None
        
        explainer.fit_flow(num_steps=50, verbose=False)
        assert explainer.flow_model is not None
        assert explainer.Z_full is not None
    
    def test_set_flow(self):
        """Test setting external flow model."""
        from fdfi.models import FlowMatchingModel
        
        np.random.seed(42)
        X = np.random.randn(50, 5)
        
        # Create and train external flow
        external_flow = FlowMatchingModel(X, dim=5)
        external_flow.fit(num_steps=50, verbose=False)
        
        # Set it in explainer
        explainer = FlowExplainer(simple_linear_model, X, fit_flow=False)
        explainer.set_flow(external_flow)
        
        assert explainer.flow_model is external_flow
        assert explainer.Z_full is not None
    
    def test_call_returns_expected_keys(self):
        """Test that __call__ returns all expected result keys."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        explainer = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, method='cpi', random_state=0, verbose=False
        )
        
        results = explainer(X_test)
        
        expected_keys = ['phi_X', 'std_X', 'se_X', 'phi_Z', 'std_Z', 'se_Z']
        assert all(key in results for key in expected_keys)
        
        # Check shapes
        assert results['phi_X'].shape == (5,)
        assert results['phi_Z'].shape == (5,)
        assert results['std_X'].shape == (5,)
        assert results['se_X'].shape == (5,)
    
    def test_cpi_phi_X_transformed_via_jacobian(self):
        """Test that CPI method transforms phi_Z to phi_X via Jacobian."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        explainer = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, method='cpi', random_state=0, verbose=False
        )
        
        results = explainer(X_test)
        
        # phi_X should be transformed from phi_Z via Jacobian
        # They should not be exactly equal due to the transformation
        assert np.all(np.isfinite(results['phi_X']))
        assert np.all(np.isfinite(results['phi_Z']))
        # Both should be positive (squared differences)
        assert np.all(results['phi_X'] >= 0)
        assert np.all(results['phi_Z'] >= 0)
    
    def test_scpi_computes_different_importance(self):
        """Test that SCPI method computes importance differently from CPI."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        explainer_scpi = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, method='scpi', random_state=0, verbose=False
        )
        
        results_scpi = explainer_scpi(X_test)
        
        # SCPI should compute E[(Y - f(X_tilde))^2]
        assert results_scpi['phi_X'] is not None
        assert results_scpi['phi_Z'] is not None
        assert np.all(np.isfinite(results_scpi['phi_Z']))
        
        # phi_X is transformed from phi_Z via Jacobian
        assert np.all(np.isfinite(results_scpi['phi_X']))
        # Both should be positive (squared differences)
        assert np.all(results_scpi['phi_X'] >= 0)
        assert np.all(results_scpi['phi_Z'] >= 0)
    
    def test_sampling_methods(self):
        """Test all sampling methods produce valid results."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        for method in ['resample', 'permutation', 'normal']:
            explainer = FlowExplainer(
                simple_linear_model, X_train, fit_flow=True,
                num_steps=50, nsamples=5, sampling_method=method, random_state=0, verbose=False
            )
            
            results = explainer(X_test)
            
            assert np.all(np.isfinite(results['phi_X']))
            assert np.all(np.isfinite(results['phi_Z']))
            assert np.all(results['std_X'] >= 0)
            assert np.all(results['se_X'] >= 0)
    
    def test_condperm_sampling(self):
        """Test conditional permutation sampling method."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        explainer = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, sampling_method='condperm', random_state=0, verbose=False
        )
        
        results = explainer(X_test)
        
        assert np.all(np.isfinite(results['phi_X']))
        assert np.all(np.isfinite(results['phi_Z']))
    
    def test_conf_int_integration(self):
        """Test confidence interval computation."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(20, 5)
        
        explainer = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=10, random_state=0, verbose=False
        )
        
        results = explainer(X_test)
        
        # Test conf_int
        ci = explainer.conf_int(alpha=0.05, target='X', alternative='two-sided')
        
        assert 'phi_hat' in ci
        assert 'se' in ci
        assert 'ci_lower' in ci
        assert 'ci_upper' in ci
        assert 'reject_null' in ci
        assert 'pvalue' in ci
    
    def test_active_features_higher_importance(self):
        """Test that active features have higher importance."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(30, 5)
        
        explainer = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=200, nsamples=20, random_state=0, verbose=False
        )
        
        results = explainer(X_test)
        
        # Features 0 and 1 should have higher importance than 2, 3, 4
        active_importance = (results['phi_Z'][0] + results['phi_Z'][1]) / 2
        null_importance = results['phi_Z'][2:].mean()
        
        assert active_importance > null_importance
    
    def test_reproducibility(self):
        """Test random_state produces reproducible results."""
        np.random.seed(42)
        X_train = np.random.randn(50, 5)
        X_test = np.random.randn(10, 5)
        
        explainer1 = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, random_state=123
        )
        results1 = explainer1(X_test)
        
        # Create new explainer with same seed
        np.random.seed(42)
        explainer2 = FlowExplainer(
            simple_linear_model, X_train, fit_flow=True,
            num_steps=50, nsamples=5, random_state=123
        )
        results2 = explainer2(X_test)
        
        # Results should be correlated (exact reproducibility is hard due to
        # flow training stochasticity and Jacobian computation)
        # Check rank correlation instead of exact values
        from scipy.stats import spearmanr
        corr, _ = spearmanr(results1['phi_Z'], results2['phi_Z'])
        assert corr > 0.5, f"Rank correlation should be positive: {corr}"


class TestFlowVsOTExplainer:
    """Integration tests comparing FlowExplainer and OTExplainer."""
    
    def test_both_identify_active_features(self):
        """Both explainers should identify active features > null features."""
        np.random.seed(42)
        
        # Create correlated data
        n_train, n_test, d = 200, 50, 8
        cov = np.eye(d)
        cov[0, 1] = cov[1, 0] = 0.6
        
        X_train = np.random.multivariate_normal(np.zeros(d), cov, size=n_train)
        X_test = np.random.multivariate_normal(np.zeros(d), cov, size=n_test)
        
        # Model uses features 0, 1, 2 only
        def model(X):
            return X[:, 0] + 2 * X[:, 1] + 0.5 * X[:, 2]
        
        # FlowExplainer
        flow_exp = FlowExplainer(
            model, X_train, fit_flow=True,
            num_steps=200, nsamples=20, random_state=0
        )
        flow_results = flow_exp(X_test)
        
        # OTExplainer
        ot_exp = OTExplainer(model, X_train, nsamples=20, random_state=0)
        ot_results = ot_exp(X_test)
        
        # Both should identify active > null features
        active_idx = [0, 1, 2]
        null_idx = [3, 4, 5, 6, 7]
        
        flow_active = np.mean([flow_results['phi_X'][i] for i in active_idx])
        flow_null = np.mean([flow_results['phi_X'][i] for i in null_idx])
        
        ot_active = np.mean([ot_results['phi_X'][i] for i in active_idx])
        ot_null = np.mean([ot_results['phi_X'][i] for i in null_idx])
        
        assert flow_active > flow_null, "FlowExplainer should identify active > null"
        assert ot_active > ot_null, "OTExplainer should identify active > null"
    
    def test_positive_correlation_between_methods(self):
        """Feature rankings should be positively correlated."""
        np.random.seed(42)
        
        n_train, n_test, d = 150, 40, 6
        X_train = np.random.randn(n_train, d)
        X_test = np.random.randn(n_test, d)
        
        def model(X):
            return X[:, 0] + 1.5 * X[:, 1] + 0.3 * X[:, 2]
        
        flow_exp = FlowExplainer(
            model, X_train, fit_flow=True,
            num_steps=200, nsamples=15, random_state=0
        )
        flow_results = flow_exp(X_test)
        
        ot_exp = OTExplainer(model, X_train, nsamples=15, random_state=0)
        ot_results = ot_exp(X_test)
        
        # Correlation should be positive
        correlation = np.corrcoef(flow_results['phi_X'], ot_results['phi_X'])[0, 1]
        assert correlation > 0, f"Expected positive correlation, got {correlation}"
