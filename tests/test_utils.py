"""
Tests for FDFI utility functions.
"""

import pytest
import numpy as np
from fdfi.utils import (
    validate_input,
    sample_background,
    get_feature_names,
    convert_to_link,
    check_additivity,
    compute_latent_independence,
    compute_mmd,
)
from fdfi.explainers import FlowExplainer


class TestValidateInput:
    """Test the validate_input function."""
    
    def test_valid_numpy_array(self):
        """Test with a valid numpy array."""
        X = np.random.randn(10, 5)
        result = validate_input(X)
        assert np.array_equal(result, X)
    
    def test_convert_list(self):
        """Test converting a list to numpy array."""
        X = [[1, 2, 3], [4, 5, 6]]
        result = validate_input(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
    
    def test_scalar_raises_error(self):
        """Test that scalar input raises ValueError."""
        with pytest.raises(ValueError, match="at least 1-dimensional"):
            validate_input(5)
    
    def test_invalid_input(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError, match="at least 1-dimensional"):
            validate_input(object())


class TestSampleBackground:
    """Test the sample_background function."""
    
    def test_sample_less_than_available(self):
        """Test sampling fewer samples than available."""
        data = np.random.randn(100, 10)
        result = sample_background(data, n_samples=20, random_state=42)
        assert result.shape == (20, 10)
    
    def test_sample_more_than_available(self):
        """Test sampling more samples than available returns all data."""
        data = np.random.randn(50, 10)
        result = sample_background(data, n_samples=100)
        assert result.shape == (50, 10)
        assert np.array_equal(result, data)
    
    def test_random_state(self):
        """Test that random_state provides reproducibility."""
        data = np.random.randn(100, 10)
        result1 = sample_background(data, n_samples=20, random_state=42)
        result2 = sample_background(data, n_samples=20, random_state=42)
        assert np.array_equal(result1, result2)


class TestGetFeatureNames:
    """Test the get_feature_names function."""
    
    def test_with_provided_names(self):
        """Test with user-provided feature names."""
        data = np.random.randn(10, 5)
        names = ["a", "b", "c", "d", "e"]
        result = get_feature_names(data, feature_names=names)
        assert result == names
    
    def test_without_provided_names(self):
        """Test automatic feature name generation."""
        data = np.random.randn(10, 5)
        result = get_feature_names(data)
        assert result == ["Feature 0", "Feature 1", "Feature 2", "Feature 3", "Feature 4"]
    
    def test_mismatch_raises_error(self):
        """Test that mismatched feature names raise ValueError."""
        data = np.random.randn(10, 5)
        names = ["a", "b", "c"]  # Only 3 names for 5 features
        with pytest.raises(ValueError, match="does not match"):
            get_feature_names(data, feature_names=names)


class TestConvertToLink:
    """Test the convert_to_link function."""
    
    def test_identity_link(self):
        """Test identity link function."""
        predictions = np.array([0.1, 0.5, 0.9])
        result = convert_to_link(predictions, link="identity")
        assert np.array_equal(result, predictions)
    
    def test_logit_link(self):
        """Test logit link function."""
        predictions = np.array([0.1, 0.5, 0.9])
        result = convert_to_link(predictions, link="logit")
        # Check that logit transformation is applied
        expected = np.log(predictions / (1 - predictions))
        np.testing.assert_allclose(result, expected)
    
    def test_invalid_link(self):
        """Test that invalid link function raises ValueError."""
        predictions = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="Unknown link function"):
            convert_to_link(predictions, link="invalid")


class TestCheckAdditivity:
    """Test the check_additivity function."""
    
    def test_perfect_additivity(self):
        """Test with perfect additivity."""
        base_value = 0.5
        shap_values = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
        # Predictions should equal base_value + sum(shap_values)
        predictions = np.array([base_value + 0.6, base_value + 0.9])
        
        satisfies, max_diff = check_additivity(shap_values, predictions, base_value)
        assert satisfies
        assert max_diff < 1e-10
    
    def test_additivity_within_tolerance(self):
        """Test additivity within tolerance."""
        base_value = 0.5
        shap_values = np.array([[0.1, 0.2, 0.3]])
        predictions = np.array([1.1 + 1e-4])  # Small deviation
        
        satisfies, max_diff = check_additivity(shap_values, predictions, base_value, tol=1e-3)
        assert satisfies
        assert max_diff < 1e-3
    
    def test_additivity_violation(self):
        """Test when additivity is violated."""
        base_value = 0.5
        shap_values = np.array([[0.1, 0.2, 0.3]])
        predictions = np.array([2.0])  # Large deviation
        
        satisfies, max_diff = check_additivity(shap_values, predictions, base_value, tol=1e-3)
        assert not satisfies
        assert max_diff > 1e-3


class TestComputeLatentIndependence:
    """Tests for compute_latent_independence."""

    def test_independent_dims_have_low_dcor(self):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((200, 3))
        _, median_dcor = compute_latent_independence(Z)
        assert median_dcor < 0.2

    def test_correlated_dims_have_higher_dcor(self):
        rng = np.random.default_rng(1)
        z1 = rng.standard_normal(200)
        z2 = z1 + rng.normal(0, 0.05, 200)
        z3 = rng.standard_normal(200)
        Z = np.vstack([z1, z2, z3]).T
        _, median_dcor = compute_latent_independence(Z)
        assert median_dcor > 0.1


class TestComputeMMD:
    """Tests for compute_mmd."""

    def test_identical_sets_have_near_zero_mmd(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 4))
        mmd = compute_mmd(X, X.copy())
        assert mmd < 1e-6

    def test_shifted_sets_increase_mmd(self):
        rng = np.random.default_rng(1)
        X_real = rng.standard_normal((200, 4))
        X_gen = X_real + 2.0  # clear shift
        mmd = compute_mmd(X_real, X_gen)
        assert mmd > 0.3


class TestFlowExplainerDiagnostics:
    """Tests for FlowExplainer diagnostics without requiring torch."""

    def test_diagnostics_populated_and_labeled(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 3))

        # Simple linear model
        def model(x):
            return x.sum(axis=1)

        explainer = FlowExplainer(
            model=model,
            data=X,
            flow_model=None,
            fit_flow=False,
            verbose=False,
            nsamples=5,
        )

        # Bypass flow decode with identity to avoid torch dependency
        explainer._decode_to_X = lambda Z, **kwargs: Z
        explainer.Z_full = X
        explainer._compute_diagnostics(X_orig=X, Z_full=X, report_title="Flow Model")

        diag = explainer.diagnostics
        assert "latent_independence_median" in diag
        assert "distribution_fidelity_mmd" in diag
        assert diag["latent_independence_label"] in {"GOOD", "MODERATE", "POOR"}
        assert diag["distribution_fidelity_label"] in {"GOOD", "MODERATE", "POOR"}
