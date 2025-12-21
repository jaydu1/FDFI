"""
Utility functions for FDFI.

This module provides helper functions used across the package.
"""

import numpy as np
from typing import Optional, Union, Any, Tuple


def validate_input(X: Any) -> np.ndarray:
    """
    Validate and convert input to numpy array.
    
    Parameters
    ----------
    X : array-like
        Input data to validate.
    
    Returns
    -------
    numpy.ndarray
        Validated numpy array.
    
    Raises
    ------
    ValueError
        If input cannot be converted to a valid numpy array.
    """
    if not isinstance(X, np.ndarray):
        try:
            X = np.asarray(X)
        except Exception as e:
            raise ValueError(f"Cannot convert input to numpy array: {e}")
    
    if X.ndim == 0:
        raise ValueError("Input must be at least 1-dimensional")
    
    return X


def sample_background(
    data: np.ndarray,
    n_samples: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Sample background data for explanations.
    
    Parameters
    ----------
    data : numpy.ndarray
        Full dataset to sample from.
    n_samples : int
        Number of samples to draw.
    random_state : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    numpy.ndarray
        Sampled background data.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_available = data.shape[0]
    if n_samples >= n_available:
        return data.copy()
    
    indices = np.random.choice(n_available, size=n_samples, replace=False)
    return data[indices]


def get_feature_names(
    data: Any,
    feature_names: Optional[list] = None
) -> list:
    """
    Get or generate feature names.
    
    Parameters
    ----------
    data : array-like
        Data to get feature names for.
    feature_names : list, optional
        User-provided feature names.
    
    Returns
    -------
    list
        Feature names.
    """
    data = validate_input(data)
    n_features = data.shape[1] if data.ndim > 1 else 1
    
    if feature_names is not None:
        if len(feature_names) != n_features:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) does not "
                f"match number of features ({n_features})"
            )
        return feature_names
    
    return [f"Feature {i}" for i in range(n_features)]


def convert_to_link(
    predictions: np.ndarray,
    link: str = "identity"
) -> np.ndarray:
    """
    Convert predictions using a link function.
    
    Parameters
    ----------
    predictions : numpy.ndarray
        Model predictions.
    link : str, default="identity"
        Link function to use. Options: "identity", "logit".
    
    Returns
    -------
    numpy.ndarray
        Transformed predictions.
    
    Raises
    ------
    ValueError
        If link function is not recognized.
    """
    if link == "identity":
        return predictions
    elif link == "logit":
        # Convert probabilities to logit scale
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
        return np.log(predictions / (1 - predictions))
    else:
        raise ValueError(f"Unknown link function: {link}")


def check_additivity(
    shap_values: np.ndarray,
    predictions: np.ndarray,
    base_value: float,
    tol: float = 1e-3
) -> Tuple[bool, float]:
    """
    Check if SHAP values satisfy the additivity property.
    
    The additivity property states that the sum of SHAP values plus
    the base value should equal the prediction.
    
    Parameters
    ----------
    shap_values : numpy.ndarray
        Feature importance values. Shape (n_samples, n_features).
    predictions : numpy.ndarray
        Model predictions. Shape (n_samples,).
    base_value : float
        Base value (expected output).
    tol : float, default=1e-3
        Tolerance for checking equality.
    
    Returns
    -------
    bool
        Whether additivity is satisfied.
    float
        Maximum absolute difference.
    """
    predicted_from_shap = base_value + shap_values.sum(axis=1)
    diff = np.abs(predicted_from_shap - predictions)
    max_diff = np.max(diff)
    
    return max_diff <= tol, max_diff
