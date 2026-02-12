"""
<<<<<<< HEAD
Utility functions for dfi.
=======
Utility functions for DFI.
>>>>>>> 914762b723966c3192bac9ea445a716d8760dd38

This module provides helper functions used across the package.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy import stats
from typing import Optional, Union, Any, Tuple


# Constants for logit link function
_LOGIT_MIN_PROB = 1e-7
_LOGIT_MAX_PROB = 1 - 1e-7


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
        predictions = np.clip(predictions, _LOGIT_MIN_PROB, _LOGIT_MAX_PROB)
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


@dataclass
class TwoComponentMixture:
    """
    Fit a two-component Gaussian mixture and extract quantiles.

    Used for:
    1. Variance floor estimation (from raw stds)
    2. Practical significance margin (from point estimates)
    """
    n_components: int = 2
    random_state: int = 0
    min_samples: int = 10

    means_: np.ndarray = field(default=None, init=False, repr=False)
    stds_: np.ndarray = field(default=None, init=False, repr=False)
    weights_: np.ndarray = field(default=None, init=False, repr=False)
    gmm_: object = field(default=None, init=False, repr=False)
    method_used_: str = field(default=None, init=False)

    def fit(self, x: np.ndarray) -> "TwoComponentMixture":
        x = np.asarray(x).flatten()

        if len(x) < self.min_samples:
            self._fit_robust(x)
        else:
            self._fit_gmm(x)

        return self

    def _fit_robust(self, x: np.ndarray) -> None:
        median = np.median(x)
        scale = self._robust_scale(x)

        self.means_ = np.array([median, median])
        self.stds_ = np.array([scale, scale])
        self.weights_ = np.array([0.5, 0.5])
        self.gmm_ = None
        self.method_used_ = "robust"

    def _fit_gmm(self, x: np.ndarray) -> None:
        try:
            from sklearn.mixture import GaussianMixture
        except Exception:
            self._fit_robust(x)
            return

        gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        gmm.fit(x.reshape(-1, 1))

        self.means_ = gmm.means_.flatten()
        self.stds_ = np.sqrt(gmm.covariances_.flatten())
        self.weights_ = gmm.weights_
        self.gmm_ = gmm
        self.method_used_ = "gmm"

    @staticmethod
    def _robust_scale(x: np.ndarray) -> float:
        if hasattr(stats, "qn_scale"):
            try:
                qn = stats.qn_scale(x)
                if qn > 0:
                    return float(qn)
            except Exception:
                pass

        mad = stats.median_abs_deviation(x, scale="normal")
        if mad > 0:
            return float(mad)

        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            return float(iqr / 1.349)

        return float((np.max(x) - np.min(x)) / 4)

    def quantile(self, q: float, component: str = "larger") -> float:
        if self.means_ is None:
            raise ValueError("Call fit() first.")

        idx = np.argmax(self.means_) if component == "larger" else np.argmin(self.means_)
        return float(self.means_[idx] + stats.norm.ppf(q) * self.stds_[idx])

    def plot(self, x: np.ndarray, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        x = np.asarray(x).flatten()
        ax.hist(x, bins="auto", density=True, alpha=0.5, **kwargs)

        x_grid = np.linspace(x.min(), x.max(), 200)
        for i in range(len(self.means_)):
            pdf = self.weights_[i] * stats.norm.pdf(
                x_grid, self.means_[i], self.stds_[i]
            )
            label = f"Component {i+1} (μ={self.means_[i]:.3f}, σ={self.stds_[i]:.3f})"
            ax.plot(x_grid, pdf, label=label)

        ax.legend()
        ax.set_title(f"Mixture fit (method: {self.method_used_})")
        return ax


def detect_feature_types(
    X: np.ndarray,
    categorical_threshold: int = 10,
    feature_types: Optional[np.ndarray] = None,
) -> dict:
    """
    Auto-detect feature types from data.

    Returns a dict with:
      - 'binary', 'categorical', 'continuous' indices
      - 'types' array of labels per feature
      - 'ranges' array of ranges per feature (for normalization)
    """
    n, d = X.shape

    if feature_types is not None:
        types = np.array(feature_types)
        if len(types) != d:
            raise ValueError(
                f"feature_types length {len(types)} != number of features {d}"
            )
    else:
        types = np.empty(d, dtype=object)
        for j in range(d):
            unique_vals = np.unique(X[:, j])
            n_unique = len(unique_vals)

            if n_unique == 2:
                types[j] = "binary"
            elif n_unique <= categorical_threshold:
                if np.allclose(X[:, j], np.round(X[:, j])):
                    types[j] = "categorical"
                else:
                    types[j] = "continuous"
            else:
                types[j] = "continuous"

    ranges = np.zeros(d)
    for j in range(d):
        if types[j] == "continuous":
            ranges[j] = X[:, j].max() - X[:, j].min()
            if ranges[j] == 0:
                ranges[j] = 1.0
        else:
            ranges[j] = 1.0

    binary_idx = [j for j in range(d) if types[j] == "binary"]
    categorical_idx = [j for j in range(d) if types[j] == "categorical"]
    continuous_idx = [j for j in range(d) if types[j] == "continuous"]

    return {
        "binary": binary_idx,
        "categorical": categorical_idx,
        "continuous": continuous_idx,
        "types": types,
        "ranges": ranges,
    }


def gower_cost_matrix(
    X: np.ndarray,
    Z: np.ndarray,
    feature_types: np.ndarray,
    feature_ranges: np.ndarray,
    feature_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Gower distance matrix for mixed-type data.
    """
    n, d = X.shape
    m = Z.shape[0]

    if feature_weights is None:
        feature_weights = np.ones(d) / d
    else:
        feature_weights = feature_weights / feature_weights.sum()

    C = np.zeros((n, m))
    for j in range(d):
        if feature_types[j] in ("binary", "categorical"):
            C += feature_weights[j] * (
                X[:, j:j + 1] != Z[:, j:j + 1].T
            ).astype(float)
        else:
            C += feature_weights[j] * np.abs(X[:, j:j + 1] - Z[:, j:j + 1].T) / feature_ranges[j]

    return C


def compute_latent_independence(Z: np.ndarray, subset_size: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Compute pairwise Distance Correlation (dCor) between latent dimensions.
    
    Measures statistical independence of latent variables. Lower values indicate
    higher independence (ideal for disentangled representations).
    
    Parameters
    ----------
    Z : np.ndarray
        Latent representations. Shape (n_samples, n_latent_dims).
    subset_size : int, optional
        If provided, sample this many points for efficiency. If None, use all.
    
    Returns
    -------
    dcor_matrix : np.ndarray
        Pairwise distance correlation matrix. Shape (d, d).
    median_dcor : float
        Median of off-diagonal distance correlations (overall independence score).
    """
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape {Z.shape}")
    
    n, d = Z.shape
    
    # Sample if needed
    if subset_size is not None and n > subset_size:
        idx = np.random.choice(n, size=subset_size, replace=False)
        Z = Z[idx]
        n = subset_size
    
    # Compute pairwise distances for each dimension
    dcor_matrix = np.zeros((d, d))
    
    for j in range(d):
        for k in range(d):
            if j == k:
                dcor_matrix[j, k] = 1.0  # Perfect correlation with itself
            else:
                # Compute distance correlation between Z[:, j] and Z[:, k]
                z_j = Z[:, j]
                z_k = Z[:, k]
                
                # Euclidean distance matrices
                dj = np.abs(z_j[:, None] - z_j[None, :])
                dk = np.abs(z_k[:, None] - z_k[None, :])
                
                # Doubly center the distance matrices
                row_mean_j = dj.mean(axis=1, keepdims=True)
                col_mean_j = dj.mean(axis=0, keepdims=True)
                grand_mean_j = dj.mean()
                Aj = dj - row_mean_j - col_mean_j + grand_mean_j
                
                row_mean_k = dk.mean(axis=1, keepdims=True)
                col_mean_k = dk.mean(axis=0, keepdims=True)
                grand_mean_k = dk.mean()
                Ak = dk - row_mean_k - col_mean_k + grand_mean_k
                
                # Distance covariance
                dcov_jk_sq = (Aj * Ak).sum() / (n * n)
                dcov_j_sq = (Aj * Aj).sum() / (n * n)
                dcov_k_sq = (Ak * Ak).sum() / (n * n)
                
                # Distance correlation
                if dcov_j_sq > 0 and dcov_k_sq > 0:
                    dcor = np.sqrt(dcov_jk_sq / np.sqrt(dcov_j_sq * dcov_k_sq))
                else:
                    dcor = 0.0
                
                dcor_matrix[j, k] = dcor
    
    # Extract off-diagonal median
    mask = ~np.eye(d, dtype=bool)
    off_diag_values = dcor_matrix[mask]
    median_dcor = np.median(off_diag_values)
    
    return dcor_matrix, median_dcor


def compute_mmd(X_real: np.ndarray, X_generated: np.ndarray, sigma: float = 1.0, 
                subset_size: Optional[int] = None) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) with Gaussian RBF kernel.
    
    Measures distributional distance between real and generated data.
    Lower values indicate better fidelity.
    
    Parameters
    ----------
    X_real : np.ndarray
        Real data. Shape (n_real, n_features).
    X_generated : np.ndarray
        Generated/reconstructed data. Shape (n_gen, n_features).
    sigma : float
        Bandwidth parameter for Gaussian kernel.
    subset_size : int, optional
        Sample this many points from each distribution for efficiency.
    
    Returns
    -------
    mmd : float
        MMD score (non-negative). Lower is better.
    """
    if X_real.ndim != 2 or X_generated.ndim != 2:
        raise ValueError("Both X_real and X_generated must be 2D arrays")
    
    if X_real.shape[1] != X_generated.shape[1]:
        raise ValueError("Feature dimensions must match")
    
    n_real = X_real.shape[0]
    n_gen = X_generated.shape[0]
    
    # Sample if needed
    if subset_size is not None:
        if n_real > subset_size:
            idx_real = np.random.choice(n_real, size=subset_size, replace=False)
            X_real = X_real[idx_real]
        if n_gen > subset_size:
            idx_gen = np.random.choice(n_gen, size=subset_size, replace=False)
            X_generated = X_generated[idx_gen]
    
    # Gaussian RBF kernel
    def gaussian_kernel(x, y, sigma):
        """Compute Gaussian kernel between two sets of points."""
        # ||x - y||^2
        sq_dist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
        return np.exp(-sq_dist / (2 * sigma ** 2))
    
    # Compute kernels
    K_xx = gaussian_kernel(X_real, X_real, sigma)
    K_yy = gaussian_kernel(X_generated, X_generated, sigma)
    K_xy = gaussian_kernel(X_real, X_generated, sigma)
    
    # MMD computation
    n, m = X_real.shape[0], X_generated.shape[0]
    mmd_sq = (K_xx.sum() / (n * n) + K_yy.sum() / (m * m) - 2 * K_xy.sum() / (n * m))
    mmd = np.sqrt(np.maximum(mmd_sq, 0.0))  # Clamp to avoid numerical issues
    
    return mmd
