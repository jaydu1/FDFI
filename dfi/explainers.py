"""
Explainer classes for dfi.

This module contains the main explainer classes for computing
flow-disentangled feature importance.
"""

import numpy as np
from typing import Optional, Union, Callable, Any, Tuple
from scipy.spatial.distance import cdist, pdist
from scipy import stats
from .utils import TwoComponentMixture, detect_feature_types, gower_cost_matrix



class Explainer:
    """
    Base class for dfi explainers.
    
    This class provides the interface for computing feature importance
    using flow-disentangled methods, similar to SHAP explainers.
    It also provides post-hoc confidence intervals via `conf_int()` and
    formatted summaries via `summary()`.
    
    Parameters
    ----------
    model : callable
        The model to explain. Should be a function that takes a numpy array
        and returns predictions.
    data : numpy.ndarray, optional
        Background data to use for explanations.
    **kwargs : dict
        Additional parameters for the explainer.
    
    Attributes
    ----------
    model : callable
        The model being explained.
    data : numpy.ndarray or None
        Background data for explanations.
    
    Examples
    --------
    >>> import numpy as np
    >>> from dfi import Explainer
    >>> 
    >>> # Define a simple model
    >>> def model(x):
    ...     return x.sum(axis=1)
    >>> 
    >>> # Create an explainer
    >>> explainer = Explainer(model)
    >>> 
    >>> # Compute explanations (when implemented)
    >>> # explanations = explainer(X_test)
    """
    
    
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: Optional[np.ndarray] = None,
        **kwargs: Any
    ):
        """Initialize the Explainer."""
        self.model = model
        self.data = data
        self.kwargs = kwargs
        self._last_results = None
        self._last_n = None
        self._var_floor_mixture = None
        self._margin_mixture = None
        self._var_floor_value = None

        fit_flow = kwargs.get("fit_flow", True)
        self.flow_engine = None
        if data is not None and fit_flow:
            try:
                from .models import FlowMatchingModel
            except ImportError as exc:
                raise ImportError(
                    "Flow matching requires torch; install it or pass fit_flow=False."
                ) from exc
            self.flow_engine = FlowMatchingModel(
                X=data,
                dim=data.shape[1],
                device=kwargs.get("device")
            )
            print("Training flow model for disentanglement...")
            self.flow_engine.fit(num_steps=kwargs.get("num_steps", 5000))

    def _cache_results(self, results: dict, n: int) -> None:
        self._last_results = results
        self._last_n = n

    def _adjust_se(
        self,
        se_raw: np.ndarray,
        var_floor_c: float = 0.1,
        var_floor_method: str = "fixed",
        var_floor_quantile: float = 0.05,
    ) -> np.ndarray:
        if self._last_n is None:
            return se_raw

        if var_floor_method == "mixture":
            self._var_floor_mixture = TwoComponentMixture().fit(se_raw)
            floor = self._var_floor_mixture.quantile(var_floor_quantile, "smaller")
            self._var_floor_value = floor
        else:
            if var_floor_c <= 0:
                return se_raw
            floor = var_floor_c / np.sqrt(self._last_n)
            self._var_floor_value = floor

        return np.sqrt(se_raw**2 + floor**2)

    def conf_int(
        self,
        alpha: float = 0.05,
        target: str = "X",
        var_floor_c: float = 0.1,
        var_floor_method: str = "fixed",
        var_floor_quantile: float = 0.95,
        margin: float = 0.0,
        margin_method: str = "fixed",
        margin_quantile: float = 0.95,
        alternative: str = "two-sided",
    ) -> dict:
        if self._last_results is None:
            raise ValueError("Run the explainer first to compute scores.")

        if target == "Z":
            phi_hat = self._last_results["phi_Z"]
            se_raw = self._last_results["se_Z"]
        else:
            phi_hat = self._last_results["phi_X"]
            se_raw = self._last_results["se_X"]

        se_adj = self._adjust_se(
            se_raw,
            var_floor_c=var_floor_c,
            var_floor_method=var_floor_method,
            var_floor_quantile=var_floor_quantile,
        )

        if margin_method == "mixture":
            self._margin_mixture = TwoComponentMixture().fit(phi_hat)
            margin = max(self._margin_mixture.quantile(margin_quantile, "smaller"), 0)
        else:
            self._margin_mixture = None

        with np.errstate(divide="ignore", invalid="ignore"):
            if alternative == "greater":
                z = stats.norm.ppf(1 - alpha)
                ci_lower = phi_hat - z * se_adj
                ci_upper = np.full_like(phi_hat, np.inf)
                reject_null = ci_lower > margin
                z_scores = (phi_hat - margin) / se_adj
                pvalues = 1 - stats.norm.cdf(z_scores)
            elif alternative == "less":
                z = stats.norm.ppf(1 - alpha)
                ci_lower = np.full_like(phi_hat, -np.inf)
                ci_upper = phi_hat + z * se_adj
                reject_null = ci_upper < margin
                z_scores = (phi_hat - margin) / se_adj
                pvalues = stats.norm.cdf(z_scores)
            elif alternative == "two-sided":
                z = stats.norm.ppf(1 - alpha / 2)
                ci_lower = phi_hat - z * se_adj
                ci_upper = phi_hat + z * se_adj
                reject_null = (ci_lower > margin) | (ci_upper < margin)
                z_scores = np.abs(phi_hat - margin) / se_adj
                pvalues = 2 * (1 - stats.norm.cdf(z_scores))
            else:
                raise ValueError(
                    "alternative must be 'greater', 'less', or 'two-sided'"
                )

            pvalues = np.where(np.isfinite(pvalues), pvalues, 1.0)

        return {
            "phi_hat": phi_hat,
            "se": se_adj,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "reject_null": reject_null,
            "pvalue": pvalues,
            "margin": margin,
            "alternative": alternative,
        }

    def summary(self, alpha: float = 0.05, print_output: bool = True, **kwargs) -> str:
        results = self.conf_int(alpha=alpha, **kwargs)
        return self._format_summary(results, alpha, print_output)

    def _format_summary(self, results: dict, alpha: float, print_output: bool = True) -> str:
        lines = []
        lines.append("=" * 78)
        lines.append("Feature Importance Results")
        lines.append("=" * 78)
        lines.append(f"Method: {self.__class__.__name__}")
        lines.append(f"Number of features: {len(results['phi_hat'])}")
        lines.append(f"Significance level: {alpha}")
        lines.append(f"Alternative: {results['alternative']}")
        if results["margin"] > 0:
            lines.append(f"Practical margin: {results['margin']:.4f}")
        lines.append("-" * 78)

        header = (
            f"{'Feature':>8} {'Estimate':>10} {'Std Err':>10} "
            f"{'CI Lower':>10} {'CI Upper':>10} {'P-value':>10} {'Sig':>5}"
        )
        lines.append(header)
        lines.append("-" * 78)

        for i in range(len(results["phi_hat"])):
            ci_lower = results["ci_lower"][i]
            ci_upper = results["ci_upper"][i]
            ci_upper_str = (
                f"{ci_upper:>10.4f}" if np.isfinite(ci_upper) else f"{'inf':>10}"
            )
            ci_lower_str = (
                f"{ci_lower:>10.4f}" if np.isfinite(ci_lower) else f"{'-inf':>10}"
            )
            pval = results["pvalue"][i]
            sig = (
                "***"
                if pval < 0.01
                else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
            )
            row = (
                f"{i:>8} {results['phi_hat'][i]:>10.4f} "
                f"{results['se'][i]:>10.4f} {ci_lower_str} {ci_upper_str} "
                f"{pval:>10.4f} {sig:>5}"
            )
            lines.append(row)

        lines.append("=" * 78)
        n_sig = np.sum(results["reject_null"])
        lines.append(f"Significant features: {n_sig} / {len(results['phi_hat'])}")
        lines.append("---")
        lines.append("Signif. codes:  0 '***' 0.01 '**' 0.05 '*' 0.1 ' ' 1")
        lines.append("=" * 78)

        output = "\n".join(lines)
        if print_output:
            print(output)
        return output
    
    def __call__(
        self,
        X: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute feature importance for the given input.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain. Shape (n_samples, n_features).
        **kwargs : dict
            Additional parameters for explanation.
        
        Returns
        -------
        numpy.ndarray
            Feature importance values. Shape (n_samples, n_features).
        
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        n_samples, n_features = X.shape
        attributions = np.zeros((n_samples, n_features))
        
        raise NotImplementedError(
            "Explainer.__call__ must be implemented by subclasses"
        )
    
    def shap_values(
        self,
        X: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute SHAP-like values (alias for __call__).
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain.
        **kwargs : dict
            Additional parameters.
        
        Returns
        -------
        numpy.ndarray
            Feature importance values.
        """
        return self(X, **kwargs)


class TreeExplainer(Explainer):
    """
    Explainer for tree-based models.
    
    This explainer is optimized for tree-based models like
    Random Forests, Gradient Boosting, etc.
    
    Parameters
    ----------
    model : object
        A tree-based model (e.g., sklearn RandomForest, XGBoost, LightGBM).
    data : numpy.ndarray, optional
        Background data.
    **kwargs : dict
        Additional parameters.
    """
    
    def __init__(
        self,
        model: Any,
        data: Optional[np.ndarray] = None,
        **kwargs: Any
    ):
        """Initialize the TreeExplainer."""
        super().__init__(model, data, **kwargs)
    
    def __call__(
        self,
        X: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute feature importance for tree-based models.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain.
        **kwargs : dict
            Additional parameters.
        
        Returns
        -------
        numpy.ndarray
            Feature importance values.
        """
        # Placeholder implementation
        raise NotImplementedError(
            "TreeExplainer is not yet fully implemented. "
            "Please refer to the documentation for upcoming features."
        )


class LinearExplainer(Explainer):
    """
    Explainer for linear models.
    
    This explainer is optimized for linear models like
    Linear Regression, Logistic Regression, etc.
    
    Parameters
    ----------
    model : object
        A linear model.
    data : numpy.ndarray, optional
        Background data.
    **kwargs : dict
        Additional parameters.
    """
    
    def __init__(
        self,
        model: Any,
        data: Optional[np.ndarray] = None,
        **kwargs: Any
    ):
        """Initialize the LinearExplainer."""
        super().__init__(model, data, **kwargs)
    
    def __call__(
        self,
        X: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute feature importance for linear models.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain.
        **kwargs : dict
            Additional parameters.
        
        Returns
        -------
        numpy.ndarray
            Feature importance values.
        """
        # Placeholder implementation
        raise NotImplementedError(
            "LinearExplainer is not yet fully implemented. "
            "Please refer to the documentation for upcoming features."
        )


class KernelExplainer(Explainer):
    """
    Explainer using kernel-based methods.
    
    This is a model-agnostic explainer that can work with any model.
    
    Parameters
    ----------
    model : callable
        The model to explain.
    data : numpy.ndarray
        Background data (required for kernel methods).
    **kwargs : dict
        Additional parameters.
    """
    
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        **kwargs: Any
    ):
        """Initialize the KernelExplainer."""
        if data is None:
            raise ValueError("KernelExplainer requires background data")
        super().__init__(model, data, **kwargs)
    
    def __call__(
        self,
        X: np.ndarray,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Compute feature importance using kernel methods.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain.
        **kwargs : dict
            Additional parameters.
        
        Returns
        -------
        numpy.ndarray
            Feature importance values.
        """
        # Placeholder implementation
        raise NotImplementedError(
            "KernelExplainer is not yet fully implemented. "
            "Please refer to the documentation for upcoming features."
        )
        
class OTExplainer(Explainer):
    """
    Optimal-transport DFI explainer using Gaussian transport.

    This is the Gaussian DFI estimator without cross-fitting.
    """
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        nsamples: int = 50,
        sampling_method: str = "resample",
        random_state: int = 0,
        **kwargs: Any
    ):
        """Initialize the OTExplainer."""
        super().__init__(model, data, fit_flow=False, **kwargs)
        self.nsamples = nsamples
        self.regularize = kwargs.get("regularize", 1e-6)
        self.sampling_method = sampling_method
        self.random_state = random_state

       
        self.mean = np.mean(data, axis=0, keepdims=True)
        
        self.cov = np.cov(data, rowvar=False, ddof=0)
        self.cov = (self.cov + self.cov.T) / 2  

      
        eigenvals, eigenvecs = np.linalg.eigh(self.cov)
        eigenvals = np.maximum(eigenvals, self.regularize) 

        
        self.L = eigenvecs @ np.diag(eigenvals**0.5) @ eigenvecs.T
       
        self.L_inv = eigenvecs @ np.diag(eigenvals**-0.5) @ eigenvecs.T

        self.Z_full = (data - self.mean) @ self.L_inv

    def __call__(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
       
        n, d = X.shape
        Z = (X - self.mean) @ self.L_inv
        
    
        y_pred = self.model(X)
        
       
        # get per-sample UEIFs in latent space (n_samples, n_features)
        ueifs_Z = self._phi_Z(Z, y_pred)

        # Jacobian sensitivity matrix (constant for linear mapping)
        H = self.L ** 2
        # Map latent-space UEIFs back to original X-space per-sample
        ueifs_X = ueifs_Z @ H.T

        # Aggregate to get mean importance and uncertainty (std) across samples
        n = ueifs_X.shape[0]
        ddof = 1 if n > 1 else 0
        phi_X = np.mean(ueifs_X, axis=0)
        std_X = np.std(ueifs_X, axis=0)
        se_X = np.std(ueifs_X, axis=0, ddof=ddof) / np.sqrt(n)

        phi_Z = np.mean(ueifs_Z, axis=0)
        std_Z = np.std(ueifs_Z, axis=0)
        se_Z = np.std(ueifs_Z, axis=0, ddof=ddof) / np.sqrt(n)

        results = {
            "phi_X": phi_X,
            "std_X": std_X,
            "se_X": se_X,
            "phi_Z": phi_Z,
            "std_Z": std_Z,
            "se_Z": se_Z,
        }
        self._cache_results(results, n)
        return results

    def _phi_Z(self, Z: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
       
        n, d = Z.shape
        ueifs_Z = np.zeros((n, d))
        
        for j in range(d):

            rng = np.random.default_rng(self.random_state + j)
            
            # Z_tilde: (nsamples, n, d)
            Z_tilde = np.tile(Z[None, :, :], (self.nsamples, 1, 1))
            if self.sampling_method == "resample":
                resample_idx = rng.choice(
                    self.Z_full.shape[0], size=(self.nsamples, n), replace=True
                )
                Z_tilde[:, :, j] = self.Z_full[resample_idx, j]
            elif self.sampling_method == "permutation":
                perm_idx = np.array([rng.permutation(n) for _ in range(self.nsamples)])
                Z_tilde[:, :, j] = Z[perm_idx, j]
            elif self.sampling_method == "normal":
                Z_tilde[:, :, j] = rng.normal(0.0, 1.0, size=(self.nsamples, n))
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
            
            Z_tilde_flat = Z_tilde.reshape(-1, d)
            X_tilde_flat = Z_tilde_flat @ self.L + self.mean
            
            y_tilde_flat = self.model(X_tilde_flat)
            y_tilde = y_tilde_flat.reshape(self.nsamples, n).mean(axis=0)
            
            ueifs_Z[:, j] = (y_pred - y_tilde) ** 2

        # Return per-sample UEIFs in latent space (no aggregation here)
        return ueifs_Z


class EOTExplainer(Explainer):
    """
    Entropic optimal-transport DFI explainer (no cross-fitting).

    Fits an entropic transport on background data, then uses it to
    resample Z for counterfactual evaluation with the provided model.
    Supports adaptive epsilon, stochastic transport sampling, and
    Gaussian/empirical targets.

    Notable options:
      - auto_epsilon: enable median-distance heuristic
      - stochastic_transport: sample from k(z|x) instead of barycentric map
      - target: "gaussian" or "empirical" transport target
    """
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        nsamples: int = 50,
        epsilon: float = 0.1,
        auto_epsilon: bool = False,
        target: str = "gaussian",
        transport_type: str = "entropic",
        alpha: float = 0.5,
        n_iter: int = 100,
        tol: float = 1e-6,
        cost_metric: str = "sqeuclidean",
        cost_fn: Optional[Callable[..., np.ndarray]] = None,
        cost_kwargs: Optional[dict] = None,
        categorical_threshold: int = 10,
        feature_types: Optional[np.ndarray] = None,
        feature_weights: Optional[np.ndarray] = None,
        sampling_method: str = "resample",
        stochastic_transport: bool = False,
        n_transport_samples: int = 10,
        random_state: int = 0,
        **kwargs: Any
    ):
        super().__init__(model, data, fit_flow=False, **kwargs)
        self.nsamples = nsamples
        self.epsilon = epsilon
        self.auto_epsilon = auto_epsilon
        self.target = target
        self.transport_type = transport_type
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol
        self.cost_metric = cost_metric
        self.cost_fn = cost_fn
        self.cost_kwargs = cost_kwargs
        self.categorical_threshold = categorical_threshold
        self.feature_types = feature_types
        self.feature_weights = feature_weights
        self.sampling_method = sampling_method
        self.stochastic_transport = stochastic_transport
        self.n_transport_samples = n_transport_samples
        self.random_state = random_state
        self.regularize = kwargs.get("regularize", 1e-6)
        self.transport_plan_ = None
        self.Z_target_ = None
        self.Z_gauss_ = None
        self.detected_types_ = None

        self.mean = np.mean(data, axis=0, keepdims=True)
        self.cov = np.cov(data - self.mean, rowvar=False, ddof=0)
        self.cov = (self.cov + self.cov.T) / 2

        eigenvals, eigenvecs = np.linalg.eigh(self.cov)
        eigenvals = np.maximum(eigenvals, self.regularize)

        self.L = eigenvecs @ np.diag(eigenvals**0.5) @ eigenvecs.T
        self.L_inv = eigenvecs @ np.diag(eigenvals**-0.5) @ eigenvecs.T

        X_centered = data - self.mean
        Z_gauss = X_centered @ self.L_inv
        self.Z_gauss_ = Z_gauss
        if self.auto_epsilon:
            self.epsilon = self._auto_epsilon(X_centered)
        Z_entropic = self._fit_entropic_Z(X_centered, Z_gauss)

        if self.transport_type == "entropic":
            self.Z_fit_ = Z_entropic
        elif self.transport_type == "semiparametric":
            self.Z_fit_ = (1.0 - self.alpha) * Z_gauss + self.alpha * Z_entropic
        else:
            raise ValueError(f"Unknown transport_type: {self.transport_type}")

    def __call__(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        n, d = X.shape
        Z = (X - self.mean) @ self.L_inv
        y_pred = self.model(X)

        ueifs_Z = self._phi_Z(Z, y_pred)

        H = self.L ** 2
        ueifs_X = ueifs_Z @ H.T

        n = ueifs_X.shape[0]
        ddof = 1 if n > 1 else 0
        phi_X = np.mean(ueifs_X, axis=0)
        std_X = np.std(ueifs_X, axis=0)
        se_X = np.std(ueifs_X, axis=0, ddof=ddof) / np.sqrt(n)

        phi_Z = np.mean(ueifs_Z, axis=0)
        std_Z = np.std(ueifs_Z, axis=0)
        se_Z = np.std(ueifs_Z, axis=0, ddof=ddof) / np.sqrt(n)

        results = {
            "phi_X": phi_X,
            "std_X": std_X,
            "se_X": se_X,
            "phi_Z": phi_Z,
            "std_Z": std_Z,
            "se_Z": se_Z,
        }
        self._cache_results(results, n)
        return results

    def _phi_Z(self, Z: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        n, d = Z.shape
        ueifs_Z = np.zeros((n, d))

        def compute_y_perm(j: int, rng: np.random.Generator, Z_pool: np.ndarray) -> np.ndarray:
            Z_tilde = np.tile(Z[None, :, :], (self.nsamples, 1, 1))

            if self.sampling_method == "resample":
                resample_idx = rng.choice(
                    Z_pool.shape[0], size=(self.nsamples, n), replace=True
                )
                Z_tilde[:, :, j] = Z_pool[resample_idx, j]
            elif self.sampling_method == "permutation":
                perm_idx = np.array([rng.permutation(n) for _ in range(self.nsamples)])
                Z_tilde[:, :, j] = Z[perm_idx, j]
            elif self.sampling_method == "normal":
                Z_tilde[:, :, j] = rng.normal(0.0, 1.0, size=(self.nsamples, n))
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

            Z_tilde_flat = Z_tilde.reshape(-1, d)
            X_tilde_flat = Z_tilde_flat @ self.L + self.mean

            y_tilde_flat = self.model(X_tilde_flat)
            return y_tilde_flat.reshape(self.nsamples, n).mean(axis=0)

        for j in range(d):
            rng = np.random.default_rng(self.random_state + j)
            if self.sampling_method == "resample" and self.stochastic_transport:
                all_y_perm = []
                for _ in range(self.n_transport_samples):
                    Z_pool = self._sample_transport_pool(rng)
                    all_y_perm.append(compute_y_perm(j, rng, Z_pool))
                y_perm = np.mean(all_y_perm, axis=0)
            else:
                y_perm = compute_y_perm(j, rng, self.Z_fit_)

            ueifs_Z[:, j] = (y_pred - y_perm) ** 2

        return ueifs_Z

    def _auto_epsilon(self, X_centered: np.ndarray) -> float:
        if X_centered.shape[0] < 2:
            return self.epsilon
        n = min(1000, X_centered.shape[0])
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(X_centered.shape[0], size=n, replace=False)
        X_sub = X_centered[idx]
        dists = pdist(X_sub)
        if dists.size == 0:
            return self.epsilon
        median_dist = np.median(dists)
        return (median_dist ** 2) / X_centered.shape[1]

    def _make_target(self, Z_gauss: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n, d = Z_gauss.shape
        if self.target == "gaussian":
            return rng.standard_normal((n, d))
        if self.target == "empirical":
            perm = rng.permutation(n)
            return Z_gauss[perm]
        raise ValueError(f"Unknown target: {self.target}")

    def _fit_entropic_Z(self, X_centered: np.ndarray, Z_gauss: np.ndarray) -> np.ndarray:
        Z_target = self._make_target(Z_gauss)
        C = self._compute_cost_matrix(X_centered, Z_target)
        P = self._sinkhorn(C)
        self.Z_target_ = Z_target
        self.transport_plan_ = P
        return P @ Z_target

    def _compute_cost_matrix(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        if self.cost_fn is not None:
            kwargs = self.cost_kwargs or {}
            return self.cost_fn(X, Z, **kwargs)

        if self.cost_metric in ("auto", "gower"):
            detected = detect_feature_types(
                X,
                categorical_threshold=self.categorical_threshold,
                feature_types=self.feature_types,
            )
            self.detected_types_ = detected
            types = detected["types"]
            ranges = detected["ranges"]
            if self.cost_metric == "auto":
                if np.all(types == "continuous"):
                    return cdist(X, Z, metric="sqeuclidean")
            return gower_cost_matrix(
                X,
                Z,
                feature_types=types,
                feature_ranges=ranges,
                feature_weights=self.feature_weights,
            )

        if self.cost_metric == "sqeuclidean":
            return cdist(X, Z, metric="sqeuclidean")

        return cdist(X, Z, metric=self.cost_metric)

    def _sample_transport_pool(self, rng: np.random.Generator) -> np.ndarray:
        if self.transport_plan_ is None or self.Z_target_ is None:
            return self.Z_fit_

        P = self.transport_plan_
        n, m = P.shape
        d = self.Z_target_.shape[1]
        Z_entropic = np.zeros((n, d))

        for i in range(n):
            probs = P[i]
            if probs.sum() <= 0:
                probs = np.ones(m) / m
            else:
                probs = probs / probs.sum()
            idx = rng.choice(m, p=probs)
            Z_entropic[i] = self.Z_target_[idx]

        if self.transport_type == "semiparametric" and self.Z_gauss_ is not None:
            return (1.0 - self.alpha) * self.Z_gauss_ + self.alpha * Z_entropic
        return Z_entropic

    def _sinkhorn(self, C: np.ndarray) -> np.ndarray:
        n, m = C.shape
        a = np.ones(n) / n
        b = np.ones(m) / m

        K = np.exp(-C / self.epsilon)
        K = np.maximum(K, 1e-300)

        u = np.ones(n)
        v = np.ones(m)

        for _ in range(self.n_iter):
            u_prev = u.copy()
            u = a / (K @ v + 1e-300)
            v = b / (K.T @ u + 1e-300)
            if np.max(np.abs(u - u_prev)) < self.tol:
                break

        P = np.diag(u) @ K @ np.diag(v)
        row_sums = P.sum(axis=1, keepdims=True)
        return P / np.maximum(row_sums, 1e-300)


DFIExplainer = OTExplainer

class FlowExplainer(Explainer):
    """
    Flow Matching DFI explainer using non-linear generative flow.

    This explainer uses a learned flow model to transform X to disentangled latent Z.
    Unlike OTExplainer which uses a constant Gaussian mapping, FlowExplainer computes
    sample-specific Jacobian matrices for mapping importance back to original space.
    """

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        nsamples: int = 50,
        sampling_method: str = "resample",
        random_state: int = 0,
        **kwargs: Any
    ):
        """Initialize the FlowExplainer.
        
        Parameters
        ----------
        model : callable
            The model to explain.
        data : np.ndarray
            Background/training data for flow model training.
        nsamples : int
            Number of samples for conditional prediction in latent space.
        sampling_method : str
            Method for sampling in latent space ("resample", "permutation", "normal").
        random_state : int
            Random seed for reproducibility.
        **kwargs : dict
            Additional arguments including num_steps for flow training.
        """
        super().__init__(model, data, fit_flow=True, **kwargs)
        self.nsamples = nsamples
        self.sampling_method = sampling_method
        self.random_state = random_state
        
        # Store background data in latent space for future sampling
        Z_full = self.flow_engine.sample_batch(data, t_span=(1, 0))
        # Convert from torch tensor to numpy if needed
        if hasattr(Z_full, 'detach'):
            Z_full = Z_full.detach().numpy()
        self.Z_full = Z_full
        
        # Compute diagnostics: latent independence and distribution fidelity
        self._compute_flow_diagnostics(data, Z_full)

    def __call__(self, X: np.ndarray, **kwargs: Any) -> dict:
        """Compute flow-disentangled feature importance.
        
        Parameters
        ----------
        X : np.ndarray
            Input data to explain. Shape (n_samples, n_features).
        
        Returns
        -------
        dict
            Dictionary with keys:
            - "phi_X": Mean importance in original space
            - "std_X": Std dev of importance in original space
            - "se_X": Standard error in original space
            - "phi_Z": Mean importance in latent space
            - "std_Z": Std dev in latent space
            - "se_Z": Standard error in latent space
        """
        n, d = X.shape
        
        # Step 1: Encode X to latent Z
        Z = self.flow_engine.sample_batch(X, t_span=(1, 0))
        # Convert from torch tensor to numpy if needed
        if hasattr(Z, 'detach'):
            Z = Z.detach().numpy()
        
        # Step 2: Get predictions on original X
        y_pred = self.model(X)
        
        # Step 3: Compute per-sample UEIFs in latent space
        ueifs_Z = self._phi_Z(Z, y_pred)
        
        # Step 4: Compute sample-specific Jacobian mappings
        # For each sample, H_i = (dX/dZ_i)^2 (element-wise square of Jacobian)
        ueifs_X = np.zeros_like(ueifs_Z)
        for i in range(n):
            H_i = self.flow_engine.Jacobi_N(Z[i:i+1], t_span=(0, 1)) ** 2  # (1, d, d)
            # Convert from torch tensor to numpy if needed
            if hasattr(H_i, 'detach'):
                H_i = H_i.detach().numpy()
            H_i = H_i[0]  # Remove batch dimension: (d, d)
            # Map latent importance to original space
            ueifs_X[i, :] = ueifs_Z[i, :] @ H_i.T
        
        # Step 5: Aggregate across samples
        ddof = 1 if n > 1 else 0
        
        phi_X = np.mean(ueifs_X, axis=0)
        std_X = np.std(ueifs_X, axis=0)
        se_X = np.std(ueifs_X, axis=0, ddof=ddof) / np.sqrt(n)
        
        phi_Z = np.mean(ueifs_Z, axis=0)
        std_Z = np.std(ueifs_Z, axis=0)
        se_Z = np.std(ueifs_Z, axis=0, ddof=ddof) / np.sqrt(n)
        
        results = {
            "phi_X": phi_X,
            "std_X": std_X,
            "se_X": se_X,
            "phi_Z": phi_Z,
            "std_Z": std_Z,
            "se_Z": se_Z,
        }
        self._cache_results(results, n)
        return results

    def _phi_Z(self, Z: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute per-sample UEIFs in latent space.
        
        Parameters
        ----------
        Z : np.ndarray
            Latent representations. Shape (n_samples, n_features).
        y_pred : np.ndarray
            Model predictions on original X. Shape (n_samples,).
        
        Returns
        -------
        np.ndarray
            Per-sample UEIFs in latent space. Shape (n_samples, n_features).
        """
        n, d = Z.shape
        ueifs_Z = np.zeros((n, d))
        
        for j in range(d):
            rng = np.random.default_rng(self.random_state + j)
            
            # Create perturbed latent samples by resampling feature j
            # Z_tilde: (nsamples, n, d)
            Z_tilde = np.tile(Z[None, :, :], (self.nsamples, 1, 1))
            
            if self.sampling_method == "resample":
                # Resample j-th dimension from background distribution
                resample_idx = rng.choice(
                    self.Z_full.shape[0], size=(self.nsamples, n), replace=True
                )
                Z_tilde[:, :, j] = self.Z_full[resample_idx, j]
            elif self.sampling_method == "permutation":
                # Randomly permute j-th dimension
                perm_idx = np.array([rng.permutation(n) for _ in range(self.nsamples)])
                Z_tilde[:, :, j] = Z[perm_idx, j]
            elif self.sampling_method == "normal":
                # Sample from standard normal
                Z_tilde[:, :, j] = rng.normal(0.0, 1.0, size=(self.nsamples, n))
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
            
            # Flatten for batch processing
            Z_tilde_flat = Z_tilde.reshape(-1, d)
            
            # Decode perturbed latent samples back to original space
            X_tilde_flat = self.flow_engine.sample_batch(Z_tilde_flat, t_span=(0, 1))
            # Convert from torch tensor to numpy if needed
            if hasattr(X_tilde_flat, 'detach'):
                X_tilde_flat = X_tilde_flat.detach().numpy()
            
            # Get predictions and average across nsamples
            y_tilde_flat = self.model(X_tilde_flat)
            y_tilde = y_tilde_flat.reshape(self.nsamples, n).mean(axis=0)
            
            # Compute UEIF for feature j
            ueifs_Z[:, j] = (y_pred - y_tilde) ** 2
        
        return ueifs_Z

    def _compute_flow_diagnostics(self, X_orig: np.ndarray, Z_full: np.ndarray) -> None:
        """
        Compute and report flow model diagnostics.
        
        Evaluates:
        1. Latent Independence: Median Distance Correlation of latent dimensions
        2. Distribution Fidelity: MMD between original and reconstructed data
        
        Parameters
        ----------
        X_orig : np.ndarray
            Original background data.
        Z_full : np.ndarray
            Latent encodings of background data (already numpy).
        """
        from . import utils
        
        # Determine subset size for large datasets
        n_samples = X_orig.shape[0]
        subset_size = min(n_samples, 1000) if n_samples > 1000 else None
        
        # 1. Compute latent independence
        dcor_matrix, median_dcor = utils.compute_latent_independence(Z_full, subset_size=subset_size)
        
        # 2. Reconstruct data from latent space
        X_hat = self.flow_engine.sample_batch(Z_full, t_span=(0, 1))
        # Convert from torch tensor to numpy if needed
        if hasattr(X_hat, 'detach'):
            X_hat = X_hat.detach().numpy()
        
        # 3. Compute distribution fidelity (MMD)
        mmd_score = utils.compute_mmd(X_orig, X_hat, subset_size=subset_size)
        
        # Store diagnostics
        self.diagnostics = {
            "latent_independence_dcor": dcor_matrix,
            "latent_independence_median": median_dcor,
            "distribution_fidelity_mmd": mmd_score,
        }
        
        # Print professional diagnostic report
        print("\n" + "=" * 50)
        print("--- Flow Model Diagnostics ---")
        print("=" * 50)
        print(f"Latent Independence (Median dCor):  {median_dcor:.6f}")
        print(f"  → Lower values (~0) indicate higher independence")
        print(f"Distribution Fidelity (MMD):       {mmd_score:.6f}")
        print(f"  → Lower values (~0) indicate better fidelity")
        print("=" * 50 + "\n")
