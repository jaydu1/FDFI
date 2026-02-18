"""
Explainer classes for DFI.

This module contains the main explainer classes for computing
disentangled feature importance.
"""

import numpy as np
from typing import Optional, Union, Callable, Any, Tuple
from scipy.spatial.distance import cdist, pdist
from scipy import stats
from .utils import (
    TwoComponentMixture,
    detect_feature_types,
    gower_cost_matrix,
    compute_latent_independence,
    compute_mmd,
)



class Explainer:
    """
    Base class for DFI explainers.
    
    This class provides the interface for computing feature importance
    using disentangled methods, similar to SHAP explainers.
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
        self.verbose = kwargs.get("verbose", False)
        self.diagnostics = None
        self.compute_diagnostics = kwargs.get("compute_diagnostics", True)
        self.diagnostics_subset_max_samples = kwargs.get(
            "diagnostics_subset_max_samples", 1000
        )
        self.latent_independence_thresholds = kwargs.get(
            "latent_independence_thresholds", (0.1, 0.25)
        )
        self.distribution_fidelity_thresholds = kwargs.get(
            "distribution_fidelity_thresholds", (0.05, 0.15)
        )

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
        var_floor_method: str = "mixture",
        var_floor_quantile: float = 0.95,
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
        var_floor_method: str = "mixture",
        var_floor_quantile: float = 0.95,
        margin: float = 0.0,
        margin_method: str = "mixture",
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

    def _log(self, message: str, level: str = "INFO") -> None:
        """Consistent, verbosity-aware logging with unified style."""
        v = getattr(self, "verbose", False)
        if v is False:
            return
        if v in ("all", True, "final"):
            print(f"[FDFI][{level.upper()}] {message}")

    @staticmethod
    def _qualitative_score(
        value: float,
        thresholds: Tuple[float, float],
        lower_is_better: bool = True,
    ) -> str:
        """
        Return GOOD / MODERATE / POOR label based on thresholds.

        thresholds: (good_cutoff, moderate_cutoff); behavior flips if lower_is_better=False.
        """
        good, moderate = thresholds
        if lower_is_better:
            if value < good:
                return "GOOD"
            if value < moderate:
                return "MODERATE"
            return "POOR"
        if value > moderate:
            return "POOR"
        if value > good:
            return "MODERATE"
        return "GOOD"

    def _decode_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode latent-space samples to original feature space.

        Subclasses with disentanglement maps (OT/EOT/Flow) should implement this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement latent decoding."
        )

    def _get_diagnostics_sources(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return default (X_orig, Z_full) used for diagnostics."""
        return self.data, getattr(self, "Z_full", None)

    def _compute_diagnostics(
        self,
        X_orig: Optional[np.ndarray] = None,
        Z_full: Optional[np.ndarray] = None,
        report_title: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Compute and store generic disentanglement diagnostics.

        Diagnostics:
        - Latent independence: median distance correlation across latent dims.
        - Distribution fidelity: Maximum Mean Discrepancy (MMD) between
          original data and reconstructions.
        """
        if not self.compute_diagnostics:
            return None

        if X_orig is None or Z_full is None:
            default_X, default_Z = self._get_diagnostics_sources()
            if X_orig is None:
                X_orig = default_X
            if Z_full is None:
                Z_full = default_Z

        if X_orig is None or Z_full is None:
            return None

        X_arr = np.asarray(X_orig)
        Z_arr = np.asarray(Z_full)
        if X_arr.ndim != 2 or Z_arr.ndim != 2:
            raise ValueError("X_orig and Z_full must both be 2D arrays.")

        n = min(X_arr.shape[0], Z_arr.shape[0])
        if n == 0:
            return None
        X_use = X_arr[:n]
        Z_use = Z_arr[:n]

        subset_size = None
        if (
            self.diagnostics_subset_max_samples is not None
            and n > self.diagnostics_subset_max_samples
        ):
            subset_size = int(self.diagnostics_subset_max_samples)

        dcor_matrix, median_dcor = compute_latent_independence(
            Z_use, subset_size=subset_size
        )
        dcor_label = self._qualitative_score(
            float(median_dcor),
            thresholds=self.latent_independence_thresholds,
            lower_is_better=True,
        )

        X_hat = self._decode_from_Z(Z_use)
        mmd_score = compute_mmd(X_use, X_hat, subset_size=subset_size)
        mmd_label = self._qualitative_score(
            float(mmd_score),
            thresholds=self.distribution_fidelity_thresholds,
            lower_is_better=True,
        )

        self.diagnostics = {
            "latent_independence_dcor": dcor_matrix,
            "latent_independence_median": float(median_dcor),
            "distribution_fidelity_mmd": float(mmd_score),
            "latent_independence_label": dcor_label,
            "distribution_fidelity_label": mmd_label,
        }

        title = report_title if report_title is not None else self.__class__.__name__
        self._log(f"{title} Diagnostics", level="diag")
        self._log(
            f"Latent independence (median dCor): {median_dcor:.6f} [{dcor_label}]  "
            "-> lower is better",
            level="diag",
        )
        self._log(
            f"Distribution fidelity (MMD):       {mmd_score:.6f} [{mmd_label}]  "
            "-> lower is better",
            level="diag",
        )
        return self.diagnostics

    def diagnose(
        self,
        X_orig: Optional[np.ndarray] = None,
        Z_full: Optional[np.ndarray] = None,
        report_title: Optional[str] = None,
    ) -> dict:
        """Public API to compute (or recompute) diagnostics."""
        diagnostics = self._compute_diagnostics(
            X_orig=X_orig,
            Z_full=Z_full,
            report_title=report_title,
        )
        if diagnostics is None:
            raise ValueError(
                "Diagnostics unavailable. Ensure diagnostics are enabled and latent data exists."
            )
        return diagnostics
    
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
        self._compute_diagnostics(report_title="OTExplainer")

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

    def _decode_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """Decode Z to X using the Gaussian OT linear map."""
        return Z @ self.L + self.mean


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
      - decode_method: "auto" (default), "knn", or "linear" latent decoder
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
        decode_method: str = "auto",
        decode_n_neighbors: int = 25,
        decode_weights: Union[str, Callable] = "distance",
        decode_leave_one_out: bool = True,
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
        self.decode_method = decode_method
        self.decode_n_neighbors = decode_n_neighbors
        self.decode_weights = decode_weights
        self.decode_leave_one_out = decode_leave_one_out
        self.decode_method_effective_ = "linear"
        self.random_state = random_state
        self.regularize = kwargs.get("regularize", 1e-6)
        self.transport_plan_ = None
        self.Z_target_ = None
        self.Z_gauss_ = None
        self._decode_model = None
        self.X_centered_ = None
        self.detected_types_ = None

        self.mean = np.mean(data, axis=0, keepdims=True)
        self.cov = np.cov(data - self.mean, rowvar=False, ddof=0)
        self.cov = (self.cov + self.cov.T) / 2

        eigenvals, eigenvecs = np.linalg.eigh(self.cov)
        eigenvals = np.maximum(eigenvals, self.regularize)

        self.L = eigenvecs @ np.diag(eigenvals**0.5) @ eigenvecs.T
        self.L_inv = eigenvecs @ np.diag(eigenvals**-0.5) @ eigenvecs.T

        X_centered = data - self.mean
        self.X_centered_ = X_centered
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

        self._build_decoder()
        self.Z_full = self.Z_fit_
        self._compute_diagnostics(report_title="EOTExplainer")

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
        """
        Auto-tune epsilon from latent geometry.

        We estimate pairwise distances in Gaussian-whitened latent space and use
        a conservative shrinkage factor to avoid over-smoothing transport plans
        on multi-modal data.
        """
        if X_centered.shape[0] < 2:
            return self.epsilon

        Z_ref = X_centered @ self.L_inv
        n = min(1000, Z_ref.shape[0])
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(Z_ref.shape[0], size=n, replace=False)
        Z_sub = Z_ref[idx]

        sq_dists = pdist(Z_sub, metric="sqeuclidean")
        if sq_dists.size == 0:
            return self.epsilon

        median_sq_dist = np.median(sq_dists)
        eps = 0.25 * (median_sq_dist / Z_ref.shape[1])
        return max(eps, 1e-3)

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

    def _build_decoder(self) -> None:
        """Build optional nonlinear decoder from latent Z to original X."""
        if self.decode_method == "linear":
            self.decode_method_effective_ = "linear"
            self._decode_model = None
            return
        if self.decode_method not in ("knn", "auto"):
            raise ValueError(
                "decode_method must be 'auto', 'linear', or 'knn'"
            )
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:
            raise ImportError(
                "decode_method='knn' requires scikit-learn."
            ) from exc

        n_neighbors = int(max(1, min(self.decode_n_neighbors, self.Z_fit_.shape[0])))
        self._decode_model = NearestNeighbors(n_neighbors=n_neighbors)
        self._decode_model.fit(self.Z_fit_)

        if self.decode_method == "knn":
            self.decode_method_effective_ = "knn"
            return

        # Auto-select the decoder with lower reconstruction MSE on training pairs.
        X_linear = self.Z_fit_ @ self.L
        mse_linear = float(np.mean((self.X_centered_ - X_linear) ** 2))
        X_knn = self._predict_knn_centered(self.Z_fit_)
        mse_knn = float(np.mean((self.X_centered_ - X_knn) ** 2))

        if mse_knn < mse_linear:
            self.decode_method_effective_ = "knn"
        else:
            self.decode_method_effective_ = "linear"
            self._decode_model = None

    def _predict_knn_centered(self, Z_arr: np.ndarray) -> np.ndarray:
        """Predict centered X from Z via the configured kNN decoder."""
        if self._decode_model is None:
            raise RuntimeError("kNN decoder is not initialized.")

        n_train = self.Z_fit_.shape[0]
        k = int(max(1, min(self.decode_n_neighbors, n_train)))
        k_query = min(
            n_train,
            k + 1 if self.decode_leave_one_out and n_train > 1 else k,
        )
        dists, nbr_idx = self._decode_model.kneighbors(
            Z_arr, n_neighbors=k_query, return_distance=True
        )

        X_centered_hat = np.zeros((Z_arr.shape[0], self.X_centered_.shape[1]))
        for i in range(Z_arr.shape[0]):
            row_idx = nbr_idx[i]
            row_dist = dists[i]

            if (
                self.decode_leave_one_out
                and row_dist.size > 1
                and row_dist[0] <= 1e-12
            ):
                row_idx = row_idx[1:]
                row_dist = row_dist[1:]

            row_idx = row_idx[:k]
            row_dist = row_dist[:k]
            X_neighbors = self.X_centered_[row_idx]

            if callable(self.decode_weights):
                w = np.asarray(self.decode_weights(row_dist))
                if w.shape != row_dist.shape:
                    raise ValueError(
                        "decode_weights callable must return shape (n_neighbors,)."
                    )
                w = np.maximum(w, 0.0)
                if w.sum() <= 0:
                    w = np.ones_like(row_dist)
                w = w / w.sum()
                X_centered_hat[i] = w @ X_neighbors
            elif self.decode_weights == "uniform":
                X_centered_hat[i] = X_neighbors.mean(axis=0)
            elif self.decode_weights == "distance":
                w = 1.0 / np.maximum(row_dist, 1e-12)
                w = w / w.sum()
                X_centered_hat[i] = w @ X_neighbors
            else:
                raise ValueError(
                    "decode_weights must be 'uniform', 'distance', or a callable."
                )
        return X_centered_hat

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

    def _decode_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode Z to X.

        - linear: Gaussian reference map X = ZL + mean
        - knn: local nonlinear inverse map fitted on (Z_fit_, X_centered)
        """
        Z_arr = np.asarray(Z)
        if Z_arr.ndim != 2:
            raise ValueError("Z must be a 2D array.")

        if self.decode_method_effective_ == "knn" and self._decode_model is not None:
            X_centered_hat = self._predict_knn_centered(Z_arr)
        else:
            X_centered_hat = Z_arr @ self.L
        return X_centered_hat + self.mean


DFIExplainer = OTExplainer


class FlowExplainer(Explainer):
    """
    Flow-based DFI explainer using normalizing flows.
    
    Implements CPI (Conditional Permutation Importance) and 
    SCPI (Sobol-CPI) methods. Both measure feature importance in Z-space:

    - CPI: Squared difference after averaging predictions: (Y - E[f(X_tilde)])^2
    - SCPI: Conditional variance of predictions: Var[f(X_tilde)]
    
    For L2 loss with independent (disentangled) features, CPI and SCPI give
    similar results. SCPI is related to the Sobol total-order sensitivity index.
    
    Z-space importance is transformed to X-space using the Jacobian of the flow
    ``phi_X[l] = sum_k H[l,k]^2 * phi_Z[k]`` where H = dX/dZ is the Jacobian
    of the decoder transformation.
    
    Parameters
    ----------
    model : callable
        The model to explain. Should take (n, d) array and return (n,) predictions.
    data : numpy.ndarray
        Background data for fitting flow and resampling. Shape (n, d).
    flow_model : object, optional
        Pre-trained flow model. If None, will create default FlowMatchingModel.
    fit_flow : bool, default=True
        Whether to fit flow model during initialization.
    nsamples : int, default=50
        Number of Monte Carlo samples per feature.
    sampling_method : str, default='resample'
        Method for generating counterfactual Z values:
        - 'resample': Sample from encoded background data
        - 'permutation': Permute within test set
        - 'normal': Sample from standard normal
        - 'condperm': Conditional permutation (regress Z_j | Z_{-j})
    permuter : object, optional
        Regressor for conditional permutation method. Defaults to LinearRegression.
    method : str, default='cpi'
        Which importance method to use:
        - 'cpi': Conditional Permutation Importance - average predictions first
        - 'scpi': Sobol-CPI - average squared differences
        - 'both': Compute both CPI and SCPI
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool or str, default='final'
        Controls training output:
        - True or 'all': Show full progress bar
        - 'final': Only print final step status (default)
        - False: Silent
    compute_diagnostics : bool, default=True
        Whether to compute disentanglement diagnostics at setup time.
    flow_solver_rtol : float, default=1e-3
        Relative tolerance for default ODE integration in flow encode/decode.
    flow_solver_atol : float, default=1e-5
        Absolute tolerance for default ODE integration in flow encode/decode.
    diagnostics_solver_rtol : float, default=1e-6
        Relative tolerance for diagnostics round-trip integration.
    diagnostics_solver_atol : float, default=1e-8
        Absolute tolerance for diagnostics round-trip integration.
    **kwargs : dict
        Additional arguments passed to FlowMatchingModel if creating default.
    
    Attributes
    ----------
    flow_model : object
        The fitted normalizing flow model.
    Z_full : numpy.ndarray
        Encoded background data in latent space.
    method : str
        The importance method being used ('cpi', 'scpi', or 'both').
    
    Examples
    --------
    >>> import numpy as np
    >>> from fdfi.explainers import FlowExplainer
    >>> 
    >>> # Define a simple model
    >>> def model(x):
    ...     return x[:, 0] + 2 * x[:, 1]
    >>> 
    >>> # Create background data
    >>> X_train = np.random.randn(200, 5)
    >>> X_test = np.random.randn(50, 5)
    >>> 
    >>> # CPI only (default)
    >>> explainer = FlowExplainer(model, X_train, method='cpi')
    >>> results = explainer(X_test)
    >>> 
    >>> # SCPI (Sobol-CPI - different averaging order)
    >>> explainer = FlowExplainer(model, X_train, method='scpi')
    >>> results = explainer(X_test)
    """
    
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        flow_model: Optional[Any] = None,
        fit_flow: bool = True,
        nsamples: int = 50,
        sampling_method: str = "resample",
        permuter: Optional[Any] = None,
        method: str = "cpi",
        random_state: Optional[int] = None,
        verbose: Union[bool, str] = "final",
        compute_diagnostics: bool = True,
        **kwargs: Any
    ):
        """Initialize the FlowExplainer."""
        # Don't fit flow in base class
        super().__init__(
            model,
            data,
            fit_flow=False,
            verbose=verbose,
            compute_diagnostics=compute_diagnostics,
            **kwargs,
        )
        
        self.nsamples = nsamples
        self.sampling_method = sampling_method
        self.random_state = random_state if random_state is not None else 0
        self.method = method.lower()
        self.verbose = verbose
        self.kwargs = kwargs
        self.flow_solver_method = kwargs.get("flow_solver_method", "dopri5")
        self.flow_solver_rtol = kwargs.get("flow_solver_rtol", 1e-3)
        self.flow_solver_atol = kwargs.get("flow_solver_atol", 1e-5)
        self.diagnostics_solver_method = kwargs.get(
            "diagnostics_solver_method", self.flow_solver_method
        )
        self.diagnostics_solver_rtol = kwargs.get("diagnostics_solver_rtol", 1e-6)
        self.diagnostics_solver_atol = kwargs.get("diagnostics_solver_atol", 1e-8)
        self.flow_training_seed = kwargs.get("flow_training_seed", self.random_state)
        
        if self.method not in ("cpi", "scpi", "both"):
            raise ValueError(f"method must be 'cpi', 'scpi', or 'both', got '{method}'")
        
        # Set up permuter for condperm method
        if permuter is None:
            from sklearn.linear_model import LinearRegression
            self.permuter = LinearRegression()
        else:
            self.permuter = permuter
        
        # Flow model setup
        self.flow_model = flow_model
        self.Z_full = None
        
        if flow_model is not None:
            # Use provided flow model
            self._encode_background()
            if self.Z_full is not None:
                self._refresh_flow_diagnostics()
        elif fit_flow and data is not None:
            # Create and fit default flow model
            self.fit_flow(num_steps=kwargs.get("num_steps", 5000), verbose=self.verbose)
    
    def fit_flow(
        self,
        X: Optional[np.ndarray] = None,
        num_steps: int = 5000,
        verbose: Union[bool, str] = None,
        **kwargs
    ) -> "FlowExplainer":
        """
        Fit the flow model on data.
        
        Can be called after initialization with fit_flow=False,
        or to refit on new data.
        
        Parameters
        ----------
        X : numpy.ndarray, optional
            Data to fit on. If None, uses self.data.
        num_steps : int, default=5000
            Number of training steps.
        verbose : bool or str, optional
            Controls training output. If None, uses self.verbose.
            - True or 'all': Show full progress bar
            - 'final': Only print final step status (default)
            - False: Silent
        **kwargs
            Additional arguments passed to flow_model.fit().
        
        Returns
        -------
        self
            For method chaining.
        """
        if X is not None:
            self.data = X
        
        if self.data is None:
            raise ValueError("No data provided for flow fitting.")
        
        try:
            from .models import FlowMatchingModel
        except ImportError as exc:
            raise ImportError(
                "Flow matching requires torch; install it with: pip install torch torchdiffeq"
            ) from exc

        import torch
        if self.flow_training_seed is not None:
            torch.manual_seed(int(self.flow_training_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(self.flow_training_seed))
            np.random.seed(int(self.flow_training_seed))
        
        d = self.data.shape[1]
        self.flow_model = FlowMatchingModel(
            X=self.data,
            dim=d,
            device=self.kwargs.get("device"),
            hidden_dim=self.kwargs.get("hidden_dim", 64),
            time_embed_dim=self.kwargs.get("time_embed_dim", 32),
            num_blocks=self.kwargs.get("num_blocks", 1),
            use_bn=self.kwargs.get("use_bn", False),
        )
        
        _verbose = verbose if verbose is not None else getattr(self, 'verbose', 'final')
        self.verbose = _verbose
        if _verbose is True or _verbose == 'all' or _verbose == 'final':
            self._log("Training flow model...", level="info")
        self.flow_model.fit(num_steps=num_steps, verbose=_verbose, **kwargs)
        
        # Encode background data
        self._encode_background()
        if self.Z_full is not None:
            self._refresh_flow_diagnostics()
        
        return self
    
    def set_flow(self, flow_model: Any) -> "FlowExplainer":
        """
        Set a user-provided flow model.
        
        The flow model must have a sample_batch(x, t_span) method where:
        - t_span=(1, 0) encodes X to Z
        - t_span=(0, 1) decodes Z to X
        
        Parameters
        ----------
        flow_model : object
            A flow model with sample_batch(x, t_span) method.
        
        Returns
        -------
        self
            For method chaining.
        """
        self.flow_model = flow_model
        self._encode_background()
        if self.Z_full is not None:
            self._refresh_flow_diagnostics()
        return self
    
    def _encode_background(self) -> None:
        """Encode background data to Z space."""
        if self.data is not None and self.flow_model is not None:
            self.Z_full = self._encode_to_Z(self.data)
    
    def _encode_to_Z(
        self,
        X: np.ndarray,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Transform X to latent space Z."""
        if self.flow_model is None:
            raise ValueError("Flow model not set. Call fit_flow() or set_flow() first.")
        
        _rtol = self.flow_solver_rtol if rtol is None else rtol
        _atol = self.flow_solver_atol if atol is None else atol
        _method = self.flow_solver_method if method is None else method

        import torch
        with torch.no_grad():
            Z = self.flow_model.sample_batch(
                X,
                t_span=(1, 0),
                rtol=_rtol,
                atol=_atol,
                method=_method,
            )
            if isinstance(Z, torch.Tensor):
                Z = Z.cpu().numpy()
        return Z
    
    def _decode_to_X(
        self,
        Z: np.ndarray,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Transform Z back to X space."""
        if self.flow_model is None:
            raise ValueError("Flow model not set. Call fit_flow() or set_flow() first.")
        
        _rtol = self.flow_solver_rtol if rtol is None else rtol
        _atol = self.flow_solver_atol if atol is None else atol
        _method = self.flow_solver_method if method is None else method

        import torch
        with torch.no_grad():
            X_hat = self.flow_model.sample_batch(
                Z,
                t_span=(0, 1),
                rtol=_rtol,
                atol=_atol,
                method=_method,
            )
            if isinstance(X_hat, torch.Tensor):
                X_hat = X_hat.cpu().numpy()
        return X_hat

    def _decode_from_Z(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode Z to X for diagnostics using tighter ODE tolerances.

        This keeps reconstruction-fidelity diagnostics focused on model fit
        rather than numerical integration error.
        """
        return self._decode_to_X(
            Z,
            rtol=self.diagnostics_solver_rtol,
            atol=self.diagnostics_solver_atol,
            method=self.diagnostics_solver_method,
        )

    def _refresh_flow_diagnostics(self) -> None:
        """Compute diagnostics from high-precision X -> Z -> X round-trip."""
        if self.data is None or self.flow_model is None:
            return
        Z_diag = self._encode_to_Z(
            self.data,
            rtol=self.diagnostics_solver_rtol,
            atol=self.diagnostics_solver_atol,
            method=self.diagnostics_solver_method,
        )
        self._compute_diagnostics(
            X_orig=self.data,
            Z_full=Z_diag,
            report_title="Flow Model",
        )
    
    def _compute_jacobian(self, Z: np.ndarray, batch_size: int = 50) -> np.ndarray:
        """
        Compute the Jacobian dX/dZ averaged over samples.
        
        For the flow decoder T: Z -> X, computes H = dX/dZ at each sample point
        and returns the average H matrix. This is used to transform Z-space
        importance to X-space importance via:
            phi_X[l] = sum_k H[l,k]^2 * phi_Z[k]
        
        Parameters
        ----------
        Z : numpy.ndarray
            Latent space data, shape (n, d).
        batch_size : int, default=50
            Batch size for Jacobian computation.
        
        Returns
        -------
        H : numpy.ndarray
            Average Jacobian matrix of shape (d, d), where H[l, k] = dX_l/dZ_k.
        """
        import torch
        from torch.autograd.functional import jacobian
        
        if self.flow_model is None:
            raise ValueError("Flow model not set. Call fit_flow() or set_flow() first.")
        
        n, d = Z.shape
        
        # Get device from flow model
        if hasattr(self.flow_model, 'device'):
            device = self.flow_model.device
        elif hasattr(self.flow_model, 'model'):
            device = next(self.flow_model.model.parameters()).device
        else:
            device = torch.device('cpu')
        
        # Define decoder function for jacobian computation
        def decoder_fn(z_single):
            """Decode a single Z vector to X."""
            z_batch = z_single.unsqueeze(0)  # (1, d)
            x_batch = self.flow_model.sample_batch(
                z_batch,
                t_span=(0, 1),
                rtol=self.flow_solver_rtol,
                atol=self.flow_solver_atol,
                method=self.flow_solver_method,
            )
            if not isinstance(x_batch, torch.Tensor):
                x_batch = torch.tensor(x_batch, dtype=torch.float32, device=device)
            return x_batch.squeeze(0)  # (d,)
        
        # Compute Jacobian for a subset of samples (for efficiency)
        n_samples = min(n, batch_size)
        indices = np.linspace(0, n - 1, n_samples, dtype=int)
        
        H_sum = np.zeros((d, d))
        
        for idx in indices:
            z_single = torch.tensor(Z[idx], dtype=torch.float32, device=device)
            # jacobian returns (output_dim, input_dim) = (d, d)
            jac = jacobian(decoder_fn, z_single)
            H_sum += jac.cpu().numpy()
        
        H_avg = H_sum / n_samples
        return H_avg
    
    def _phi_Z(self, Z: np.ndarray, y: np.ndarray) -> tuple:
        """
        Compute CPI and SCPI importance in Z-space.
        
        CPI: Squared difference after averaging predictions: (Y - E[Ỹ])^2
        SCPI: Conditional variance of counterfactual predictions: Var[Ỹ]
        
        For L2 loss with independent features, CPI ≈ SCPI because both measure
        the explanatory power of feature j through disentangled perturbation.
        
        Parameters
        ----------
        Z : numpy.ndarray
            Latent space data, shape (n, d).
        y : numpy.ndarray
            Model predictions, shape (n,).
        
        Returns
        -------
        ueifs_cpi : numpy.ndarray
            CPI per-sample importance, shape (n, d).
        ueifs_scpi : numpy.ndarray
            SCPI per-sample importance, shape (n, d).
        """
        n, d = Z.shape
        ueifs_cpi = np.zeros((n, d))
        ueifs_scpi = np.zeros((n, d))
        
        for j in range(d):
            rng = np.random.default_rng(self.random_state + j)
            
            # Create B copies of Z: shape (B, n, d)
            Z_tilde = np.tile(Z[None, :, :], (self.nsamples, 1, 1))
            
            # Replace j-th component based on sampling method
            if self.sampling_method == "resample":
                if self.Z_full is None:
                    raise ValueError("Z_full is None. Call fit_flow() first.")
                resample_idx = rng.choice(
                    self.Z_full.shape[0], size=(self.nsamples, n), replace=True
                )
                Z_tilde[:, :, j] = self.Z_full[resample_idx, j]
            elif self.sampling_method == "permutation":
                perm_idx = np.array([rng.permutation(n) for _ in range(self.nsamples)])
                Z_tilde[:, :, j] = Z[perm_idx, j]
            elif self.sampling_method == "normal":
                Z_tilde[:, :, j] = rng.normal(0.0, 1.0, size=(self.nsamples, n))
            elif self.sampling_method == "condperm":
                # Conditional permutation using permuter
                from sklearn.base import clone
                from sklearn.utils import shuffle
                Z_minus_j = np.delete(Z, j, axis=1)
                z_j = Z[:, j]
                rg = clone(self.permuter)
                rg.fit(Z_minus_j, z_j)
                z_j_hat = rg.predict(Z_minus_j)
                eps_z = z_j - z_j_hat
                for b in range(self.nsamples):
                    eps_perm = shuffle(eps_z, random_state=self.random_state + j * 1000 + b)
                    Z_tilde[b, :, j] = z_j_hat + eps_perm
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
            
            # Decode to X space
            Z_tilde_flat = Z_tilde.reshape(-1, d)
            X_tilde_flat = self._decode_to_X(Z_tilde_flat)
            
            # Evaluate model: shape (nsamples, n)
            y_tilde_flat = self.model(X_tilde_flat)
            y_tilde = y_tilde_flat.reshape(self.nsamples, n)
            
            # CPI: average predictions first, then squared difference
            # Formula: φ_j^CPI = (Y - E[Ỹ])²
            y_tilde_mean = y_tilde.mean(axis=0)
            ueifs_cpi[:, j] = (y - y_tilde_mean) ** 2
            
            # SCPI (Sobol-CPI): variance of counterfactual predictions
            # Formula: φ_j^SCPI = Var(Ỹ) = E[Ỹ²] - E[Ỹ]²
            # This measures the conditional variance, matching Sobol total-order index
            # For L2 loss with independent features, SCPI ≈ CPI
            ueifs_scpi[:, j] = y_tilde.var(axis=0)
        
        return ueifs_cpi, ueifs_scpi
    
    def __call__(self, X: np.ndarray, **kwargs: Any) -> dict:
        """
        Compute feature importance.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data to explain. Shape (n_samples, n_features).
        **kwargs : dict
            Additional parameters (unused, for API compatibility).
        
        Returns
        -------
        dict
            Dictionary containing:
            - phi_Z: Z-space importance (d,) - CPI or SCPI depending on method
            - std_Z: Standard deviation (d,)
            - se_Z: Standard error (d,)
            - phi_X: X-space importance (d,) - transformed via Jacobian
            - std_X: Standard deviation (d,)
            - se_X: Standard error (d,)
            When method='both', also includes phi_Z_scpi, std_Z_scpi, se_Z_scpi.
        """
        n, d = X.shape
        
        # Encode to Z space
        Z = self._encode_to_Z(X)
        
        # Get model predictions on decoded X (for consistency with flow)
        X_hat = self._decode_to_X(Z)
        y_pred = self.model(X_hat)
        
        # Compute both CPI and SCPI in Z-space (per-sample UEIFs)
        ueifs_cpi, ueifs_scpi = self._phi_Z(Z, y_pred)
        
        # Compute Jacobian H = dX/dZ (averaged over samples)
        H = self._compute_jacobian(Z)
        
        # Sensitivity matrix: H_sq[l,k] = H[l,k]^2
        # Maps Z-space importance to X-space: phi_X[l] = sum_k H[l,k]^2 * phi_Z[k]
        H_sq = H ** 2
        
        # Transform per-sample UEIFs from Z-space to X-space
        # ueifs_X[i, l] = sum_k H_sq[l, k] * ueifs_Z[i, k]
        ueifs_cpi_X = ueifs_cpi @ H_sq.T
        ueifs_scpi_X = ueifs_scpi @ H_sq.T
        
        # Aggregate statistics
        ddof = 1 if n > 1 else 0
        results = {}
        
        if self.method == "cpi":
            # Z-space (CPI)
            phi_Z = np.mean(ueifs_cpi, axis=0)
            std_Z = np.std(ueifs_cpi, axis=0)
            se_Z = np.std(ueifs_cpi, axis=0, ddof=ddof) / np.sqrt(n)
            # X-space (transformed via Jacobian)
            phi_X = np.mean(ueifs_cpi_X, axis=0)
            std_X = np.std(ueifs_cpi_X, axis=0)
            se_X = np.std(ueifs_cpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            results.update({
                "phi_Z": phi_Z,
                "std_Z": std_Z,
                "se_Z": se_Z,
                "phi_X": phi_X,
                "std_X": std_X,
                "se_X": se_X,
            })
        elif self.method == "scpi":
            # Z-space (SCPI)
            phi_Z = np.mean(ueifs_scpi, axis=0)
            std_Z = np.std(ueifs_scpi, axis=0)
            se_Z = np.std(ueifs_scpi, axis=0, ddof=ddof) / np.sqrt(n)
            # X-space (transformed via Jacobian)
            phi_X = np.mean(ueifs_scpi_X, axis=0)
            std_X = np.std(ueifs_scpi_X, axis=0)
            se_X = np.std(ueifs_scpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            results.update({
                "phi_Z": phi_Z,
                "std_Z": std_Z,
                "se_Z": se_Z,
                "phi_X": phi_X,
                "std_X": std_X,
                "se_X": se_X,
            })
        else:  # method == "both"
            # CPI (Z-space and X-space)
            phi_Z_cpi = np.mean(ueifs_cpi, axis=0)
            std_Z_cpi = np.std(ueifs_cpi, axis=0)
            se_Z_cpi = np.std(ueifs_cpi, axis=0, ddof=ddof) / np.sqrt(n)
            phi_X_cpi = np.mean(ueifs_cpi_X, axis=0)
            std_X_cpi = np.std(ueifs_cpi_X, axis=0)
            se_X_cpi = np.std(ueifs_cpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            # SCPI (Z-space and X-space)
            phi_Z_scpi = np.mean(ueifs_scpi, axis=0)
            std_Z_scpi = np.std(ueifs_scpi, axis=0)
            se_Z_scpi = np.std(ueifs_scpi, axis=0, ddof=ddof) / np.sqrt(n)
            phi_X_scpi = np.mean(ueifs_scpi_X, axis=0)
            std_X_scpi = np.std(ueifs_scpi_X, axis=0)
            se_X_scpi = np.std(ueifs_scpi_X, axis=0, ddof=ddof) / np.sqrt(n)
            results.update({
                # Default to CPI for phi_Z/phi_X
                "phi_Z": phi_Z_cpi,
                "std_Z": std_Z_cpi,
                "se_Z": se_Z_cpi,
                "phi_X": phi_X_cpi,
                "std_X": std_X_cpi,
                "se_X": se_X_cpi,
                # SCPI with suffix
                "phi_Z_scpi": phi_Z_scpi,
                "std_Z_scpi": std_Z_scpi,
                "se_Z_scpi": se_Z_scpi,
                "phi_X_scpi": phi_X_scpi,
                "std_X_scpi": std_X_scpi,
                "se_X_scpi": se_X_scpi,
            })
        
        self._cache_results(results, n)
        return results
