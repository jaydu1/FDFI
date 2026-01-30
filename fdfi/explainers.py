"""
Explainer classes for FDFI.

This module contains the main explainer classes for computing
flow-disentangled feature importance.
"""

import numpy as np
import torch
from typing import Optional, Union, Callable, Any, Tuple
from .models import FlowMatchingModel



class Explainer:
    """
    Base class for FDFI explainers.
    
    This class provides the interface for computing feature importance
    using flow-disentangled methods, similar to SHAP explainers.
    
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
    >>> from fdfi import Explainer
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
        
        self.flow_engine = FlowMatchingModel(
            X=data,
            dim=data.shape[1],
            device=kwargs.get("device")
        )
        print("Training flow model for disentanglement...")
        self.flow_engine.fit(num_steps=kwargs.get("num_steps", 5000))
    
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
        
class DFIExplainer(Explainer):
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        data: np.ndarray,
        nsamples: int = 50,
        **kwargs: Any
    ):
        """Initialize the DFIExplainer."""
        super().__init__(model, data, **kwargs)
        self.nsamples = nsamples
        self.regularize = kwargs.get("regularize", 1e-6)

       
        self.mean = np.mean(data, axis=0, keepdims=True)
        
        self.cov = np.cov(data, rowvar=False)
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
        phi_X = np.mean(ueifs_X, axis=0)
        std_X = np.std(ueifs_X, axis=0)

        phi_Z = np.mean(ueifs_Z, axis=0)
        std_Z = np.std(ueifs_Z, axis=0)

        return {
            "phi_X": phi_X,
            "std_X": std_X,
            "phi_Z": phi_Z,
            "std_Z": std_Z,
        }

    def _phi_Z(self, Z: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
       
        n, d = Z.shape
        ueifs_Z = np.zeros((n, d))
        
        for j in range(d):

            rng = np.random.default_rng(j)
            resample_idx = rng.choice(self.Z_full.shape[0], size=(self.nsamples, n), replace=True)
            
            # Z_tilde: (nsamples, n, d)
            Z_tilde = np.tile(Z[None, :, :], (self.nsamples, 1, 1))
            Z_tilde[:, :, j] = self.Z_full[resample_idx, j]
            
            Z_tilde_flat = Z_tilde.reshape(-1, d)
            X_tilde_flat = Z_tilde_flat @ self.L + self.mean
            
            y_tilde_flat = self.model(X_tilde_flat)
            y_tilde = y_tilde_flat.reshape(self.nsamples, n).mean(axis=0)
            
            ueifs_Z[:, j] = (y_pred - y_tilde) ** 2

        # Return per-sample UEIFs in latent space (no aggregation here)
        return ueifs_Z