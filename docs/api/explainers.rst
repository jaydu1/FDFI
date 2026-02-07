Explainers
==========

Overview
--------

The ``dfi.explainers`` module provides classes for computing flow-disentangled 
feature importance. The main classes are:

Base Explainer
--------------

.. autoclass:: dfi.explainers.Explainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Tree-Based Models
-----------------

.. autoclass:: dfi.explainers.TreeExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Linear Models
-------------

.. autoclass:: dfi.explainers.LinearExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Kernel Methods
--------------

.. autoclass:: dfi.explainers.KernelExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Gaussian Optimal Transport (OTExplainer)
----------------------------------------

The ``OTExplainer`` implements Gaussian optimal-transport DFI (Disentangled 
Feature Importance) without cross-fitting. This is the recommended starting 
point for most use cases.

.. autoclass:: dfi.explainers.OTExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

**Example:**

.. code-block:: python

   import numpy as np
   from dfi.explainers import OTExplainer

   # Create model and data
   def model(X):
       return X[:, 0] + 2 * X[:, 1]

   X_background = np.random.randn(100, 10)
   X_test = np.random.randn(10, 10)

   # Create explainer and compute importance
   explainer = OTExplainer(model, data=X_background, nsamples=50)
   results = explainer(X_test)

   print("Feature importance (X-space):", results["phi_X"])
   print("Standard errors:", results["se_X"])

Entropic Optimal Transport (EOTExplainer)
-----------------------------------------

The ``EOTExplainer`` uses entropic optimal transport with Sinkhorn iterations. 
It supports adaptive epsilon, stochastic transport sampling, and both Gaussian 
and empirical transport targets.

.. autoclass:: dfi.explainers.EOTExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

**Example with advanced options:**

.. code-block:: python

   from dfi.explainers import EOTExplainer

   explainer = EOTExplainer(
       model.predict,
       X_background,
       auto_epsilon=True,           # Adaptive regularization
       stochastic_transport=True,   # Sample from transport kernel
       n_transport_samples=10,      # Number of transport samples
       target="gaussian",           # or "empirical"
   )
   results = explainer(X_test)

DFIExplainer Alias
------------------

``DFIExplainer`` is an alias for ``OTExplainer`` for backward compatibility:

.. data:: dfi.explainers.DFIExplainer

   Alias for :class:`OTExplainer`.
