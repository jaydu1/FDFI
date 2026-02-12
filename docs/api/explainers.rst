Explainers
==========

Overview
--------

The ``fdfi.explainers`` module provides classes for computing flow-disentangled 
feature importance. The main classes are:

Base Explainer
--------------

.. autoclass:: fdfi.explainers.Explainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Tree-Based Models
-----------------

.. autoclass:: fdfi.explainers.TreeExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Linear Models
-------------

.. autoclass:: fdfi.explainers.LinearExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Kernel Methods
--------------

.. autoclass:: fdfi.explainers.KernelExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

Gaussian Optimal Transport (OTExplainer)
----------------------------------------

The ``OTExplainer`` implements Gaussian optimal-transport DFI (Disentangled 
Feature Importance) without cross-fitting. This is the recommended starting 
point for most use cases.

.. autoclass:: fdfi.explainers.OTExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

**Example:**

.. code-block:: python

   import numpy as np
   from fdfi.explainers import OTExplainer

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

.. autoclass:: fdfi.explainers.EOTExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

**Example with advanced options:**

.. code-block:: python

   from fdfi.explainers import EOTExplainer

   explainer = EOTExplainer(
       model.predict,
       X_background,
       auto_epsilon=True,           # Adaptive regularization
       stochastic_transport=True,   # Sample from transport kernel
       n_transport_samples=10,      # Number of transport samples
       target="gaussian",           # or "empirical"
   )
   results = explainer(X_test)

Flow-Based DFI (FlowExplainer)
------------------------------

The ``FlowExplainer`` implements Flow-Disentangled Feature Importance using 
normalizing flows. It supports both CPI (Conditional Permutation Importance) 
and SCPI (Sobol-CPI). The key difference is the order of averaging:

- **CPI**: Average predictions first, then squared difference: $(Y - E[f(\tilde{X})])^2$
- **SCPI**: Squared differences first, then average: $E[(Y - f(\tilde{X}_b))^2]$

.. autoclass:: fdfi.explainers.FlowExplainer
   :members:
   :special-members: __init__, __call__
   :show-inheritance:

**Example with CPI (default):**

.. code-block:: python

   from fdfi.explainers import FlowExplainer

   explainer = FlowExplainer(
       model.predict,
       X_background,
       fit_flow=True,           # Fit normalizing flow during init
       method='cpi',            # CPI (default)
       num_steps=200,           # Flow training iterations
       nsamples=50,             # Monte Carlo samples
       random_state=42,
   )
   results = explainer(X_test)

   print("Z-space importance (CPI):", results["phi_Z"])
   print("Confidence intervals:")
   ci = explainer.conf_int(alpha=0.05, target="Z")

**Example with SCPI (Sobol-CPI):**

.. code-block:: python

   from fdfi.explainers import FlowExplainer

   explainer = FlowExplainer(
       model.predict,
       X_background,
       fit_flow=True,
       method='scpi',           # SCPI (Sobol-CPI)
       num_steps=200,
       nsamples=50,
   )
   results = explainer(X_test)

   print("Importance (SCPI):", results["phi_Z"])

**Using external flow models:**

.. code-block:: python

   from fdfi.explainers import FlowExplainer
   from fdfi.models import FlowMatchingModel

   # Train flow externally
   flow = FlowMatchingModel(X_background, dim=X_background.shape[1])
   flow.fit(num_steps=500, verbose='final')

   # Use in explainer
   explainer = FlowExplainer(model.predict, X_background, fit_flow=False)
   explainer.set_flow(flow)
   results = explainer(X_test)

DFIExplainer Alias
------------------

``DFIExplainer`` is an alias for ``OTExplainer`` for backward compatibility:

.. data:: fdfi.explainers.DFIExplainer

   Alias for :class:`OTExplainer`.
