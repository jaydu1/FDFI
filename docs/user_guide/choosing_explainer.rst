Choosing an Explainer
=====================

DFI provides several explainer classes for different use cases. This guide 
helps you choose the right one.

Quick Decision Guide
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Situation
     - Recommended
     - Notes
   * - General use, continuous data
     - ``OTExplainer``
     - Fast, stable, good default
   * - Non-Gaussian data
     - ``EOTExplainer``
     - Adaptive epsilon, more flexible
   * - Complex multimodal data
     - ``FlowExplainer``
     - Learns data distribution via normalizing flow
   * - Mixed data types
     - ``EOTExplainer`` with Gower
     - Use ``cost_metric="gower"``
   * - Tree-based models
     - ``TreeExplainer``
     - Optimized for RF, XGBoost, etc.
   * - Linear models
     - ``LinearExplainer``
     - Exact for linear models
   * - Any black-box model
     - ``OTExplainer`` or ``KernelExplainer``
     - Model-agnostic

OTExplainer (Gaussian OT)
-------------------------

**Best for:** Continuous data that is approximately Gaussian

**Pros:**

- Fast closed-form computation
- Stable and reliable
- Good starting point for most problems

**Cons:**

- Assumes Gaussian structure
- May be suboptimal for heavy-tailed or multimodal data

**Example:**

.. code-block:: python

   from fdfi.explainers import OTExplainer

   explainer = OTExplainer(
       model.predict,
       data=X_background,
       nsamples=50,            # Monte Carlo samples per feature
       sampling_method="resample",  # or "permutation", "normal"
   )
   results = explainer(X_test)

EOTExplainer (Entropic OT)
--------------------------

**Best for:** Non-Gaussian, multimodal, or mixed-type data

**Pros:**

- Relaxes Gaussian assumption
- Adaptive regularization (``auto_epsilon=True``)
- Supports categorical features via Gower distance
- Stochastic transport for variance reduction

**Cons:**

- Slower than Gaussian OT
- More hyperparameters to tune

**Key options:**

.. code-block:: python

   from fdfi.explainers import EOTExplainer

   explainer = EOTExplainer(
       model.predict,
       data=X_background,
       # Regularization
       auto_epsilon=True,      # Auto-tune from median distance
       epsilon=0.1,            # Manual epsilon (if auto_epsilon=False)
       
       # Transport target
       target="gaussian",      # or "empirical"
       
       # Stochastic transport
       stochastic_transport=True,
       n_transport_samples=10,
       
       # Cost function for mixed data
       cost_metric="sqeuclidean",  # or "gower", "auto"
   )

FlowExplainer (Flow-Based DFI)
------------------------------

**Best for:** Complex, non-Gaussian data where normalizing flows can capture the
underlying distribution structure

**Pros:**

- Handles complex, multimodal distributions
- Maps data to Gaussian latent space via learned normalizing flow
- Supports both CPI and SCPI (Sobol-CPI) methods with different averaging orders
- Flexible flow training and pre-trained model support

**Cons:**

- Requires PyTorch and torchdiffeq dependencies
- Flow training can be slow for large datasets

**Key options:**

.. code-block:: python

   from fdfi.explainers import FlowExplainer

   explainer = FlowExplainer(
       model.predict,
       data=X_background,
       
       # Flow fitting
       fit_flow=True,          # Fit flow during init (or fit later)
       num_steps=200,          # Flow training iterations
       
       # Method selection
       method='cpi',           # 'cpi', 'scpi', or 'both'
       
       # Counterfactual sampling
       nsamples=50,            # Monte Carlo samples per feature
       sampling_method='resample',  # 'resample', 'permutation', 'normal', 'condperm'
       
       # Reproducibility
       random_state=42,
   )
   
   results = explainer(X_test)

**Understanding CPI vs SCPI:**

- **CPI (Conditional Permutation Importance)**: Average predictions first, then 
  compute squared difference:
  
  .. math::
  
     \phi_j^{CPI} = (Y - E_b[f(\tilde{X}_b^{(j)})])^2
  
- **SCPI (Sobol-CPI)**: Compute squared differences first, then average (Sobol
  sensitivity index formulation):
  
  .. math::
  
     \phi_j^{SCPI} = E_b[(Y - f(\tilde{X}_b^{(j)}))^2]

**External flow models:**

.. code-block:: python

   from fdfi.models import FlowMatchingModel

   # Train flow externally with custom settings
   flow = FlowMatchingModel(X_background, dim=X_background.shape[1])
   flow.fit(num_steps=500, verbose='final')

   # Use pre-trained flow in explainer
   explainer = FlowExplainer(model.predict, X_background, fit_flow=False)
   explainer.set_flow(flow)

Shared Diagnostics (OT / EOT / Flow)
------------------------------------

All disentangled explainers expose a shared ``diagnostics`` payload:

- ``latent_independence_median`` with qualitative label
- ``distribution_fidelity_mmd`` with qualitative label

Lower is better for both metrics. Labels use the same thresholds across
explainers:

- ``GOOD``: dCor < 0.10, MMD < 0.05
- ``MODERATE``: dCor < 0.25, MMD < 0.15
- ``POOR``: otherwise

.. code-block:: python

   explainer = OTExplainer(model.predict, X_background)
   diag = explainer.diagnostics
   print(diag["latent_independence_median"], diag["latent_independence_label"])
   print(diag["distribution_fidelity_mmd"], diag["distribution_fidelity_label"])

TreeExplainer
-------------

**Best for:** Tree ensemble models (Random Forest, Gradient Boosting, XGBoost, 
LightGBM)

**Pros:**

- Optimized tree traversal algorithms
- Exact or approximate Shapley computation

**Note:** Currently a placeholder—full implementation coming soon.

.. code-block:: python

   from fdfi.explainers import TreeExplainer
   from sklearn.ensemble import RandomForestRegressor

   model = RandomForestRegressor().fit(X_train, y_train)
   explainer = TreeExplainer(model, data=X_background)

LinearExplainer
---------------

**Best for:** Linear models (Linear/Logistic Regression, Ridge, Lasso)

**Pros:**

- Exact Shapley values for linear models
- Very fast computation

**Note:** Currently a placeholder—full implementation coming soon.

.. code-block:: python

   from fdfi.explainers import LinearExplainer
   from sklearn.linear_model import LinearRegression

   model = LinearRegression().fit(X_train, y_train)
   explainer = LinearExplainer(model, data=X_background)

KernelExplainer
---------------

**Best for:** Any model where you have no prior knowledge of structure

**Pros:**

- Works with any callable model
- Fully model-agnostic

**Cons:**

- Slowest method
- Can have high variance

**Note:** Currently a placeholder—full implementation coming soon.

.. code-block:: python

   from fdfi.explainers import KernelExplainer

   explainer = KernelExplainer(model.predict, data=X_background)

Hyperparameter Guidelines
-------------------------

nsamples
~~~~~~~~

Number of Monte Carlo samples for counterfactual estimation.

- **Low (10-30)**: Fast but high variance
- **Medium (50-100)**: Good balance (recommended)
- **High (200+)**: Low variance but slow

sampling_method
~~~~~~~~~~~~~~~

How to generate counterfactual feature values:

- ``"resample"``: Sample from background data (default, preserves marginal)
- ``"permutation"``: Permute within test set (no new values)
- ``"normal"``: Sample from standard normal (strong Gaussian assumption)

epsilon (EOTExplainer)
~~~~~~~~~~~~~~~~~~~~~~

Entropic regularization strength:

- **Small (0.01)**: Sharp transport, may be unstable
- **Medium (0.1)**: Good balance
- **Large (1.0+)**: Smooth transport, loses structure
- **auto_epsilon=True**: Recommended, auto-tunes from data

target (EOTExplainer)
~~~~~~~~~~~~~~~~~~~~~

Transport target distribution:

- ``"gaussian"``: Standard normal target (default)
- ``"empirical"``: Permuted data as target

Computing Confidence Intervals
------------------------------

All explainers support post-hoc confidence intervals:

.. code-block:: python

   # Compute importance
   results = explainer(X_test)

   # Get confidence intervals
   ci = explainer.conf_int(
       alpha=0.05,
       target="X",              # or "Z" for latent space
       alternative="two-sided", # or "greater", "less"
       var_floor_method="mixture",  # Stabilize small variances
       margin=0.0,              # Practical significance threshold
   )

   print("Significant features:", np.where(ci["reject_null"])[0])
