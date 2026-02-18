Utilities
=========

Overview
--------

The ``fdfi.utils`` module provides helper functions and classes used across 
the FDFI package.

Input Validation
----------------

.. autofunction:: fdfi.utils.validate_input

Data Sampling
-------------

.. autofunction:: fdfi.utils.sample_background

Feature Names
-------------

.. autofunction:: fdfi.utils.get_feature_names

Link Functions
--------------

.. autofunction:: fdfi.utils.convert_to_link

The following link functions are supported:

- ``"identity"``: No transformation (default)
- ``"logit"``: Logit transformation for probability outputs

Additivity Check
----------------

.. autofunction:: fdfi.utils.check_additivity

This function verifies the SHAP additivity property:

.. math::

   f(x) = \\phi_0 + \\sum_{j=1}^{d} \\phi_j

where :math:`\\phi_0` is the base value and :math:`\\phi_j` are the feature 
attributions.

Feature Type Detection
----------------------

.. autofunction:: fdfi.utils.detect_feature_types

This function auto-detects whether features are binary, categorical, or 
continuous based on the data distribution.

Gower Distance
--------------

.. autofunction:: fdfi.utils.gower_cost_matrix

Computes the Gower distance matrix for mixed-type data (continuous, binary, 
and categorical features). Used by ``EOTExplainer`` when ``cost_metric="gower"``
or ``cost_metric="auto"``.

Diagnostics Utilities
---------------------

.. autofunction:: fdfi.utils.compute_latent_independence

.. autofunction:: fdfi.utils.compute_mmd

Statistical Utilities
---------------------

TwoComponentMixture
~~~~~~~~~~~~~~~~~~~

.. autoclass:: fdfi.utils.TwoComponentMixture
   :members:
   :special-members: __init__
   :show-inheritance:

The ``TwoComponentMixture`` class fits a two-component Gaussian mixture model 
and is used for:

1. **Variance floor estimation**: Determining a minimum variance threshold 
   for stable confidence intervals
2. **Practical significance margins**: Estimating reasonable effect size 
   thresholds

**Example:**

.. code-block:: python

   from fdfi.utils import TwoComponentMixture
   import numpy as np

   # Fit mixture to standard errors
   se_values = np.array([0.01, 0.02, 0.15, 0.18, 0.20, 0.25])
   mixture = TwoComponentMixture().fit(se_values)

   # Get quantile from smaller component
   floor = mixture.quantile(0.95, component="smaller")
   print(f"Variance floor: {floor}")

   # Visualize the fit
   mixture.plot(se_values)
