Plotting Functions
==================

Overview
--------

The ``fdfi.plots`` module provides visualization functions for feature 
importance, similar to SHAP's plotting capabilities. These functions help 
you understand and communicate model explanations.

.. note::

   Plotting functions require the ``plots`` extra dependency:
   
   .. code-block:: bash

      pip install -e ".[plots]"

Summary Plot
------------

.. autofunction:: fdfi.plots.summary_plot

The summary plot shows the distribution of feature importance values across 
all samples, with features ordered by their overall importance.

**Example:**

.. code-block:: python

   from fdfi.plots import summary_plot

   # After computing explanations
   results = explainer(X_test)
   
   # Create summary visualization
   summary_plot(results["phi_X"], features=X_test, feature_names=feature_names)

Waterfall Plot
--------------

.. autofunction:: fdfi.plots.waterfall_plot

The waterfall plot shows how each feature contributes to pushing the 
prediction from the base value for a single sample.

**Example:**

.. code-block:: python

   from fdfi.plots import waterfall_plot

   # Explain a single prediction
   waterfall_plot(
       results["phi_X"][0],  # First sample
       feature_names=feature_names,
       max_display=10
   )

Force Plot
----------

.. autofunction:: fdfi.plots.force_plot

The force plot is an interactive visualization showing feature contributions 
as forces pushing the prediction higher or lower.

**Example:**

.. code-block:: python

   from fdfi.plots import force_plot

   force_plot(
       base_value=0.5,
       shap_values=results["phi_X"][0],
       feature_names=feature_names
   )

Dependence Plot
---------------

.. autofunction:: fdfi.plots.dependence_plot

The dependence plot shows the relationship between a feature's value and 
its contribution to the model output, optionally colored by another feature 
to show interaction effects.

**Example:**

.. code-block:: python

   from fdfi.plots import dependence_plot

   # Show dependence for feature 0
   dependence_plot(
       feature_idx=0,
       shap_values=results["phi_X"],
       features=X_test,
       feature_names=feature_names
   )
