Plotting Functions
==================

Overview
--------

The ``fdfi.plots`` module provides static Matplotlib visualizations for FDFI
diagnostics and results. All functions return Matplotlib objects and accept
``show=False`` for tests, scripts, and documentation builds.

Typical inputs come directly from explainer results:

.. code-block:: python

   results = explainer(X_test)
   ci = explainer.conf_int(alpha=0.05, target="X")
   feature_names = [f"X{i}" for i in range(X_test.shape[1])]

   summary_bar(results["phi_X"], results["se_X"], feature_names, show=False)
   summary_plot(explainer.ueifs_X, features=X_test, feature_names=feature_names, show=False)
   confidence_interval_plot(ci, feature_names=feature_names, show=False)
   diagnostics_plot(explainer.diagnostics, feature_names=feature_names, show=False)

Global Importance
-----------------

.. autofunction:: fdfi.plots.summary_bar

``summary_bar`` is the public global bar-chart API for aggregate arrays such as
``results["phi_X"]`` and ``results["phi_Z"]``. It sorts features by absolute
importance, sanitizes missing or infinite standard errors, supports optional
feature-color mappings, and returns the sorted table used for plotting.

Summary Plot
------------

.. autofunction:: fdfi.plots.summary_plot

``summary_plot`` draws a SHAP-like dot summary for 2D per-sample attribution
arrays such as ``explainer.ueifs_X`` or ``explainer.ueifs_Z``. For 1D aggregate
arrays, it delegates to ``summary_bar``.

Single-Explanation Views
------------------------

.. autofunction:: fdfi.plots.waterfall_plot

.. autofunction:: fdfi.plots.force_plot

Use these for one row of per-sample UEIFs:

.. code-block:: python

   waterfall_plot(explainer.ueifs_X[0], feature_names=feature_names, show=False)
   force_plot(0.0, explainer.ueifs_X[0], feature_names=feature_names, show=False)

Dependence Plot
---------------

.. autofunction:: fdfi.plots.dependence_plot

``dependence_plot`` accepts integer or string feature identifiers and can color
points by an interaction feature:

.. code-block:: python

   dependence_plot(
       "X0",
       explainer.ueifs_X,
       X_test,
       feature_names=feature_names,
       interaction_index="X1",
       show=False,
   )

Feature Correlation
-------------------

.. autofunction:: fdfi.plots.correlation_heatmap

The heatmap uses Pearson correlation and hierarchical clustering based on
``1 - abs(correlation)`` to reveal correlated feature blocks in the background
data.

Inference and Diagnostics
-------------------------

.. autofunction:: fdfi.plots.confidence_interval_plot

.. autofunction:: fdfi.plots.diagnostics_plot

``confidence_interval_plot`` accepts dictionaries returned by ``conf_int()``,
including feature-level and grouped outputs. ``diagnostics_plot`` accepts the
shared diagnostics dictionaries exposed by ``OTExplainer``, ``EOTExplainer``,
and ``FlowExplainer``.
