FDFI Documentation
==================

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

**FDFI** (Flow-Disentangled Feature Importance) is a Python library for computing
feature importance using disentangled methods, inspired by SHAP. This package
implements both OT-based DFI and flow-based FDFI methods.

Key Features
------------

- 🎯 **Multiple Explainer Types**: Tree, Linear, Kernel, and Optimal Transport explainers
- 🧭 **OT-Based DFI**: Gaussian OT (OTExplainer) and Entropic OT (EOTExplainer)
- 🔍 **Shared Diagnostics**: Latent independence and fidelity checks for OT/EOT/Flow
- 📊 **Statistical Inference**: Confidence intervals and hypothesis testing
- 🔧 **Easy to Use**: Simple API similar to SHAP
- 🚀 **Extensible**: Built with modularity for future enhancements

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from fdfi.explainers import OTExplainer

   # Define your model
   def model(X):
       return X.sum(axis=1)

   # Create background data
   X_background = np.random.randn(100, 10)

   # Create an explainer
   explainer = OTExplainer(model, data=X_background, nsamples=50)

   # Explain test instances
   X_test = np.random.randn(10, 10)
   results = explainer(X_test)

   # Get confidence intervals
   ci = explainer.conf_int(alpha=0.05, target="X", alternative="two-sided")

Installation
------------

.. code-block:: bash

   git clone https://github.com/jaydu1/FDFI.git
   cd FDFI
   pip install -e .

   # With optional dependencies
   pip install -e ".[plots]"   # For visualization
   pip install -e ".[flow]"    # For flow matching models
   pip install -e ".[dev]"     # For development

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/concepts
   user_guide/choosing_explainer
   user_guide/faq

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
