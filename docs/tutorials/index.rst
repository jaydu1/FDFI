Tutorials
=========

This section contains hands-on tutorials for learning FDFI. Each tutorial 
is a Jupyter notebook that you can run interactively.

.. toctree::
   :maxdepth: 1

   quickstart
   ot_explainer
   eot_explainer
   flow_explainer
   confidence_intervals

Getting Started
---------------

If you're new to FDFI, start with the **Quickstart** tutorial to learn the 
basics in 5 minutes.

Tutorial Overview
-----------------

:doc:`quickstart`
   Learn the basics of FDFI in 5 minutes. Create your first explainer, 
   compute feature importance, and interpret the results.

:doc:`ot_explainer`
   Deep dive into the Gaussian OT explainer. Learn about the mathematical 
   foundation, hyperparameters, shared diagnostics, and when to use it.

:doc:`eot_explainer`
   Mixed-type-first EOT tutorial. Starts with active-feature screening using
   Gower cost, then covers epsilon, stochastic transport, target choice, and
   shared diagnostics.

:doc:`flow_explainer`
   Master Flow-DFI with normalizing flows. Learn about CPI vs SCPI methods,
   custom flow models, shared diagnostics, and when to choose FlowExplainer.

:doc:`confidence_intervals`
   Statistical inference with FDFI. Learn to compute confidence intervals, 
   perform hypothesis testing, and identify significant features.

Running the Tutorials
---------------------

You can run these tutorials in several ways:

**Option 1: Jupyter Notebook**

.. code-block:: bash

   cd docs/tutorials
   jupyter notebook

**Option 2: JupyterLab**

.. code-block:: bash

   cd docs/tutorials
   jupyter lab

**Option 3: VS Code**

Open the ``.ipynb`` files directly in VS Code with the Jupyter extension.

**Option 4: Google Colab**

Upload the notebooks to Google Colab and run in the cloud.

Prerequisites
-------------

Make sure you have FDFI installed with plotting support:

.. code-block:: bash

   pip install -e ".[plots]"
