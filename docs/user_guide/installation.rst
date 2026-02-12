Installation
============

DFI can be installed from source. We recommend using a conda environment 
for managing dependencies.

Using Conda (Recommended)
-------------------------

Create and activate a conda environment:

.. code-block:: bash

   conda create -n fdfi python=3.10
   conda activate fdfi

Then install DFI:

.. code-block:: bash

   git clone https://github.com/jaydu1/FDFI.git
   cd FDFI
   pip install -e .

From Source with pip
--------------------

.. code-block:: bash

   git clone https://github.com/jaydu1/FDFI.git
   cd FDFI
   pip install -e .

Optional Dependencies
---------------------

DFI has optional dependency groups for different use cases:

**Plotting support** (matplotlib, seaborn):

.. code-block:: bash

   pip install -e ".[plots]"

**Flow matching models** (PyTorch, torchdiffeq):

.. code-block:: bash

   pip install -e ".[flow]"

**Development tools** (pytest, black, flake8, mypy):

.. code-block:: bash

   pip install -e ".[dev]"

**Documentation building** (Sphinx, RTD theme):

.. code-block:: bash

   pip install -e ".[docs]"

**All optional dependencies**:

.. code-block:: bash

   pip install -e ".[all]"

Using environment.yml
---------------------

You can also use the provided conda environment file:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate fdfi

Requirements
------------

**Core requirements:**

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0

**Optional requirements:**

- matplotlib >= 3.5.0 (for plotting)
- seaborn >= 0.12.0 (for plotting)
- torch >= 2.0.0 (for flow matching)
- torchdiffeq >= 0.2.3 (for flow matching)
- scikit-learn (for mixture models in utilities)

Verifying Installation
----------------------

After installation, verify that FDFI is working:

.. code-block:: python

   import fdfi
   print(fdfi.__version__)

   # Test basic functionality
   import numpy as np
   from fdfi.explainers import OTExplainer

   def model(X):
       return X.sum(axis=1)

   X = np.random.randn(50, 5)
   explainer = OTExplainer(model, data=X, nsamples=20)
   results = explainer(X[:5])
   print("Installation successful!")

Troubleshooting
---------------

**ImportError for torch or torchdiffeq**

If you see import errors related to PyTorch, you need to install the flow 
dependencies:

.. code-block:: bash

   pip install -e ".[flow]"

Or pass ``fit_flow=False`` when creating explainers to disable flow matching:

.. code-block:: python

   explainer = Explainer(model, data=X, fit_flow=False)

**Matplotlib backend issues**

If you encounter issues with matplotlib on headless servers:

.. code-block:: python

   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
