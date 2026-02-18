Core Concepts
=============

This page introduces the key concepts behind DFI (Disentangled Feature 
Importance) and FDFI (Flow-DFI), and how they relate to other interpretability methods.

What is Feature Importance?
---------------------------

Feature importance quantifies **how much each input feature contributes** to a 
model's predictions. Given a model :math:`f(x)` and an input 
:math:`x = (x_1, ..., x_d)`, we want to compute attributions 
:math:`\\phi = (\\phi_1, ..., \\phi_d)` where :math:`\\phi_j` represents the 
importance of feature :math:`j`.

SHAP and Shapley Values
-----------------------

SHAP (SHapley Additive exPlanations) computes feature importance using 
**Shapley values** from cooperative game theory. For a prediction :math:`f(x)`, 
SHAP values satisfy:

1. **Efficiency**: :math:`\\sum_j \\phi_j = f(x) - E[f(X)]`
2. **Symmetry**: Features with equal contributions get equal attributions
3. **Null**: Features that don't affect the output get zero attribution
4. **Linearity**: Attributions combine linearly for ensemble models

DFI is inspired by SHAP but uses **optimal transport** methods to compute 
feature importance.

Disentangled Feature Importance (DFI)
-------------------------------------

DFI introduces **disentangled feature importance** using optimal transport 
to create counterfactual distributions. The key insight is:

   *To measure the importance of feature j, compare the model output when 
   feature j comes from the data distribution vs. when it's replaced by 
   an independent sample.*

Mathematically, let :math:`Z = L^{-1}(X - \\mu)` be the whitened 
(disentangled) representation where features are uncorrelated. The 
**Unit Effect Independent Feature** (UEIF) for feature :math:`j` is:

.. math::

   \\text{UEIF}_j(x) = \\left( f(x) - E[f(\\tilde{X}^{(j)})] \\right)^2

where :math:`\\tilde{X}^{(j)}` has feature :math:`j` replaced with an 
independent sample from the marginal distribution.

Gaussian vs Entropic OT
-----------------------

DFI provides two main approaches:

**Gaussian OT (OTExplainer)**

- Assumes data is approximately Gaussian
- Uses closed-form Gaussian optimal transport: :math:`Z = L^{-1}(X - \\mu)`
- Fast and stable
- Best for continuous, roughly normal data

**Entropic OT (EOTExplainer)**

- Relaxes Gaussian assumption using Sinkhorn algorithm
- Adaptive regularization via median distance heuristic
- Supports mixed data types (continuous + categorical)
- Better for non-Gaussian or mixed-type data

Flow-Disentangled Feature Importance (Flow-DFI)
-----------------------------------------------

**FlowExplainer** uses **normalizing flows** to learn a flexible, data-driven
transformation between the original feature space X and a disentangled latent
space Z where features are approximately independent.

**CPI (Conditional Permutation Importance)**

Averages predictions first, then computes squared difference:

.. math::

   \phi_{Z,j}^{CPI} = (Y - \mathbb{E}_b[f(\tilde{X}_b^{(j)})])^2

where :math:`\tilde{X}_b^{(j)} = T^{-1}(\tilde{Z}_b^{(j)})` and 
:math:`\tilde{Z}_b^{(j)}` has the j-th component replaced with sample b.

**SCPI (Sobol-CPI)**

Computes squared differences first for each Monte Carlo sample, then averages:

.. math::

   \phi_{Z,j}^{SCPI} = \mathbb{E}_b[(Y - f(\tilde{X}_b^{(j)}))^2]

This is equivalent to the Sobol sensitivity index formulation. The key 
difference from CPI is the **order of averaging**.

**Jacobian Transformation to X-space**

Both CPI and SCPI compute importance in the disentangled Z-space. To attribute
importance to the original features :math:`X_l`, we use the **Jacobian** of the
decoder transformation :math:`T^{-1}: Z \to X`:

.. math::

   \phi_{X,l} = \sum_{k=1}^{d} H_{lk}^2 \cdot \phi_{Z,k}

where :math:`H = \frac{\partial X}{\partial Z}` is the Jacobian matrix evaluated
at the data points. This correctly accounts for how changes in each latent 
dimension Z_k affect each original feature X_l.

For linear transformations (as in OTExplainer), this reduces to :math:`\phi_X = H^T H \phi_Z`
where :math:`H = L` is the Cholesky factor. For normalizing flows, the Jacobian
varies with position and is computed via automatic differentiation.

**When to Use FlowExplainer**

- Complex non-linear dependencies between features
- Non-Gaussian data distributions  
- When OT assumptions are too restrictive
- When you have sufficient data (>500 samples) to train the flow

Shared Disentanglement Diagnostics
----------------------------------

All disentangled explainers (OT, EOT, Flow) report two common diagnostics:

- **Latent independence**: median pairwise distance correlation in latent space.
- **Distribution fidelity**: MMD between original data and reconstructed data.

Both are "lower is better" metrics and are reported with qualitative labels
(``GOOD``, ``MODERATE``, ``POOR``) using shared thresholds.

Relationship to Other Methods
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Method
     - Comparison to DFI
   * - **SHAP KernelExplainer**
     - Model-agnostic like DFI, but uses sampling-based Shapley estimation. 
       DFI uses OT-based counterfactuals.
   * - **SHAP TreeExplainer**
     - Exact Shapley for trees. DFI works with any model.
   * - **LIME**
     - Local linear approximation. DFI considers global distribution.
   * - **Permutation Importance**
     - Breaks feature dependencies. DFI preserves correlation structure.
   * - **Integrated Gradients**
     - Requires gradients. DFI is gradient-free.

Confidence Intervals
--------------------

A key advantage of DFI is **built-in uncertainty quantification**. The 
``conf_int()`` method provides:

- Standard errors computed across samples
- Confidence intervals using normal approximation
- P-values for testing :math:`H_0: \\phi_j = 0` or :math:`H_0: \\phi_j \\leq \\delta`
- Variance floor methods for stable inference with small effects

This enables **statistical feature selection**: identify features that are 
significantly different from zero or a practical threshold.

Further Reading
---------------

- `SHAP paper <https://arxiv.org/abs/1705.07874>`_: Lundberg & Lee (2017)
- `Optimal Transport <https://arxiv.org/abs/1803.00567>`_: Peyré & Cuturi (2019)
- `Entropic OT <https://arxiv.org/abs/1306.0895>`_: Cuturi (2013)
