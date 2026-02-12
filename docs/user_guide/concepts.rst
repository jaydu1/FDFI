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
