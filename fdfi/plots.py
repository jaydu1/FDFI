"""
Flow-Disentangled Feature Importance (FDFI): Publication-Grade Visualizations

This module provides standardized plotting utilities for FDFI analysis, including
correlation heatmaps with hierarchical clustering, global importance bar charts
with uncertainty quantification, and coefficient-of-variation scatter plots for
diagnosing attribution reliability.

All functions follow NumPy documentation standards and integrate seamlessly with
the Jupyter notebook ecosystem for reproducible research.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster import hierarchy


def correlation_heatmap(X_background, feature_names, savepath=None, **kwargs):
    """
    Plot a hierarchical-clustered Pearson correlation matrix heatmap.
    
    This function computes the Pearson correlation matrix of the feature matrix `X_background`,
    then reorders rows and columns using hierarchical clustering based on absolute
    correlation distance. The resulting visualization reveals collinear blocks among
    features, which is the fundamental motivation for the FDFI disentanglement step.
    
    The clustering uses average linkage and distance metric d = 1 - |correlation|,
    which is semantically appropriate for grouping correlated features near each other.
    
    Parameters
    ----------
    X_background : ndarray, shape (n_samples, n_features)
        Background/training feature matrix used to compute correlation structure.
        Rows are samples, columns are features.
        
        **Important:** This should be a representative background dataset (e.g., training data),
        NOT a small test batch. Correlation estimates are sample-size dependent; small samples
        yield unreliable estimates. A warning is issued if n_samples < 50.
    feature_names : list of str
        Human-readable names of the features (length must match X_background.shape[1]).
    savepath : str, optional
        If provided, saves the figure to this path (e.g., 'fig1_correlation.pdf').
        Default is None (figure is not saved).
    **kwargs
        Additional keyword arguments passed to matplotlib:
        
        figsize : tuple, default (10, 8)
            Width and height of the figure in inches.
        cmap : str, default 'RdBu_r'
            Colormap for the heatmap (reversed red-blue).
        vmin, vmax : float, default -1, 1
            Limits of the color scale (for Pearson correlations).
        fontsize : int, default 9
            Font size for axis labels.
        title_fontsize : int, default 11
            Font size for the title.
        dpi : int, default 150
            Dots per inch for saving.
        bbox_inches : str, default 'tight'
            Bounding box setting for saved figures.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the heatmap.
    feature_names_reordered : list of str
        Feature names in the new (clustered) order, reflecting the reordered matrix.
    
    Notes
    -----
    **Hierarchical Clustering:**
    The function constructs a distance matrix d_ij = 1 - |rho_ij|, where rho_ij
    is the Pearson correlation. It then applies scipy.cluster.hierarchy.linkage
    with method='average' to compute a hierarchical clustering. The resulting
    dendrogram is traversed (no_plot=True) to extract the optimal leaf order.
    
    **Reproducibility:**
    The printed summary reports the top-3 absolute pairwise correlations from the
    *original* (unclustered) matrix to maintain reproducibility and statistical
    transparency.
    
    **Sample Size Warning:**
    Correlation estimates stabilize with larger samples. If n_samples < 50,
    a warning is issued recommending a larger background dataset for more
    reliable correlation structure.
    
    **Academic Context:**
    High off-diagonal entries in the reordered heatmap reveal collinearity, which
    violates the feature independence assumption in standard Shapley estimators
    (e.g., KernelSHAP). FDFI addresses this via a learned normalising flow that
    maps the correlated feature space to an independent latent space.
    
    Examples
    --------
    >>> import numpy as np
    >>> from dfi.plots import correlation_heatmap
    >>> np.random.seed(42)
    >>> X_background = np.random.randn(100, 5)
    >>> feature_names = ['F1', 'F2', 'F3', 'F4', 'F5']
    >>> fig, ax, names_reord = correlation_heatmap(X_background, feature_names, 
    ...                                               savepath='corr_heatmap.pdf')
    >>> plt.show()
    
    >>> # Example with custom styling
    >>> fig, ax, names_reord = correlation_heatmap(
    ...     X_background, feature_names,
    ...     figsize=(12, 10),
    ...     fontsize=10,
    ...     title_fontsize=13,
    ...     cmap='coolwarm'
    ... )
    """
    
    # --- Input Validation ---
    X_background = np.asarray(X_background)
    if X_background.ndim != 2:
        raise ValueError(f"X_background must be 2-dimensional; got shape {X_background.shape}")
    
    n_samples, n_features = X_background.shape
    if n_features != len(feature_names):
        raise ValueError(
            f"Number of features ({n_features}) does not match length of "
            f"feature_names ({len(feature_names)})"
        )
    
    if n_samples < 50:
        warnings.warn(
            f"X_background has only {n_samples} samples. Correlation estimates are "
            "unreliable with small sample sizes. Recommend n_samples ≥ 50 for "
            "statistically stable correlation structure.",
            UserWarning
        )
    
    # Extract kwargs with sensible defaults
    figsize = kwargs.get('figsize', (10, 8))
    cmap = kwargs.get('cmap', 'RdBu_r')
    vmin = kwargs.get('vmin', -1)
    vmax = kwargs.get('vmax', 1)
    fontsize = kwargs.get('fontsize', 9)
    title_fontsize = kwargs.get('title_fontsize', 11)
    dpi = kwargs.get('dpi', 150)
    bbox_inches = kwargs.get('bbox_inches', 'tight')
    
    # Compute Pearson correlation matrix
    df_features = pd.DataFrame(X_background, columns=feature_names)
    corr = df_features.corr()
    
    # --- Hierarchical Clustering Reordering ---
    # Distance = 1 - |correlation|. This places positively and negatively
    # correlated features close together.
    dist_matrix = 1 - np.abs(corr.values)
    linkage_matrix = hierarchy.linkage(distance.squareform(dist_matrix), method='average')
    dendro = hierarchy.dendrogram(linkage_matrix, no_plot=True)
    leaf_order = dendro['leaves']
    
    # Reorder correlation matrix and feature names
    corr_reordered = corr.values[leaf_order, :][:, leaf_order]
    feature_names_reordered = [feature_names[i] for i in leaf_order]
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_reordered, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    
    ax.set_xticks(range(len(feature_names_reordered)))
    ax.set_yticks(range(len(feature_names_reordered)))
    ax.set_xticklabels(feature_names_reordered, rotation=45, ha='right', fontsize=fontsize)
    ax.set_yticklabels(feature_names_reordered, fontsize=fontsize)
    ax.set_title(
        'Pearson Correlation Matrix — Feature Space\n'
        r'(Hierarchical‑clustering reordering reveals collinear blocks)',
        fontsize=title_fontsize, pad=12
    )
    
    plt.tight_layout()
    
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches)
    
    # --- Print Diagnostic Summary ---
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    top3 = upper.stack().abs().sort_values(ascending=False).head(3)
    print('Top-3 absolute pairwise correlations (from original matrix):')
    print(top3.to_string())
    
    return fig, ax, feature_names_reordered


def summary_bar(phi_X, se_X, feature_names, group_colors=None, savepath=None, **kwargs):
    """
    Plot global FDFI feature importance as a bar chart with uncertainty quantification.
    
    This function visualizes the global feature importance scores (phi_X) as a bar chart,
    with bootstrap standard errors (se_X) displayed as error bars. Features are sorted
    in descending order of importance for rapid visual prioritization. Optional group
    coloring enables domain-specific categorization (e.g., FHR baseline vs. accelerations).
    
    When `group_colors` is `None`, bars are colored dynamically using a colormap
    gradient reflecting the magnitude of each feature's importance, providing
    visual consistency across diverse datasets.
    
    This plot directly addresses the Uncertainty Quantification requirement of
    reproducible statistical software: unlike point-estimate methods, error bars
    allow practitioners to distinguish between high-confidence and high-variance
    attribution scores.
    
    Parameters
    ----------
    phi_X : ndarray, shape (n_features,)
        Global feature importance scores. Typically computed as the mean absolute
        Shapley value (|phi|) across a set of test instances.
    se_X : ndarray, shape (n_features,)
        Bootstrap standard errors for phi_X. Represents the uncertainty in each
        importance estimate. NaN and inf values are automatically sanitized:
        NaN → 0.0, inf → max finite value (or 0.0 if all infinite).
    feature_names : list of str
        Human-readable feature names. Length must match phi_X.shape[0].
    group_colors : dict, optional
        Mapping from feature name (str) to color hex code (e.g., '#FF5733').
        Features not in this dict default to '#888888' (gray).
        If None, bars are colored dynamically using a colormap gradient based
        on importance values. Default is None.
        
        Example: {'LB': '#4878a8', 'ASTV': '#4878a8', 'Mean': '#c9a227', ...}
    savepath : str, optional
        If provided, saves the figure to this path (e.g., 'importance_bar.pdf').
        Default is None.
    **kwargs
        Additional keyword arguments for matplotlib styling:
        
        figsize : tuple, default (13, 5)
            Figure dimensions in inches.
        title_fontsize : int, default 11
            Font size for the title.
        label_fontsize : int, default 11
            Font size for axis labels (xlabel, ylabel).
        tick_fontsize : int, default 9
            Font size for tick labels.
        capsize : float, default 4
            Width of the error bar caps.
        elinewidth : float, default 1.3
            Line width of error bars and caps.
        dpi : int, default 150
            Dots per inch for saving.
        bbox_inches : str, default 'tight'
            Bounding box setting for saved figures.
        cmap : str, default 'viridis'
            Colormap for dynamic coloring when group_colors is None.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    importance_df : DataFrame
        Sorted DataFrame with columns ['feature', 'phi', 'se'].
        Rows are sorted by 'phi' in descending order.
    
    Notes
    -----
    **Interpretation:**
    - Bar height: point estimate of feature importance (mean |phi_X|).
    - Error bar: 1 bootstrap standard error (se_X).
    - Confidence intervals: multiply se_X by 1.96 (for 95% CI) or 2.576 (for 99% CI).
    
    **Feature Sorting:**
    The returned DataFrame is sorted by phi_X descending, making it trivial to
    extract the top-k most important features for downstream analysis.
    
    **Error Bar Sanitization:**
    The function automatically handles NaN and inf values in se_X, converting them
    to safe numeric defaults. This prevents matplotlib crashes from malformed
    upstream confidence interval estimates.
    
    **Reducing Error Bars:**
    To reduce the width of error bars (and identify significance more sharply),
    increase the `nsamples` parameter in `FlowExplainer.__init__()`.
    
    Examples
    --------
    >>> import numpy as np
    >>> from dfi.plots import summary_bar
    >>> np.random.seed(42)
    >>> phi_X = np.array([0.050, 0.025, 0.032, 0.015, 0.008])
    >>> se_X = np.array([0.005, 0.003, 0.004, 0.002, 0.001])
    >>> feature_names = ['LB', 'ASTV', 'Mean', 'Width', 'Mode']
    >>> group_colors = {
    ...     'LB': '#4878a8', 'ASTV': '#4878a8', 'Mean': '#c9a227',
    ...     'Width': '#3a9e8c', 'Mode': '#c9a227'
    ... }
    >>> fig, ax, df = summary_bar(
    ...     phi_X, se_X, feature_names,
    ...     group_colors=group_colors,
    ...     savepath='importance_bar.pdf'
    ... )
    >>> plt.show()
    
    >>> # Example: dynamic coloring (no group_colors)
    >>> fig, ax, df = summary_bar(phi_X, se_X, feature_names)
    >>> plt.show()
    
    >>> # Example: access top-3 features
    >>> top3_features = df.head(3)['feature'].values
    >>> print(f"Top-3 features: {top3_features}")
    """
    
    # Extract kwargs with defaults
    figsize = kwargs.get('figsize', (13, 5))
    title_fontsize = kwargs.get('title_fontsize', 11)
    label_fontsize = kwargs.get('label_fontsize', 11)
    tick_fontsize = kwargs.get('tick_fontsize', 9)
    capsize = kwargs.get('capsize', 4)
    elinewidth = kwargs.get('elinewidth', 1.3)
    dpi = kwargs.get('dpi', 150)
    bbox_inches = kwargs.get('bbox_inches', 'tight')
    colormap_name = kwargs.get('cmap', 'viridis')
    
    # Ensure inputs are numpy arrays
    phi_X = np.asarray(phi_X)
    se_X = np.asarray(se_X)
    
    # --- Sanitize se_X: handle NaN and inf ---
    se_X_clean = se_X.copy()
    # First, identify all finite values
    finite_mask = np.isfinite(se_X_clean)
    
    # Get max finite value for replacement of inf
    if np.any(finite_mask):
        max_finite = np.max(se_X_clean[finite_mask])
    else:
        max_finite = 0.0
    
    # Replace NaN with 0.0
    se_X_clean = np.where(np.isnan(se_X_clean), 0.0, se_X_clean)
    # Replace inf with max_finite
    se_X_clean = np.where(np.isinf(se_X_clean), max_finite, se_X_clean)
    
    # Create DataFrame and sort by importance (descending)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'phi': np.abs(phi_X),
        'se': se_X_clean
    }).sort_values('phi', ascending=False).reset_index(drop=True)
    
    # Assign colors: dynamic gradient if group_colors is None, else categorical
    if group_colors is None:
        # Dynamic colormap-based coloring
        cmap = plt.get_cmap(colormap_name)
        phi_normalized = importance_df['phi'].values
        phi_min = phi_normalized.min()
        phi_max = phi_normalized.max()
        if phi_max > phi_min:
            phi_normalized = (phi_normalized - phi_min) / (phi_max - phi_min)
        else:
            phi_normalized = np.ones_like(phi_normalized) * 0.5
        colours = [cmap(val) for val in phi_normalized]
    else:
        colours = [group_colors.get(f, '#888888') for f in importance_df['feature']]
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(
        importance_df['feature'],
        importance_df['phi'],
        yerr=importance_df['se'],
        color=colours,
        edgecolor='white',
        linewidth=0.5,
        capsize=capsize,
        error_kw=dict(elinewidth=elinewidth, ecolor='#222222', capthick=elinewidth),
        zorder=3,
    )
    
    ax.set_xlabel('Feature', fontsize=label_fontsize)
    ax.set_ylabel(r'Mean $|\phi_X|$ (FDFI Importance)', fontsize=label_fontsize)
    ax.set_title(
        'Global FDFI Feature Importance with Uncertainty Quantification\n'
        r'(Error bars = bootstrap standard error; features sorted by decreasing importance)',
        fontsize=title_fontsize, pad=10
    )
    ax.set_xticklabels(importance_df['feature'], rotation=45, ha='right', fontsize=tick_fontsize)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches)
    
    return fig, ax, importance_df


def cv_scatter(phi_X, se_X, feature_names, y_cutoff=3.0, group_colors=None, 
               savepath=None, **kwargs):
    """
    Plot Coefficient of Variation (CV) against feature importance for reliability diagnosis.
    
    This function diagnoses the reliability and statistical stability of FDFI attributions
    by scattering the Coefficient of Variation (CV = se_X / (phi_X + epsilon)) against the importance
    scores (phi_X). Features with CV > y_cutoff are *not dropped*: they are still shown
    on the plot by visually capping their CV values near the top border, and are also
    reported separately as statistically unreliable attributions (high noise relative
    to signal).
    
    This plot operationalizes the principle of statistical transparency: it exposes
    which attributions are well-determined and which are poorly estimated, guiding
    the practitioner on where to invest computational resources (via increasing
    `nsamples` in FlowExplainer) to reduce uncertainty.
    
    Parameters
    ----------
    phi_X : ndarray, shape (n_features,)
        Feature importance scores. Typically mean |phi_X| across instances.
    se_X : ndarray, shape (n_features,)
        Bootstrap standard errors for phi_X, or enlarged standard errors from
        `explainer.conf_int()` (which include variance-floor and margin adjustments).
    feature_names : list of str
        Human-readable feature names. Length must match phi_X.shape[0].
    y_cutoff : float, default 3.0
        Coefficient of Variation threshold for visual clarity.
        Features with CV > y_cutoff are visually capped at the top of the plot
        (hollow downward triangles) and reported in console output.
        
        Recommended range: 2.0–5.0.
        - y_cutoff=2.0: stringent; excludes features where SE > 0.5 * phi.
        - y_cutoff=3.0: moderate; good balance for most applications.
        - y_cutoff=5.0: lenient; includes more features but may compress visualization.
    group_colors : dict, optional
        Mapping from feature name to color hex code. Features not in dict
        default to '#888888' (gray).
        If None, all points are colored '#4878a8' (blue).
    savepath : str, optional
        If provided, saves the figure to this path. Default is None.
    **kwargs
        Additional keyword arguments:
        
        figsize : tuple, default (10, 4)
            Figure dimensions in inches.
        title_fontsize : int, default 11
            Font size for the title.
        label_fontsize : int, default 10
            Font size for axis labels.
        tick_fontsize : int, default 7.5
            Font size for point annotations (feature names).
        dpi : int, default 150
            Dots per inch for saving.
        bbox_inches : str, default 'tight'
            Bounding box setting.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    cv_df_filtered : DataFrame
        DataFrame of plotted (included) features with columns
        ['feature', 'phi', 'se', 'cv']. Sorted by 'phi' descending.
    cv_df_outliers : DataFrame
        DataFrame of high-CV features with same columns.
    
    Notes
    -----
    **Interpretation:**
    
    Region 1 — Well-determined (CV < 1.0):
        Standard error < importance score. These attributions are reliable.
    
    Region 2 — Uncertain (1.0 ≤ CV ≤ y_cutoff):
        Standard error ≥ importance score but within acceptable range.
        Interpretation: use these results cautiously or increase `nsamples`.
    
    Region 3 — Noise (CV > y_cutoff):
        Standard error >> importance score. Shown as capped hollow triangles.
        Interpretation: statistical signal is overwhelmed by noise;
        confidence in ranking is low. Consider increasing `nsamples`.
    
    **Red Dashed Line (CV = 1.0):**
    Marks the transition between well-determined and uncertain regimes.
    Features on or above this line warrant careful interpretation.
    
    **Zero Importance Handling:**
    Features with exactly zero importance but non-zero SE will appear with
    high CV due to numerical epsilon added for stability (1e-8). These are
    correctly reported as unreliable and will be shown as capped outliers if
    CV > y_cutoff.
    
    **Reducing CV:**
    Increase `nsamples` in `FlowExplainer()`. Doubling nsamples reduces
    standard errors by approximately sqrt(2), thus reducing CV by ~0.7x.
    
    Examples
    --------
    >>> import numpy as np
    >>> from dfi.plots import cv_scatter
    >>> np.random.seed(42)
    >>> phi_X = np.array([0.050, 0.025, 0.032, 0.015, 0.008])
    >>> se_X = np.array([0.005, 0.010, 0.080, 0.002, 0.015])
    >>> feature_names = ['LB', 'ASTV', 'Mean', 'Width', 'Mode']
    >>> group_colors = {
    ...     'LB': '#4878a8', 'ASTV': '#4878a8', 'Mean': '#c9a227',
    ...     'Width': '#3a9e8c', 'Mode': '#c9a227'
    ... }
    >>> fig, ax, df_filt, df_out = cv_scatter(
    ...     phi_X, se_X, feature_names, y_cutoff=3.0,
    ...     group_colors=group_colors,
    ...     savepath='cv_scatter.pdf'
    ... )
    >>> plt.show()
    
    >>> # Inspect which features were excluded
    >>> if len(df_out) > 0:
    ...     print("Excluded features (high CV):")
    ...     print(df_out[['feature', 'cv']])
    """
    
    # Extract kwargs with defaults
    figsize = kwargs.get('figsize', (10, 4))
    title_fontsize = kwargs.get('title_fontsize', 11)
    label_fontsize = kwargs.get('label_fontsize', 10)
    tick_fontsize = kwargs.get('tick_fontsize', 7.5)
    dpi = kwargs.get('dpi', 150)
    bbox_inches = kwargs.get('bbox_inches', 'tight')
    
    # Ensure inputs are numpy arrays
    phi_X = np.asarray(phi_X)
    se_X = np.asarray(se_X)
    
    # --- Create DataFrame and compute CV ---
    cv_df = pd.DataFrame({
        'feature': feature_names,
        'phi': np.abs(phi_X),
        'se': se_X
    })
    
    # Compute CV with numerical epsilon for stability (prevents silent NaN dropping)
    epsilon = 1e-8
    cv_df['cv'] = cv_df['se'] / (cv_df['phi'] + epsilon)
    
    # --- Partition by cutoff (do not drop outliers from the plot) ---
    cv_df_filtered = cv_df[cv_df['cv'] <= y_cutoff].copy().sort_values('phi', ascending=False)
    cv_df_outliers = cv_df[cv_df['cv'] > y_cutoff].copy().sort_values('phi', ascending=False)
    
    # --- Print Diagnostic Summary ---
    print("\nCoefficient of Variation (CV) Filtering Summary:")
    print(f"  Total features: {len(cv_df)}")
    print(f"  Displayed (CV ≤ {y_cutoff}): {len(cv_df_filtered)}")
    print(f"  High-CV (CV > {y_cutoff}): {len(cv_df_outliers)}")
    if len(cv_df_outliers) > 0:
        excluded_str = ', '.join(cv_df_outliers['feature'].values)
        print(f"  High-CV features (statistically unreliable): {excluded_str}")
    
    # --- Assign colors ---
    if group_colors is None:
        scatter_colours_inlier = ['#4878a8'] * len(cv_df_filtered)
        scatter_colours_outlier = ['#4878a8'] * len(cv_df_outliers)
    else:
        scatter_colours_inlier = [group_colors.get(f, '#888888') for f in cv_df_filtered['feature']]
        scatter_colours_outlier = [group_colors.get(f, '#888888') for f in cv_df_outliers['feature']]
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # Visual cap for outliers (place at the top edge but keep them visible)
    # Assumption: y-limit top is y_cutoff + 0.5, so default cap is y_cutoff + 0.4.
    y_top = y_cutoff + 0.5
    outlier_cap = kwargs.get('outlier_cap', y_cutoff + 0.4)
    outlier_cap = min(outlier_cap, y_top - 0.05)  # keep inside the frame

    # Inliers: solid circles (original behaviour)
    if len(cv_df_filtered) > 0:
        ax.scatter(
            cv_df_filtered['phi'], cv_df_filtered['cv'],
            c=scatter_colours_inlier, s=85, marker='o',
            edgecolors='white', linewidths=0.7, zorder=3
        )

    # Outliers: hollow downward triangles, capped to top edge
    if len(cv_df_outliers) > 0:
        ax.scatter(
            cv_df_outliers['phi'], np.full(len(cv_df_outliers), outlier_cap),
            s=100, marker='v',
            facecolors='none', edgecolors=scatter_colours_outlier,
            linewidths=1.2, zorder=4
        )
    
    # --- Annotate points with feature names ---
    for _, row in cv_df_filtered.iterrows():
        ax.annotate(
            row['feature'],
            xy=(row['phi'], row['cv']),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=tick_fontsize,
            color='#333333'
        )
    for _, row in cv_df_outliers.iterrows():
        ax.annotate(
            row['feature'],
            xy=(row['phi'], outlier_cap),
            xytext=(3, 3),
            textcoords='offset points',
            fontsize=tick_fontsize,
            color='#333333'
        )
    
    # --- Reference line at CV = 1.0 ---
    ax.axhline(1.0, color='crimson', linestyle='--', linewidth=1.1,
               label='CV = 1 (se = phi)', zorder=2)
    
    # --- Axes and labels ---
    ax.set_xlabel(r'Mean $|\phi_X|$ (importance)', fontsize=label_fontsize)
    ax.set_ylabel('Coefficient of Variation (se / phi)', fontsize=label_fontsize)
    ax.set_ylim(-0.05, y_top)
    ax.set_title(
        'Attribution Reliability: Importance vs. Coefficient of Variation\n'
        f'(Note: Features with CV > {y_cutoff} are capped at the top edge)',
        fontsize=title_fontsize, pad=12
    )

    # Legend: add a proxy artist explaining the capped outliers
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0], [0],
            marker='v', linestyle='None',
            markerfacecolor='none', markeredgecolor='#333333',
            markersize=8, label=f'CV > {y_cutoff}'
        )
    )
    ax.legend(handles=handles, fontsize=9, loc='upper right')
    ax.grid(linestyle='--', alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches=bbox_inches)
    
    return fig, ax, cv_df_filtered, cv_df_outliers