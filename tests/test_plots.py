"""Tests for FDFI plotting functions."""

import ast
import json
import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from fdfi.plots import (
    confidence_interval_plot,
    correlation_heatmap,
    dependence_plot,
    diagnostics_plot,
    force_plot,
    summary_bar,
    summary_plot,
    waterfall_plot,
)


@pytest.fixture
def plot_data():
    """Small deterministic plotting fixture."""
    rng = np.random.default_rng(7)
    values = rng.normal(size=(40, 5))
    features = rng.normal(size=(40, 5))
    names = [f"F{i}" for i in range(5)]
    return values, features, names


def close(fig):
    """Close a figure in tests."""
    plt.close(fig)


class TestSummaryPlot:
    """Tests for summary_plot."""

    def test_summary_plot_2d_returns_matplotlib_objects(self, plot_data):
        values, features, names = plot_data

        fig, ax = summary_plot(
            values,
            features=features,
            feature_names=names,
            max_display=4,
            show=False,
        )

        assert fig is not None
        assert ax is not None
        assert len(ax.get_yticklabels()) == 4
        close(fig)

    def test_summary_plot_1d_delegates_to_summary_bar(self, plot_data):
        values, _, names = plot_data
        phi = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])

        fig, ax, table = summary_plot(
            phi,
            se_X=se,
            feature_names=names,
            show=False,
        )

        assert fig is not None
        assert ax is not None
        assert isinstance(table, pd.DataFrame)
        assert list(table.columns) == ["feature", "phi", "se"]
        assert table["phi"].is_monotonic_decreasing
        close(fig)


class TestSummaryBar:
    """Tests for summary_bar."""

    def test_summary_bar_handles_none_and_finite_se(self):
        phi = np.array([0.1, -0.5, 0.2])

        fig, ax, table = summary_bar(phi, show=False)

        assert fig is not None
        assert ax is not None
        assert np.allclose(table["se"], 0.0)
        assert table["feature"].iloc[0] == "Feature 1"
        close(fig)

    def test_summary_bar_sanitizes_nan_inf_and_negative_se(self):
        phi = np.array([0.1, 0.4, 0.2, 0.3])
        se = np.array([np.nan, np.inf, -0.5, 0.1])
        names = ["a", "b", "c", "d"]

        fig, _, table = summary_bar(phi, se, names, show=False)

        assert np.all(np.isfinite(table["se"]))
        assert np.all(table["se"] >= 0)
        assert table.loc[table["feature"] == "a", "se"].item() == 0.0
        close(fig)

    def test_summary_bar_uses_group_colors_and_savepath(self, tmp_path):
        phi = np.array([0.1, 0.4, 0.2])
        se = np.array([0.01, 0.02, 0.03])
        names = ["a", "b", "c"]
        savepath = tmp_path / "summary_bar.png"

        fig, ax, _ = summary_bar(
            phi,
            se,
            names,
            group_colors={"a": "#111111", "b": "#222222"},
            savepath=str(savepath),
            show=False,
        )

        assert savepath.exists()
        assert len(ax.patches) == 3
        close(fig)

    def test_summary_bar_validates_shapes(self):
        with pytest.raises(ValueError):
            summary_bar([1.0, 2.0], se_X=[0.1], show=False)


class TestCorePlots:
    """Tests for waterfall, force, and dependence plots."""

    def test_waterfall_plot_groups_remaining_features(self):
        values = np.array([0.5, -0.2, 0.1, 0.05, 0.03])

        fig, ax = waterfall_plot(
            values,
            feature_names=[f"F{i}" for i in range(5)],
            max_display=3,
            base_value=1.0,
            show=False,
        )

        assert fig is not None
        assert ax is not None
        assert len(ax.patches) == 3
        assert "remaining" in ax.get_yticklabels()[-1].get_text()
        close(fig)

    def test_force_plot_accepts_2d_values(self, plot_data):
        values, _, names = plot_data

        fig, ax = force_plot(
            0.25,
            values,
            feature_names=names,
            max_display=4,
            show=False,
        )

        assert fig is not None
        assert ax is not None
        assert len(ax.patches) == 4
        close(fig)

    def test_dependence_plot_resolves_int_and_string_features(self, plot_data):
        values, features, names = plot_data

        fig_int, ax_int = dependence_plot(
            0,
            values,
            features,
            feature_names=names,
            show=False,
        )
        fig_str, ax_str = dependence_plot(
            "F1",
            values,
            features,
            feature_names=names,
            interaction_index="F2",
            show=False,
        )

        assert ax_int.get_xlabel() == "F0"
        assert ax_str.get_xlabel() == "F1"
        close(fig_int)
        close(fig_str)

    def test_dependence_plot_validates_string_feature_names(self, plot_data):
        values, features, names = plot_data

        with pytest.raises(ValueError):
            dependence_plot("missing", values, features, feature_names=names, show=False)


class TestCorrelationHeatmap:
    """Tests for correlation_heatmap."""

    def test_correlation_heatmap_returns_clustered_names(self):
        rng = np.random.default_rng(3)
        x0 = rng.normal(size=80)
        X = np.column_stack([x0, x0 + rng.normal(scale=0.01, size=80), rng.normal(size=80)])
        names = ["x0", "x1", "x2"]

        fig, ax, reordered = correlation_heatmap(X, names, show=False)

        assert fig is not None
        assert ax is not None
        assert set(reordered) == set(names)
        close(fig)

    def test_correlation_heatmap_warns_for_small_sample_size(self):
        X = np.random.default_rng(4).normal(size=(12, 3))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig, _, _ = correlation_heatmap(X, ["a", "b", "c"], show=False)

        assert any("correlation estimates" in str(item.message) for item in caught)
        close(fig)

    def test_correlation_heatmap_validates_shape_and_names(self):
        with pytest.raises(ValueError):
            correlation_heatmap(np.arange(5), ["a"], show=False)

        with pytest.raises(ValueError):
            correlation_heatmap(np.ones((5, 2)), ["a"], show=False)


class TestInferenceAndDiagnosticsPlots:
    """Tests for confidence interval and diagnostics plots."""

    def test_confidence_interval_plot_accepts_conf_int_style_dict(self):
        ci = {
            "score": np.array([0.3, 0.1, 0.2]),
            "se": np.array([0.02, 0.01, 0.03]),
            "ci_lower": np.array([0.25, 0.08, 0.12]),
            "ci_upper": np.array([0.35, 0.12, 0.28]),
            "reject_null": np.array([True, False, True]),
            "ranking": np.array([1, 3, 2]),
            "margin": 0.0,
        }

        fig, ax = confidence_interval_plot(
            ci,
            feature_names=["a", "b", "c"],
            max_display=2,
            show=False,
        )

        assert fig is not None
        assert ax is not None
        assert len(ax.get_yticklabels()) == 2
        fig.canvas.draw()
        close(fig)

    def test_confidence_interval_plot_uses_group_names(self):
        ci = {
            "score": np.array([0.3, 0.1]),
            "ci_lower": np.array([0.2, -np.inf]),
            "ci_upper": np.array([np.inf, 0.2]),
            "reject_null": np.array([True, False]),
            "groups": ["clinical", "genomic"],
        }

        fig, ax = confidence_interval_plot(ci, show=False)

        assert [tick.get_text() for tick in ax.get_yticklabels()] == [
            "clinical",
            "genomic",
        ]
        close(fig)

    def test_confidence_interval_plot_validates_max_display(self):
        ci = {
            "score": np.array([0.3]),
            "ci_lower": np.array([0.2]),
            "ci_upper": np.array([0.4]),
        }

        with pytest.raises(ValueError):
            confidence_interval_plot(ci, max_display=0, show=False)

    def test_diagnostics_plot_accepts_existing_diagnostic_keys(self):
        diagnostics = {
            "latent_independence_dcor": np.eye(3),
            "latent_independence_median": 0.04,
            "distribution_fidelity_mmd": 0.08,
            "latent_independence_label": "GOOD",
            "distribution_fidelity_label": "MODERATE",
        }

        fig, axes = diagnostics_plot(
            diagnostics,
            feature_names=["z0", "z1", "z2"],
            show=False,
        )

        assert fig is not None
        assert len(axes) == 2
        close(fig)

    def test_diagnostics_plot_validates_dcor_matrix_shape(self):
        diagnostics = {
            "latent_independence_dcor": np.ones((2, 3)),
            "latent_independence_median": 0.04,
        }

        with pytest.raises(ValueError):
            diagnostics_plot(diagnostics, show=False)


class TestTutorialSmoke:
    """Smoke checks for visualization tutorial integration."""

    def test_visualization_tutorial_is_valid_json(self):
        with open("docs/tutorials/visualization.ipynb", encoding="utf-8") as handle:
            notebook = json.load(handle)

        assert notebook["nbformat"] == 4
        assert any(
            "correlation_heatmap" in "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )

    def test_visualization_tutorial_imports_existing_plot_functions(self):
        with open("docs/tutorials/visualization.ipynb", encoding="utf-8") as handle:
            notebook = json.load(handle)

        imported = set()
        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            for match in re.finditer(r"from fdfi\.plots import \((.*?)\)", source, re.S):
                imported.update(
                    name.strip() for name in match.group(1).split(",") if name.strip()
                )
            for line in source.splitlines():
                if line.startswith("from fdfi.plots import ") and "(" not in line:
                    imported.update(
                        name.strip()
                        for name in line.replace("from fdfi.plots import ", "").split(",")
                        if name.strip()
                    )

        module = ast.parse("from fdfi import plots")
        assert module is not None
        import fdfi.plots as plots

        expected = {
            "confidence_interval_plot",
            "correlation_heatmap",
            "dependence_plot",
            "diagnostics_plot",
            "force_plot",
            "summary_bar",
            "summary_plot",
            "waterfall_plot",
        }
        assert expected <= imported
        for name in imported:
            assert hasattr(plots, name)
