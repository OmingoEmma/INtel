"""SHAP explainability utilities for tree-based models.

This module provides functions to compute SHAP values using TreeExplainer and
to generate and persist global and local explanation artifacts (PNG and HTML)
for downstream consumption (e.g., Streamlit dashboards).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# Use non-interactive backend for headless environments
matplotlib.use("Agg")


logger = logging.getLogger(__name__)


def _ensure_directories(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def get_shap_explanation(
    model: object,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> shap.Explanation:
    """Compute SHAP values using TreeExplainer.

    Parameters
    ----------
    model: object
        A trained tree-based model (e.g., RandomForestRegressor).
    X: pd.DataFrame
        Feature matrix for which SHAP values will be computed.
    feature_names: Optional[List[str]]
        Names of the features. If None, uses X.columns.

    Returns
    -------
    shap.Explanation
        SHAP Explanation object containing values, base values, and data.

    Raises
    ------
    ValueError
        If SHAP computation fails or inputs are invalid.
    """
    if X is None or len(X) == 0:
        raise ValueError("X must be a non-empty DataFrame for SHAP computation")

    try:
        if feature_names is None:
            feature_names = list(X.columns)

        # TreeExplainer as required
        explainer = shap.TreeExplainer(model)
        shap_values_array: np.ndarray = explainer.shap_values(X)

        # Ensure 2D array for regression
        if isinstance(shap_values_array, list):
            # For classifiers TreeExplainer may return a list per class
            shap_values_array = np.array(shap_values_array).mean(axis=0)

        # Construct Explanation for compatibility with shap.plots.*
        base_values = np.full(shape=(X.shape[0],), fill_value=np.ravel(explainer.expected_value)[0])
        explanation = shap.Explanation(
            values=shap_values_array,
            base_values=base_values,
            data=X.to_numpy(),
            feature_names=feature_names,
        )
        return explanation
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to compute SHAP values: %s", exc)
        raise ValueError(f"Failed to compute SHAP values: {exc}") from exc


def _save_current_fig(path: str) -> str:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure to %s", path)
    return path


def _select_top_instance_by_abs_shap_sum(explanation: shap.Explanation) -> int:
    values = explanation.values
    # Sum absolute SHAP values per row
    row_scores = np.abs(values).sum(axis=1)
    return int(np.argmax(row_scores))


def compute_and_save_shap_artifacts(
    model: object,
    X: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    reports_dir: str = "reports",
    figures_dir: str = "reports/figures",
    max_display: int = 10,
) -> Dict[str, str]:
    """Compute SHAP values and persist global and local explanation artifacts.

    Generates and saves:
    - Global summary beeswarm (PNG)
    - Global feature importance bar plot (PNG)
    - Local waterfall for most impactful instance (PNG)
    - Global force plot (HTML)
    - Local force plot (HTML)
    - SHAP values JSON for dashboard use (reports)

    Parameters
    ----------
    model: object
        Trained model.
    X: pd.DataFrame
        Feature matrix to explain (kept reasonably small for performance).
    feature_names: Optional[List[str]]
        Feature names; defaults to X.columns.
    reports_dir: str
        Directory to store HTML and JSON outputs.
    figures_dir: str
        Directory to store PNG plots.
    max_display: int
        Maximum number of features to display in waterfall/plots.

    Returns
    -------
    Dict[str, str]
        Mapping of artifact names to file paths.
    """
    _ensure_directories([reports_dir, figures_dir])
    ts = _timestamp()

    explanation = get_shap_explanation(model, X, feature_names)

    # Derive feature names
    if feature_names is None:
        feature_names = list(getattr(X, "columns", [str(i) for i in range(X.shape[1])]))

    artifacts: Dict[str, str] = {}

    # Global beeswarm summary (PNG)
    try:
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)
        artifacts["summary_png"] = _save_current_fig(os.path.join(figures_dir, f"shap_summary_{ts}.png"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save SHAP beeswarm plot: %s", exc)

    # Global bar plot (PNG)
    try:
        plt.figure(figsize=(10, 6))
        shap.plots.bar(explanation, max_display=max_display, show=False)
        artifacts["bar_png"] = _save_current_fig(os.path.join(figures_dir, f"shap_feature_importance_{ts}.png"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save SHAP bar plot: %s", exc)

    # Local waterfall for the most impactful instance (PNG)
    try:
        idx = _select_top_instance_by_abs_shap_sum(explanation)
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation[idx], max_display=max_display, show=False)
        artifacts["waterfall_png"] = _save_current_fig(os.path.join(figures_dir, f"shap_waterfall_{ts}.png"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save SHAP waterfall plot: %s", exc)

    # Force plots (HTML) - global and local
    try:
        global_force = shap.force_plot(
            base_value=np.mean(explanation.base_values),
            shap_values=explanation.values,
            features=X,
            feature_names=feature_names,
        )
        global_html_path = os.path.join(reports_dir, f"shap_force_summary_{ts}.html")
        shap.save_html(global_html_path, global_force)
        artifacts["summary_html"] = global_html_path
        logger.info("Saved SHAP global force plot to %s", global_html_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save SHAP global force HTML: %s", exc)

    try:
        local_idx = _select_top_instance_by_abs_shap_sum(explanation)
        local_force = shap.force_plot(
            base_value=float(explanation.base_values[local_idx]),
            shap_values=explanation.values[local_idx],
            features=X.iloc[local_idx],
            feature_names=feature_names,
        )
        local_html_path = os.path.join(reports_dir, f"shap_force_local_{ts}.html")
        shap.save_html(local_html_path, local_force)
        artifacts["local_html"] = local_html_path
        logger.info("Saved SHAP local force plot to %s", local_html_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save SHAP local force HTML: %s", exc)

    # Save raw SHAP values JSON for dashboard integration
    try:
        shap_df = pd.DataFrame(explanation.values, columns=feature_names)
        shap_df.insert(0, "base_value", explanation.base_values)
        shap_json_path = os.path.join(reports_dir, f"shap_values_{ts}.json")
        shap_df.to_json(shap_json_path, orient="records")
        artifacts["values_json"] = shap_json_path
        logger.info("Saved SHAP values JSON to %s", shap_json_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save SHAP values JSON: %s", exc)

    # Also save a compact manifest to help dashboards discover files
    try:
        manifest_path = os.path.join(reports_dir, f"shap_manifest_{ts}.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=2)
        artifacts["manifest_json"] = manifest_path
        logger.info("Saved SHAP manifest to %s", manifest_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to save SHAP manifest: %s", exc)

    return artifacts

# Backwards-compatible alias for other callers
save_shap_artifacts = compute_and_save_shap_artifacts

