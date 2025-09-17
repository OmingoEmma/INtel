"""
RiskIntel Chapter 5: SHAP explainability artifacts.

Outputs:
- artifacts/figs/shap_summary.png
- artifacts/figs/shap_waterfall_high.png
- artifacts/figs/shap_waterfall_low.png

CLI:
  python src/explain.py [--csv data/processed/merged_features.csv] [--demo]
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    shap = None  # type: ignore
    _HAS_SHAP = False

try:
    from .ri_utils import ensure_dirs, infer_columns, temporal_split
except Exception:  # pragma: no cover
    import os as _os
    import importlib.util as _importlib_util
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    _UTILS_PATH = _os.path.join(_HERE, "utils.py")
    _spec = _importlib_util.spec_from_file_location("ri_local_utils", _UTILS_PATH)
    if _spec is None or _spec.loader is None:
        raise
    _mod = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ensure_dirs = getattr(_mod, "ensure_dirs")
    infer_columns = getattr(_mod, "infer_columns")
    temporal_split = getattr(_mod, "temporal_split")


CSV_PATH = "data/processed/merged_features.csv"


def _synthetic_demo_like_evaluate() -> pd.DataFrame:
    # Minimal synthetic consistent with evaluate.py
    rng = np.random.default_rng(42)
    countries = ["C01", "C02", "C03", "C04", "C05"]
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    rows = []
    for c in countries:
        cpi = np.cumsum(rng.normal(0, 0.05, len(dates))) + 2
        unemp = 5 + rng.normal(0, 0.2, len(dates))
        fxv = np.abs(rng.normal(0.3, 0.1, len(dates)))
        sent = rng.normal(0, 0.5, len(dates))
        risk = 0.5 * cpi + 0.3 * unemp + 0.2 * fxv - 0.2 * sent + rng.normal(0, 0.5, len(dates))
        for i, d in enumerate(dates):
            rows.append({
                "country": c,
                "date": d.strftime("%Y-%m-%d"),
                "cpi_yoy": cpi[i],
                "unemp_rate": unemp[i],
                "fx_vol": fxv[i],
                "sentiment_mean": sent[i],
                "sentiment_z": (sent[i] - sent.mean()) / (sent.std() + 1e-6),
                "sentiment_n": int(rng.integers(5, 40)),
                "risk_score": risk[i],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="RiskIntel SHAP explainability")
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--demo", action="store_true", help="Generate synthetic data if CSV missing")
    args = parser.parse_args()

    ensure_dirs(["artifacts/figs", "docs"]) 

    if not os.path.exists(args.csv):
        if args.demo:
            df = _synthetic_demo_like_evaluate()
            os.makedirs(os.path.dirname(args.csv), exist_ok=True)
            df.to_csv(args.csv, index=False)
            print(f"[demo] Wrote synthetic dataset to {args.csv}")
        else:
            raise FileNotFoundError(f"Data missing at {args.csv}. Run with --demo to generate synthetic example.")
    else:
        df = pd.read_csv(args.csv)

    mapping = infer_columns(df)
    date_col = mapping.get("date", "date")
    country_col = mapping.get("country", "country")
    target_col = mapping.get("risk_score", "risk_score")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values([date_col, country_col]).reset_index(drop=True)

    train_df, test_df = temporal_split(df, date_col=date_col, test_ratio=0.2)

    # Prepare features
    sentiment_cols = [c for c in mapping.get("__sentiment_cols", "").split(",") if c]
    macro_cols = [c for c in mapping.get("__macro_cols", "").split(",") if c]
    feature_cols_all = macro_cols + sentiment_cols

    X_train = train_df[feature_cols_all]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols_all]
    y_test = test_df[target_col]

    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    summary_path = "artifacts/figs/shap_summary.png"
    high_path = "artifacts/figs/shap_waterfall_high.png"
    low_path = "artifacts/figs/shap_waterfall_low.png"

    if _HAS_SHAP:
        # SHAP summary
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150)
        plt.close()
    else:
        # Fallback: feature importances bar chart
        importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values()
        plt.figure(figsize=(9, 6))
        importances.plot(kind="barh")
        plt.title("Global feature importance (RF, SHAP unavailable)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(summary_path, dpi=150)
        plt.close()

    # Choose high/low risk instances by prediction
    preds = pd.Series(y_pred, index=X_test.index)
    high_thresh = preds.quantile(0.9)
    low_thresh = preds.quantile(0.1)
    high_idx = preds[preds >= high_thresh].index[0]
    low_idx = preds[preds <= low_thresh].index[0]

    if _HAS_SHAP:
        sv_high = shap_values[X_test.index.get_loc(high_idx)]
        base_value = explainer.expected_value
        shap.plots.waterfall(
            shap.Explanation(values=sv_high, base_values=base_value, data=X_test.loc[high_idx], feature_names=list(X_test.columns)),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(high_path, dpi=150)
        plt.close()

        sv_low = shap_values[X_test.index.get_loc(low_idx)]
        shap.plots.waterfall(
            shap.Explanation(values=sv_low, base_values=base_value, data=X_test.loc[low_idx], feature_names=list(X_test.columns)),
            show=False,
        )
        plt.tight_layout()
        plt.savefig(low_path, dpi=150)
        plt.close()
    else:
        # Fallback: approximate local contributions using centered features and feature importances
        def _approx_contrib(x_row: pd.Series, X_ref: pd.DataFrame) -> pd.Series:
            centered = x_row - X_ref.mean()
            weights = pd.Series(model.feature_importances_, index=X_ref.columns)
            return centered * (weights / (weights.abs().sum() + 1e-12))

        for idx, out_path, label in [(high_idx, high_path, "High-risk approx contributions"), (low_idx, low_path, "Low-risk approx contributions")]:
            contrib = _approx_contrib(X_test.loc[idx], X_train)
            top = contrib.abs().sort_values(ascending=False).head(15).sort_values()
            plt.figure(figsize=(9, 6))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in top]
            top.plot(kind="barh", color=colors)
            plt.title(label)
            plt.xlabel("Approx contribution (arb units)")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()

    print(f"Wrote {summary_path}")
    print(f"Wrote {high_path}")
    print(f"Wrote {low_path}")

    print("\nInterpretation (brief):")
    print("- Global summary highlights top drivers across the test set.")
    print("- Local waterfalls illustrate how features push risk up/down for specific days.")

    # Checklist
    try:
        from .ri_utils import print_checklist  # type: ignore
    except Exception:
        import os as _os
        import importlib.util as _importlib_util
        _HERE = _os.path.dirname(_os.path.abspath(__file__))
        _UTILS_PATH = _os.path.join(_HERE, "utils.py")
        _spec = _importlib_util.spec_from_file_location("ri_local_utils", _UTILS_PATH)
        if _spec is None or _spec.loader is None:
            raise
        _mod = _importlib_util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        print_checklist = getattr(_mod, "print_checklist")

    print_checklist([
        (summary_path, os.path.exists(summary_path)),
        (high_path, os.path.exists(high_path)),
        (low_path, os.path.exists(low_path)),
    ])


if __name__ == "__main__":
    main()

