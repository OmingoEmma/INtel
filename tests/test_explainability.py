import os
import json
import glob
import pandas as pd
import joblib

from src.explainability.shap_explainer import get_shap_explanation, compute_and_save_shap_artifacts


def _latest(pattern: str):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


def test_shap_explanation_shapes():
    model = joblib.load("models/risk_model.pkl")
    df = pd.read_csv("data/processed/merged_features.csv")
    features = ["sentiment_score", "gdp", "unemployment", "cpi"]
    df = df.dropna(subset=features)
    X = df[features].head(100)

    explanation = get_shap_explanation(model, X, features)

    assert explanation.values.shape[0] == X.shape[0]
    assert explanation.values.shape[1] == X.shape[1]
    assert len(explanation.base_values) == X.shape[0]


def test_compute_and_save_artifacts():
    model = joblib.load("models/risk_model.pkl")
    df = pd.read_csv("data/processed/merged_features.csv")
    features = ["sentiment_score", "gdp", "unemployment", "cpi"]
    df = df.dropna(subset=features)
    X = df[features].head(200)

    artifacts = compute_and_save_shap_artifacts(model, X, feature_names=features)

    # PNGs
    assert any(name.endswith("_png") for name in artifacts.keys())
    for key in ["summary_png", "bar_png", "waterfall_png"]:
        if key in artifacts:
            assert os.path.exists(artifacts[key])

    # HTMLs
    for key in ["summary_html", "local_html"]:
        if key in artifacts:
            assert os.path.exists(artifacts[key])
            with open(artifacts[key], "r", encoding="utf-8") as f:
                html = f.read(128)
                assert "<html" in html.lower() or "<div" in html.lower()

    # JSON values and manifest
    for key in ["values_json", "manifest_json"]:
        if key in artifacts:
            assert os.path.exists(artifacts[key])
            with open(artifacts[key], "r", encoding="utf-8") as f:
                content = f.read(64)
                assert content

