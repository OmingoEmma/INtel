"""Training script for the financial risk model with SHAP explainability.

This module trains a RandomForestRegressor on engineered features, persists the
trained model and predictions, and generates SHAP explainability artifacts for
both global and local model behavior analysis.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from src.explainability.shap_explainer import compute_and_save_shap_artifacts


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(path: str = "data/processed/merged_features.csv") -> pd.DataFrame:
    """Load processed features from CSV.

    Parameters
    ----------
    path: str
        Path to the processed features CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe of features.
    """
    return pd.read_csv(path)

def add_fake_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add a synthetic risk score target for training/demo purposes.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe containing economic features.

    Returns
    -------
    pd.DataFrame
        Dataframe with an added `risk_score` column.
    """
    # You’ll replace this later with a real scoring model or human labels
    np.random.seed(42)
    df["risk_score"] = (
        0.4 * df["sentiment_score"].fillna(0)
        - 0.2 * df["gdp"].fillna(0)
        + 0.3 * df["unemployment"].fillna(0)
        + 0.1 * df["cpi"].fillna(0)
        + np.random.normal(0, 1, len(df))  # noise
    )
    return df

def train_and_save_model(df: pd.DataFrame) -> Dict[str, str]:
    """Train model, save artifacts, and generate SHAP explanations.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe containing features and target (risk_score).

    Returns
    -------
    Dict[str, str]
        Mapping of generated SHAP artifact names to their file paths.
    """
    features = ["sentiment_score", "gdp", "unemployment", "cpi"]
    target = "risk_score"

    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = X, X, y, y

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    
    df.loc[:, "predicted_risk"] = model.predict(X)
    df.to_csv("data/processed/predicted_risks.csv", index=False)
    logger.info("Saved: predicted_risks.csv")


    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    logger.info("MAE: %.4f", mae)
    logger.info("R2 Score: %.4f", r2)

    os.makedirs("models", exist_ok=True)
    os.makedirs("examples", exist_ok=True)

    joblib.dump(model, "models/risk_model.pkl")

    # Save sample predictions
    df_preds = pd.DataFrame({
        "actual": y_test,
        "predicted": preds
    }).reset_index(drop=True)

    df_preds.head(5).to_json("examples/sample_predictions.json", orient="records", indent=2)

    # SHAP explainability artifacts
    artifacts: Dict[str, str] = {}
    try:
        # For SHAP, use a manageable subset to avoid heavy computation
        X_for_shap = X_test.copy()
        if len(X_for_shap) > 1000:
            X_for_shap = X_for_shap.sample(n=1000, random_state=42)

        artifacts = compute_and_save_shap_artifacts(
            model=model,
            X=X_for_shap,
            feature_names=list(X.columns),
            reports_dir="reports",
            figures_dir="reports/figures",
            max_display=10,
        )
        logger.info("Generated SHAP artifacts: %s", artifacts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("SHAP explainability failed: %s", exc)

    return artifacts

if __name__ == "__main__":
    df = load_data()
    df = add_fake_risk_score(df)
    # After risk score is added
    df.to_csv("data/processed/merged_features.csv", index=False)

    train_and_save_model(df)
