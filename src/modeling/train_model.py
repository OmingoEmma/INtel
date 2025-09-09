# src/modeling/train_model.py

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

def load_data(path="data/processed/merged_features.csv"):
    return pd.read_csv(path)

def add_fake_risk_score(df):
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

def train_and_save_model(df):
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
    print("Saved: predicted_risks.csv")


    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("examples", exist_ok=True)

    joblib.dump(model, "models/risk_model.pkl")

    # Save sample predictions
    df_preds = pd.DataFrame({
        "actual": y_test,
        "predicted": preds
    }).reset_index(drop=True)

    df_preds.head(5).to_json("examples/sample_predictions.json", orient="records", indent=2)

if __name__ == "__main__":
    df = load_data()
    df = add_fake_risk_score(df)
    # After risk score is added
    df.to_csv("data/processed/merged_features.csv", index=False)

    train_and_save_model(df)
