# src/explainability/explain_model.py

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load model and data
model = joblib.load("models/risk_model.pkl")
df = pd.read_csv("data/processed/merged_features.csv")

# Only use rows without missing values for explainability
features = ["sentiment_score", "gdp", "unemployment", "cpi"]
df_clean = df.dropna(subset=features)

# SHAP expects a model and a matrix of features
X = df_clean[features]

# Create SHAP explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Save summary plot
os.makedirs("reports/figures", exist_ok=True)
plt.title("SHAP Feature Impact Summary")
shap.plots.beeswarm(shap_values, show=False)
plt.savefig("reports/figures/shap_summary.png", bbox_inches="tight")

# Save SHAP values as JSON for dashboard
shap_df = pd.DataFrame(shap_values.values, columns=features)
shap_df["risk_score"] = model.predict(X)
shap_df.to_json("examples/shap_explanations.json", orient="records", indent=2)

print("SHAP summary plot saved to reports/figures/shap_summary.png")
print(" SHAP JSON saved to examples/shap_explanations.json")
