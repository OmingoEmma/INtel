# src/dashboard/streamlit_app.py

import streamlit as st
import pandas as pd
import json
from PIL import Image

# Load risk predictions
df = pd.read_csv("data/processed/merged_features.csv")
sample_preds = pd.read_json("examples/sample_predictions.json")

# Load SHAP explanation plot
shap_img = "reports/figures/shap_summary.png"

# Title
st.title(" RiskIntel Dashboard")
st.markdown("**Country-Level Risk Monitoring Using Media Sentiment + Macroeconomics**")

# Show raw data
if st.checkbox("Show Dataset"):
    st.dataframe(df.head(10))

# Sample predictions
st.subheader(" Sample Risk Predictions")
st.dataframe(sample_preds)

# SHAP Explanation
st.subheader(" SHAP Feature Importance")
st.image(Image.open(shap_img), caption="SHAP Summary Plot", use_column_width=True)

# Filter by country
st.subheader(" Filter by Country")
country = st.selectbox("Select a country:", sorted(df["country"].dropna().unique()))
st.write(df[df["country"] == country][["title", "sentiment_score", "gdp", "unemployment", "risk_score"]])
