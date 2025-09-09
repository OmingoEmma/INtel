import joblib
import pandas as pd

def test_model_load():
    model = joblib.load("models/risk_model.pkl")
    assert model is not None

def test_prediction_shape():
    model = joblib.load("models/risk_model.pkl")
    df = pd.read_csv("data/processed/merged_features.csv")

    # Drop rows with missing values (NaNs) to avoid prediction failure
    df = df.dropna(subset=["sentiment_score", "gdp", "cpi", "unemployment"])

    X = df[["sentiment_score", "gdp", "unemployment", "cpi"]]

    preds = model.predict(X)

    assert preds.shape[0] == X.shape[0]
