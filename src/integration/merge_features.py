import pandas as pd
import os
import json

# --- STEP 1: Extract GPE entities like countries from tuples
def extract_country_from_entities(entities):
    for ent in entities:
        if isinstance(ent, tuple) and ent[1] == "GPE":
            return ent[0].title()
    return None

# --- STEP 2: Preprocess NLP outputs and add country + dummy date
def preprocess_merged_data():
    filepath = os.path.join(os.path.dirname(__file__), "../../data/processed/analyzed_articles.csv")
    df = pd.read_csv(filepath, converters={"entities": eval})
    df["country"] = df["entities"].apply(extract_country_from_entities)
    df["date"] = "2024-01-01"
    return df

# --- STEP 3: Merge with economic indicators
def merge_with_economic_indicators(news_df):
    econ_path = os.path.join(os.path.dirname(__file__), "../../data/processed/economic_indicators.csv")
    econ_df = pd.read_csv(econ_path)
    merged_df = pd.merge(news_df, econ_df, how="left", on=["country", "date"])
    return merged_df

# --- STEP 4: Save outputs
def save_outputs(df):
    os.makedirs(os.path.join(os.path.dirname(__file__), "../../data/processed"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "../../examples"), exist_ok=True)
    df.to_csv(os.path.join(os.path.dirname(__file__), "../../data/processed/merged_features.csv"), index=False)

    sample = df[["title", "country", "sentiment_score", "gdp", "cpi", "unemployment"]].head(3).to_dict(orient="records")
    with open(os.path.join(os.path.dirname(__file__), "../../examples/sample_output.json"), "w") as f:
        json.dump(sample, f, indent=2)

# --- STEP 5: Execute full process when running directly
if __name__ == "__main__":
    news_df = preprocess_merged_data()
    merged_df = merge_with_economic_indicators(news_df)
    save_outputs(merged_df)
