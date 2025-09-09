# SmartAudit-Lite: AI-Based Country-Level Risk Assessment from Media Content

This project is the codebase for my MSc Computer Science thesis at Middlesex University, titled:

> "SmartAudit-Lite: An AI-Based System for Real-Time Country-Level Loan and Business Risk Assessment via Media Content Analysis"

It uses a modular NLP pipeline to extract sentiment and named entities from real-world news data. The project structure follows a clean research code layout, with Jupyter notebooks, Python modules, and versioned data folders.

---

## 📁 Repository Structure

RiskIntel/
├── notebooks/
│ ├── 01_data_ingestion.ipynb
│ ├── 02_preprocessing.ipynb
│ └── 03_nlp_pipeline.ipynb
├── src/
│ ├── data_ingestion/ingest_news.py
│ ├── preprocessing/clean_text.py
│ └── nlp/
│ ├── sentiment_analysis.py
│ └── transformers_model.py (planned)
├── data/
│ ├── raw/news_sample.json
│ ├── processed/cleaned_articles.csv
│ └── processed/analyzed_articles.csv
├── tests/
│ └── test_clean_text.py
├── .env
├── requirements.txt
└── README.md

---

##  Progress Log

###  Day 1 – Environment & Data Ingestion

- Set up conda environment (`.venv`, `requirements.txt`)
- Created modular folder structure: `src/`, `notebooks/`, `data/`
- Wrote `ingest_news.py` to load raw `.json` news data into DataFrame
- Logged ingestion workflow in `01_data_ingestion.ipynb`
- Saved raw data to `data/raw/news_sample.json`

> 📂 Output: `data/raw/news_sample.json`

---

### Day 2 – Text Cleaning & Preprocessing

- Built reusable cleaning function in `clean_text.py`
  - Lowercasing, punctuation removal, stopword removal, lemmatization
- Applied `clean_text()` to all raw news text
- Validated results in `02_preprocessing.ipynb`
- Saved cleaned data to `data/processed/cleaned_articles.csv`

> 📂 Output: `data/processed/cleaned_articles.csv`

---

###  Day 3 – NLP Pipeline: Sentiment & Entity Extraction

- Performed sentiment scoring using `TextBlob`
- Performed Named Entity Recognition (NER) using `spaCy`
  - Used `en_core_web_sm` model
  - Extracted geopolitical entities (e.g. “Kenya”, “UK”)
- Appended `sentiment_score` and `entities` columns to DataFrame
- Saved final NLP output to `analyzed_articles.csv`
- Tracked results in `03_nlp_pipeline.ipynb`

| title                 | sentiment_score | entities        |
|----------------------|-----------------|-----------------|
| Kenya inflation news | 0.0681          | [(kenya, GPE)]  |
| UK economy outlook   | 0.1667          | [(uk, GPE)]     |
| Empty article        | 0.0000          | []              |

> 📂 Output: `data/processed/analyzed_articles.csv`

---

## 📌 Next Steps

###  Day 4: Risk Labeling & Weak Supervision
- Assign initial risk categories to news (e.g., high/med/low)
- Explore keyword-based rule engine or Snorkel-style weak supervision
- Output: `risk_labeled_articles.csv`

###  Day 5: Exploratory Data Visualizations
- Word clouds, frequency plots, named entity bar charts
- Sentiment distribution histograms
- Country-specific risk trends

---

##  Tech Stack

- Python 3.10
- Jupyter Notebooks
- `pandas`, `spacy`, `textblob`, `nltk`
- Modular scripts in `src/`

---

##  Author

Emma Adhiambo Omingo  
MSc Computer Science, Middlesex University  
GitHub: [@EmmaOmingo](https://github.com/EmmaOmingo)  
Project Code Name: `RiskIntel`  
Thesis Title: *SmartAudit-Lite: AI-Based Real-Time Country Risk Scoring*

---

##  Citation (If reusing)



> Omingo, E. (2025). *SmartAudit-Lite: An AI-Based System for Real-Time Country-Level Loan and Business Risk Assessment via Media Content Analysis*. MSc Thesis, Middlesex University.

