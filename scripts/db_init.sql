-- RiskIntel database initialization script
-- Tables: articles, macro_indicators, risk_scores

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Articles table
CREATE TABLE IF NOT EXISTS articles (
    id SERIAL PRIMARY KEY,
    source TEXT,
    author TEXT,
    title TEXT,
    description TEXT,
    url TEXT UNIQUE,
    published_at TIMESTAMPTZ,
    content TEXT,
    country_iso CHAR(3),
    ner_countries TEXT[],
    sentiment_score DOUBLE PRECISION,
    location_confidence DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Helpful indexes for articles
CREATE INDEX IF NOT EXISTS idx_articles_country_published ON articles (country_iso, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_ner_countries_gin ON articles USING GIN (ner_countries);

-- Macro indicators table
CREATE TABLE IF NOT EXISTS macro_indicators (
    id SERIAL PRIMARY KEY,
    country_iso CHAR(3) NOT NULL,
    date DATE NOT NULL,
    gdp REAL,
    cpi REAL,
    unemployment REAL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(country_iso, date)
);
CREATE INDEX IF NOT EXISTS idx_macro_country_date ON macro_indicators (country_iso, date DESC);

-- Risk scores table
CREATE TABLE IF NOT EXISTS risk_scores (
    id SERIAL PRIMARY KEY,
    country_iso CHAR(3) NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    score DOUBLE PRECISION,
    score_p90 DOUBLE PRECISION,
    score_p10 DOUBLE PRECISION,
    method VARCHAR(64) NOT NULL,
    top_factors JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_risk_country_asof ON risk_scores (country_iso, as_of DESC);
CREATE INDEX IF NOT EXISTS idx_risk_method ON risk_scores (method);
