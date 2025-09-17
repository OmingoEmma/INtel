"""
Utility functions for RiskIntel dissertation artefacts.

This module provides helpers for:
- Column name inference with fuzzy matching
- Temporal train/test split
- Bootstrap confidence intervals for evaluation metrics
- Country-level naïve persistence baseline
- Small formatting and IO utilities

CLI usage (for quick verification):
  python -m src.utils --help
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Column inference
# -----------------------------

EXPECTED_COLUMNS = {
    "date": ["date", "as_of_date", "dt"],
    "country": ["country", "country_code", "iso", "iso_code"],
    "risk_score": ["risk_score", "target", "y"],
    # Sentiment features (prefix-based in code too)
    "sentiment_mean": ["sentiment_mean", "sent_mean", "news_sentiment_mean"],
    "sentiment_z": ["sentiment_z", "sentiment_zscore", "sent_z", "sentiment_std"],
    "sentiment_n": ["sentiment_n", "num_articles", "articles_n", "sent_count"],
    # Macro examples with common variants
    "gdp_yoy": ["gdp_yoy", "gdp_growth", "gdp_yoy_pct"],
    "cpi_yoy": ["cpi_yoy", "inflation", "cpi"],
    "unemp_rate": ["unemp_rate", "unemployment", "u_rate"],
    "fx_vol": ["fx_vol", "fx_volatility", "fxv"],
}


def _best_match(name: str, candidates: Iterable[str]) -> Optional[str]:
    candidates = list(candidates)
    if not candidates:
        return None
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None


def infer_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Infer column mapping for expected names with fuzzy matching and prefixes.

    Expected keys: date, country, risk_score, and any columns starting with
    "sentiment_" are treated as sentiment features.

    Returns a dict mapping expected names to the actual column names present.
    Also returns discovered sentiment and macro feature lists under special keys:
      - __sentiment_cols
      - __macro_cols

    The function prefers exact/alias matches, then fuzzy matches.
    """
    lower_cols = {c.lower(): c for c in df.columns}

    mapping: Dict[str, str] = {}

    # Exact/alias matching first
    for expected, aliases in EXPECTED_COLUMNS.items():
        for alias in aliases:
            if alias.lower() in lower_cols:
                mapping[expected] = lower_cols[alias.lower()]
                break

    # Fuzzy fallback for critical keys
    for expected in ["date", "country", "risk_score"]:
        if expected not in mapping:
            best = _best_match(expected, lower_cols.keys())
            if best is not None:
                mapping[expected] = lower_cols[best]

    # Sentiment features by prefix and fuzzy
    sentiment_cols = [c for c in df.columns if c.lower().startswith("sentiment_")]
    if not sentiment_cols:
        # Try fuzzy by common base names
        for candidate in ["sentiment_mean", "sentiment_z", "sentiment_n"]:
            best = _best_match(candidate, lower_cols.keys())
            if best is not None:
                sentiment_cols.append(lower_cols[best])
    # Unique and preserve order
    seen = set()
    sentiment_cols = [c for c in sentiment_cols if not (c in seen or seen.add(c))]

    # Macro columns: numeric, excluding keys/target/sentiment
    protected = set([mapping.get("date", "date"), mapping.get("country", "country"), mapping.get("risk_score", "risk_score")])
    protected |= set(sentiment_cols)
    macro_cols = [c for c in df.columns if c not in protected and pd.api.types.is_numeric_dtype(df[c])]

    mapping["__sentiment_cols"] = ",".join(sentiment_cols)
    mapping["__macro_cols"] = ",".join(macro_cols)

    return mapping


# -----------------------------
# Temporal split
# -----------------------------

def temporal_split(df: pd.DataFrame, date_col: str, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal 80/20 split by unique date values. Ensures train dates < test dates.
    """
    df_sorted = df.sort_values(by=[date_col]).reset_index(drop=True)
    unique_dates = df_sorted[date_col].dropna().sort_values().unique()
    if len(unique_dates) < 5:
        # Fallback: simple row split if not enough unique dates
        split_idx = int((1 - test_ratio) * len(df_sorted))
        return df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]
    cutoff_idx = int((1 - test_ratio) * len(unique_dates))
    cutoff_date = unique_dates[cutoff_idx]
    train = df_sorted[df_sorted[date_col] < cutoff_date]
    test = df_sorted[df_sorted[date_col] >= cutoff_date]
    return train, test


# -----------------------------
# Bootstrap CI
# -----------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    B: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute percentile bootstrap CI for a metric on paired (y_true, y_pred).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return np.nan, np.nan
    stats = np.empty(B, dtype=float)
    indices = np.arange(n)
    for b in range(B):
        sample_idx = rng.choice(indices, size=n, replace=True)
        stats[b] = metric_fn(y_true[sample_idx], y_pred[sample_idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# -----------------------------
# Naïve persistence baseline
# -----------------------------

def naive_persistence(y: pd.Series, countries: pd.Series, dates: pd.Series, train_mean: float) -> pd.Series:
    """
    Country-level naïve persistence: predict y_t as y_{t-1} within country;
    if missing (no t-1 in country), fill with provided train_mean.
    Expects aligned Series with matching index.
    """
    df_tmp = pd.DataFrame({"y": y, "country": countries, "date": dates})
    df_tmp = df_tmp.sort_values(["country", "date"]).copy()
    df_tmp["y_pred"] = df_tmp.groupby("country")["y"].shift(1)
    # Fill missing with global train mean
    df_tmp["y_pred"].fillna(train_mean, inplace=True)
    # Restore original order
    df_tmp = df_tmp.loc[y.index]
    return df_tmp["y_pred"].astype(float)


# -----------------------------
# Small helpers
# -----------------------------

def format_ci(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "NA–NA"
    return f"{lo:.3f}–{hi:.3f}"


def ensure_dirs(paths: Iterable[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_markdown_table(
    path: str,
    header: List[str],
    rows: List[List[str]],
    notes: Optional[str] = None,
    title: Optional[str] = None,
    interpretation: Optional[str] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: List[str] = []
    if title:
        lines.append(f"### {title}")
        lines.append("")
    # Header
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    # Rows
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    if notes:
        lines.append("")
        lines.append("**Notes**: " + notes.strip())
    if interpretation:
        lines.append("")
        lines.append("**Interpretation**: " + interpretation.strip())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_checklist(items: List[Tuple[str, bool]]) -> None:
    print("\nChecklist:")
    for label, ok in items:
        status = "[x]" if ok else "[ ]"
        print(f"  {status} {label}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="RiskIntel utils quick checks")
    parser.add_argument("--inspect", type=str, default=None, help="Path to CSV to infer columns")
    args = parser.parse_args()
    if args.inspect:
        df = pd.read_csv(args.inspect)
        mapping = infer_columns(df)
        print(json.dumps(mapping, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()

