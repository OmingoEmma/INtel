import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv


NEWSAPI_BASE_URL = "https://newsapi.org/v2/top-headlines"
COUNTRIES = ["gb", "ke"]
CATEGORY = "business"

FINANCE_KEYWORDS = [
    "bank",
    "inflation",
    "interest",
    "loan",
    "bond",
    "debt",
    "default",
    "credit",
    "equity",
    "market",
    "gdp",
    "unemployment",
    "cpi",
    "forex",
    "currency",
    "fiscal",
    "budget",
    "rating",
    "imf",
    "world bank",
    "central bank",
    "treasury",
]


def ensure_directories() -> Tuple[Path, Path]:
    data_raw_dir = Path("data/raw")
    logs_dir = Path("logs")
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return data_raw_dir, logs_dir


def setup_logger(logs_dir: Path) -> logging.Logger:
    logger = logging.getLogger("ingestion")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_path = logs_dir / "ingestion.log"
    handler = RotatingFileHandler(str(log_path), maxBytes=1_000_000, backupCount=5)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Also log to console for immediate feedback
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def load_api_key() -> str:
    load_dotenv(override=False)
    api_key = os.getenv("NEWSAPI_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "NEWSAPI_KEY is not set. Set it in the environment or .env file."
        )
    return api_key


def fetch_headlines(country: str, api_key: str, logger: logging.Logger) -> Tuple[List[Dict], bool]:
    params = {
        "country": country,
        "category": CATEGORY,
        "apiKey": api_key,
        # Keep pageSize moderate to avoid rate limits; default is 20
        "pageSize": 50,
    }
    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=20)
    except requests.RequestException as exc:
        logger.error(f"Request error for country={country}: {exc}")
        return [], False

    if response.status_code == 429:
        logger.warning(
            f"HTTP 429 rate limit for country={country}. Will apply 15-minute backoff."
        )
        return [], True

    if not response.ok:
        logger.error(
            f"Failed to fetch headlines for country={country}. Status={response.status_code} Body={response.text[:200]}"
        )
        return [], False

    payload = response.json()
    articles = payload.get("articles", []) or []
    logger.info(
        f"Fetched {len(articles)} headlines for country={country}, status={payload.get('status')}"
    )
    return articles, False


def filter_financial_articles(articles: List[Dict]) -> List[Dict]:
    if not articles:
        return []
    keywords = [kw.lower() for kw in FINANCE_KEYWORDS]
    filtered: List[Dict] = []
    for art in articles:
        text_fields = " ".join(
            str(art.get(field, "") or "") for field in ("title", "description", "content")
        ).lower()
        if any(kw in text_fields for kw in keywords):
            filtered.append(art)
    return filtered


def dedupe_by_url(articles: List[Dict]) -> List[Dict]:
    seen = set()
    unique: List[Dict] = []
    for art in articles:
        url = art.get("url")
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        unique.append(art)
    return unique


def save_articles(articles: List[Dict], data_raw_dir: Path, logger: logging.Logger) -> Path:
    now_utc = datetime.now(timezone.utc)
    filename = now_utc.strftime("news_%d%m%y_%H%M.json")
    save_path = data_raw_dir / filename
    manifest = {
        "saved_at_utc": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(articles),
        "articles": articles,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(articles)} articles to {save_path}")
    return save_path


def trigger_rescore(skip_rescore: bool, logger: logging.Logger) -> None:
    if skip_rescore:
        logger.info("Skipping rescore as per --no-rescore flag.")
        return
    try:
        # Deferred import and subprocess to avoid heavy deps on import
        import subprocess

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env.setdefault("PYTHONPATH", ".")
        logger.info("Triggering training/SHAP pipeline: python -m src.modeling.train_model")
        result = subprocess.run(
            ["python", "-m", "src.modeling.train_model"],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Rescore pipeline completed successfully.")
        else:
            logger.error(
                f"Rescore pipeline failed with code {result.returncode}. Stdout: {result.stdout[:400]} Stderr: {result.stderr[:400]}"
            )
    except Exception as exc:
        logger.error(f"Failed to trigger rescore pipeline: {exc}")


def poll_once(api_key: str, data_raw_dir: Path, logger: logging.Logger) -> Tuple[Path, bool]:
    all_articles: List[Dict] = []
    saw_rate_limit = False
    for country in COUNTRIES:
        articles, rate_limited = fetch_headlines(country, api_key, logger)
        saw_rate_limit = saw_rate_limit or rate_limited
        all_articles.extend(articles)

    filtered = filter_financial_articles(all_articles)
    deduped = dedupe_by_url(filtered)
    logger.info(
        f"Collected={len(all_articles)} Filtered(finance)={len(filtered)} Deduped={len(deduped)}"
    )
    save_path = save_articles(deduped, data_raw_dir, logger)
    return save_path, saw_rate_limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate live NewsAPI ingestion for UK & Kenya with autosave and rescore",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--once",
        action="store_true",
        help="Fetch once and exit",
    )
    group.add_argument(
        "--watch",
        action="store_true",
        help="Continuously fetch on an interval (minutes)",
    )
    parser.add_argument(
        "--interval-mins",
        type=float,
        default=30.0,
        help="Polling interval in minutes for --watch (default: 30)",
    )
    parser.add_argument(
        "--no-rescore",
        action="store_true",
        help="Do not trigger the training/SHAP pipeline after save",
    )
    return parser.parse_args()


def main() -> None:
    data_raw_dir, logs_dir = ensure_directories()
    logger = setup_logger(logs_dir)
    logger.info("Starting live NewsAPI ingestion simulation")

    try:
        api_key = load_api_key()
    except Exception as exc:
        # Log and raise to show clear failure in CLI
        logger.error(str(exc))
        raise

    args = parse_args()

    if args.once:
        save_path, rate_limited = poll_once(api_key, data_raw_dir, logger)
        logger.info(f"Batch complete. Output: {save_path}")
        trigger_rescore(args.no_rescore, logger)
        return

    if args.watch:
        logger.info(
            f"Entering watch mode with interval {args.interval_mins} minutes (default 30)."
        )
        interval_secs_default = max(60.0, args.interval_mins * 60.0)
        while True:
            start_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info(f"Polling cycle started at {start_ts}")
            try:
                save_path, rate_limited = poll_once(api_key, data_raw_dir, logger)
                logger.info(f"Batch complete. Output: {save_path}")
                trigger_rescore(args.no_rescore, logger)
            except Exception as exc:
                logger.error(f"Unexpected error in polling cycle: {exc}")
                rate_limited = False  # Avoid forcing backoff on generic errors

            sleep_secs = 15 * 60 if rate_limited else interval_secs_default
            if rate_limited:
                logger.warning(
                    "Rate limited detected in last cycle. Backing off for 15 minutes before next attempt."
                )
            else:
                logger.info(
                    f"Sleeping for {int(sleep_secs // 60)} minutes until next cycle."
                )
            time.sleep(sleep_secs)


if __name__ == "__main__":
    main()

