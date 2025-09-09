import logging
import os
import re
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover - torch may not be installed at import time
    torch = None  # type: ignore

try:
    from transformers import pipeline
except Exception as _e:  # pragma: no cover - transformers may not be installed at import time
    pipeline = None  # type: ignore

try:
    import boto3
except Exception:  # pragma: no cover - boto3 may not be installed
    boto3 = None  # type: ignore


# Initialize logger
LOGGER = logging.getLogger("nlp.sentiment")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)


# Lazy-initialized global for FinBERT pipeline
_FINBERT_PIPELINE: Optional[Any] = None


def _detect_device() -> Any:
    """Return device for transformers pipeline: GPU index if available, else CPU.

    Returns 0 for first CUDA device when available, otherwise returns 'cpu'.
    """
    try:
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            LOGGER.info("Detected GPU: using CUDA device 0 out of %s", num_devices)
            return 0
    except Exception as exc:  # pragma: no cover - environment specific
        LOGGER.warning("GPU detection failed: %s", exc)
    LOGGER.info("Using CPU for FinBERT inference")
    return "cpu"


def _init_finbert_pipeline() -> Optional[Any]:
    """Initialize and cache the FinBERT transformers pipeline."""
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE

    if pipeline is None:
        LOGGER.error("transformers is not available; FinBERT cannot be initialized")
        return None

    try:
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        device = _detect_device()
        LOGGER.info("Initializing FinBERT pipeline (ProsusAI/finbert)…")
        _FINBERT_PIPELINE = pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=device,
            return_all_scores=True,
        )
        LOGGER.info("FinBERT pipeline initialized successfully")
        return _FINBERT_PIPELINE
    except Exception as exc:
        LOGGER.exception("Failed to initialize FinBERT pipeline: %s", exc)
        _FINBERT_PIPELINE = None
        return None


def _preprocess_financial_text(text: str) -> str:
    """Lightweight normalization for financial text prior to model inference."""
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if not s:
        return ""
    # Remove URLs
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    # Normalize cashtags like $AAPL -> AAPL
    s = re.sub(r"\$([A-Za-z]{1,10})", r"\1", s)
    # Remove zero-width chars
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _canonicalize_finbert_label(label: str) -> str:
    l = label.lower()
    if "pos" in l:
        return "positive"
    if "neg" in l:
        return "negative"
    if "neu" in l:
        return "neutral"
    return l


def _scores_to_output(label_to_score: Dict[str, float]) -> Dict[str, Any]:
    pos = float(label_to_score.get("positive", 0.0))
    neg = float(label_to_score.get("negative", 0.0))
    neu = float(label_to_score.get("neutral", 0.0))
    # Compute polarity in [-1, 1]
    polarity = max(min(pos - neg, 1.0), -1.0)
    # Choose sentiment with highest probability
    if pos >= neg and pos >= neu:
        sentiment = "positive"
        confidence = pos
    elif neg >= pos and neg >= neu:
        sentiment = "negative"
        confidence = neg
    else:
        sentiment = "neutral"
        confidence = neu
    return {
        "sentiment": sentiment,
        "polarity": float(polarity),
        "confidence": float(confidence),
    }


def _finbert_analyze_batch(texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
    clf = _init_finbert_pipeline()
    if clf is None:
        raise RuntimeError("FinBERT pipeline is unavailable")

    outputs: List[Dict[str, Any]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # For empty strings, short circuit to neutral
        if all(t == "" for t in batch):
            for _ in batch:
                outputs.append({
                    "sentiment": "neutral",
                    "polarity": 0.0,
                    "confidence": 0.0,
                    "source": "finbert",
                })
            continue
        try:
            preds = clf(batch, truncation=True, max_length=512)
            # preds: List[List[{label, score}]] with return_all_scores=True
            for scores in preds:
                label_map: Dict[str, float] = {}
                for item in scores:
                    label_map[_canonicalize_finbert_label(item.get("label", ""))] = float(item.get("score", 0.0))
                base = _scores_to_output(label_map)
                base["source"] = "finbert"
                outputs.append(base)
        except Exception as exc:
            LOGGER.exception("FinBERT inference error: %s", exc)
            raise
    return outputs


def _aws_comprehend_analyze_batch(texts: List[str], region: Optional[str] = None) -> List[Dict[str, Any]]:
    if boto3 is None:
        raise RuntimeError("boto3 is not available for AWS Comprehend fallback")

    region_name = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    client = boto3.client("comprehend", region_name=region_name)

    outputs: List[Dict[str, Any]] = [
        {"sentiment": "neutral", "polarity": 0.0, "confidence": 0.0, "source": "aws_comprehend"}
        for _ in texts
    ]
    max_batch = 25
    for i in range(0, len(texts), max_batch):
        batch = texts[i : i + max_batch]
        # AWS Comprehend text size limit ~5000 bytes; truncate conservatively
        safe_batch = [(t or "")[:4500] for t in batch]
        try:
            resp = client.batch_detect_sentiment(TextList=safe_batch, LanguageCode="en")
        except Exception as exc:
            LOGGER.exception("AWS Comprehend batch_detect_sentiment failed: %s", exc)
            # On total failure leave defaults
            continue

        # Fill successful results
        for item in resp.get("ResultList", []):
            idx = int(item.get("Index", 0))
            absolute_idx = i + idx
            scores = item.get("SentimentScore", {})
            pos = float(scores.get("Positive", 0.0))
            neg = float(scores.get("Negative", 0.0))
            neu = float(scores.get("Neutral", 0.0))
            label_map = {"positive": pos, "negative": neg, "neutral": neu}
            base = _scores_to_output(label_map)
            base["source"] = "aws_comprehend"
            outputs[absolute_idx] = base

        # Log errors for failed items
        for err in resp.get("ErrorList", []):
            idx = err.get("Index")
            code = err.get("ErrorCode")
            msg = err.get("ErrorMessage")
            LOGGER.error("AWS Comprehend error for index %s: %s - %s", idx, code, msg)

    return outputs


def analyze_sentiments(texts: List[str]) -> List[Dict[str, Any]]:
    """Batch analyze sentiments using FinBERT with AWS Comprehend fallback.

    Returns list[dict] with keys: sentiment, polarity, confidence, source.
    """
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")

    preprocessed: List[str] = [_preprocess_financial_text(t) for t in texts]

    # Attempt FinBERT inference first
    try:
        return _finbert_analyze_batch(preprocessed)
    except Exception:
        LOGGER.warning("Falling back to AWS Comprehend for sentiment analysis")
        try:
            return _aws_comprehend_analyze_batch(preprocessed)
        except Exception as exc:
            LOGGER.exception("All sentiment analysis methods failed: %s", exc)
            # Graceful degradation: return neutral defaults
            return [
                {"sentiment": "neutral", "polarity": 0.0, "confidence": 0.0, "source": "unavailable"}
                for _ in texts
            ]


def score_sentiment(text: str) -> float:
    """Backward-compatible API: return polarity in [-1, 1] for a single text."""
    if not text or not isinstance(text, str):
        return 0.0
    try:
        result = analyze_sentiments([text])
        if not result:
            return 0.0
        return float(result[0].get("polarity", 0.0))
    except Exception as exc:  # pragma: no cover - unexpected runtime failure
        LOGGER.exception("score_sentiment failed: %s", exc)
        return 0.0

