import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values, Json

from src.utils.config import load_config

logger = logging.getLogger(__name__)


def _parse_dsn_from_cfg(cfg: Dict[str, Any]) -> str:
    """Compose a DSN for psycopg2 from configuration.
    Supports `DATABASE_URL` style or discrete fields.
    """
    pg = cfg.get("postgres", {})
    if pg.get("database_url"):
        return pg["database_url"]
    host = pg.get("host", "127.0.0.1")
    port = pg.get("port", 5432)
    dbname = pg.get("dbname", "riskintel")
    user = pg.get("user", "postgres")
    password = pg.get("password", "")
    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


def get_conn(cfg: Dict[str, Any]) -> psycopg2.extensions.connection:
    """Create a new PostgreSQL connection from configuration."""
    dsn = _parse_dsn_from_cfg(cfg)
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    return conn


@contextmanager
def db_cursor(conn: psycopg2.extensions.connection):
    """Context manager yielding a cursor with commit/rollback semantics."""
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def upsert_articles(conn: psycopg2.extensions.connection, rows: Sequence[Dict[str, Any]]) -> int:
    """UPSERT a batch of article rows by URL."""
    if not rows:
        return 0

    columns = [
        "source",
        "author",
        "title",
        "description",
        "url",
        "published_at",
        "content",
        "country_iso",
        "ner_countries",
        "sentiment_score",
        "location_confidence",
    ]

    values: List[Tuple[Any, ...]] = []
    for row in rows:
        published_at = row.get("published_at")
        if isinstance(published_at, str):
            # PostgreSQL can parse ISO strings directly
            pass
        values.append(
            (
                row.get("source"),
                row.get("author"),
                row.get("title"),
                row.get("description"),
                row.get("url"),
                published_at,
                row.get("content"),
                row.get("country_iso"),
                row.get("ner_countries"),
                row.get("sentiment_score"),
                row.get("location_confidence"),
            )
        )

    insert_sql = sql.SQL(
        """
        INSERT INTO articles ({fields})
        VALUES %s
        ON CONFLICT (url) DO UPDATE SET
            source = EXCLUDED.source,
            author = EXCLUDED.author,
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            published_at = EXCLUDED.published_at,
            content = EXCLUDED.content,
            country_iso = COALESCE(EXCLUDED.country_iso, articles.country_iso),
            ner_countries = COALESCE(EXCLUDED.ner_countries, articles.ner_countries),
            sentiment_score = COALESCE(EXCLUDED.sentiment_score, articles.sentiment_score),
            location_confidence = COALESCE(EXCLUDED.location_confidence, articles.location_confidence)
        """
    ).format(fields=sql.SQL(", ").join(map(sql.Identifier, columns)))

    with db_cursor(conn) as cur:
        execute_values(cur, insert_sql.as_string(cur), values)
    return len(values)


def upsert_macro(conn: psycopg2.extensions.connection, rows: Sequence[Dict[str, Any]]) -> int:
    """UPSERT macro indicator rows by (country_iso, date)."""
    if not rows:
        return 0

    columns = ["country_iso", "date", "gdp", "cpi", "unemployment"]
    values: List[Tuple[Any, ...]] = [
        (
            row.get("country_iso"),
            row.get("date"),
            row.get("gdp"),
            row.get("cpi"),
            row.get("unemployment"),
        )
        for row in rows
    ]

    insert_sql = sql.SQL(
        """
        INSERT INTO macro_indicators ({fields})
        VALUES %s
        ON CONFLICT (country_iso, date) DO UPDATE SET
            gdp = COALESCE(EXCLUDED.gdp, macro_indicators.gdp),
            cpi = COALESCE(EXCLUDED.cpi, macro_indicators.cpi),
            unemployment = COALESCE(EXCLUDED.unemployment, macro_indicators.unemployment)
        """
    ).format(fields=sql.SQL(", ").join(map(sql.Identifier, columns)))

    with db_cursor(conn) as cur:
        execute_values(cur, insert_sql.as_string(cur), values)
    return len(values)


def insert_risk_score(conn: psycopg2.extensions.connection, row: Dict[str, Any]) -> None:
    """Insert a single risk score record."""
    sql_text = (
        "INSERT INTO risk_scores (country_iso, as_of, score, score_p90, score_p10, method, top_factors) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
    )
    with db_cursor(conn) as cur:
        tf = row.get("top_factors")
        tf_param = Json(tf) if isinstance(tf, dict) else tf
        cur.execute(
            sql_text,
            (
                row.get("country_iso"),
                row.get("as_of"),
                row.get("score"),
                row.get("score_p90"),
                row.get("score_p10"),
                row.get("method", "baseline_v1"),
                tf_param,
            ),
        )


def get_last_published_at(conn: psycopg2.extensions.connection) -> Optional[datetime]:
    """Return the most recent published_at timestamp from articles, if any."""
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(published_at) FROM articles")
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None


def get_recent_countries(conn: psycopg2.extensions.connection, hours: int = 24) -> List[str]:
    """Return countries that had articles in the last N hours."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT country_iso
            FROM articles
            WHERE country_iso IS NOT NULL
              AND published_at >= now() - INTERVAL '%s hours'
            """,
            (hours,),
        )
        return [r[0] for r in cur.fetchall() if r[0]]
