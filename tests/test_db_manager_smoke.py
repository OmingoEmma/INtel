# tests/test_db_manager_smoke.py
import datetime
from src.utils.config import load_config
from src.data_ingestion import database_manager as db


def main():
    # Load config and connect
    cfg = load_config()
    conn = db.get_conn(cfg)

    print("✅ Connected to DB")

    # --- Test articles upsert ---
    articles = [
        {
            "source": "UnitTest",
            "author": "Tester",
            "title": "Smoke Test Article",
            "description": "DB manager smoke test",
            "url": "http://example.com/test-article",
            "published_at": datetime.datetime.utcnow(),
            "content": "Some test content",
            "country_iso": "KEN",
            "ner_countries": ["KEN"],
            "sentiment_score": 0.25,
            "location_confidence": 0.9,
        }
    ]
    n_articles = db.upsert_articles(conn, articles)
    print(f"✅ Inserted/updated {n_articles} article(s)")

    # --- Test macro upsert ---
    macros = [
        {
            "country_iso": "KEN",
            "date": datetime.date.today(),
            "gdp": 95.2,
            "cpi": 3.5,
            "unemployment": 7.4,
        }
    ]
    n_macros = db.upsert_macro(conn, macros)
    print(f"✅ Inserted/updated {n_macros} macro row(s)")

    # --- Test risk score insert ---
    risk_row = {
        "country_iso": "KEN",
        "as_of": datetime.datetime.utcnow(),
        "score": 55.5,
        "score_p90": 70.0,
        "score_p10": 40.0,
        "method": "smoke_test",
        "top_factors": {"sentiment": -0.2, "gdp": 95.2},
    }
    db.insert_risk_score(conn, risk_row)
    print("✅ Inserted risk score row")

    # --- Test last published_at ---
    last_pub = db.get_last_published_at(conn)
    print(f"✅ Last published_at: {last_pub}")

    # --- Test recent countries ---
    recent = db.get_recent_countries(conn, hours=24)
    print(f"✅ Recent countries: {recent}")

    conn.close()
    print("🎉 Smoke test completed successfully.")


if __name__ == "__main__":
    main()
