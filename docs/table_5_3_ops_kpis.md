### Table 5.3 — Operational KPIs

| Metric | Value |
| --- | --- |
| Freshness breaches (>24h) | 351 |
| Total retry attempts | 580 |
| Retry successes | 292 |
| Retry success rate | 0.503 |
| Alert count: ALERT_SENTIMENT_SHOCK | 83 |
| Alert count: DATA_STALE | 351 |

**Notes**: Freshness breach is defined as last_ingest_hours > 24. Retry rate computed as successes/attempts when both fields are present. Alert counts are illustrative aggregations for the excerpt.

**Interpretation**: The pipeline largely succeeds but occasional freshness breaches and retries indicate integration risks. Sentiment shock alerts provide early warning signals when news volume is low or volatility spikes.
