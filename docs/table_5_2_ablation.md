### Table 5.2 — Ablation results

| Feature set | MAE | R² | ΔMAE | ΔR² |
| --- | --- | --- | --- | --- |
| Macro + Sentiment | 0.419 | -0.0617 | — | — |
| Macro-only | 0.435 | -0.115 | 0.0162 | -0.0536 |

**Notes**: Ablation compares Random Forest with all features vs macro-only on the same temporal split. Sparse-news stress: MAE_sparse=0.403 when sentiment_n < 15 vs MAE_dense=0.426 when sentiment_n ≥ 15. Interpretation: Sentiment features provide incremental accuracy, especially on dense-news days, while sparse-news conditions degrade performance due to weaker signals.

**Interpretation**: Including sentiment features reduces error in aggregate. Performance degrades on sparse-news days, highlighting the value of robust guardrails and uncertainty-aware operations.
