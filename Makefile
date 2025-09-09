PY := python
EXPORT_ENV := export MPLBACKEND=Agg; export PYTHONPATH=.

ingest-live-once:
	$(EXPORT_ENV); $(PY) -m src.ingestion.simulate_live_news --once

ingest-live-watch:
	$(EXPORT_ENV); $(PY) -m src.ingestion.simulate_live_news --watch --interval-mins 30

rescore:
	$(EXPORT_ENV); $(PY) -m src.modeling.train_model

