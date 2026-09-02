# Migration utilities

Files in this directory are explicit, one-off migration tools. They are not
imported or invoked by current inference or evaluation runs.

- `backfill_pipeline_provenance.py` repairs historical run and evaluation
  metadata. Run it from the repository root as
  `python -m tools.migrations.backfill_pipeline_provenance`. It is dry-run by
  default; `--apply` is required to write changes.

Keep a migration utility only while historical data may still require it. New
runs must write the current schema directly rather than relying on backfills.
