# Migration utilities

Files in this directory are explicit, one-off migration tools. They are not
imported or invoked by current inference or evaluation runs.

- `backfill_pipeline_provenance.py` repairs historical run and evaluation
  metadata. Run it from the repository root as
  `python -m tools.migrations.backfill_pipeline_provenance`. It is dry-run by
  default; `--apply` is required to write changes.
- `migrate_legacy_ipa.py` recomputes generic `ipa/` results with the exact
  installed Africa G2P identity and admits only row- and metric-identical runs
  to `ipa/<g2p_system_id>/`. It is dry-run by default. Applying also requires
  `--reference-manifest`; legacy files are preserved.

  ```bash
  python -m tools.migrations.migrate_legacy_ipa \
    --dataset-root input_output_data/input/<dataset>

  python -m tools.migrations.migrate_legacy_ipa \
    --dataset-root input_output_data/input/<dataset> \
    --reference-manifest input_output_data/output/experiments/<dataset>/manifests/ref_manifest.raw_segments.jsonl \
    --apply
  ```

Keep a migration utility only while historical data may still require it. New
runs must write the current schema directly rather than relying on backfills.
