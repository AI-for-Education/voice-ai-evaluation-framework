#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from egra_eval.data.linking import add_audio_keys
from egra_eval.data.textgrid_io import add_refs_from_textgrid
from egra_eval.data.passage_merge import attach_passage_texts
from egra_eval.data.nemo_manifest import load_many_manifests
from egra_eval.normalize.textnorm import normalize
from egra_eval.data.dataset_layout import resolve_dataset_paths, DatasetLayoutError


def _abs_audio_path(audio_root: Path, learner_id: str, audio_file: str) -> str:
    return str((audio_root / learner_id / Path(audio_file).name).as_posix())


def _write_jsonl(rows, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


def _maybe_norm(s: str, do_norm: bool) -> str:
    if not s:
        return s
    return normalize(s) if do_norm else s


def main():
    ap = argparse.ArgumentParser(description="Create NeMo JSONL manifests for offline scoring.")
    ap.add_argument("--dataset_root", default=None, help="Root folder containing 0_Audio/ and 2_TextGrid/.")
    ap.add_argument("--dataset_annotator", default=None, help="Specific annotator folder under 2_TextGrid/ to use.")
    ap.add_argument("--output_dir", default=None, help="Directory where manifests will be written (defaults next to dataset).")

    ap.add_argument("--egra_csv", default=None)
    ap.add_argument("--passages_csv", default=None)
    ap.add_argument("--textgrids_dir", default=None, help="Root with <learner_id> subfolders.")
    ap.add_argument("--audio_root", default=None, help="Root containing learner audio folders.")
    ap.add_argument("--nemo_hyp_manifest", default=None,
                    help="ASR output JSONL with audio_filepath + pred_text")
    ap.add_argument("--out_ref", default=None, help="Output JSONL: text=REF, pred_text from ASR if available.")
    ap.add_argument("--out_can", default=None, help="Output JSONL: text=CAN, pred_text from ASR if available.")
    ap.add_argument("--normalize_for_nemo", action="store_true",
                    help="If set, normalize both text and pred_text like in the main pipeline (recommended).")
    args = ap.parse_args()

    if args.dataset_root:
        try:
            layout = resolve_dataset_paths(args.dataset_root, annotator=args.dataset_annotator)
        except DatasetLayoutError as exc:  # pragma: no cover - CLI validation
            raise SystemExit(str(exc)) from exc

        args.egra_csv = args.egra_csv or str(layout.canonical_csv)
        args.textgrids_dir = args.textgrids_dir or str(layout.textgrid_root)
        args.audio_root = args.audio_root or str(layout.audio_root)
        args.passages_csv = args.passages_csv or ""

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = layout.root / "nemo_asr_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        args.out_ref = args.out_ref or str(output_dir / "ref_manifest_norm.jsonl")
        args.out_can = args.out_can or str(output_dir / "can_manifest_norm.jsonl")
        args.nemo_hyp_manifest = args.nemo_hyp_manifest or str(output_dir / "transcriptions.jsonl")
    else:
        if not all((args.egra_csv, args.textgrids_dir, args.audio_root, args.nemo_hyp_manifest, args.out_ref, args.out_can)):
            raise SystemExit("Please provide either --dataset_root or explicit paths for CSVs, audio/textgrid roots, manifests, and outputs.")
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if args.out_ref is None:
                args.out_ref = str(output_dir / "ref_manifest_norm.jsonl")
            if args.out_can is None:
                args.out_can = str(output_dir / "can_manifest_norm.jsonl")

    if not Path(args.nemo_hyp_manifest).exists():
        raise SystemExit(f"NeMo hypothesis manifest not found: {args.nemo_hyp_manifest}")

    # 1) EGRA rows + join keys
    egra = pd.read_csv(args.egra_csv)
    egra = add_audio_keys(egra, audio_col="audio_file")

    # 2) REF from TextGrid (tier 'child')
    egra = add_refs_from_textgrid(
        egra,
        base_dir=args.textgrids_dir,
        textgrid_col="textgrid",
        tier_name="child",
        logger=None,
    )

    # 3) CAN from passages table
    if args.passages_csv and Path(args.passages_csv).exists():
        egra = attach_passage_texts(egra, args.passages_csv, logger=None)

    # 4) Audio absolute path
    audio_root = Path(args.audio_root)
    egra["audio_abs"] = [
        _abs_audio_path(audio_root, r["learner_id"], r["audio_file"]) for _, r in egra.iterrows()
    ]

    # 5) Load HYP (pred_text) from NeMo manifest
    hyp_df = load_many_manifests([args.nemo_hyp_manifest], audio_key="audio_filepath", hyp_key="pred_text")
    hyp_map = {str(Path(r["audio_path"]).as_posix()): (r.get("hyp_text") or "") for _, r in hyp_df.iterrows()}

    # 6) Build REF manifest
    rows_ref = []
    for _, r in egra.iterrows():
        audio_path = r["audio_abs"]
        ref = (r.get("ref_text") or "").strip()
        pred = hyp_map.get(audio_path, "")

        if not ref or not pred:
            continue

        row = {
            "audio_filepath": audio_path,
            "text": _maybe_norm(ref, args.normalize_for_nemo),
            "pred_text": _maybe_norm(pred, args.normalize_for_nemo),
        }
        rows_ref.append(row)
    _write_jsonl(rows_ref, args.out_ref)

    # 7) Build CAN manifest
    rows_can = []
    for _, r in egra.iterrows():
        audio_path = r["audio_abs"]
        can = (r.get("canonical_text") or "").strip()
        pred = hyp_map.get(audio_path, "")

        if not can or not pred:
            continue

        row = {
            "audio_filepath": audio_path,
            "text": _maybe_norm(can, args.normalize_for_nemo),
            "pred_text": _maybe_norm(pred, args.normalize_for_nemo),
        }
        rows_can.append(row)
    _write_jsonl(rows_can, args.out_can)

    print(f"Wrote: {args.out_ref}  (rows={len(rows_ref)})")
    print(f"Wrote: {args.out_can}  (rows={len(rows_can)})")


if __name__ == "__main__":
    main()
