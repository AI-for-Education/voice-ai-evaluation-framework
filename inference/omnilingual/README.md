# Omnilingual ASR

This integration keeps Fairseq2 out of the shared ASR image because Fairseq2's
native wheel must match the exact PyTorch ABI. The dedicated image pins
Omnilingual ASR 0.2.0, Fairseq2 0.5.2, and PyTorch 2.7.1/CUDA 12.8.

Two Swahili profiles are ready:

- `profiles/omniasr-llm-3b-v2-swh-latn.yaml` is the primary, language-conditioned
  LLM 3B v2 comparison (about 10 GiB inference VRAM at the owner's benchmark).
- `profiles/omniasr-ctc-3b-v2-swh.yaml` is the faster CTC 3B v2 comparison
  (about 8 GiB inference VRAM). Its language field is metadata only because the
  CTC model does not accept language conditioning.

Preparation and inference are deliberately separate. Preparation has network
access and writes one model-specific Fairseq2 cache. Inference requires the
successful preparation marker, mounts that cache read-only, and has networking
disabled.

From Git Bash, prepare exactly the profile you intend to run:

```bash
./run_omnilingual_prepare.sh \
  --inference_profile inference/omnilingual/profiles/omniasr-llm-3b-v2-swh-latn.yaml
```

Then run a smoke test:

```bash
./run_omnilingual_inference.sh \
  --inference_profile inference/omnilingual/profiles/omniasr-llm-3b-v2-swh-latn.yaml \
  --audio_manifest path/to/transcriptions.jsonl \
  --output_root input_output_data/output \
  --batch_size 1 \
  --smoke_test
```

Replace the profile path with the CTC profile for that comparison. The adapter
resamples to 16 kHz, uses the owner's normalization/frontend, splits longer
inputs into sequential 30-second chunks, and preserves the owner's decoded text
without postprocessing.
