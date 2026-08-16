# Exp41 mobile-target ONNX validation on PC

This backend tests two deployment artifacts derived from the existing internal
Exp41 NeMo checkpoint:

- `model.onnx`: FP32 portable reference.
- `model.int8.onnx`: dynamic per-channel QUInt8 weight quantization intended as
  the first mobile-target candidate.

They are two representations of the **same trained model**, not two newly
trained models. Both use the same 16 kHz audio, 80-feature frontend,
subsampling factor 8, SentencePiece vocabulary, and greedy CTC decoding.

## What this first implementation does

- Exports the exact local `model_exp41_avg.nemo` checkpoint after requiring its
  known SHA-256.
- Writes `vocab.txt` with the CTC blank as the final token, following the
  desktop convention used by `onnx-asr`.
- Records source and derived-file hashes in `artifact_metadata.json`.
- Resumes safely: a rerun verifies and skips a completed stage; it never
  overwrites an existing model file.
- Runs FP32 and INT8 through the normal framework manifest/output contract on
  PC CPU.

This step does **not** build an APK, package assets for Android, or emulate an
ARM phone. It tests model-format correctness and PC CPU accuracy/performance.
True phone latency, memory, battery use, Android audio preprocessing, and ARM
operator behavior still require a later physical-device test.

## Prepare the artifacts

Build the updated shared image once:

```bash
docker compose build onnxruntime-prepare
```

Then export FP32 and quantize INT8:

```bash
./run_onnxruntime_prepare.sh
```

The command writes only below:

```text
inference/onnxruntime/models/swahili-exp41-ctc/
├── artifact_metadata.json
├── config.json
├── model.onnx
├── model.int8.onnx
└── vocab.txt
```

If interrupted after FP32 completes, run the same command again. It verifies
FP32 and continues with INT8. If either existing artifact fails its integrity
check, preparation stops and reports the problem instead of deleting or
replacing it.

## Smoke tests

Use the same segment manifest for native NeMo, FP32 ONNX, and INT8 ONNX:

```bash
MANIFEST="input_output_data/output/segments/<run>/manifest.jsonl"

./run_onnxruntime_inference.sh \
  --model_config inference/onnxruntime/profiles/swahili-exp41-ctc-onnx-fp32.yaml \
  --audio_manifest "$MANIFEST" \
  --batch_size 1 \
  --num_threads 2 \
  --smoke_test

./run_onnxruntime_inference.sh \
  --model_config inference/onnxruntime/profiles/swahili-exp41-ctc-onnx-int8.yaml \
  --audio_manifest "$MANIFEST" \
  --batch_size 1 \
  --num_threads 2 \
  --smoke_test
```

No segmentation command is needed again. These commands read the already
segmented audio paths from the manifest and write the standard
`transcriptions.jsonl` plus `run_metadata.json` below
`input_output_data/output/smoke_tests/transcripts/`.

## Real tests

After smoke-test parity is acceptable, remove `--smoke_test` and keep the same
manifest and fixed thread/batch settings:

```bash
./run_onnxruntime_inference.sh \
  --model_config inference/onnxruntime/profiles/swahili-exp41-ctc-onnx-fp32.yaml \
  --audio_manifest "$MANIFEST" \
  --batch_size 8 \
  --num_threads 2

./run_onnxruntime_inference.sh \
  --model_config inference/onnxruntime/profiles/swahili-exp41-ctc-onnx-int8.yaml \
  --audio_manifest "$MANIFEST" \
  --batch_size 8 \
  --num_threads 2
```

Run the existing manifest/evaluation commands on each generated transcript
directory. Compare native NeMo, FP32, and INT8 as deployment variants with:

- identical input audio paths and references;
- exact-match transcript disagreement rate;
- WER/MER changes;
- wall time and peak memory;
- artifact file size.

## Android-parity accuracy proxy

This is a separate result instance; it does not replace the PC `onnx-asr`
baseline above. It runs the exact public mobile artifact with the pinned Android
runtime version and mirrors the demo's PCM16 reader, linear resampler, log-mel
frontend, and greedy CTC decoder.

The local ignored bundle is:

```text
inference/onnxruntime/models/swahili-exp41-ctc-android/
├── config.json
├── model.int8.onnx
└── vocab.txt
```

`model.int8.onnx` must be the artifact linked from the pinned mobile repository:

- size: `139737438` bytes
- SHA-256: `bef2607617aef3520a5313d76c08a03eda692f69f1f1561bc3e980dab49afaa7`

Build the isolated image once:

```bash
docker compose build onnxruntime-android-asr
```

Run it with the existing segmented manifest:

```bash
./run_onnxruntime_android_inference.sh \
  --model_config inference/onnxruntime/profiles/swahili-exp41-ctc-android-int8.yaml \
  --audio_manifest "$MANIFEST" \
  --batch_size 1 \
  --num_threads 1
```

The launcher rejects other batch or thread values. Backend startup also checks
the artifact hash and size, vocabulary/blank id, ONNX input/output contract,
quantized operator counts, and `onnxruntime==1.22.0`. Focused frontend and
decoder tests live in `tests/test_onnxruntime_android_backend.py`; the opt-in
real-artifact test is `tests/test_onnxruntime_android_artifact_smoke.py`.

This is an Android-behaviour **accuracy proxy**, not ARM emulation. Physical
phone tests remain authoritative for latency, memory, battery, and final
cross-runtime numerical parity.
## Provenance and reuse note

The conversion layout was checked against
`AI-for-Education/nemoasr-android` revision
`6cd031267aed45e47347bc05a0eb63eee285f0d8`. That repository has no license
file at this revision, so redistribution/reuse permission should be confirmed
with its owner before any external release. The PC wrapper is pinned to
`onnx-asr==0.12.0` (source tag revision
`b9e0ce0ae3223b3d24ce5a22a5e701a726ca35fc`).
