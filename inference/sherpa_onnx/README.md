# Sherpa-ONNX inference

This package runs the BookBot streaming Zipformer model through Sherpa-ONNX.
It uses the same inference profile, input manifest, progress display, timestamped output,
and `transcriptions.jsonl` contract as the NeMo and Transformers launchers.

## Why this has its own folder

Hugging Face hosts the model files, but the files are ONNX encoder, decoder,
and joiner graphs. Hugging Face Transformers cannot load that model. This is a
separate inference route: Zipformer runs through Sherpa-ONNX using ONNX
Runtime. Sherpa-ONNX is the inference library; ONNX Runtime is the inference
engine installed in the container. Neither one is bundled inside the model
files.

```text
inference/sherpa_onnx/
├── backend.py       # Loads the ONNX graphs and transcribes audio
├── infer.py         # Shared-style command-line interface
├── models/          # Local model artifacts; not tracked by Git
└── profiles/        # Tracked model configuration
```

The profile matrix is:

- FP32 ONNX: `zipformer-streaming-robust-sw-v4.yaml` and its
  `-modified-beam4` pair.
- INT8 ONNX: `zipformer-streaming-robust-sw-v4-onnx-int8.yaml` and its
  `-modified-beam4` pair.
- INT8 ORT: `zipformer-streaming-robust-sw-v4-ort-int8.yaml` and its
  `-modified-beam4` pair.

Every quantized profile owns both `loader.artifact_format` and
`loader.artifact_precision`. The backend therefore fails closed if the selected
encoder, decoder, and joiner triplet is missing or ambiguous; it never chooses
INT8 merely because those files happen to share a directory with FP32 graphs.

Sherpa-ONNX's official streaming recognizer configuration documents both
`greedy_search` and `modified_beam_search`, and defines 4 as the default
maximum number of active paths for beam-style decoding. See the
[official OnlineRecognizerConfig reference](https://k2-fsa.github.io/sherpa/onnx/c-api/html/structsherpa__onnx_1_1cxx_1_1OnlineRecognizerConfig.html).

FP32 and INT8 ONNX expect:

```text
models/sherpa-onnx-zipformer-streaming-robust-sw-v4/
```

INT8 ORT expects:

```text
models/sherpa-onnx-ort-zipformer-streaming-robust-sw-v4/
```

## Run

```bash
./run_sherpa_onnx_inference.sh \
  --inference_profile inference/sherpa_onnx/profiles/zipformer-streaming-robust-sw-v4.yaml \
  --audio_manifest input_output_data/output/experiments/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --batch_size 8
```

The completed modified-beam baseline uses the identical manifest input and
`transcriptions.jsonl` output schema as greedy. Its unique profile ID keeps its
timestamped transcript and evaluation directories separate, and the evidence
record identifies four active paths as the documented Sherpa-ONNX search
default.

Add `--smoke_test` when using a small test manifest. Normal and smoke-test
outputs follow the same paths as the other inference routes:

```text
input_output_data/output/transcripts/<inference-setup-id>_<timestamp>/
input_output_data/output/smoke_tests/transcripts/<inference-setup-id>_<timestamp>/
```

The Compose service reserves the GPU by default. The inference adapter chooses Sherpa's
CUDA provider from the CUDA package installed by `docker/Dockerfile`; use
`--provider cpu` for the BookBot ORT artifacts, which crash in the pinned CUDA runtime.
The resolved provider, decoding method, and (for modified beam)
`max_active_paths` are recorded in `run_metadata.json`.

The model outputs **phonemes**, not written Swahili words. Keep its transcript
as a phoneme-system result and do not feed it directly into orthographic
WER/MER evaluation. A phoneme metric or a separately defined phoneme-to-word
system is required for a fair evaluation.
