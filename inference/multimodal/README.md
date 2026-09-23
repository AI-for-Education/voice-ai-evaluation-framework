# Multimodal audio inference

This package gives prompt-driven multimodal models a separate inference flow
while preserving the repository's shared input and output contracts. It does not
change the Transformers CTC/Whisper adapters, NeMo, Sherpa-ONNX, transcript
manifest shape, or evaluation commands.

The name *multimodal* describes the model interface rather than the result
format. Gemma 4 E2B can accept several kinds of input—text, audio, images, and
video—through one chat-style generation model. This profile uses only two of
those inputs: a fixed text instruction and audio. Its output is ordinary text,
so evaluation still reads the same `pred_text` field as every ASR route.

## Local model storage

All prompt-driven multimodal snapshots live below `inference/multimodal/models/`:

```text
inference/multimodal/models/
├── gemma-4-E2B-it/
├── paza-Phi-4-multimodal-instruct/
└── Qwen2.5-Omni-7B/
```

All three snapshots now have model-family profiles and adapters. Gemma and Paza are validated on the current host; Qwen is explicitly hardware-gated. Inference is offline and never downloads weights.

## Run Gemma 4 E2B

```bash
./run_multimodal_inference.sh \
  --inference_profile inference/multimodal/profiles/gemma-4-E2B-sw.yaml \
  --audio_manifest input_output_data/output/experiments/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --batch_size 1
```

For a directory of WAV files, replace `--audio_manifest ...` with
`--root_audio_dir <directory>`. Add `--smoke_test` for output below
`input_output_data/output/smoke_tests/transcripts/`.

The checked-in profile freezes the Swahili-only transcription instruction,
disables thinking, loads BF16 weights locally with `trust_remote_code: false`,
and generates at most 512 new tokens per chunk. Gemma accepts at most 30 seconds
of audio, so longer files are split into consecutive 30-second chunks with no
overlap. Chunk transcripts are joined in source order and written as one result
for the original file. The prompt, chunk policy, counts, model class, processor
class, device, dtype, and generation settings are recorded in `run_metadata.json`.

The current `voice-ai-evaluation-framework-asr:latest` image already contains
the required dependencies. The Compose service reuses that image and mounts
`inference/multimodal/models` read-only at `/models`; no image rebuild or extra
download is required for Gemma 4 E2B.


## Inference setup and execution environment

The source folder is organized by inference workflow, while Docker images are
organized by dependency compatibility. All three models accept prompt-plus-audio
inputs and produce the same text transcript contract, so their adapters,
profiles, weights, chunking policy, and output handling remain together here.
They use separate images because their required Transformers versions differ.

Images provide compatible dependency versions; YAML profiles remain authoritative
for each model's prompt, dtype, attention implementation, generation arguments,
audio policy, and hardware gate. Each profile carries structured evidence and
labels owner-published, supported, project, and unvalidated choices separately.

| Model | Launcher | Image | Current-host status |
| --- | --- | --- | --- |
| Gemma 4 E2B | `run_multimodal_inference.sh` | `voice-ai-evaluation-framework-asr:latest` | Implemented and GPU smoke-tested. |
| Paza Phi-4 | `run_phi4_multimodal_inference.sh` | `voice-ai-evaluation-framework-phi4:latest` | Image build and one-file GPU inference passed on the current host. |
| Qwen2.5-Omni-7B | `run_qwen_omni_inference.sh` | `voice-ai-evaluation-framework-qwen-omni:latest` | Structure only on the current 16 GiB host; the tracked BF16 profile requires at least 40 GiB VRAM. |

Model weights are never copied into an image. Every service mounts this
directory's `models/` store read-only, so separate images do not duplicate the
roughly 50 GiB of downloaded snapshots.

## Build and run Paza Phi-4

Start Docker Desktop, then build only the dedicated Phi-4 image:

```bash
docker compose build phi4-multimodal-asr
```

Run a one-file smoke test before a full evaluation:

```bash
./run_phi4_multimodal_inference.sh \
  --inference_profile inference/multimodal/profiles/paza-phi4-multimodal-sw.yaml \
  --audio_manifest <one-row-manifest.jsonl> \
  --batch_size 1 \
  --smoke_test
```

The adapter makes a temporary processor-only view under
`inference/multimodal/tmp/phi4/`, rewriting remote-prefixed `auto_map`
entries to the Python files bundled in the pinned local snapshot. The original
model directory remains unchanged and read-only. Trusted code is enabled only
for this adapter, remains offline, and is recorded in run metadata.

The profile uses BF16, SDPA, `device_map=auto`, a 13 GiB GPU cap, a 12 GiB CPU
cap, and disk offload if needed. On the current 15.9 GiB GPU, Accelerate placed
the effective BF16 model fully on GPU. The first successful one-file smoke run
loaded five shards in about 127 seconds, decoded a 4.096-second file in about
16 seconds, returned `A`, and reported zero errors.

## Qwen2.5-Omni text-only structure

The pinned owner card reports a 31.11 GiB theoretical BF16 minimum for the 7B
model with no input and warns that practical usage is normally at least 1.2
times higher. The profile therefore rounds the supported lower bound up to a
40 GiB preflight gate. It calls `disable_talker()` and fixes
`return_audio=False`; the owner says disabling the talker saves only about 2 GiB,
which is not enough to make the current 16 GiB host runnable.

The owner does not publish a Swahili ASR prompt, output-token cap, or input
duration. The profile freezes a project prompt, `max_new_tokens: 256`, and
deterministic 30-second zero-overlap chunks to bound per-call activation memory.
BF16 comes from the pinned config; SDPA is the built-image compatibility choice.

The image can be prepared now:

```bash
docker compose --profile large-gpu build qwen-omni-asr
```

The run command is for a future suitable GPU host:

```bash
./run_qwen_omni_inference.sh \
  --inference_profile inference/multimodal/profiles/qwen2.5-omni-7b-sw-text.yaml \
  --audio_manifest <segment-manifest.jsonl> \
  --batch_size 1
```

On this 16 GiB machine that command is expected to stop during hardware
preflight. This is deliberate: it avoids a long model load ending in an
out-of-memory failure and prevents an unsupported run from looking valid.
