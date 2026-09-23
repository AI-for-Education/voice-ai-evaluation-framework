# Native BookBot TorchScript inference

This adapter runs BookBot's published streaming Zipformer JIT artifact through
PyTorch. It consumes audio, not a natural-language prompt, and emits the IPA
phoneme sequence defined by the packaged `tokens.txt`.

The two tracked profiles use the same pinned model:

- `zipformer-streaming-robust-sw-v4-torchscript.yaml`: BookBot's documented
  greedy JIT path.
- `zipformer-streaming-robust-sw-v4-torchscript-modified-beam4.yaml`: the same
  JIT with Icefall-compatible modified beam search and four active paths.

The native feature path is owner-aligned rather than byte-for-byte identical.
It preserves BookBot's published geometry: 16 kHz audio, 80-bin online Kaldi
fbank, dither disabled, `snip_edges=false`, a 32-frame encoder chunk (64 input
frames), 13 padding frames, 0.25-second waveform feeds, and 0.3-second tail
padding. The project substitutes `kaldi-native-fbank` for the owner's
`kaldifeat`, and the shared loader mean-downmixes and resamples instead of
rejecting non-16-kHz input and selecting only channel one.

As in the owner JIT example, a partial final encoder window is not generally
flushed. The adapter adds zero padding only when an input plus the owner tail
would otherwise produce no encoder window; this prevents the three sub-window
items in the frozen manifest from becoming silent successful empty results.
The effective policy and padding count are recorded in `run_metadata.json`.

BookBot publishes a JIT command only for greedy decoding. The beam profile is
therefore recorded as an unvalidated project adaptation of Icefall's stateful,
one-symbol-per-frame streaming modified-beam algorithm, not as an
owner-published JIT command. Four active paths is separately recorded as the
Icefall default. Keep the two setup IDs separate in comparisons.

Run either profile with the shared manifest/output contract:

```bash
./run_torch_inference.sh \
  --inference_profile inference/torch/profiles/zipformer-streaming-robust-sw-v4-torchscript.yaml \
  --audio_manifest input_output_data/output/experiments/<experiment>/manifests/ref_manifest.raw_segments.jsonl \
  --batch_size 1
```

Use `--smoke_test` for development inputs. Results are phonemes and must not be
scored directly as orthographic Swahili WER.
