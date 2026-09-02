# Vocabulary

Use these five terms in normal project documentation. They describe the
workflow without requiring readers to understand every implementation layer.

| Term | Meaning in this repository |
|---|---|
| **Model** | The learned speech-recognition system. A model is not a launcher, container, or decoding choice. |
| **Inference setup** | The exact model artifact, input processing, precision, decoding, and other controls selected for a run. One model may have several inference setups. |
| **Execution stack** | The software and hardware used to execute an inference setup. |
| **Run** | One execution of an inference setup on a defined input set. |
| **Evaluation** | Scoring a run's predictions against the appropriate references. A **benchmark** is an evaluation performed under a standardized procedure with fixed inputs and metrics. |

For example, “run the benchmark” means applying the fixed evaluation
procedure. A model, inference setup, or score is not itself “a benchmark.” A
score produced by that procedure is a **benchmark result**.

## Supporting terms

Use the following terms only when their detail helps the reader:

| Term | Precise meaning |
|---|---|
| **Model artifact** | The local files containing model parameters and configuration, such as a `.nemo` archive, a Transformers checkpoint directory, or ONNX/ORT graphs. |
| **Checkpoint** | A model artifact saved in the model author's training or native library format. An exported ONNX artifact is a model artifact, but it is not a checkpoint. |
| **Input processing** | Audio decoding, resampling, feature extraction, and any declared chunking performed before or as part of inference. |
| **Decoding** | Converting model outputs into the final hypothesis, such as greedy, beam, language-model, or transducer search. |
| **Inference library** | The model-facing library called by an adapter, such as Transformers, NeMo, Sherpa-ONNX, or `onnx-asr`. |
| **Inference engine** | The lower-level system that executes model operations, such as PyTorch or ONNX Runtime. |
| **Execution environment** | The process/container, operating system, packages, and hardware available to a run. |
| **Inference adapter** | Repository code that translates the common run interface into calls to a particular inference library. Internal `Backend` class names are compatibility names for these adapters. |

The **execution stack** groups the runner, inference library, inference engine,
execution environment, and hardware. Ordinary guides should name the actual
route directly—for example, “Zipformer runs through Sherpa-ONNX using ONNX
Runtime.” Decompose the stack only in provenance or troubleshooting material.

## ONNX and ORT files

**ONNX Runtime** is the inference engine used by both the direct ONNX and
Sherpa-ONNX routes in this repository. It is installed in the execution
environment; it is not contained inside a model artifact.

- `.onnx` stores a model graph and parameters in the ONNX format.
- `.ort` stores an ONNX Runtime–optimized model representation. It can reduce
  loading work and support a reduced ONNX Runtime build, but it does not bundle
  ONNX Runtime itself or the repository's run configuration.
- A full ONNX Runtime build supports a broad operator set. A reduced build is
  compiled for a known set of models/operators to reduce deployment size. Both
  are ONNX Runtime.

## Schema v2 and compatibility names

New inference profiles use `inference_setup_id`, `inference_library`, and one
`execution_stack` contract reference. New run metadata uses the same terms.
During migration, readers also accept v1 `id`, `framework`, and `runtime`
fields, and new outputs dual-write marked v1 aliases. The aliases describe
compatibility only; they are not preferred vocabulary.
