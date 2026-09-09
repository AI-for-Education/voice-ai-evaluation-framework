# docker/Dockerfile
FROM debian:12

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# before any pip install lines
ENV PIP_BREAK_SYSTEM_PACKAGES=1


# Basic deps (audio I/O + build essentials for a few pip wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates \
    python3 python3-pip python3-dev \
    git \
    ffmpeg sox libsndfile1 \
    build-essential cmake \
    libboost-program-options-dev libboost-system-dev libboost-thread-dev \
    libbz2-dev liblzma-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# Debian 12 ships an older pip that cannot normalize some dependency names
# exposed by the PyTorch wheel index (for example typing-extensions).
RUN python3 -m pip install --upgrade \
    pip==25.1.1 \
    setuptools==80.9.0 \
    wheel==0.45.1

# ----------------------------
# Choose PyTorch build: cu128 (default) | cpu
# Override with --build-arg TORCH_CUDA=cpu for a CPU-only image.
# ----------------------------
ARG TORCH_CUDA=cu128
ENV TORCH_CUDA=${TORCH_CUDA}

# Install PyTorch matching torchaudio/vision for the chosen backend
# (The CUDA 12.8 wheels include kernels for NVIDIA Blackwell / sm_120 GPUs.)
RUN if [ "$TORCH_CUDA" = "cpu" ]; then \
      pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cpu ; \
    else \
      pip3 install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
        --index-url https://download.pytorch.org/whl/cu128 ; \
    fi

# Core Python dependencies
RUN pip3 install \
    numpy==1.26.2 \
    tqdm==4.67.1 \
    soundfile==0.13.1 \
    librosa==0.10.1 \
    PyYAML==6.0.2 \
    sentencepiece==0.2.0 \
    pandas==2.2.2 \
    praatio==6.2.0 \
    africa-g2p==0.1.0 \
    jiwer==3.1.0

# babygruut and the Swahili lexicon/CRF resources are pinned to one upstream
# revision. Evaluation always passes turso_config=None, so these local assets
# are the only pronunciation source (SQLite lexicon first, CRF fallback).
RUN pip3 install \
    "babygruut @ git+https://github.com/bookbot-hive/babygruut.git@85eadf1743455ba57ccb1baccf39485dc8628c33" \
    "gruut-lang-sw @ git+https://github.com/bookbot-hive/babygruut.git@85eadf1743455ba57ccb1baccf39485dc8628c33#subdirectory=gruut-lang-sw"

# BookBot's packaged 5-gram CTC decoder. These are additive: greedy CTC,
# speech-seq2seq, NeMo, and evaluation code do not import them.
RUN pip3 install \
    pyctcdecode==0.5.0 \
    kenlm==0.3.0

RUN pip3 install \
    psutil==5.9.5 \
    loguru==0.7.0 \
    python-stdnum==1.20 \
    cython==3.0.11 \
    hydra-core==1.3.2 \
    pytorch-lightning==2.2.4 \
    transformers==5.15.0 \
    huggingface-hub==1.27.0 \
    inflect==7.2.1 \
    editdistance==0.8.1 \
    "lhotse>=1.33.0" \
    pyannote.core==5.0.0 \
    webdataset==0.1.62 \
    datasets==2.18.0 \
    pyannote.metrics==3.2.1 \
    ipython==8.24.0

# NeMo ASR
RUN pip3 install "nemo_toolkit[asr]==2.3"

# PC validation for NeMo-derived FP32/INT8 ONNX deployment artifacts. The
# explicit Python environment stays compatible with NeMo's NumPy/protobuf pins;
# onnx-asr adds the matching feature preprocessor and greedy CTC wrapper.
RUN pip3 install \
    numpy==1.26.2 \
    protobuf==4.24.4 \
    onnxruntime==1.23.2 \
    onnx-asr==0.12.0

# Streaming BookBot Zipformer. This wheel matches the default CUDA 12 image;
# Sherpa selects its CPU provider automatically when CUDA is unavailable.
RUN if [ "$TORCH_CUDA" = "cpu" ]; then \
      pip3 install sherpa-onnx==1.13.5 ; \
    else \
      pip3 install sherpa-onnx==1.13.5+cuda12.cudnn9.onnxruntime1.27.1 \
        --find-links https://k2-fsa.github.io/sherpa/onnx/cuda.html ; \
    fi \
 && pip3 check

# Kaldi-compatible online features for BookBot's native TorchScript Zipformer.
# This lightweight extractor has no PyTorch ABI coupling; decoding stays in the
# already pinned PyTorch runtime above. Keep this additive layer last so the
# established ASR dependency image remains cacheable.
RUN pip3 install kaldi-native-fbank==1.22.3 \
 && pip3 check

# The XLS-R-53 eSpeak comparator uses Transformers' phoneme tokenizer. Keep
# this additive layer last so established ASR runtimes retain their cache.
RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng \
 && rm -rf /var/lib/apt/lists/* \
 && pip3 install phonemizer==3.4.0 \
 && pip3 check

# App
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/dist-packages/nvidia/nvjitlink/lib
ENTRYPOINT ["python3"]
