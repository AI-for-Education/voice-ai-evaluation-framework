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
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Choose PyTorch build: cpu | cu121
# You can override at build time:
#   docker compose build --build-arg TORCH_CUDA=cpu
#   docker compose build --build-arg TORCH_CUDA=cu121
# ----------------------------
ARG TORCH_CUDA=cpu
ENV TORCH_CUDA=${TORCH_CUDA}

# Install PyTorch matching torchaudio/vision for the chosen backend
# (These wheels include CUDA runtime when cu121 is selected.)
RUN if [ "$TORCH_CUDA" = "cpu" ]; then \
      pip3 install torch==2.3.1+cpu torchvision==0.18.1+cpu torchaudio==2.3.1+cpu \
        --index-url https://download.pytorch.org/whl/cpu ; \
    else \
      pip3 install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 ; \
    fi

# Core Python dependencies
RUN pip3 install \
    numpy==1.26.2 \
    tqdm==4.67.1 \
    soundfile==0.13.1 \
    librosa==0.10.1 \
    sentencepiece==0.2.0 \
    pandas==2.2.2 \
    torchaudio==2.3.1 \
    praatio==6.2.0 \
    jiwer==3.1.0

RUN pip3 install \
    psutil==5.9.5 \
    loguru==0.7.0 \
    python-stdnum==1.20 \
    cython==3.0.11 \
    hydra-core==1.3.2 \
    pytorch-lightning==2.2.4 \
    transformers==4.40.2 \
    huggingface-hub==0.23.0 \
    inflect==7.2.1 \
    editdistance==0.8.1 \
    lhotse>=1.33.0 \
    pyannote.core==5.0.0 \
    webdataset==0.1.62 \
    datasets==2.18.0 \
    pyannote.metrics==3.2.1 \
    ipython==8.24.0

# NeMo ASR
RUN pip3 install "nemo_toolkit[asr]==2.4.1"

# App
WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python3"]
