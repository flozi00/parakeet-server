# parakeet-server — OpenAI-compatible ASR API (onnx-asr + TensorRT/NeMo)
#
# Just-in-time build: a small Python base, then pip install the requirements.
# The entire CUDA/TensorRT runtime is pulled from pip wheels (see
# requirements.txt: onnxruntime-gpu, tensorrt-cu12-libs, nvidia-*-cu12), so no
# pre-baked multi-GB CUDA image is needed. The host still needs the NVIDIA
# driver + nvidia-container-toolkit so the container can see the GPU.

FROM python:3.10-slim

# libsndfile1: soundfile decodes wav/flac/ogg uploads.
# curl: the HEALTHCHECK below.
# build-essential + libgomp1: a few wheels compile/link against OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 curl build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so the layer is cached across app.py edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Where the runtime ONNX export (and any HF downloads) land. Mount a volume
# here in production so the one-time .nemo -> ONNX export survives restarts.
ENV PORT=8000 \
    ONNX_CACHE_DIR=/root/.cache/huggingface/onnx_export \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8000

# /health answers 200 as soon as the HTTP server is up (status: degraded until
# the model loads), so it is a valid liveness/readiness probe.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["python", "app.py"]
