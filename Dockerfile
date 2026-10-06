# parakeet-server — OpenAI-compatible ASR API (onnx-asr + TensorRT/NeMo)
#
# Just-in-time build: a small Python base, then pip install the requirements.
# The entire CUDA/TensorRT runtime is pulled from pip wheels (see
# requirements.txt: onnxruntime-gpu, tensorrt-cu12-libs, nvidia-*-cu12), so no
# pre-baked multi-GB CUDA image is needed. The host still needs the NVIDIA
# driver + nvidia-container-toolkit so the container can see the GPU.

FROM python:3.12-slim

# libsndfile1: soundfile decodes wav/flac/ogg uploads.
# build-essential + libgomp1: a few wheels compile/link against OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libsndfile1 build-essential libgomp1 git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so the layer is cached across app.py edits.
# --extra-index-url pypi.nvidia.com: tensorrt-cu12-libs publishes only an
# sdist stub to PyPI (the real manylinux wheels live on the NVIDIA index), so
# a plain PyPI install of the pinned TRT version fails to build.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com \
        -r requirements.txt

# Numba discovers NVVM/libdevice through CUDA_HOME, but does not discover
# NVIDIA's pip wheel layout automatically. Expose the compiler wheel as a
# toolkit and its CUDA runtime libraries at the expected lib64 location.
RUN ln -s /usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc /usr/local/cuda \
    && ln -s ../cuda_runtime/lib /usr/local/cuda/lib64
ENV CUDA_HOME=/usr/local/cuda

# This check needs no GPU and prevents shipping another inference-only image
# whose first training request fails because NVVM or libdevice is missing.
RUN python -c "from numba.cuda.cudadrv.libs import open_cudalib, open_libdevice; open_cudalib('nvvm'); open_cudalib('cudart'); assert open_libdevice()"

COPY app.py diarization.py training.py ./

# Where the runtime ONNX export (and any HF downloads) land. Mount a volume
# here in production so the one-time .nemo -> ONNX export survives restarts.
ENV PORT=8000 \
    ONNX_CACHE_DIR=/root/.cache/huggingface/onnx_export \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8000

# Two probes, matching app.py: /health answers 200 ("degraded") the moment
# uvicorn binds, /ready flips to 200 only once the ASR model is loaded. This
# Docker-level check is liveness-only (Kubernetes uses the probes in
# k8s/deployment.yaml instead); python/urllib keeps curl out of the image.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health',timeout=8)" || exit 1

CMD ["python", "app.py"]
