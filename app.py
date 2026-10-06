import asyncio
import gc
import hashlib
import io
import json
import logging
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from diarization import NemotronDiarizer, transcribe_turns
from training import OnlineTrainer, decode_example

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global batch_processor
    logger.info("Starting ASR Proxy Server...")
    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to initialize model on startup: {e}")
    batch_processor = BatchProcessor(BATCH_SIZE, BATCH_TIMEOUT_MS, MAX_QUEUE_SIZE)
    batch_processor.start()
    logger.info(
        f"Batch processor started (batch_size={BATCH_SIZE}, "
        f"timeout={BATCH_TIMEOUT_MS}ms, max_queue={MAX_QUEUE_SIZE})"
    )
    yield
    logger.info("Shutting down ASR Proxy Server...")
    if batch_processor:
        await batch_processor.stop()


app = FastAPI(title="Openai ASR Server", middleware=middleware, lifespan=lifespan)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Silence httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Environment variables
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "primeline/parakeet-primeline")
ASR_MODEL_PATH = os.getenv("ASR_MODEL_PATH", None)
ASR_QUANTIZATION = os.getenv("ASR_QUANTIZATION", None) or None
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "tensorrt")
TRT_FP16_ENABLE = os.getenv("TRT_FP16_ENABLE", "true").lower() == "true"
TRT_MAX_WORKSPACE_GB = int(os.getenv("TRT_MAX_WORKSPACE_GB", "6"))
USE_VAD = os.getenv("USE_VAD", "true").lower() == "true"
DIARIZATION_ENABLED = os.getenv("DIARIZATION_ENABLED", "false").lower() == "true"
diarizer = NemotronDiarizer()
online_trainer = OnlineTrainer()
training_active = False

# NeMo export settings (only used when ONNX files not cached)
NEMO_REPO_ID = os.getenv("NEMO_REPO_ID", "primeline/parakeet-primeline")
NEMO_FILENAME = os.getenv("NEMO_FILENAME", "2_95_WER.nemo")
ONNX_CACHE_DIR = Path(os.getenv("ONNX_CACHE_DIR", "/root/.cache/huggingface/onnx_export"))

# Batching configuration
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
BATCH_TIMEOUT_MS = float(os.getenv("BATCH_TIMEOUT_MS", "100"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "64"))

# Global state
asr_model = None
model_loading = False
batch_processor: Optional["BatchProcessor"] = None


def _ensure_tensorrt_on_ld_path():
    """Make the pip-installed TensorRT (tensorrt-cu12-libs) visible to
    onnxruntime.

    onnxruntime's TRT provider does a plain dlopen("libnvinfer.so.10"), which
    only searches the LD_LIBRARY_PATH captured at process start — the pip
    wheel ships its .so files inside site-packages/tensorrt_libs/lib, so the
    dlopen fails with "libnvinfer.so.10: cannot open shared object file" and
    ORT silently falls the whole session back to CPU. Prepend the wheel's lib
    dir to LD_LIBRARY_PATH and re-exec once so the dynamic loader picks it up
    (setting the var after start has no effect on dlopen in glibc).
    """
    if ASR_PROVIDER != "tensorrt" or os.name != "posix":
        return
    try:
        import tensorrt_libs
    except ImportError:
        logger.warning(
            "ASR_PROVIDER=tensorrt but tensorrt-cu12-libs is not installed; "
            "onnxruntime will fall back to CPU/CUDA."
        )
        return
    if not tensorrt_libs.__file__:
        return
    pkg_dir = Path(tensorrt_libs.__file__).parent
    lib_dir = next(
        (p for p in (pkg_dir / "lib", pkg_dir) if any(p.glob("libnvinfer.so*"))),
        None,
    )
    if lib_dir is None:
        logger.warning("tensorrt_libs found but no libnvinfer.so* inside it.")
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if str(lib_dir) in current.split(":"):
        return  # already on the path (this is the re-exec'd process)
    os.environ["LD_LIBRARY_PATH"] = (
        f"{lib_dir}:{current}" if current else str(lib_dir)
    )
    # The early return above makes this loop-free: after re-exec the dir is
    # already on LD_LIBRARY_PATH and we fall through unchanged.
    logger.info(f"Re-exec with LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")
    os.execv(sys.executable, [sys.executable, *sys.argv])


# Run at import time, before uvicorn/onnxruntime are loaded: glibc resolves
# dlopen("libnvinfer.so.10") against the LD_LIBRARY_PATH captured at process
# start, so the re-exec must happen before ORT ever loads. Import-time also
# covers `uvicorn app:app` (not just `python app.py`).
_ensure_tensorrt_on_ld_path()


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------


@dataclass
class _BatchItem:
    """A single queued transcription request."""
    waveform: np.ndarray
    sample_rate: int
    future: asyncio.Future
    filename: str
    diarize: bool = False
    submit_time: float = field(default_factory=time.time)


class BatchProcessor:
    """Collects concurrent transcription requests and processes them with
    controlled GPU concurrency.  Incoming requests are queued; a background
    task drains up to *max_batch_size* items (or fewer after *batch_timeout_ms*)
    and runs inference sequentially on a single-thread executor so the event
    loop stays responsive while only one GPU call is in-flight at a time."""

    def __init__(self, max_batch_size: int, batch_timeout_ms: float, max_queue_size: int):
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._queue: asyncio.Queue[_BatchItem] = asyncio.Queue(maxsize=max_queue_size)
        self._gpu_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
        self._task: Optional[asyncio.Task] = None
        self._training_future: Optional[asyncio.Future] = None
        self._stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "items_failed": 0,
        }

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Finish an accepted training run/checkpoint upload even on shutdown.
        await asyncio.to_thread(self._gpu_executor.shutdown, wait=True)

    # -- public API -----------------------------------------------------------

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        return {**self._stats, "queue_depth": self._queue.qsize()}

    async def submit(self, waveform: np.ndarray, sample_rate: int, filename: str,
                     diarize: bool = False) -> dict:
        """Submit audio for transcription.  Blocks until result is ready.
        Raises asyncio.QueueFull when the service is overloaded."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        item = _BatchItem(waveform=waveform, sample_rate=sample_rate,
                          future=future, filename=filename, diarize=diarize)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            raise HTTPException(
                status_code=429,
                detail=f"Transcription queue full ({self._queue.maxsize}). Try again later.",
            )
        return await future

    async def submit_training(self, examples, steps, batch_size, learning_rate):
        if self._training_future is not None and not self._training_future.done():
            raise HTTPException(409, "A training run is already pending or running.")
        future = asyncio.get_running_loop().run_in_executor(
            self._gpu_executor, _run_training, examples, steps, batch_size, learning_rate,
        )
        self._training_future = future
        # Retrieve errors even if the requesting client disconnects.
        future.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        return await asyncio.shield(future)

    # -- internal -------------------------------------------------------------

    async def _collect_batch(self) -> list[_BatchItem]:
        """Wait for at least one item, then collect up to max_batch_size
        within the timeout window."""
        batch: list[_BatchItem] = []
        # Block until the first item arrives
        batch.append(await self._queue.get())

        deadline = asyncio.get_event_loop().time() + self.batch_timeout_ms / 1000.0
        while len(batch) < self.max_batch_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(item)
            except asyncio.TimeoutError:
                break
        return batch

    def _run_inference(self, waveform: np.ndarray, sample_rate: int, diarize: bool = False):
        """Blocking inference — called inside the thread-pool executor."""
        load_model()
        if asr_model is None:
            raise HTTPException(503, "ASR model not available.")
        if diarize:
            try:
                turns = diarizer.diarize(waveform, sample_rate)
            except Exception as exc:
                logger.exception("Diarization failed")
                raise HTTPException(503, "Diarization unavailable; check server logs and NeMo installation.") from exc
            return transcribe_turns(
                waveform, sample_rate, turns, asr_model.recognize, _extract_text
            )
        # VAD inference is lazy: consume the generator on the GPU worker too.
        return _materialize_items(asr_model.recognize(waveform, sample_rate=sample_rate))

    async def _process_loop(self):
        """Background loop: collect batches and process items."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                batch = await self._collect_batch()
                logger.info(f"Batch collected: {len(batch)} item(s)")
                self._stats["batches_processed"] += 1

                for item in batch:
                    if item.future.cancelled():
                        continue
                    try:
                        start_time = time.time()
                        result = await loop.run_in_executor(
                            self._gpu_executor,
                            self._run_inference,
                            item.waveform,
                            item.sample_rate,
                            item.diarize,
                        )
                        elapsed = round(time.time() - start_time, 3)
                        queue_wait = round(start_time - item.submit_time, 3)

                        if item.diarize:
                            result["transcription_time"] = elapsed
                            result["queue_wait_time"] = queue_wait
                            if not item.future.done():
                                item.future.set_result(result)
                            self._stats["items_processed"] += 1
                            continue

                        # Materialize once — a VAD result is a generator that
                        # would otherwise be exhausted by _extract_text before
                        # _result_to_segments ever sees it.
                        items = _materialize_items(result)
                        full_text = _extract_text(items)
                        segments = _result_to_segments(items)

                        total_duration = 0.0
                        if segments:
                            total_duration = max(seg["end"] for seg in segments)
                        if total_duration == 0.0:
                            total_duration = round(len(item.waveform) / item.sample_rate, 3)

                        response = {
                            "text": full_text,
                            "segments": segments,
                            "language": "en",
                            "duration": total_duration,
                            "transcription_time": elapsed,
                            "queue_wait_time": queue_wait,
                            "task": "transcribe",
                        }
                        if not item.future.done():
                            item.future.set_result(response)
                        self._stats["items_processed"] += 1
                        logger.info(
                            f"Batch item '{item.filename}': {elapsed}s inference, "
                            f"{queue_wait}s queue wait, {len(full_text)} chars"
                        )
                    except Exception as e:
                        if not item.future.done():
                            item.future.set_exception(e)
                        self._stats["items_failed"] += 1
                        logger.error(f"Batch item '{item.filename}' failed: {e}")

            except asyncio.CancelledError:
                # Drain remaining items on shutdown
                while not self._queue.empty():
                    try:
                        item = self._queue.get_nowait()
                        if not item.future.done():
                            item.future.set_exception(
                                HTTPException(status_code=503, detail="Server shutting down")
                            )
                    except asyncio.QueueEmpty:
                        break
                raise
            except Exception as e:
                logger.error(f"Batch processing loop error: {e}", exc_info=True)
                await asyncio.sleep(0.1)  # avoid tight error loop


def _log_active_providers(model) -> None:
    """Log the execution providers each ONNX Runtime session actually got.

    onnx-asr holds its rt.InferenceSession objects on private attributes of
    the underlying Asr instance (e.g. _encoder / _decoder_joint for the
    conformer TDT model); the adapters expose that instance as `.asr`. Walk
    it and report what ORT registered so a silent TRT/CUDA -> CPU fallback is
    visible in the startup log.
    """
    try:
        inner = getattr(model, "asr", model)
        sessions = {}
        for attr in vars(inner).values():
            get_providers = getattr(attr, "get_providers", None)
            if callable(get_providers):
                try:
                    sessions[id(attr)] = get_providers()
                except Exception:
                    pass
        seen = set()
        for providers in sessions.values():
            key = tuple(providers)
            if key in seen:
                continue
            seen.add(key)
            logger.info(f"ORT session providers: {list(providers)}")
        if seen and all(p and p[0] == "CPUExecutionProvider" for p in seen):
            logger.warning(
                "All ONNX Runtime sessions fell back to CPUExecutionProvider — "
                "TensorRT/CUDA were requested but not registered. Transcription "
                "will be slow. Check the provider_bridge errors above "
                "(libnvinfer / libcudnn load failures)."
            )
    except Exception as e:  # diagnostics only — never break startup
        logger.debug(f"Could not inspect ORT providers: {e}")


def _build_providers():
    """Build ONNX Runtime provider list based on configuration."""
    if ASR_PROVIDER == "tensorrt":
        return [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_max_workspace_size": TRT_MAX_WORKSPACE_GB * 1024**3,
                    "trt_fp16_enable": TRT_FP16_ENABLE,
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    elif ASR_PROVIDER == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]


def _ensure_onnx_export(nemo_checkpoint: Path | None = None):
    """Export .nemo model to ONNX if not already cached. Returns local ONNX path."""
    onnx_dir = ONNX_CACHE_DIR / NEMO_REPO_ID.replace("/", "_")
    if nemo_checkpoint is not None:
        # Every published training run gets a new cache; never reuse base ONNX
        # files or TensorRT engines with stale weights.
        key = hashlib.sha256(str(nemo_checkpoint.resolve()).encode()).hexdigest()[:16]
        onnx_dir = ONNX_CACHE_DIR / f"trained_{key}"
    marker = onnx_dir / "config.json"

    if marker.exists():
        logger.info(f"ONNX export found at {onnx_dir}, skipping export.")
        return str(onnx_dir)

    logger.info(f"No ONNX export cached. Exporting {NEMO_REPO_ID}/{NEMO_FILENAME}...")
    onnx_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    from nemo.collections.asr.models import ASRModel

    # Download .nemo checkpoint
    nemo_path = str(nemo_checkpoint) if nemo_checkpoint else hf_hub_download(
        repo_id=NEMO_REPO_ID, filename=NEMO_FILENAME,
    )
    logger.info(f"Downloaded .nemo to {nemo_path}, loading model for export...")

    # Load on CPU to minimise GPU memory during export
    model = ASRModel.restore_from(nemo_path, map_location="cpu")
    model.eval()

    # Export to ONNX
    onnx_path = str(onnx_dir / "model.onnx")
    logger.info(f"Exporting to ONNX: {onnx_path}")
    model.export(onnx_path)

    # NeMo produces model_encoder.onnx + model_decoder_joint.onnx
    # onnx-asr expects encoder-model.onnx + decoder_joint-model.onnx
    renames = {
        "model_encoder.onnx": "encoder-model.onnx",
        "model_decoder_joint.onnx": "decoder_joint-model.onnx",
        "model_encoder.onnx.data": "encoder-model.onnx.data",
    }
    for src_name, dst_name in renames.items():
        src = onnx_dir / src_name
        if src.exists():
            src.rename(onnx_dir / dst_name)
            logger.info(f"Renamed {src_name} -> {dst_name}")

    # Write vocab.txt
    vocab_path = onnx_dir / "vocab.txt"
    with vocab_path.open("wt") as f:
        for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
            f.write(f"{token} {i}\n")
    logger.info(f"Wrote vocab ({i+1} tokens) to {vocab_path}")

    # Write config.json (written last — acts as completion marker)
    config = {
        "model_type": "nemo-conformer-tdt",
        "features_size": 128,
        "subsampling_factor": 8,
        "max_tokens_per_step": 10,
    }
    with marker.open("w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Wrote config.json to {marker}")

    # Free NeMo/torch memory before loading with onnx-asr
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    logger.info(f"ONNX export complete at {onnx_dir}")
    return str(onnx_dir)


def load_model():
    """Load published training weights at startup, or retry on a later request."""
    global asr_model, model_loading

    if asr_model is not None:
        return

    if model_loading:
        max_wait = 120
        waited = 0
        while model_loading and waited < max_wait:
            time.sleep(0.5)
            waited += 0.5
        return

    model_loading = True
    try:
        import onnx_asr

        # If model path not set, ensure ONNX export exists
        checkpoint = online_trainer.current_checkpoint()
        model_path = _ensure_onnx_export(checkpoint) if checkpoint else ASR_MODEL_PATH
        if not model_path:
            model_path = _ensure_onnx_export()

        providers = _build_providers()
        logger.info(
            f"Loading ASR model: {ASR_MODEL_NAME} from {model_path} "
            f"(quantization={ASR_QUANTIZATION}, providers={[p if isinstance(p, str) else p[0] for p in providers]})"
        )

        model = onnx_asr.load_model(
            ASR_MODEL_NAME,
            path=model_path,
            quantization=ASR_QUANTIZATION,
            providers=providers,
        )

        # Use timestamps adapter for segment-level results
        asr_model = model.with_timestamps()

        if USE_VAD:
            vad = onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])
            asr_model = model.with_vad(vad).with_timestamps()
            logger.info("VAD (Silero) enabled for long audio support.")

        # Report which execution providers the sessions actually got. ORT
        # silently falls back to CPU when a provider fails to register, which
        # previously hid a broken TensorRT setup behind a "loaded successfully"
        # log line.
        _log_active_providers(model)

        logger.info("ASR model loaded successfully.")

        # Warmup. flo.wav is not baked into the image, so fall back to a
        # synthetic 1 s tone — the point is to exercise the preprocessor +
        # ORT sessions (CUDA context, TRT execution) before real traffic.
        warmup_audio_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "flo.wav"
        )
        warmup_iterations = 3
        if os.path.exists(warmup_audio_path):
            warmup_input = warmup_audio_path
        else:
            logger.info(
                f"Warmup audio not found at {warmup_audio_path}; "
                "using synthetic tone."
            )
            _t = np.linspace(0, 1.0, 16000, dtype=np.float32)
            warmup_input = (0.1 * np.sin(2 * np.pi * 440.0 * _t)).astype(np.float32)
        logger.info(f"Performing {warmup_iterations} warmup transcriptions...")
        for i in range(warmup_iterations):
            try:
                warmup_start = time.time()
                warmup_result = asr_model.recognize(warmup_input)
                warmup_time = time.time() - warmup_start
                if i == 0 or i == warmup_iterations - 1:
                    text = _extract_text(warmup_result)
                    logger.info(
                        f"Warmup {i+1}/{warmup_iterations}: {warmup_time:.2f}s - '{text[:80]}'"
                    )
            except Exception as e:
                logger.warning(f"Warmup {i+1}/{warmup_iterations} failed (non-fatal): {e}")

    except Exception as e:
        asr_model = None
        logger.critical(f"FATAL: Could not load ASR model. Error: {e}")
        raise
    finally:
        model_loading = False


def _run_training(examples, steps, batch_size, learning_rate):
    """The same executor owns training, export, reloading, and all inference."""
    global asr_model, training_active
    training_active = True
    # Free ONNX and diarization GPU allocations for the full NeMo optimizer.
    asr_model = None
    diarizer.model = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result = online_trainer.run(
            examples, steps, batch_size, learning_rate, NEMO_REPO_ID, NEMO_FILENAME,
        )
    except Exception as exc:
        # Failed training/upload does not advance the active checkpoint.
        # Trainer error tracebacks can retain the entire GPU model/optimizer.
        # Logs already contain the training traceback; release its frame locals
        # before restoring inference so an OOM does not keep those allocations.
        cause = exc
        seen = set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            traceback.clear_frames(cause.__traceback__)
            cause = cause.__cause__ or cause.__context__
        gc.collect()
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
            sys.modules["torch"].cuda.empty_cache()
        try:
            load_model()
        except Exception:
            logger.exception("Could not restore ASR after failed training")
        raise
    else:
        try:
            load_model()
        except Exception as exc:
            logger.exception("Checkpoint published but inference reload failed")
            online_trainer.status = {**online_trainer.status, "state": "reload_failed"}
            raise HTTPException(503, detail={
                "message": "Checkpoint published, but inference reload failed; check server logs.",
                "run_id": result["run_id"], "checkpoint_published": True,
                "commit_sha": result["commit_sha"],
            }) from exc
        return {**result, "inference_reloaded": True}
    finally:
        training_active = False


def _materialize_items(result):
    """Normalize any onnx-asr recognize() return value into a list of result
    items so it can be traversed more than once.

    onnx-asr returns different shapes depending on the adapter:
      * ``str`` for the plain text adapter,
      * a single ``TimestampedResult`` for ``.with_timestamps()`` (no VAD),
      * an ``Iterator`` of ``SegmentResult`` / ``TimestampedSegmentResult`` for
        the VAD adapters (a generator that can only be consumed once).
    """
    if result is None:
        return []
    if isinstance(result, str):
        return [result]
    # A single dataclass result exposes .text and is not itself an iterator.
    if hasattr(result, "text") and not hasattr(result, "__next__"):
        return [result]
    # Iterator / generator / list of segment results.
    try:
        return list(result)
    except TypeError:
        return [result]


def _extract_text(result):
    """Extract the full transcript text from any onnx-asr result type."""
    parts = []
    for item in _materialize_items(result):
        if isinstance(item, str):
            parts.append(item)
        elif hasattr(item, "text") and item.text:
            parts.append(item.text)
    return " ".join(p.strip() for p in parts if p and p.strip()).strip()


def _avg_logprob(item):
    """Mean token log-probability for a result item, or None when unavailable."""
    logprobs = getattr(item, "logprobs", None)
    if logprobs:
        return round(sum(logprobs) / len(logprobs), 4)
    return None


def _result_to_segments(result):
    """Convert an onnx-asr result into OpenAI verbose_json compatible segments.

    Mapping rules:
      * VAD results (``TimestampedSegmentResult`` / ``SegmentResult``) carry
        per-speech-segment ``start``/``end`` — one OpenAI segment each.
      * A non-VAD ``TimestampedResult`` only carries token-level timestamps for
        the whole utterance — emit a single segment spanning first..last token.
      * A plain text result has no timing — emit a single 0..0 segment.

    ``tokens`` is intentionally left empty: onnx-asr returns string subword
    tokens whereas the OpenAI schema expects integer token ids, so emitting the
    raw tokens would violate the schema for strict clients.
    """
    segments = []

    for idx, item in enumerate(_materialize_items(result)):
        if isinstance(item, str):
            text = item.strip()
        else:
            text = item.text.strip() if hasattr(item, "text") and item.text else ""

        if hasattr(item, "start") and hasattr(item, "end"):
            # SegmentResult / TimestampedSegmentResult (VAD path).
            start = round(item.start, 3)
            end = round(item.end, 3)
        elif getattr(item, "timestamps", None):
            # TimestampedResult without VAD — token-level timestamps only.
            start = round(item.timestamps[0], 3)
            end = round(item.timestamps[-1], 3)
        elif text:
            # Plain text result without any timing information.
            start = 0.0
            end = 0.0
        else:
            continue

        segments.append({
            "id": idx,
            "seek": 0,
            "start": start,
            "end": end,
            "text": text,
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": _avg_logprob(item),
            "compression_ratio": None,
            "no_speech_prob": None,
        })

    return segments


def _ensure_response_segments(response: dict) -> dict:
    segments = response.get("segments")
    if isinstance(segments, list) and segments:
        return response

    text = response.get("text")
    duration = response.get("duration")
    if not isinstance(text, str) or not text.strip():
        response.setdefault("segments", [])
        return response

    try:
        end = float(duration)
    except (TypeError, ValueError):
        end = 0.0

    if end <= 0:
        response.setdefault("segments", [])
        return response

    response["segments"] = [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": round(end, 3),
            "text": text.strip(),
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": None,
            "compression_ratio": None,
            "no_speech_prob": None,
        }
    ]
    return response



@app.get("/health")
async def health_check(deep: bool = False):
    """Health check endpoint."""
    base_status = {
        "model_loaded": asr_model is not None,
        "model_name": ASR_MODEL_NAME,
        "provider": ASR_PROVIDER,
        "quantization": ASR_QUANTIZATION,
        "vad_enabled": USE_VAD,
        "diarization_enabled": DIARIZATION_ENABLED,
        "diarization_loaded": diarizer.model is not None,
        "training": {"active": training_active, **online_trainer.status},
        "batch": batch_processor.stats if batch_processor else None,
    }

    if not deep:
        base_status["status"] = "training" if training_active else ("healthy" if asr_model else "degraded")
        return base_status

    try:
        if not batch_processor:
            base_status["status"] = "unhealthy"
            base_status["error"] = "Model not loaded"
            return JSONResponse(content=base_status, status_code=503)

        warmup_audio_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "flo.wav"
        )
        if not os.path.exists(warmup_audio_path):
            base_status["status"] = "degraded"
            base_status["error"] = "Health check audio file not found"
            return base_status

        start_time = time.time()
        waveform, sample_rate = sf.read(warmup_audio_path, dtype="float32")
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        result = await batch_processor.submit(waveform, sample_rate, "health-check")
        transcription_time = round(time.time() - start_time, 3)

        text = result["text"]
        if len(text) < 3:
            base_status["status"] = "unhealthy"
            base_status["error"] = "Transcription returned empty or too short result"
            return JSONResponse(content=base_status, status_code=503)

        base_status["status"] = "healthy"
        base_status["transcription_test"] = {
            "success": True,
            "text_length": len(text),
            "transcription_time_seconds": transcription_time,
        }
        return base_status

    except Exception as e:
        logger.error(f"Deep health check failed: {e}")
        base_status["status"] = "unhealthy"
        base_status["error"] = str(e)
        return JSONResponse(content=base_status, status_code=503)


@app.get("/ready")
async def readiness():
    """Readiness probe: 503 until the ASR model is actually loaded.

    /health answers 200 ("degraded") as soon as the HTTP server is up, which
    makes it a fine liveness probe but a useless readiness gate — Kubernetes
    would route traffic to a pod that cannot transcribe yet (the first start
    also runs a one-time .nemo -> ONNX/TensorRT export that can take many
    minutes). This endpoint flips to 200 exactly when the model can serve.
    """
    if training_active or asr_model is None:
        return JSONResponse(
            content={"status": "training" if training_active else "not_ready",
                     "model_loaded": asr_model is not None}, status_code=503
        )
    return {"status": "ready", "model_loaded": True, "model_name": ASR_MODEL_NAME}


@app.post("/audio/transcriptions")
@app.post("/v1/audio/transcriptions")
async def transcribe_rest(
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    response_format: str | None = Form(default=None),
    stream: bool = Form(default=False),
    chunking_strategy: str | None = Form(default=None),
    known_speaker_names: list[str] | None = Form(default=None, alias="known_speaker_names[]"),
    known_speaker_references: list[str] | None = Form(default=None, alias="known_speaker_references[]"),
    timestamp_granularities: list[str] | None = Form(
        default=None, alias="timestamp_granularities[]"
    ),
):
    """Handles audio transcription via REST API (OpenAI compatible).
    Requests are queued and processed in batches for scalability."""
    wants_diarization = response_format == "diarized_json"
    if wants_diarization:
        if stream:
            raise HTTPException(400, "Streaming diarization is not supported; use stream=false.")
        if timestamp_granularities:
            raise HTTPException(400, "diarized_json provides speaker segment timestamps, not timestamp_granularities[].")
        if known_speaker_names or known_speaker_references:
            raise HTTPException(400, "Known-speaker reference matching is not supported.")
        if chunking_strategy not in (None, "auto"):
            raise HTTPException(400, "Only chunking_strategy=auto is supported for diarization.")
        if not DIARIZATION_ENABLED:
            raise HTTPException(503, "Diarization is disabled; set DIARIZATION_ENABLED=true.")
    if not batch_processor:
        raise HTTPException(status_code=503, detail="Batch processor not ready.")

    logger.info(
        f"transcribe_rest: Received request for file: {file.filename}, "
        f"model={model or ASR_MODEL_NAME}, response_format={response_format}, "
        f"timestamp_granularities={timestamp_granularities}"
    )

    try:
        # Read audio bytes and decode with soundfile (handles wav, flac, ogg, etc.)
        audio_bytes = await file.read()
        try:
            waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except (sf.LibsndfileError, ValueError) as exc:
            raise HTTPException(400, "Invalid or unsupported audio file.") from exc

        # Convert stereo to mono if needed
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if len(waveform) == 0 or not np.isfinite(waveform).all():
            raise HTTPException(400, "Audio must contain finite, non-empty samples.")

        logger.info(
            f"transcribe_rest: Audio loaded, {len(waveform)} samples, {sample_rate}Hz, "
            f"{len(waveform) / sample_rate:.1f}s — submitting to batch queue "
            f"(depth={batch_processor.queue_size})"
        )

        response = await batch_processor.submit(
            waveform, sample_rate, file.filename or "unknown", diarize=wants_diarization
        )
        if not wants_diarization:
            response = _ensure_response_segments(response)

        logger.info(
            f"transcribe_rest: Completed for '{file.filename}', "
            f"{response['transcription_time']}s inference, "
            f"{response.get('queue_wait_time', 0)}s queued"
        )

        if wants_diarization:
            return JSONResponse(content={
                key: response[key] for key in ("task", "duration", "text", "segments")
            })
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"transcribe_rest: Unhandled exception: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _train_uploads(files, transcriptions, steps, batch_size, learning_rate):
    online_trainer.check_available()
    if ASR_QUANTIZATION is not None:
        raise HTTPException(503, "Online training requires ASR_QUANTIZATION to be unset for ONNX reload.")
    if len(files) != len(transcriptions):
        raise HTTPException(400, "Provide one transcription for each audio file, in the same order.")
    if not 1 <= len(files) <= online_trainer.max_examples:
        raise HTTPException(400, f"Provide between 1 and {online_trainer.max_examples} pairs.")
    if not 1 <= steps <= online_trainer.max_steps:
        raise HTTPException(400, f"steps must be between 1 and {online_trainer.max_steps}.")
    if not 1 <= batch_size <= online_trainer.max_examples:
        raise HTTPException(400, f"batch_size must be between 1 and {online_trainer.max_examples}.")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise HTTPException(400, "learning_rate must be finite and in (0, 1].")
    if batch_processor is None:
        raise HTTPException(503, "Batch processor not ready.")
    examples = []
    total_bytes = 0
    for file, transcription in zip(files, transcriptions):
        audio = await file.read(online_trainer.max_audio_bytes - total_bytes + 1)
        total_bytes += len(audio)
        if total_bytes > online_trainer.max_audio_bytes:
            raise HTTPException(413, "Training audio exceeds TRAINING_MAX_AUDIO_BYTES.")
        examples.append(decode_example(audio, transcription))
    return await batch_processor.submit_training(examples, steps, batch_size, learning_rate)


@app.post("/audio/training")
@app.post("/v1/audio/training")
async def train_single(
    file: UploadFile = File(...),
    transcription: str = Form(...),
    steps: int = Form(default=1),
    batch_size: int = Form(default=1),
    learning_rate: float = Form(default=1e-5),
):
    """Fine-tune on one audio/transcript pair, publish, then reload inference."""
    return await _train_uploads([file], [transcription], steps, batch_size, learning_rate)


@app.post("/audio/training/batch")
@app.post("/v1/audio/training/batch")
async def train_batch(
    files: list[UploadFile] = File(...),
    transcriptions: list[str] = Form(...),
    steps: int = Form(default=1),
    batch_size: int = Form(default=1),
    learning_rate: float = Form(default=1e-5),
):
    """Repeated files/transcriptions fields pair by position in the request."""
    return await _train_uploads(files, transcriptions, steps, batch_size, learning_rate)


def main():
    """Main application entry point."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "warning")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        workers=None,
        forwarded_allow_ips="*",
        proxy_headers=True,
        timeout_keep_alive=900,
        reload=False,
    )


if __name__ == "__main__":
    main()
