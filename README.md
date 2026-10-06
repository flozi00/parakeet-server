# parakeet-server

Parakeet transcription server with optional NVIDIA Nemotron speaker diarization.

## Online training

`POST /v1/audio/training` accepts one audio/transcription pair.
`POST /v1/audio/training/batch` accepts multiple pairs. Both are also available
without `/v1`. Requests use multipart form data and wait for training, checkpoint
upload, ONNX export, and inference reload to finish.

Configure the training destination on the server:

```bash
docker build -t parakeet-server .
export HF_TOKEN=hf_your_write_token
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN \
  -e TRAINING_REPO_ID=your-account/parakeet-online \
  -v parakeet-cache:/root/.cache/huggingface parakeet-server
```

The token must have write access to the destination and read access to the
original `NEMO_REPO_ID` checkpoint. The server creates the destination as a
**private model repo** if it does not exist. An existing public repo is rejected
before training. Training starts from `NEMO_REPO_ID` / `NEMO_FILENAME` (defaults:
`primeline/parakeet-primeline` / `2_95_WER.nemo`), then continues from the most
recent successfully published weights. An ONNX-only source cannot be trained;
configure the matching original `.nemo` model. Leave `ASR_QUANTIZATION` unset.

Send one pair:

```bash
curl http://localhost:8000/v1/audio/training \
  -F file=@utterance.wav \
  -F 'transcription=The correct transcription.' \
  -F steps=3 \
  -F learning_rate=0.00001
```

Send a batch by repeating `files` and `transcriptions` fields. The first file
pairs with the first transcription, the second file with the second, and so on:

```bash
curl http://localhost:8000/v1/audio/training/batch \
  -F files=@first.wav -F 'transcriptions=First transcript.' \
  -F files=@second.wav -F 'transcriptions=Second transcript.' \
  -F steps=5 -F batch_size=2 -F learning_rate=0.00001
```

| Form field | Default | Meaning |
| --- | --- | --- |
| `steps` | `1` | Optimizer updates per request; repeats the supplied examples across epochs as needed |
| `batch_size` | `1` | Pairs per update, capped to the number supplied |
| `learning_rate` | `0.00001` | Constant AdamW learning rate, greater than zero and at most one |

Stereo audio is mixed to mono and resampled to the model's sample rate.
Empty audio, non-finite samples, blank transcriptions, and mismatched batches
are rejected before training. Supported formats match SoundFile (WAV, FLAC,
OGG, etc.). Each request uses a fresh AdamW optimizer; model weights carry over
between requests. The saved Lightning checkpoint includes optimizer state for
manual continuation outside this API.

Each successful run commits these files together to the Hub:

```text
runs/<run_id>/model.nemo       # complete model, including tokenizer
runs/<run_id>/trainer.ckpt     # Lightning model + optimizer state
runs/<run_id>/metadata.json   # training options, completed steps, final loss
latest.json                  # pointer to the most recently published run
```

Audio and transcriptions are temporary training inputs and are not uploaded.
Previous run files remain available. The response includes `run_id`, `repo_id`,
`checkpoint`, `trainer_checkpoint`, `steps_completed`, `loss`, `commit_sha`,
`commit_url`, and `inference_reloaded: true`.

Training and inference share one worker. Training unloads ONNX and diarization
models to free GPU memory, then exports the new weights into a separate ONNX
cache and reloads inference. Queued transcription requests wait until this is
complete; deep health inference uses that same queue. `/health` reports training
state and `/ready` returns 503 during training and reload. Only one training
request may be pending/running; another returns 409. Use one server process and
one replica per training repository. Hub commits also check that the repository
has not changed during the run.

On a training failure, the previous published weights are reloaded. If upload
fails, the API returns 502 with the run ID and the retained local checkpoint
path; that run does not become active. A 503 with `checkpoint_published: true`
means upload succeeded but inference reload failed. A client disconnect does
not cancel an accepted run, and shutdown waits for its checkpoint operation.
Restarting uses the locally published checkpoint, or downloads the Hub's
`latest.json` checkpoint if the local volume is unavailable. Keep the cache
volume persistent to retain unpublished checkpoints and avoid repeated exports.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `TRAINING_REPO_ID` | unset | Private Hub model repository, e.g. `your-account/parakeet-online` |
| `HF_TOKEN` | unset | Hugging Face write token; a cached Hub login also works |
| `TRAINING_CHECKPOINT_DIR` | `/root/.cache/huggingface/training` | Local run checkpoints, isolated by destination repo |
| `TRAINING_DEVICE` | `cuda` if available, else `cpu` | NeMo training device |
| `TRAINING_PRECISION` | `32-true` | Lightning precision setting |
| `TRAINING_MAX_STEPS` | `100` | Maximum optimizer steps per request |
| `TRAINING_MAX_EXAMPLES` | `64` | Maximum pairs and requested batch size |
| `TRAINING_MAX_AUDIO_BYTES` | `67108864` | Maximum total uploaded audio bytes per request (64 MiB) |

Full-model training needs memory for weights, gradients, and AdamW optimizer
state, beyond inference memory. Tiny online batches can overfit and forget
previous data; assess recognition quality on held-out audio for your use case.

## Speaker diarization

Enable `DIARIZATION_ENABLED=true` and request `response_format=diarized_json`
on `POST /v1/audio/transcriptions` (also available without `/v1`).
Parakeet still transcribes the speech; Nemotron identifies speaker turns.
The existing ASR response behavior is unchanged for other response formats.

```bash
docker build -t parakeet-server .
docker run --gpus all -p 8000:8000 \
  -e DIARIZATION_ENABLED=true \
  -v parakeet-cache:/root/.cache/huggingface parakeet-server

curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F model=primeline/parakeet-primeline \
  -F response_format=diarized_json
```

Using the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
with open("meeting.wav", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="primeline/parakeet-primeline",
        file=audio,
        response_format="diarized_json",
    )
for segment in result.segments:
    print(segment.speaker, segment.start, segment.end, segment.text)
```

Example response (illustrative):

```json
{
  "task": "transcribe",
  "duration": 5.0,
  "text": "Hallo! Guten Tag.",
  "segments": [
    {"id": "0", "type": "transcript.text.segment", "speaker": "A", "start": 0.2, "end": 1.1, "text": "Hallo!"},
    {"id": "1", "type": "transcript.text.segment", "speaker": "B", "start": 1.5, "end": 3.0, "text": "Guten Tag."}
  ]
}
```

`duration` includes silence and describes the complete input, in seconds.
Segment IDs are strings; speaker labels are `A`, `B`, etc. in first-arrival
order and are local to a request. A returning speaker keeps its label.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `DIARIZATION_ENABLED` | `false` | Enable `diarized_json`; disabled requests return HTTP 503 |
| `DIARIZATION_MODEL_NAME` | `nvidia/Nemotron-3-Diarization` | NeMo-compatible checkpoint |
| `DIARIZATION_DEVICE` | `cuda` when available, otherwise `cpu` | PyTorch device, independent of the ASR ONNX provider |

The model downloads and loads on the first diarization request, then stays
resident. Allow extra time for this request and additional GPU memory for both
models. `/health` reports `diarization_enabled` and `diarization_loaded`;
`/ready` continues to describe ASR readiness. If model loading or diarization
fails, the request returns HTTP 503 and ordinary transcription remains available.
Use `HF_TOKEN` if needed for authenticated Hugging Face downloads.

## Processing and limitations

- Stereo is mixed to mono. Nemotron receives the entire recording resampled to
  16 kHz and uses its chunked offline configuration with a speaker cache.
- Each detected turn is transcribed with Parakeet using the original audio.
  ASR crops are limited to 30 seconds; long turns retain their speaker label.
  Crop boundaries can reduce recognition accuracy or split words, and timestamps
  describe speaker turns/crops, not forced-aligned word boundaries.
- Nemotron supports up to eight speakers. This is diarization, not voice identity
  verification or source separation. Overlapping turns are retained, but ASR
  sees the mixed audio: overlapping speech may be duplicated or attributed to
  the wrong speaker. Evaluate representative recordings before production use.
- Silence with no detected turns returns empty text and segments.
- This implementation supports non-streaming file transcription. `stream=true`,
  `known_speaker_names[]`, `known_speaker_references[]`, and
  `timestamp_granularities[]` are rejected for `diarized_json` with HTTP 400.
  `chunking_strategy=auto` is accepted; custom chunking configurations are rejected.
- Model loading, diarization and ASR run on the same single-worker request queue.
  Long recordings delay other requests; the existing queue limit applies.

## Runtime and tests

The image now uses Python 3.12 and a pinned NeMo Speech source revision containing
Nemotron-3's high-resolution Sortformer implementation (NeMo 3.0.0 lacks that
code). The existing ONNX Runtime 1.23.2 / TensorRT 10.9 pins are retained. PyTorch is
pinned to 2.8.0 so it shares their CUDA 12 runtime wheels instead of adding a
second CUDA 13 stack. The image workflow frees unused runner SDKs before
building to leave room for BuildKit's unpacked wheels and exported layers.
The image build needs GitHub access for the pinned NeMo dependency.

CPU tests use fake model outputs to check the API and processing contract:

```bash
pip install fastapi python-multipart numpy soundfile scipy httpx pytest
python -m pytest -q
```

These tests do not validate model accuracy, GPU memory use, or the complete
Docker dependency stack. Before deployment, build the image and run the curl
example against a real two-speaker WAV; verify speaker continuity, timestamps,
and ordinary ASR with your chosen GPU/provider.

Training tests cover multipart pairing, validation, checkpoint publication,
failed-upload retention, weight continuation, and serialization with inference.
The real Lightning optimizer test runs when `torch`, `lightning`, and `omegaconf`
are installed (otherwise it is skipped). It uses a tiny stand-in model to verify
updates across epochs and optimizer checkpoint state. Before deployment, run a
real Parakeet training request on your GPU, check the private Hub commit, and
transcribe audio with the reloaded model. This also validates the NeMo TDT loss
and ONNX export against your actual checkpoint and runtime.

### Live private Hub upload test

The local smoke test uses the real training HTTP endpoints, Lightning optimizer
loop, checkpoint saving, and production Hub publication code. It substitutes a
tiny CPU model for NeMo and a test inference adapter for ONNX, so it can run on
a laptop. Its `model.nemo` payload is a PyTorch test state dict, **not a NeMo
archive or a usable Parakeet model**. Use a dedicated test repository; the script
refuses an existing repo whose `latest.json` points to non-test checkpoints.

```bash
uv venv .venv-test --python 3.12
uv pip install --python .venv-test/bin/python \
  fastapi python-multipart numpy soundfile scipy httpx pytest \
  torch lightning omegaconf 'huggingface_hub>=0.34,<1.0'

# Use HF_TOKEN with write access, or an existing hf auth login.
hf auth whoami
.venv-test/bin/python scripts/hub_upload_smoke.py \
  your-account/parakeet-server-hub-smoke-test
```

The script creates a private model repo if needed, performs a single-pair run
and a batch run, and leaves both checkpoints in that repo for inspection. It
checks the immutable Hub commits, SHA-256 hashes of downloaded files, latest-run
pointer, model weight continuation, optimizer state, inference reload through
the test adapter, fresh-cache resume, and denial of anonymous access. It prints
a JSON report and does not run as part of ordinary pytest or CI.

Verified locally on **2026-10-05** with Python 3.12, CPU PyTorch 2.14.1,
Lightning 2.6.6, and huggingface_hub 0.36.2:

| Request | Pairs | Completed updates | Hub commit |
| --- | --- | --- | --- |
| `/v1/audio/training` | 1 | 3 | [24b2cc7](https://huggingface.co/flozi00/parakeet-server-hub-smoke-test/commit/24b2cc75404a64bd7e739e9fd30aa1b87cf72f31) |
| `/v1/audio/training/batch` | 2 | 2 | [c7782f6](https://huggingface.co/flozi00/parakeet-server-hub-smoke-test/commit/c7782f64320f19be088540e79e31070b0af445b6) |

Both requests returned HTTP 200. The fixture weight advanced from 0 to 0.3 to
0.5. All six downloaded run files matched their local SHA-256 hashes, the repo
remained private, anonymous access was denied, and a fresh cache loaded the
second run. Links require access to the private test repository. The full
machine-readable result is in [docs/hub-upload-smoke-2026-10-05.json](docs/hub-upload-smoke-2026-10-05.json).
This verifies checkpoint publication, not Parakeet recognition quality, NeMo TDT
training, GPU memory use, or ONNX/TensorRT export.

Sources:
- [NVIDIA model card and inference configuration](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- [Pinned NeMo Speech implementation](https://github.com/NVIDIA-NeMo/Speech/tree/cf724ac337d1ebc7d0dda1e23fb80916f52927a5)
- [OpenAI diarized response schema](https://github.com/openai/openai-python/blob/main/src/openai/types/audio/transcription_diarized.py)
- [OpenAI diarized segment schema](https://github.com/openai/openai-python/blob/main/src/openai/types/audio/transcription_diarized_segment.py)
