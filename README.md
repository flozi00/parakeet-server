# parakeet-server

Parakeet transcription server with optional NVIDIA Nemotron speaker diarization.

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
code). The existing ONNX Runtime 1.23.2 / TensorRT 10.9 pins are retained.
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

Sources:
- [NVIDIA model card and inference configuration](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- [Pinned NeMo Speech implementation](https://github.com/NVIDIA-NeMo/Speech/tree/cf724ac337d1ebc7d0dda1e23fb80916f52927a5)
- [OpenAI diarized response schema](https://github.com/openai/openai-python/blob/main/src/openai/types/audio/transcription_diarized.py)
- [OpenAI diarized segment schema](https://github.com/openai/openai-python/blob/main/src/openai/types/audio/transcription_diarized_segment.py)
