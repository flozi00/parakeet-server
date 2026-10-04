import asyncio
import io
import os
import sys
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

os.environ["ASR_PROVIDER"] = "cpu"
import app as server
from diarization import NemotronDiarizer, normalize_turns, transcribe_turns


def test_normalize_clips_sorts_and_retains_overlap():
    assert normalize_turns(["1 9 speaker_1", "-1 2 speaker_0", "4 4 speaker_2"], 3) == [
        (0, 2, "speaker_0"), (1, 3, "speaker_1")
    ]


@pytest.mark.parametrize("line", ["nan 2 speaker_0", "0 inf speaker_0", "invalid"])
def test_invalid_model_output_fails(line):
    with pytest.raises(ValueError):
        normalize_turns([line], 3)


def test_turn_transcription_speakers_crops_and_duration():
    recognize = Mock(side_effect=["Hello", "Hi", "Again"])
    waveform = np.arange(80000, dtype=np.float32)
    result = transcribe_turns(waveform, 16000,
        [(0, 1, "speaker_7"), (1.5, 2, "speaker_2"), (3, 4, "speaker_7")],
        recognize, server._extract_text)
    assert result["duration"] == 5
    assert result["text"] == "Hello Hi Again"
    assert [s["speaker"] for s in result["segments"]] == ["A", "B", "A"]
    assert [s["id"] for s in result["segments"]] == ["0", "1", "2"]
    assert all(s["type"] == "transcript.text.segment" for s in result["segments"])
    np.testing.assert_array_equal(recognize.call_args_list[1].args[0], waveform[24000:32000])


def test_long_turn_is_bounded_without_resetting_speaker():
    recognize = Mock(return_value="words")
    result = transcribe_turns(np.zeros(65 * 16000), 16000, [(0, 65, "speaker_0")],
                             recognize, server._extract_text)
    assert [(s["start"], s["end"]) for s in result["segments"]] == [(0, 30), (30, 60), (60, 65)]
    assert {s["speaker"] for s in result["segments"]} == {"A"}


def test_silence_and_empty_asr():
    recognize = Mock(return_value="")
    audio = np.zeros(16000)
    assert transcribe_turns(audio, 16000, [], recognize, server._extract_text)["segments"] == []
    recognize.assert_not_called()
    assert transcribe_turns(audio, 16000, [(0, 1, "s")], recognize, server._extract_text)["text"] == ""


def test_micro_turns_shorter_than_vad_window_are_dropped():
    """Regression: <36 ms diarization turns crashed Silero VAD's sliding window.

    Long recordings produce 10-30 ms micro-turns; crops shorter than the
    576-sample VAD window raise "window shape cannot be larger than input
    array shape" inside numpy sliding_window_view. They must be dropped.
    """
    recognize = Mock(return_value="words")
    audio = np.zeros(3 * 16000, dtype=np.float32)
    turns = [(0, 0.01, "speaker_0"), (0.02, 0.03, "speaker_1"), (1, 2, "speaker_2")]
    result = transcribe_turns(audio, 16000, turns, recognize, server._extract_text)
    # Only the 1 s turn survives; recognize is never called with <576 samples.
    assert [s["speaker"] for s in result["segments"]] == ["A"]
    assert all(s["end"] - s["start"] >= (512 + 64) / 16000 for s in result["segments"])
    for call in recognize.call_args_list:
        assert call.args[0].shape[0] >= 512 + 64


def test_model_resamples_once_and_passes_complete_recording(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext, Tensor=type("Tensor", (), {})))
    backend = NemotronDiarizer()
    backend.model = Mock()
    backend.model.diarize.return_value = [["0 1 speaker_0"]]
    assert backend.diarize(np.zeros(48000, dtype=np.float32), 48000) == [(0, 1, "speaker_0")]
    kwargs = backend.model.diarize.call_args.kwargs
    assert kwargs["sample_rate"] == 16000
    assert kwargs["batch_size"] == 1
    assert kwargs["audio"][0].shape == (16000,)
    assert kwargs["audio"][0].dtype == np.float32


def test_model_load_is_cached_and_configures_offline_mode(monkeypatch):
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models",
                        SimpleNamespace(SortformerEncLabelModel=SimpleNamespace(from_pretrained=factory)))
    monkeypatch.delenv("DIARIZATION_DEVICE", raising=False)
    backend = NemotronDiarizer()
    assert backend.load() is backend.load() is model
    factory.assert_called_once_with("nvidia/Nemotron-3-Diarization", map_location="cpu")
    assert model.sortformer_modules.chunk_len == 340
    model._check_streaming_parameters.assert_called_once()


def wav_bytes(stereo=False, empty=False):
    buf = io.BytesIO()
    sf.write(buf, np.zeros((0 if empty else 16000, 2) if stereo else (0 if empty else 16000,)),
             16000, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "DIARIZATION_ENABLED", True)
    monkeypatch.setattr(server, "load_model", lambda: None)
    monkeypatch.setattr(server, "asr_model", SimpleNamespace(recognize=lambda *a, **k: "Hello"))
    monkeypatch.setattr(server, "diarizer", SimpleNamespace(model=object(),
                        diarize=lambda *a: [(0, 1, "speaker_0")]))
    with TestClient(server.app) as client:
        yield client


@pytest.mark.parametrize("path", ["/audio/transcriptions", "/v1/audio/transcriptions"])
def test_http_diarized_schema(client, path):
    response = client.post(path, data={"model": "parakeet", "response_format": "diarized_json",
                                      "chunking_strategy": "auto"},
                           files={"file": ("audio.wav", wav_bytes(stereo=True), "audio/wav")})
    assert response.status_code == 200
    assert response.json() == {
        "task": "transcribe", "duration": 1.0, "text": "Hello",
        "segments": [{"id": "0", "type": "transcript.text.segment", "speaker": "A",
                      "start": 0.0, "end": 1.0, "text": "Hello"}],
    }


@pytest.mark.parametrize("option", [{"stream": "true"}, {"known_speaker_names[]": "Flo"},
    {"known_speaker_references[]": "data:audio/wav;base64,..."},
    {"chunking_strategy": "{}"}, {"timestamp_granularities[]": "word"}])
def test_unsupported_diarization_options(client, option):
    response = client.post("/v1/audio/transcriptions", data={"response_format": "diarized_json", **option},
                           files={"file": ("audio.wav", wav_bytes())})
    assert response.status_code == 400


def test_disabled_diarization_leaves_normal_asr_available(client, monkeypatch):
    monkeypatch.setattr(server, "DIARIZATION_ENABLED", False)
    files = {"file": ("audio.wav", wav_bytes())}
    assert client.post("/v1/audio/transcriptions", data={"response_format": "diarized_json"}, files=files).status_code == 503
    result = client.post("/v1/audio/transcriptions", files=files)
    assert result.status_code == 200
    assert result.json()["text"] == "Hello"
    assert "speaker" not in result.json()["segments"][0]


def test_model_failure_does_not_break_asr(client, monkeypatch):
    monkeypatch.setattr(server.diarizer, "diarize", Mock(side_effect=RuntimeError("model unavailable")))
    files = {"file": ("audio.wav", wav_bytes())}
    assert client.post("/v1/audio/transcriptions", data={"response_format": "diarized_json"}, files=files).status_code == 503
    assert client.post("/v1/audio/transcriptions", files=files).status_code == 200


@pytest.mark.parametrize("audio", [b"not an audio file", wav_bytes(empty=True)])
def test_bad_audio(client, audio):
    assert client.post("/v1/audio/transcriptions", data={"response_format": "diarized_json"},
                       files={"file": ("audio.wav", audio)}).status_code == 400


def test_generator_and_diarization_execute_on_same_worker(monkeypatch):
    threads = []
    def recognize(*args, **kwargs):
        threads.append(threading.get_ident())
        yield SimpleNamespace(text="Hello")
    def diarize(*args):
        threads.append(threading.get_ident())
        return [(0, 1, "speaker_0")]
    monkeypatch.setattr(server, "asr_model", SimpleNamespace(recognize=recognize))
    monkeypatch.setattr(server, "diarizer", SimpleNamespace(diarize=diarize))
    async def run():
        processor = server.BatchProcessor(2, 1, 2)
        processor.start()
        try:
            results = await asyncio.gather(processor.submit(np.zeros(16000), 16000, "a"),
                processor.submit(np.zeros(16000), 16000, "b", diarize=True))
            assert [r["text"] for r in results] == ["Hello", "Hello"]
        finally:
            await processor.stop()
    asyncio.run(run())
    assert len(threads) == 3 and len(set(threads)) == 1
    assert threads[0] != threading.get_ident()
