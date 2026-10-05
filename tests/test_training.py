import asyncio
import io
import json
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

os.environ["ASR_PROVIDER"] = "cpu"
import app as server
from training import OnlineTrainer, TrainingExample, decode_example, write_manifest


def wav_bytes(sample_rate=16000, stereo=False, empty=False):
    buffer = io.BytesIO()
    shape = (0 if empty else sample_rate, 2) if stereo else (0 if empty else sample_rate,)
    sf.write(buffer, np.zeros(shape), sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def backend(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAINING_REPO_ID", "owner/private-model")
    monkeypatch.setenv("TRAINING_CHECKPOINT_DIR", str(tmp_path))
    trainer = OnlineTrainer()
    trainer._resolved = True
    return trainer


@pytest.fixture
def client(monkeypatch, backend):
    monkeypatch.setattr(server, "online_trainer", backend)
    monkeypatch.setattr(server, "ASR_QUANTIZATION", None)
    monkeypatch.setattr(server, "load_model", lambda: None)
    monkeypatch.setattr(server, "asr_model", SimpleNamespace(recognize=lambda *a, **k: "Hello"))
    run = Mock(return_value={"run_id": "test-run", "inference_reloaded": True})
    monkeypatch.setattr(server, "_run_training", run)
    with TestClient(server.app) as http:
        yield http, run


@pytest.mark.parametrize("path", ["/audio/training", "/v1/audio/training"])
def test_single_training_api(client, path):
    http, run = client
    response = http.post(path, data={"transcription": " Hallo! ", "steps": "3"},
                         files={"file": ("../../ignored.wav", wav_bytes(stereo=True))})
    assert response.status_code == 200
    examples, steps, batch_size, lr = run.call_args.args
    assert len(examples) == 1
    assert examples[0].transcription == "Hallo!"
    assert examples[0].waveform.shape == (16000,)
    assert (steps, batch_size, lr) == (3, 1, 1e-5)


@pytest.mark.parametrize("path", ["/audio/training/batch", "/v1/audio/training/batch"])
def test_batch_pairs_preserve_request_order(client, path):
    http, run = client
    response = http.post(path, files=[
        ("files", ("first.wav", wav_bytes(8000))),
        ("transcriptions", (None, "first transcript")),
        ("files", ("second.wav", wav_bytes(16000))),
        ("transcriptions", (None, "second transcript")),
        ("batch_size", (None, "2")),
    ])
    assert response.status_code == 200
    examples = run.call_args.args[0]
    assert [(e.sample_rate, e.transcription) for e in examples] == [
        (8000, "first transcript"), (16000, "second transcript"),
    ]


@pytest.mark.parametrize("data,audio", [
    ({"transcription": " "}, wav_bytes()),
    ({"transcription": "text"}, b"not audio"),
    ({"transcription": "text"}, wav_bytes(empty=True)),
    ({"transcription": "text", "steps": "0"}, wav_bytes()),
    ({"transcription": "text", "steps": "101"}, wav_bytes()),
    ({"transcription": "text", "batch_size": "0"}, wav_bytes()),
    ({"transcription": "text", "learning_rate": "nan"}, wav_bytes()),
    ({"transcription": "text", "learning_rate": "inf"}, wav_bytes()),
    ({"transcription": "text", "learning_rate": "-0.1"}, wav_bytes()),
])
def test_invalid_requests_never_train(client, data, audio):
    http, run = client
    assert http.post("/v1/audio/training", data=data, files={"file": ("a.wav", audio)}).status_code == 400
    run.assert_not_called()


def test_batch_mismatch_limit_and_missing_fields(client, backend):
    http, run = client
    files = [("files", ("a.wav", wav_bytes())), ("files", ("b.wav", wav_bytes()))]
    assert http.post("/v1/audio/training/batch", files=files,
                     data={"transcriptions": "only one"}).status_code == 400
    backend.max_examples = 1
    assert http.post("/v1/audio/training/batch", files=files,
                     data={"transcriptions": ["one", "two"]}).status_code == 400
    assert http.post("/v1/audio/training", files={"file": ("a.wav", wav_bytes())}).status_code == 422
    run.assert_not_called()


def test_unconfigured_and_oversized_requests(client, backend):
    http, run = client
    def post():
        return http.post("/v1/audio/training", data={"transcription": "text"},
                         files={"file": ("a.wav", wav_bytes())})
    backend.repo_id = ""
    assert post().status_code == 503
    backend.repo_id = "owner/private-model"
    backend.max_audio_bytes = 10
    assert post().status_code == 413
    run.assert_not_called()


def test_manifest_resamples_and_preserves_unicode(tmp_path):
    example = decode_example(wav_bytes(8000, stereo=True), "Grüße! 👋")
    path = write_manifest([example], tmp_path, 16000)
    row = json.loads(path.read_text())
    waveform, rate = sf.read(row["audio_filepath"])
    assert rate == 16000 and len(waveform) == 16000
    assert row["text"] == "Grüße! 👋" and row["duration"] == 1


@pytest.fixture
def hub(monkeypatch, tmp_path):
    class EntryNotFoundError(Exception):
        pass
    class LocalEntryNotFoundError(EntryNotFoundError):
        pass
    class RepositoryNotFoundError(Exception):
        pass
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", SimpleNamespace(
        EntryNotFoundError=EntryNotFoundError, LocalEntryNotFoundError=LocalEntryNotFoundError,
        RepositoryNotFoundError=RepositoryNotFoundError,
    ))
    base = tmp_path / "base.nemo"
    base.write_bytes(b"base")
    api = Mock()
    api.model_info.return_value = SimpleNamespace(private=True, sha="before")
    api.create_commit.return_value = SimpleNamespace(oid="after", commit_url="https://huggingface.co/commit/after")
    def fetch(repo_id, filename, **kwargs):
        if filename == "latest.json":
            raise EntryNotFoundError("empty repo")
        return str(base)
    download = Mock(side_effect=fetch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: api, get_token=lambda: "test-token", hf_hub_download=download,
        CommitOperationAdd=lambda **kwargs: SimpleNamespace(**kwargs),
    ))
    return api, download, base


def fake_fit(source, examples, run_dir, steps, batch_size, learning_rate):
    (run_dir / "model.nemo").write_bytes(b"updated weights")
    (run_dir / "trainer.ckpt").write_bytes(b"optimizer state")
    return {"steps_completed": steps, "loss": 0.25}


def test_each_run_publishes_atomic_checkpoint_and_continues_previous_weights(backend, hub, monkeypatch):
    api, download, base = hub
    fit = Mock(side_effect=fake_fit)
    monkeypatch.setattr(backend, "_fit", fit)
    examples = [decode_example(wav_bytes(), "hello")]
    first = backend.run(examples, 2, 1, 1e-5, "base/repo", "base.nemo")
    first_checkpoint = backend.current_checkpoint()
    second = backend.run(examples, 1, 1, 1e-5, "base/repo", "base.nemo")
    assert first["run_id"] != second["run_id"]
    assert fit.call_args_list[0].args[0] == base
    assert fit.call_args_list[1].args[0] == first_checkpoint
    assert first_checkpoint.exists()
    assert json.loads((backend.directory / "latest.json").read_text())["run_id"] == second["run_id"]
    api.create_repo.assert_called_with(repo_id="owner/private-model", repo_type="model", private=True, exist_ok=True)
    commit = api.create_commit.call_args.kwargs
    assert commit["parent_commit"] == "before"
    assert [operation.path_in_repo for operation in commit["operations"]] == [
        f"runs/{second['run_id']}/model.nemo", f"runs/{second['run_id']}/trainer.ckpt",
        f"runs/{second['run_id']}/metadata.json", "latest.json",
    ]
    assert not list(backend.directory.rglob("*.wav"))
    # Published local weights survive a new server instance.
    restored = OnlineTrainer()
    assert restored.current_checkpoint() == backend.current_checkpoint()


def test_public_destination_rejected_before_training(backend, hub, monkeypatch):
    api, _, _ = hub
    api.model_info.return_value.private = False
    fit = Mock()
    monkeypatch.setattr(backend, "_fit", fit)
    with pytest.raises(HTTPException) as error:
        backend.run([], 1, 1, 1e-5, "base/repo", "base.nemo")
    assert error.value.status_code == 409
    fit.assert_not_called()
    api.create_commit.assert_not_called()


def test_restart_resolves_hub_pointer_at_one_revision(backend, hub, tmp_path):
    _, download, base = hub
    pointer = tmp_path / "remote-latest.json"
    pointer.write_text(json.dumps({"checkpoint": "runs/remote/model.nemo"}))
    download.side_effect = lambda repo, filename, **kwargs: str(pointer if filename == "latest.json" else base)
    backend._resolved = False
    assert backend.current_checkpoint() == base
    assert download.call_args_list[0].kwargs == {"revision": "before"}
    assert download.call_args_list[1].args == ("owner/private-model", "runs/remote/model.nemo")
    assert download.call_args_list[1].kwargs == {"revision": "before"}


def test_new_destination_does_not_block_base_inference(backend, hub):
    api, _, _ = hub
    missing = sys.modules["huggingface_hub.utils"].RepositoryNotFoundError
    api.model_info.side_effect = missing("new destination")
    backend._resolved = False
    assert backend.current_checkpoint() is None


def test_hub_network_failure_is_not_treated_as_empty_repo(backend, hub):
    _, download, _ = hub
    offline = sys.modules["huggingface_hub.utils"].LocalEntryNotFoundError
    download.side_effect = offline("offline and no cached checkpoint")
    backend._resolved = False
    with pytest.raises(offline):
        backend.current_checkpoint()


@pytest.mark.parametrize("stage", ["training", "upload"])
def test_failed_run_does_not_advance_checkpoint(backend, hub, monkeypatch, stage):
    api, _, base = hub
    backend._checkpoint = base
    fit = Mock(side_effect=RuntimeError("training failed") if stage == "training" else fake_fit)
    monkeypatch.setattr(backend, "_fit", fit)
    if stage == "upload":
        api.create_commit.side_effect = RuntimeError("Hub unavailable")
    with pytest.raises(HTTPException) as error:
        backend.run([decode_example(wav_bytes(), "hello")], 1, 1, 1e-5, "base/repo", "base.nemo")
    assert error.value.status_code == (502 if stage == "upload" else 500)
    assert error.value.detail["checkpoint_saved"] == (stage == "upload")
    assert backend.current_checkpoint() == base
    assert not (backend.directory / "latest.json").exists()
    if stage == "upload":
        assert Path(error.value.detail["local_checkpoint"]).exists()


def test_training_blocks_inference_and_survives_client_cancellation(monkeypatch):
    started, release = threading.Event(), threading.Event()
    order, threads = [], []
    def train(*args):
        threads.append(threading.get_ident())
        started.set()
        assert release.wait(5)
        order.append("training_saved")
        return {"run_id": "one"}
    def recognize(*args, **kwargs):
        threads.append(threading.get_ident())
        order.append("inference")
        return "updated"
    monkeypatch.setattr(server, "_run_training", train)
    monkeypatch.setattr(server, "load_model", lambda: None)
    monkeypatch.setattr(server, "asr_model", SimpleNamespace(recognize=recognize))
    async def run():
        processor = server.BatchProcessor(1, 1, 2)
        processor.start()
        try:
            training = asyncio.create_task(processor.submit_training([], 1, 1, 1e-5))
            assert await asyncio.to_thread(started.wait, 3)
            inference = asyncio.create_task(processor.submit(np.zeros(16000), 16000, "audio"))
            training.cancel()
            with pytest.raises(asyncio.CancelledError):
                await training
            with pytest.raises(HTTPException) as error:
                await processor.submit_training([], 1, 1, 1e-5)
            assert error.value.status_code == 409
            await asyncio.sleep(0.02)
            assert not inference.done() and order == []
            release.set()
            assert (await asyncio.wait_for(inference, 3))["text"] == "updated"
        finally:
            release.set()
            await processor.stop()
    asyncio.run(run())
    assert order == ["training_saved", "inference"]
    assert len(set(threads)) == 1


@pytest.mark.parametrize("fails", [False, True])
def test_training_releases_models_and_reloads_even_on_failure(monkeypatch, fails):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(server, "asr_model", object())
    monkeypatch.setattr(server, "diarizer", SimpleNamespace(model=object()))
    def train(*args):
        assert server.training_active and server.asr_model is None and server.diarizer.model is None
        if fails:
            raise HTTPException(502, "upload failed")
        return {"run_id": "test", "commit_sha": "sha"}
    load = Mock()
    monkeypatch.setattr(server, "online_trainer", SimpleNamespace(run=train))
    monkeypatch.setattr(server, "load_model", load)
    if fails:
        with pytest.raises(HTTPException):
            server._run_training([], 1, 1, 1e-5)
    else:
        assert server._run_training([], 1, 1, 1e-5)["inference_reloaded"] is True
    load.assert_called_once()
    assert not server.training_active


def test_published_run_reports_reload_failure(monkeypatch, backend):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(server, "asr_model", object())
    monkeypatch.setattr(server, "diarizer", SimpleNamespace(model=None))
    monkeypatch.setattr(server, "online_trainer", backend)
    monkeypatch.setattr(backend, "run", Mock(return_value={"run_id": "published", "commit_sha": "after"}))
    monkeypatch.setattr(server, "load_model", Mock(side_effect=RuntimeError("export failed")))
    with pytest.raises(HTTPException) as error:
        server._run_training([], 1, 1, 1e-5)
    assert error.value.status_code == 503
    assert error.value.detail["checkpoint_published"] is True
    assert error.value.detail["commit_sha"] == "after"
    assert backend.status["state"] == "reload_failed"
    assert not server.training_active


def test_deep_health_uses_inference_queue(client, monkeypatch):
    http, _ = client
    async def submit(waveform, sample_rate, filename):
        assert sample_rate == 16000 and filename == "health-check"
        return {"text": "Hello"}
    monkeypatch.setattr(server.batch_processor, "submit", submit)
    monkeypatch.setattr(server.os.path, "exists", lambda path: True)
    monkeypatch.setattr(server.sf, "read", lambda *args, **kwargs: (np.zeros(16000), 16000))
    monkeypatch.setattr(server, "load_model", Mock(side_effect=AssertionError("must run on worker")))
    assert http.get("/health?deep=true").json()["status"] == "healthy"


def test_trained_export_uses_new_cache_for_each_checkpoint(monkeypatch, tmp_path, hub):
    _, download, _ = hub
    monkeypatch.setattr(server, "ONNX_CACHE_DIR", tmp_path / "onnx")
    exported = []
    def export(path):
        exported.append(path)
        directory = Path(path).parent
        (directory / "model_encoder.onnx").write_bytes(b"encoder")
        (directory / "model_decoder_joint.onnx").write_bytes(b"decoder")
    model = SimpleNamespace(eval=lambda: None, export=export,
                            tokenizer=SimpleNamespace(vocab=["a", "b"]))
    restore = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", SimpleNamespace(
        ASRModel=SimpleNamespace(restore_from=restore),
    ))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)))
    first, second = tmp_path / "first.nemo", tmp_path / "second.nemo"
    first.write_bytes(b"first weights")
    second.write_bytes(b"second weights")
    first_cache = server._ensure_onnx_export(first)
    second_cache = server._ensure_onnx_export(second)
    assert first_cache != second_cache
    assert server._ensure_onnx_export(first) == first_cache
    assert len(exported) == 2
    assert [call.args[0] for call in restore.call_args_list] == [str(first), str(second)]
    download.assert_not_called()
    for cache in (first_cache, second_cache):
        assert (Path(cache) / "encoder-model.onnx").is_file()
        assert (Path(cache) / "decoder_joint-model.onnx").is_file()
        assert json.loads((Path(cache) / "config.json").read_text())["model_type"] == "nemo-conformer-tdt"
