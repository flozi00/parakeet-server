"""Live private-Hub upload test using real CPU updates on a tiny test fixture.

Run only against a dedicated test repository. The stand-in .nemo payload is a
PyTorch state dict; this test does not run Parakeet, NeMo, or ONNX inference.
"""

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

from tiny_asr import TinyASR
from training import OnlineTrainer


class TinyOnlineTrainer(OnlineTrainer):
    def _fit(self, *args, **kwargs):
        factory = SimpleNamespace(ASRModel=SimpleNamespace(restore_from=TinyASR.restore_from))
        with patch.dict(sys.modules, {"nemo.collections.asr.models": factory}):
            metrics = super()._fit(*args, **kwargs)
        return {**metrics, "checkpoint_format": "pytorch-test-fixture",
                "test": "parakeet-server-hub-upload-smoke"}


def audio_file():
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(8000), 8000, format="WAV")
    return ("synthetic.wav", buffer.getvalue(), "audio/wav")


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run(repo_id):
    api = HfApi()
    # Do not replace a real model repository's latest.json with test fixtures.
    try:
        info = api.model_info(repo_id)
    except RepositoryNotFoundError:
        info = None
    if info is not None:
        if not info.private:
            raise RuntimeError("The smoke-test destination must be private.")
        if "latest.json" in api.list_repo_files(repo_id):
            existing = json.loads(Path(hf_hub_download(repo_id, "latest.json", revision=info.sha)).read_text())
            if existing.get("test") != "parakeet-server-hub-upload-smoke":
                raise RuntimeError("Destination contains real checkpoints; use a dedicated test repo.")

    results = []
    with tempfile.TemporaryDirectory(prefix="parakeet-hub-smoke-") as temp, patch.dict(os.environ, {
        "ASR_PROVIDER": "cpu", "TRAINING_REPO_ID": repo_id,
        "TRAINING_CHECKPOINT_DIR": temp, "TRAINING_DEVICE": "cpu", "TRAINING_PRECISION": "32-true",
    }):
        import app as server

        backend = TinyOnlineTrainer()
        seed = Path(temp) / "seed.nemo"
        TinyASR().save_to(seed)
        backend._checkpoint, backend._resolved = seed, True

        def load_test_inference():
            state = torch.load(backend.current_checkpoint(), map_location="cpu", weights_only=True)
            server.asr_model = SimpleNamespace(recognize=lambda *a, **k: f"fixture weight {state['weight'].item()}")

        previous_weight = 0.0
        with patch.object(server, "online_trainer", backend), \
             patch.object(server, "ASR_QUANTIZATION", None), \
             patch.object(server, "load_model", load_test_inference), \
             TestClient(server.app) as client:
            for pairs, steps in ((1, 3), (2, 2)):
                path = "/v1/audio/training" if pairs == 1 else "/v1/audio/training/batch"
                if pairs == 1:
                    response = client.post(path, files={"file": audio_file()}, data={
                        "transcription": "synthetic fixture", "steps": str(steps), "learning_rate": "0.1",
                    })
                else:
                    response = client.post(path, files=[
                        ("files", audio_file()), ("transcriptions", (None, "first synthetic fixture")),
                        ("files", audio_file()), ("transcriptions", (None, "second synthetic fixture")),
                    ], data={"steps": str(steps), "batch_size": "2", "learning_rate": "0.1"})
                if response.status_code != 200:
                    raise RuntimeError(f"Training API returned {response.status_code}: {response.text}")
                result = response.json()
                assert result["steps_completed"] == steps and result["examples"] == pairs
                assert result["inference_reloaded"] is True
                run_dir = backend.directory / "runs" / result["run_id"]
                files = api.list_repo_files(repo_id, revision=result["commit_sha"])
                hashes = {}
                for name in ("model.nemo", "trainer.ckpt", "metadata.json"):
                    remote = f"runs/{result['run_id']}/{name}"
                    assert remote in files
                    downloaded = hf_hub_download(repo_id, remote, revision=result["commit_sha"], force_download=True)
                    assert sha256(downloaded) == sha256(run_dir / name)
                    hashes[name] = sha256(downloaded)
                pointer = hf_hub_download(repo_id, "latest.json", revision=result["commit_sha"], force_download=True)
                assert json.loads(Path(pointer).read_text())["run_id"] == result["run_id"]
                checkpoint = torch.load(run_dir / "trainer.ckpt", map_location="cpu", weights_only=False)
                assert checkpoint["global_step"] == steps
                assert checkpoint["optimizer_states"][0]["state"][0]["step"].item() == steps
                weight = torch.load(run_dir / "model.nemo", map_location="cpu", weights_only=True)["weight"].item()
                assert weight > previous_weight
                previous_weight = weight
                inference = client.post("/v1/audio/transcriptions", files={"file": audio_file()})
                assert inference.status_code == 200 and str(weight) in inference.json()["text"]
                assert not any(name.endswith((".wav", ".jsonl")) for name in files)
                results.append({"endpoint": path, "run_id": result["run_id"],
                                "steps_completed": steps, "examples": pairs, "weight": weight,
                                "commit_sha": result["commit_sha"], "commit_url": result["commit_url"],
                                "sha256": hashes})

            # A fresh server/cache resolves the latest weights directly from the Hub.
            with patch.dict(os.environ, {"TRAINING_CHECKPOINT_DIR": str(Path(temp) / "fresh-cache")}):
                restored = OnlineTrainer().current_checkpoint()
            assert sha256(restored) == results[-1]["sha256"]["model.nemo"]

    assert api.model_info(repo_id).private is True
    try:
        HfApi(token=False).model_info(repo_id)
    except RepositoryNotFoundError as exc:
        assert exc.response is not None and exc.response.status_code in (401, 404)
    else:
        raise AssertionError("Anonymous access to the private model repo was allowed.")
    return {"tested_at": datetime.now(timezone.utc).isoformat(), "repo_id": repo_id,
            "private": True, "anonymous_access_denied": True, "hub_resume_verified": True,
            "checkpoint_format": "pytorch-test-fixture", "runs": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", help="Dedicated private model test repo, e.g. owner/parakeet-server-hub-smoke-test")
    report = run(parser.parse_args().repo_id)
    print(json.dumps(report, indent=2))
