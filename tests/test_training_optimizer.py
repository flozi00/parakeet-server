"""Exercise the real Lightning optimizer loop with a tiny NeMo-compatible model.

This checks optimizer updates, step counts across epochs, and checkpoint state;
it does not test Parakeet accuracy or NeMo's GPU TDT loss implementation.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from training import OnlineTrainer, TrainingExample


def test_real_optimizer_steps_and_checkpoint_roundtrip(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightning.pytorch")
    pytest.importorskip("omegaconf")
    from tiny_asr import TinyASR
    monkeypatch.setenv("TRAINING_DEVICE", "cpu")
    monkeypatch.setenv("TRAINING_PRECISION", "32-true")

    model = TinyASR()
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", SimpleNamespace(
        ASRModel=SimpleNamespace(restore_from=lambda *args, **kwargs: model),
    ))
    source = tmp_path / "base.nemo"
    source.write_bytes(b"placeholder")
    examples = [TrainingExample(np.zeros(8000, dtype=np.float32), 8000, "hello")]
    metrics = OnlineTrainer()._fit(source, examples, tmp_path, 3, 8, 0.1)
    # One example means three optimizer steps must span three epochs.
    assert metrics["steps_completed"] == 3 and metrics["loss"] < 1
    state = torch.load(tmp_path / "model.nemo", weights_only=True)
    assert 0 < state["weight"].item() < 1
    checkpoint = torch.load(tmp_path / "trainer.ckpt", weights_only=False)
    assert checkpoint["global_step"] == 3
    assert checkpoint["optimizer_states"][0]["state"][0]["step"].item() == 3
    torch.testing.assert_close(checkpoint["state_dict"]["weight"], state["weight"])
