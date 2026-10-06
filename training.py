"""Request-scoped NeMo fine-tuning and durable, private Hub checkpoints.

All methods that touch the model run on the server's single GPU worker.
"""

import gc
import hashlib
import io
import json
import logging
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    waveform: np.ndarray
    sample_rate: int
    transcription: str


def decode_example(audio: bytes, transcription: str) -> TrainingExample:
    if not transcription.strip():
        raise HTTPException(400, "Transcription must not be blank.")
    try:
        waveform, sample_rate = sf.read(io.BytesIO(audio), dtype="float32")
    except (sf.LibsndfileError, ValueError) as exc:
        raise HTTPException(400, "Invalid or unsupported audio file.") from exc
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise HTTPException(400, "Audio must contain finite, non-empty samples.")
    return TrainingExample(waveform, sample_rate, transcription.strip())


def write_manifest(examples: list[TrainingExample], directory: Path, sample_rate: int) -> Path:
    """Normalize uploads to the model's sample rate; never trust upload paths."""
    manifest = directory / "manifest.jsonl"
    with manifest.open("w") as output:
        for index, example in enumerate(examples):
            audio = example.waveform
            if example.sample_rate != sample_rate:
                divisor = np.gcd(example.sample_rate, sample_rate)
                audio = resample_poly(audio, sample_rate // divisor, example.sample_rate // divisor)
            path = directory / f"{index}.wav"
            sf.write(path, audio, sample_rate, subtype="FLOAT")
            output.write(json.dumps({
                "audio_filepath": str(path),
                "duration": len(audio) / sample_rate,
                "text": example.transcription,
            }, ensure_ascii=False) + "\n")
    return manifest


class OnlineTrainer:
    def __init__(self):
        self.repo_id = os.getenv("TRAINING_REPO_ID", "")
        root = Path(os.getenv("TRAINING_CHECKPOINT_DIR", "/root/.cache/huggingface/training"))
        # Isolate local state when the configured destination repo changes.
        self.directory = root / hashlib.sha256(self.repo_id.encode()).hexdigest()[:16]
        self.max_examples = int(os.getenv("TRAINING_MAX_EXAMPLES", "64"))
        self.max_steps = int(os.getenv("TRAINING_MAX_STEPS", "100"))
        self.max_audio_bytes = int(os.getenv("TRAINING_MAX_AUDIO_BYTES", str(64 * 1024**2)))
        self._checkpoint: Path | None = None
        self._resolved = False
        self.status = {"state": "idle", "last_run": None}

    def check_available(self):
        if not self.repo_id:
            raise HTTPException(503, "Set TRAINING_REPO_ID to a private Hugging Face model repo.")

    def current_checkpoint(self) -> Path | None:
        """Resolve the Hub's latest published weights once per server instance."""
        if not self.repo_id or self._resolved:
            return self._checkpoint

        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError

        try:
            info = HfApi().model_info(self.repo_id)
        except RepositoryNotFoundError:
            # New destination repos are created by the first training request.
            # The Hub can return 401 for a repo that does not exist yet. Access
            # and visibility are checked again before accepting model updates.
            logger.warning("Training repo unavailable; serving the base model until it is accessible.")
            self._resolved = True
            return None
        if not info.private:
            raise RuntimeError("TRAINING_REPO_ID must be private.")
        try:
            latest = hf_hub_download(self.repo_id, "latest.json", revision=info.sha)
        except LocalEntryNotFoundError:
            raise  # A network/cache failure is not an empty training repository.
        except EntryNotFoundError:
            logger.info("No published checkpoint in %s; starting from the base model.", self.repo_id)
            self._resolved = True
            return None
        metadata = json.loads(Path(latest).read_text())
        # Consult the Hub even with a persistent volume: another server may
        # have published newer weights since this instance last ran.
        checkpoint = self.directory / metadata["checkpoint"]
        pointer = self.directory / "latest.json"
        if (pointer.is_file() and checkpoint.is_file()
                and json.loads(pointer.read_text()).get("checkpoint") == metadata["checkpoint"]):
            self._checkpoint = checkpoint
        else:
            self._checkpoint = Path(hf_hub_download(
                self.repo_id, metadata["checkpoint"], revision=info.sha,
            ))
        logger.info("Resuming published checkpoint %s from %s at %s.",
                    metadata["checkpoint"], self.repo_id, info.sha)
        self._resolved = True
        return self._checkpoint

    def _private_repo(self):
        from huggingface_hub import HfApi, get_token

        if not get_token():
            raise HTTPException(503, "Training requires HF_TOKEN (or an authenticated Hugging Face login).")
        api = HfApi()
        api.create_repo(repo_id=self.repo_id, repo_type="model", private=True, exist_ok=True)
        info = api.model_info(self.repo_id)
        # create_repo(private=True) does not change an existing repo's visibility.
        if not info.private:
            raise HTTPException(409, "TRAINING_REPO_ID is public; use a private model repo.")
        return api, info.sha

    def _fit(self, source: Path, examples: list[TrainingExample], run_dir: Path,
             steps: int, batch_size: int, learning_rate: float) -> dict:
        import torch
        from lightning.pytorch import Callback, Trainer
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf, open_dict

        class LossMonitor(Callback):
            last_loss = None

            def on_before_backward(self, trainer, pl_module, loss):
                if not torch.isfinite(loss).all():
                    raise RuntimeError("Non-finite training loss; checkpoint will not be published.")
                self.last_loss = float(loss.detach().cpu())

        model = trainer = None
        try:
            device = os.getenv("TRAINING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            if device not in ("cpu", "cuda"):
                raise RuntimeError("TRAINING_DEVICE must be cpu or cuda.")
            monitor = LossMonitor()
            trainer = Trainer(
                accelerator="gpu" if device == "cuda" else "cpu", devices=1,
                max_steps=steps, max_epochs=-1,
                precision=os.getenv("TRAINING_PRECISION", "32-true"),
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
                gradient_clip_val=1.0, log_every_n_steps=steps + 1, callbacks=[monitor],
                default_root_dir=str(run_dir),
            )
            model = ASRModel.restore_from(str(source), map_location="cpu")
            model.set_trainer(trainer)
            model.unfreeze()
            sample_rate = int(model.cfg.preprocessor.sample_rate)
            with tempfile.TemporaryDirectory(prefix="parakeet-training-") as temp:
                manifest = write_manifest(examples, Path(temp), sample_rate)
                model.setup_training_data(OmegaConf.create({
                    "manifest_filepath": str(manifest), "sample_rate": sample_rate,
                    "batch_size": min(batch_size, len(examples)), "shuffle": True,
                    "num_workers": 0, "pin_memory": device == "cuda", "drop_last": False,
                    "is_tarred": False, "use_lhotse": False,
                    "use_start_end_token": False,
                    "min_duration": None, "max_duration": None,
                }))
                # Each request uses a fresh optimizer with a constant learning rate.
                model.setup_optimization(OmegaConf.create({
                    "name": "adamw", "lr": learning_rate, "weight_decay": 0.0,
                }))
                model.train()
                trainer.fit(model)
                if trainer.global_step != steps:
                    raise RuntimeError(f"Training completed {trainer.global_step} steps; expected {steps}.")
                # Do not persist request-specific manifest paths in model config.
                with open_dict(model.cfg):
                    model.cfg.train_ds = None
                    model.cfg.validation_ds = None
                    model.cfg.test_ds = None
                trainer.save_checkpoint(str(run_dir / "trainer.pending.ckpt"))
                (run_dir / "trainer.pending.ckpt").replace(run_dir / "trainer.ckpt")
                model.eval()
                model.save_to(str(run_dir / "model.pending.nemo"))
                (run_dir / "model.pending.nemo").replace(run_dir / "model.nemo")
            return {"steps_completed": trainer.global_step, "loss": monitor.last_loss}
        finally:
            del model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self, examples: list[TrainingExample], steps: int, batch_size: int,
            learning_rate: float, base_repo: str, base_filename: str) -> dict:
        self.check_available()
        run_id = uuid.uuid4().hex
        run_dir = self.directory / "runs" / run_id
        self.status = {"state": "training", "run_id": run_id, "last_run": self.status["last_run"]}
        stage = "initialization"
        commit = None
        start = time.monotonic()
        try:
            api, parent_commit = self._private_repo()
            if self._checkpoint is None:
                self._resolved = False
            source = self.current_checkpoint()
            if source is None:
                from huggingface_hub import hf_hub_download
                source = Path(hf_hub_download(base_repo, base_filename))
            run_dir.mkdir(parents=True)
            stage = "training"
            metrics = self._fit(source, examples, run_dir, steps, batch_size, learning_rate)
            metadata = {
                "run_id": run_id, "repo_id": self.repo_id,
                "checkpoint": f"runs/{run_id}/model.nemo",
                "trainer_checkpoint": f"runs/{run_id}/trainer.ckpt",
                "examples": len(examples), "steps": steps, "batch_size": batch_size,
                "learning_rate": learning_rate, **metrics,
            }
            payload = json.dumps(metadata, indent=2).encode()
            (run_dir / "metadata.json").write_bytes(payload)
            stage = "upload"
            from huggingface_hub import CommitOperationAdd
            commit = api.create_commit(
                repo_id=self.repo_id, repo_type="model", parent_commit=parent_commit,
                commit_message=f"Online training {run_id}: {steps} steps, {len(examples)} examples",
                operations=[
                    CommitOperationAdd(path_in_repo=f"runs/{run_id}/{name}", path_or_fileobj=str(run_dir / name))
                    for name in ("model.nemo", "trainer.ckpt", "metadata.json")
                ] + [CommitOperationAdd(path_in_repo="latest.json", path_or_fileobj=payload)],
            )
            # Only published runs advance the weights used for subsequent requests.
            self._checkpoint = run_dir / "model.nemo"
            self._resolved = True
            stage = "local_pointer"
            temporary = self.directory / "latest.json.tmp"
            temporary.write_bytes(payload)
            temporary.replace(self.directory / "latest.json")
            result = {**metadata, "commit_sha": commit.oid, "commit_url": commit.commit_url,
                      "training_time": round(time.monotonic() - start, 3)}
            self.status = {"state": "idle", "last_run": result}
            return result
        except HTTPException:
            self.status = {**self.status, "state": "failed", "stage": stage}
            raise
        except Exception as exc:
            logger.exception("Training run %s failed during %s", run_id, stage)
            saved = (run_dir / "model.nemo").is_file()
            self.status = {**self.status, "state": "failed", "stage": stage,
                           "checkpoint_saved": saved}
            raise HTTPException(502 if stage == "upload" else 500, detail={
                "message": f"Training run failed during {stage}; check server logs.",
                "run_id": run_id, "stage": stage, "checkpoint_saved": saved,
                "local_checkpoint": str(run_dir / "model.nemo") if saved else None,
                "checkpoint_published": commit is not None,
                "commit_sha": commit.oid if commit is not None else None,
            }) from exc
