"""Request-scoped NeMo fine-tuning and durable, private Hub checkpoints.

All methods that touch the model run on the server's single GPU worker.
"""

import gc
import hashlib
import io
import json
import logging
import os
import re
import shutil
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
        # Every run publishes a 2.5 GB model and a 7.5 GB trainer checkpoint.
        # Only the newest trainer checkpoint can be resumed from, so older ones
        # are deleted on publish; model weights are kept for the last N runs so
        # a bad update can still be rolled back.
        self.keep_runs = max(1, int(os.getenv("TRAINING_KEEP_RUNS", "5")))
        self._checkpoint: Path | None = None
        self._metadata: dict | None = None
        self._revision: str | None = None
        self._resolved = False
        self.status = {"state": "idle", "last_run": None}

    def check_available(self):
        if not self.repo_id:
            raise HTTPException(503, "Set TRAINING_REPO_ID to a private Hugging Face model repo.")

    def check_token(self) -> dict:
        """Report at startup whether HF_TOKEN can still write to the repo.

        An `hf auth login` OAuth token expires after ~30 days. Training then
        fails on its first Hub call, which is how two days of requests were
        lost without a single line in the log until the request arrived.
        """
        if not self.repo_id:
            return {"configured": False}
        from huggingface_hub import HfApi, get_token

        if not get_token():
            logger.warning("TRAINING_REPO_ID is set but no HF token is available; "
                           "every training request will fail.")
            return {"configured": True, "token": False}
        try:
            auth = (HfApi().whoami() or {}).get("auth") or {}
        except Exception as exc:
            logger.warning("HF token check failed (%s); training may fail.", exc)
            return {"configured": True, "token": None}
        expires = auth.get("expiresAt")
        if expires:
            logger.warning(
                "HF_TOKEN is an OAuth token that expires at %s. Replace it with a "
                "fine-grained write token (no expiry) before then, or training stops.",
                expires)
        else:
            logger.info("HF token is a long-lived access token; training can publish.")
        return {"configured": True, "token": True, "auth": auth.get("type"),
                "expires_at": expires}

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
        self._metadata = metadata
        self._revision = info.sha
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

    def resume_state(self) -> tuple[Path | None, int]:
        """Local path of the published trainer checkpoint and its step count.

        The trainer checkpoint is what makes a request continue the previous
        one: it carries the Adam moments and the cumulative step counter. It is
        ~7.5 GB, so it is fetched once per server instance and only when a
        training request actually needs it.
        """
        if not self.repo_id:
            return None, 0
        self.current_checkpoint()
        metadata = self._metadata or {}
        global_step = int(metadata.get("global_step") or metadata.get("steps_completed") or 0)
        name = metadata.get("trainer_checkpoint")
        if not name:
            # No optimizer state to restore, so the lineage restarts at step 0
            # even though the published weights are further along. Reporting the
            # old step count here would make the trainer run that many extra
            # steps (max_steps is absolute).
            return None, 0
        path = self.directory / name
        if not path.is_file():
            from huggingface_hub import hf_hub_download
            logger.info("Downloading trainer checkpoint %s from %s (optimizer state).",
                        name, self.repo_id)
            try:
                path = Path(hf_hub_download(self.repo_id, name, revision=self._revision))
            except Exception as exc:
                # A pruned or never-uploaded checkpoint is not fatal: the weights
                # are still valid, only the optimizer moments are gone.
                if self._status(exc) != 404:
                    raise
                logger.warning("Published trainer checkpoint %s is missing from %s; "
                               "starting a fresh optimizer at step 0.", name, self.repo_id)
                return None, 0
        return path, global_step

    @staticmethod
    def _status(exc) -> int | None:
        return getattr(getattr(exc, "response", None), "status_code", None)

    def _repo_info(self, api):
        """`model_info`, or None when the destination repo does not exist yet."""
        try:
            return api.model_info(self.repo_id)
        except Exception as exc:
            status = self._status(exc)
            if status == 404:
                return None
            if status in (401, 403):
                # An `hf auth login` OAuth token expires after ~30 days; only a
                # fine-grained/classic write token keeps working unattended.
                raise HTTPException(503, (
                    "HF_TOKEN is expired or has no write access to "
                    f"{self.repo_id}. Replace it with a fine-grained write token "
                    "(no expiry) and restart the pod."
                )) from exc
            raise

    def _private_repo(self):
        from huggingface_hub import HfApi, get_token

        if not get_token():
            raise HTTPException(503, "Training requires HF_TOKEN (or an authenticated Hugging Face login).")
        api = HfApi()
        # Look before creating: a fine-grained token scoped to one repo can
        # write to it but is not allowed to create repos, and create_repo
        # would fail with 403 even though publishing works.
        info = self._repo_info(api)
        if info is None:
            try:
                api.create_repo(repo_id=self.repo_id, repo_type="model", private=True, exist_ok=True)
            except Exception as exc:
                if self._status(exc) in (401, 403):
                    raise HTTPException(503, (
                        f"HF_TOKEN cannot create {self.repo_id}. Replace it with a "
                        "fine-grained write token (no expiry) and restart the pod."
                    )) from exc
                raise
            info = self._repo_info(api)
            if info is None:
                raise HTTPException(503, f"Training repo {self.repo_id} is not accessible.")
        # create_repo(private=True) does not change an existing repo's visibility.
        if not info.private:
            raise HTTPException(409, "TRAINING_REPO_ID is public; use a private model repo.")
        return api, info.sha

    def _fit(self, source: Path, examples: list[TrainingExample], run_dir: Path,
             steps: int, batch_size: int, learning_rate: float,
             resume: Path | None = None, start_step: int = 0) -> dict:
        import torch
        from lightning.pytorch import Callback, Trainer
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf, open_dict

        class LossMonitor(Callback):
            def __init__(self):
                self.losses: list[float] = []

            def on_before_backward(self, trainer, pl_module, loss):
                if not torch.isfinite(loss).all():
                    raise RuntimeError("Non-finite training loss; checkpoint will not be published.")
                value = float(loss.detach().cpu())
                self.losses.append(value)
                logger.info("Training step %d/%d: loss %.6f",
                            trainer.global_step + 1, trainer.max_steps, value)

        model = trainer = None
        total_steps = start_step + steps
        try:
            device = os.getenv("TRAINING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            if device not in ("cpu", "cuda"):
                raise RuntimeError("TRAINING_DEVICE must be cpu or cuda.")
            monitor = LossMonitor()
            trainer = Trainer(
                accelerator="gpu" if device == "cuda" else "cpu", devices=1,
                max_steps=total_steps, max_epochs=-1,
                precision=os.getenv("TRAINING_PRECISION", "32-true"),
                logger=False, enable_checkpointing=False, enable_progress_bar=False,
                enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
                gradient_clip_val=1.0, log_every_n_steps=1, callbacks=[monitor],
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
                model.setup_optimization(OmegaConf.create({
                    "name": "adamw", "lr": learning_rate, "weight_decay": 0.0,
                }))
                model.train()
                # Resume the published trainer checkpoint: it carries the Adam
                # moments and the cumulative step counter, so this request
                # continues the previous one. A cold AdamW step is ~lr of sign
                # noise and never accumulates, which is why the loss never fell.
                trainer.fit(model, ckpt_path=str(resume) if resume else None)
                if trainer.global_step != total_steps:
                    raise RuntimeError(
                        f"Training completed {trainer.global_step} steps; expected {total_steps}.")
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
                # Best effort, and only after both artifacts are on disk: a
                # probe that cannot run must never cost a finished run.
                loss_after = self._probe_loss(model, trainer, manifest, sample_rate, batch_size)
            return {
                "steps_completed": steps,
                "global_step": trainer.global_step,
                "resumed_from_step": start_step,
                "loss": monitor.losses[-1] if monitor.losses else None,
                "loss_before": monitor.losses[0] if monitor.losses else None,
                "loss_after": loss_after,
            }
        finally:
            del model, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _probe_loss(model, trainer, manifest: Path, sample_rate: int,
                    batch_size: int) -> float | None:
        """Loss on this request's own examples after the update (best effort).

        The first training step's loss is measured before the update, so the
        pair is the only per-request evidence that the step helped. NeMo's
        validation loop is the documented way to score a batch; a model or
        Lightning version that cannot run it must not fail the run.
        """
        from omegaconf import OmegaConf

        try:
            model.setup_validation_data(OmegaConf.create({
                "manifest_filepath": str(manifest), "sample_rate": sample_rate,
                "batch_size": max(1, batch_size), "shuffle": False,
                "num_workers": 0, "pin_memory": False, "drop_last": False,
                "is_tarred": False, "use_lhotse": False,
                "use_start_end_token": False,
                "min_duration": None, "max_duration": None,
            }))
            limit = trainer.limit_val_batches
            trainer.limit_val_batches = 1
            try:
                results = trainer.validate(model, verbose=False)
            finally:
                trainer.limit_val_batches = limit
            values = []
            for result in results:
                value = result.get("val_loss")
                if value is None:
                    continue
                values.append(float(value.detach().cpu()) if hasattr(value, "detach") else float(value))
            return sum(values) / len(values) if values else None
        except Exception:
            logger.warning("Post-training loss probe unavailable; reporting the training loss only.",
                           exc_info=True)
            return None

    def _prune_operations(self, api, run_id: str) -> list:
        """Delete superseded trainer checkpoints and runs beyond TRAINING_KEEP_RUNS.

        Every run used to add a 7.5 GB trainer checkpoint that nothing ever read
        back, so the repository grew by ~10 GB per request. Only the newest
        trainer checkpoint can be resumed from; older ones are dead weight.
        Model weights are kept for the last `keep_runs` runs for rollback.
        """
        from huggingface_hub import CommitOperationDelete

        operations = []
        previous = (self._metadata or {}).get("trainer_checkpoint")
        try:
            files = set(api.list_repo_files(self.repo_id, repo_type="model"))
            commits = api.list_repo_commits(self.repo_id, repo_type="model")
        except Exception:
            logger.warning("Could not list %s to prune old runs; keeping them.",
                           self.repo_id, exc_info=True)
            files, commits = None, []
        published = []
        for entry in commits:  # newest first
            match = re.match(r"Online training ([0-9a-f]{32}):", entry.title or "")
            if match and match.group(1) not in published:
                published.append(match.group(1))
        # Only the newest trainer checkpoint can be resumed from.
        checkpoints = {f"runs/{stale}/trainer.ckpt" for stale in published}
        if previous:
            checkpoints.add(previous)
        checkpoints.discard(f"runs/{run_id}/trainer.ckpt")
        operations += [CommitOperationDelete(path_in_repo=path)
                       for path in sorted(checkpoints)
                       if files is None or path in files]
        for stale in published[max(0, self.keep_runs - 1):]:
            operations += [
                CommitOperationDelete(path_in_repo=path)
                for path in (f"runs/{stale}/{name}" for name in ("model.nemo", "metadata.json"))
                if files is None or path in files
            ]
        return operations

    def _cleanup_local_runs(self, keep: Path) -> None:
        """Drop superseded local run directories; the Hub keeps the history."""
        for path in (self.directory / "runs").glob("*"):
            if path != keep:
                shutil.rmtree(path, ignore_errors=True)

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
            if self._checkpoint is None or not self._checkpoint.is_file():
                self._resolved = False
            source = self.current_checkpoint()
            if source is None:
                from huggingface_hub import hf_hub_download
                source = Path(hf_hub_download(base_repo, base_filename))
            resume, start_step = self.resume_state()
            run_dir.mkdir(parents=True)
            stage = "training"
            metrics = self._fit(source, examples, run_dir, steps, batch_size, learning_rate,
                                resume=resume, start_step=start_step)
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
                commit_message=(f"Online training {run_id}: steps {start_step}->"
                                f"{metadata['global_step']}, {len(examples)} examples"),
                operations=[
                    CommitOperationAdd(path_in_repo=f"runs/{run_id}/{name}", path_or_fileobj=str(run_dir / name))
                    for name in ("model.nemo", "trainer.ckpt", "metadata.json")
                ] + [CommitOperationAdd(path_in_repo="latest.json", path_or_fileobj=payload)]
                + self._prune_operations(api, run_id),
            )
            # Only published runs advance the weights used for subsequent requests.
            self._checkpoint = run_dir / "model.nemo"
            self._metadata = metadata
            self._resolved = True
            stage = "local_pointer"
            temporary = self.directory / "latest.json.tmp"
            temporary.write_bytes(payload)
            temporary.replace(self.directory / "latest.json")
            self._cleanup_local_runs(run_dir)
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
