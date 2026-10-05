"""CPU-only stand-in for exercising the NeMo training contract in tests.

Its model.nemo file is a PyTorch state dict, not a NeMo archive. It must never
be used as an ASR model or mixed with real Parakeet checkpoints.
"""

import json

import lightning.pytorch as pl
import soundfile as sf
import torch
from omegaconf import OmegaConf


class TinyASR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.cfg = OmegaConf.create({"preprocessor": {"sample_rate": 16000}})

    def set_trainer(self, trainer):
        self.trainer = trainer

    def unfreeze(self):
        self.requires_grad_(True)

    def setup_training_data(self, config):
        with open(config.manifest_filepath) as manifest:
            rows = [json.loads(row) for row in manifest]
        for row in rows:
            audio, sample_rate = sf.read(row["audio_filepath"])
            assert sample_rate == config.sample_rate
            assert len(audio) / sample_rate == row["duration"]
            assert row["text"].strip()
        self.loader = torch.utils.data.DataLoader(
            [torch.tensor(1.0) for _ in rows], batch_size=config.batch_size,
        )

    def setup_optimization(self, config):
        self.optim_config = config

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.optim_config.lr,
                                weight_decay=self.optim_config.weight_decay)

    def train_dataloader(self):
        return self.loader

    def training_step(self, batch, batch_idx):
        return (self.weight - batch.mean()) ** 2

    def save_to(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def restore_from(cls, path, **kwargs):
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model
