"""Optional Nemotron diarization, executed only on the serialized GPU worker."""

import math
import os

import numpy as np


class NemotronDiarizer:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            import torch
            from nemo.collections.asr.models import SortformerEncLabelModel

            device = os.getenv("DIARIZATION_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
            model = SortformerEncLabelModel.from_pretrained(
                os.getenv("DIARIZATION_MODEL_NAME", "nvidia/Nemotron-3-Diarization"),
                map_location=device,
            )
            model.eval()
            # NVIDIA's offline-quality chunked configuration. The speaker cache
            # is maintained across chunks within one diarize() call.
            model.sortformer_modules.spkcache_len = 264
            model.sortformer_modules.chunk_len = 340
            model.sortformer_modules.chunk_right_context = 40
            model.sortformer_modules.fifo_len = 40
            model.sortformer_modules.spkcache_update_period = 300
            model._check_streaming_parameters()
            self.model = model
        return self.model

    def diarize(self, waveform, sample_rate):
        import torch
        from scipy.signal import resample_poly

        if sample_rate != 16000:
            divisor = math.gcd(sample_rate, 16000)
            waveform = resample_poly(waveform, 16000 // divisor, sample_rate // divisor)
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        model = self.load()
        with torch.inference_mode():
            result = model.diarize(audio=[waveform], batch_size=1, sample_rate=16000)
        return normalize_turns(result[0], len(waveform) / 16000)


def normalize_turns(lines, duration):
    """Parse NeMo's 'start end speaker_N' records; retain overlapping turns."""
    turns = []
    for line in lines:
        start, end, speaker = line.split()
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Non-finite diarization timestamp")
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            turns.append((start, end, speaker))
    return sorted(turns, key=lambda turn: (turn[0], turn[1], turn[2]))


def transcribe_turns(waveform, sample_rate, turns, recognize, extract_text):
    """Transcribe each speaker turn with Parakeet, limiting ASR crops to 30 s.

    Crops use the original mono mixture, not source-separated audio. Overlap
    can therefore produce duplicated/misattributed words; see README.
    """
    # Silero VAD (and other frame-based preprocessors) build a sliding window
    # of 512+64 samples; crops shorter than that window crash numpy's
    # sliding_window_view ("window shape cannot be larger than input array
    # shape"). Diarization post-processing routinely emits 10-30 ms
    # micro-turns on long recordings, so drop them before ASR.
    min_crop = (512 + 64) * sample_rate // 16000
    turns = [turn for turn in turns if round(turn[1] * sample_rate) - round(turn[0] * sample_rate) >= min_crop]
    segments, labels = [], {}
    for start, end, speaker in turns:
        label = labels.setdefault(speaker, chr(ord("A") + len(labels)))
        first = max(0, round(start * sample_rate))
        last = min(len(waveform), round(end * sample_rate))
        for left in range(first, last, 30 * sample_rate):
            right = min(left + 30 * sample_rate, last)
            text = extract_text(recognize(waveform[left:right], sample_rate=sample_rate))
            if text.strip():
                segments.append({
                    "id": str(len(segments)),
                    "type": "transcript.text.segment",
                    "speaker": label,
                    "start": left / sample_rate,
                    "end": right / sample_rate,
                    "text": text.strip(),
                })
    segments.sort(key=lambda segment: (segment["start"], segment["end"]))
    for index, segment in enumerate(segments):
        segment["id"] = str(index)
    return {
        "task": "transcribe",
        "duration": len(waveform) / sample_rate,
        "text": " ".join(segment["text"] for segment in segments),
        "segments": segments,
    }
