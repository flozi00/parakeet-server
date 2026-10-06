"""Stream YODAS3's WAV preview, check captions with ASR, and train passing pairs.

Uses Python's standard library only; no pip install or external tools required.
The preview has limited recordings and audio duration, not the full WebM corpus.
Start a training-enabled server; see README.md. JSONL results go to stdout.
"""

import argparse
import io
import json
import math
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from itertools import islice
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4
import wave

DATASET = "espnet/yodas3"
VIEWER = "https://datasets-server.huggingface.co"


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str


def emit(**result):
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def normalize(text):
    """Ignore case, punctuation and spacing while preserving letters/numbers."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("'", "").replace("’", "")
    return " ".join("".join(
        char if unicodedata.category(char)[0] in "LNM" else " "
        for char in text
    ).split())


def word_error_rate(reference, hypothesis):
    """Levenshtein word edits / reference words; insertions can make WER > 1."""
    reference, hypothesis = normalize(reference).split(), normalize(hypothesis).split()
    if not reference:
        raise ValueError("Reference has no words after normalization.")
    previous = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, predicted in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (word != predicted)))
        previous = current
    return previous[-1] / len(reference)


def caption_segments(transcript, min_seconds, max_seconds, min_words):
    """YODAS caption start/duration are milliseconds, not seconds."""
    if not transcript:
        return []
    captions = json.loads(transcript)
    if not isinstance(captions, list):
        raise ValueError("Transcript must be a JSON list of captions.")
    parsed = []
    for index, caption in enumerate(captions):
        try:
            start = float(caption["start"]) / 1000
            duration = float(caption["duration"]) / 1000
            text = caption["text"].strip()
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
        if (math.isfinite(start) and math.isfinite(duration) and start >= 0
                and duration > 0 and math.isfinite(start + duration)):
            parsed.append(Segment(index, start, start + duration, text))
    parsed.sort(key=lambda segment: segment.start)
    segments = []
    for i, segment in enumerate(parsed):
        # YouTube auto-captions often stay visible while the next caption is
        # spoken. End at the next caption's start to avoid including its words.
        # Keep even short/wordless captions as boundaries before filtering.
        if i > 0 and parsed[i - 1].start == segment.start:
            continue
        end = segment.end
        if i + 1 < len(parsed):
            if parsed[i + 1].start == segment.start:
                continue  # Simultaneous caption tracks are ambiguous.
            end = min(end, parsed[i + 1].start)
        if (segment.end - segment.start <= max_seconds
                and min_seconds <= end - segment.start <= max_seconds
                and len(normalize(segment.text).split()) >= min_words):
            segments.append(Segment(segment.index, segment.start, end, segment.text))
    return segments


def request_json(url, *, fields=None, audio=None, timeout=60):
    """GET JSON or POST one multipart WAV using only urllib."""
    headers = {"Accept": "application/json"}
    body = None
    if fields is not None:
        boundary = uuid4().hex
        parts = []
        for name, value in fields.items():
            parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                          f'\r\n\r\n{value}\r\n').encode("utf-8"))
        if audio is not None:
            parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                          'filename="segment.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()
                         + audio + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        body = b"".join(parts)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    method = "GET" if fields is None else "POST"
    parsed = urlsplit(url)
    endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    # Dataset Viewer occasionally answers with a transient 500. Retry only
    # read-only requests; never repeat a POST that might have accepted training.
    attempts = 4 if method == "GET" else 1
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, data=body, headers=headers), timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            with exc:
                detail = exc.read(4096).decode("utf-8", errors="replace")
            message = f"{method} {endpoint}: HTTP {exc.code}: {detail}"
            if attempt + 1 == attempts or exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(message) from exc
        except OSError as exc:
            message = f"{method} {endpoint}: {exc}"
            if attempt + 1 == attempts:
                raise RuntimeError(message) from exc
        delay = 2 ** attempt
        print(f"{message}; retrying in {delay}s ({attempt + 2}/{attempts}).",
              file=sys.stderr, flush=True)
        time.sleep(delay)


def stream_recordings(language, shards, offset, timeout):
    """Page preview metadata as JSON; download audio only for matching rows."""
    while True:
        url = f"{VIEWER}/rows?" + urlencode({
            "dataset": DATASET, "config": "preview", "split": "train",
            "offset": offset, "length": 10,
        })
        page = request_json(url, timeout=min(timeout, 60))
        entries = page["rows"]
        if not entries:
            if offset < page["num_rows_total"]:
                raise RuntimeError("Dataset Viewer returned an incomplete page; try again later.")
            return
        for entry in entries:
            row = entry["row"]
            if row.get("lang") != language or (shards and row.get("shard") not in shards):
                continue
            if "transcript" in entry.get("truncated_cells", []):
                emit(event="skip_recording", id=row["id"], reason="truncated_transcript")
                continue
            yield row
        offset += len(entries)
        if offset >= page["num_rows_total"]:
            return


def crop_wav(reader, segment):
    """Read a caption from a forward-only PCM WAV stream; preserve its format."""
    rate = reader.getframerate()
    start, end = round(segment.start * rate), round(segment.end * rate)
    if start < reader.tell() or end <= start or end > reader.getnframes():
        raise ValueError("Caption window is outside the WAV preview or out of order.")
    frame_bytes = reader.getnchannels() * reader.getsampwidth()
    # Discard gaps in bounded chunks: HTTP responses cannot seek backwards.
    while reader.tell() < start:
        count = min(start - reader.tell(), 65536)
        if len(reader.readframes(count)) != count * frame_bytes:
            raise ValueError("WAV preview ended before the caption.")
    frames = reader.readframes(end - start)
    if len(frames) != (end - start) * frame_bytes:
        raise ValueError("WAV preview ended within the caption.")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(reader.getnchannels())
        writer.setsampwidth(reader.getsampwidth())
        writer.setframerate(rate)
        writer.writeframes(frames)
    return output.getvalue()


def check_and_train(audio, reference, args, context):
    """Train with the dataset caption and exactly the audio used for checking."""
    response = request_json(f"{args.server_url.rstrip('/')}/audio/transcriptions",
                            audio=audio, fields={"response_format": "json"}, timeout=args.timeout)
    hypothesis = response["text"]
    if not isinstance(hypothesis, str):
        raise ValueError("Transcription response text must be a string.")
    wer = word_error_rate(reference, hypothesis)
    passed = bool(normalize(hypothesis)) and wer <= args.max_wer
    emit(event="quality", **context, reference=reference, hypothesis=hypothesis,
         wer=wer, max_wer=args.max_wer, quality_passed=passed, dry_run=args.dry_run)
    if not passed or args.dry_run:
        return passed, False

    # No retries: a timeout/disconnect does not cancel a server training run,
    # and retrying could train the same pair twice.
    try:
        result = request_json(f"{args.server_url.rstrip('/')}/audio/training", audio=audio,
                              fields={"transcription": reference, "steps": args.steps,
                                      "learning_rate": args.learning_rate}, timeout=args.timeout)
        if result.get("inference_reloaded") is not True:
            raise ValueError("Training response did not confirm inference reload.")
    except (OSError, RuntimeError, ValueError) as exc:
        emit(event="training_error", **context, error=str(exc))
        raise RuntimeError(f"Training did not confirm success: {exc}. Inspect server /health and "
                           "the Hub checkpoint before rerunning; an accepted run may continue.") from exc
    emit(event="trained", **context, result=result)
    return True, True


def run(args):
    checked = accepted = trained = 0
    emit(event="source", dataset=DATASET, config="preview", language=args.language,
         offset=args.offset, dry_run=args.dry_run)
    for row in islice(stream_recordings(args.language, args.shards, args.offset, args.timeout),
                      args.max_recordings):
        context = {"id": row["id"], "shard": row["shard"]}
        try:
            segments = caption_segments(row.get("transcript"), args.min_seconds,
                                        args.max_seconds, args.min_words)
        except (ValueError, TypeError) as exc:
            emit(event="skip_recording", **context, reason=str(exc))
            continue
        if not segments or not row.get("audio"):
            emit(event="skip_recording", **context, reason="no_eligible_captions_or_audio")
            continue
        source = next((audio["src"] for audio in row["audio"]
                       if audio.get("type") == "audio/wav"), None)
        if not source:
            emit(event="skip_recording", **context, reason="no_wav_preview")
            continue
        # Finish the HTTP download before slow checkpoint/export operations.
        # A temporary file keeps one short preview on disk, not in RAM, and is
        # removed even if transcription or training fails.
        with tempfile.TemporaryFile() as recording:
            with urlopen(source, timeout=args.timeout) as response:
                shutil.copyfileobj(response, recording, length=65536)
            recording.seek(0)
            try:
                reader = wave.open(recording, "rb")
            except (wave.Error, EOFError) as exc:
                emit(event="skip_recording", **context, reason=str(exc))
                continue
            with reader:
                for segment in segments:
                    if checked >= args.max_segments:
                        break
                    segment_context = {**context, "segment": segment.index,
                                       "start": segment.start, "end": segment.end}
                    if round(segment.end * reader.getframerate()) > reader.getnframes():
                        emit(event="skip_segment", **segment_context, reason="outside_wav_preview")
                        break
                    try:
                        audio = crop_wav(reader, segment)
                    except (ValueError, wave.Error, EOFError) as exc:
                        emit(event="skip_segment", **segment_context, reason=str(exc))
                        break
                    passed, updated = check_and_train(audio, segment.text, args, segment_context)
                    checked += 1
                    accepted += passed
                    trained += updated
        if checked >= args.max_segments:
            break
    emit(event="summary", checked=checked, accepted=accepted, trained=trained,
         dry_run=args.dry_run)
    if checked == 0:
        print("No caption audio was checked. The WAV preview has few recordings; "
              "try removing --shards or choose another --language/--offset.", file=sys.stderr)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://localhost:8000/v1",
                        help="API base URL, including /v1 if desired")
    parser.add_argument("--language", default="de", help="YODAS language directory (default: de)")
    parser.add_argument("--shards", nargs="+", help="Optional 4-digit shard filter within the preview")
    parser.add_argument("--offset", type=int, default=0, help="Starting Dataset Viewer row offset")
    parser.add_argument("--max-recordings", type=int, default=10)
    parser.add_argument("--max-segments", type=int, default=20, help="Maximum ASR checks")
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-wer", type=float, default=0.15,
                        help="Accept normalized word error rate <= this value (default: 0.15)")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=3600, help="HTTP timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Transcribe and score without training")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z]{2,3}", args.language):
        parser.error("--language must be a YODAS language code such as de or en")
    if args.shards and any(not re.fullmatch(r"\d{4}", shard) for shard in args.shards):
        parser.error("--shards must contain 4-digit IDs such as 0000 0001")
    if min(args.max_recordings, args.max_segments, args.min_words, args.steps) < 1:
        parser.error("recording, segment, word and step limits must be positive")
    if args.offset < 0:
        parser.error("--offset must not be negative")
    if not (math.isfinite(args.min_seconds) and math.isfinite(args.max_seconds)
            and 0 < args.min_seconds <= args.max_seconds <= 30):
        parser.error("duration limits must satisfy 0 < min <= max <= 30 seconds")
    if not math.isfinite(args.max_wer) or not 0 <= args.max_wer <= 1:
        parser.error("--max-wer must be in [0, 1]")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1:
        parser.error("--learning-rate must be in (0, 1]")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except (OSError, RuntimeError, ValueError, wave.Error, EOFError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
