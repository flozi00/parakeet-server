"""Run without packages: python -S -m unittest discover -s tests -p 'test_stream_yodas3_training.py'."""

import contextlib
import io
import json
import struct
import subprocess
import sys
import threading
import unittest
import wave
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from scripts import stream_yodas3_training as sample


def wav_bytes(seconds=4, rate=100):
    frames = b"".join(struct.pack("<hh", i, -i) for i in range(int(seconds * rate)))
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return output.getvalue(), frames


class ReadOnly:
    """An HTTP-like source that cannot seek or tell."""

    def __init__(self, data):
        self.buffer = io.BytesIO(data)

    def read(self, count=-1):
        return self.buffer.read(count)


@contextlib.contextmanager
def http_server(handler):
    class Handler(BaseHTTPRequestHandler):
        def respond(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            status, content_type, payload = handler(self.path, self.headers, body)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = respond

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def json_reply(value, status=200):
    return status, "application/json", json.dumps(value).encode()


def multipart_fields(headers, body):
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {headers['Content-Type']}\r\n\r\n".encode() + body
    )
    return {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
            for part in message.iter_parts()}


class SampleTests(unittest.TestCase):
    def test_word_error_rate(self):
        for reference, hypothesis, expected in [
            ("Hello, WORLD!", "hello world", 0),
            ("DON’T stop", "dont stop", 0),
            ("Äpfel sind süß.", "äpfel sind süss", 0),
            ("one two three four", "one two wrong four", 0.25),
            ("one two three four", "one two four", 0.25),
            ("one two", "one extra two", 0.5),
            ("one two", "", 1),
            ("one", "one two three", 2),
        ]:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertEqual(sample.word_error_rate(reference, hypothesis), expected)

    def test_caption_boundaries_and_invalid_captions(self):
        captions = json.dumps([
            {"start": 2000, "duration": 2000, "text": "  The later caption.  "},
            {"start": 0, "duration": 4000, "text": "The first caption."},
            {"start": 3500, "duration": 1000, "text": "Short"},
            {"start": 10000, "duration": 500, "text": "Too short caption"},
            {"start": 11000, "duration": 31000, "text": "Too long caption"},
            {"start": -1, "duration": 1000, "text": "Negative start caption"},
            {"start": 43000, "duration": 1000, "text": "?! ..."},
            {"start": "NaN", "duration": 1000, "text": "Not finite caption"},
            {"start": 0, "duration": 1000, "text": None},
        ])
        self.assertEqual(sample.caption_segments(captions, 1, 30, 3), [
            sample.Segment(1, 0, 2, "The first caption."),
            sample.Segment(0, 2, 3.5, "The later caption."),
        ])
        self.assertEqual(sample.caption_segments(None, 1, 30, 3), [])

    def test_simultaneous_captions_are_skipped(self):
        captions = json.dumps([
            {"start": 0, "duration": 1000, "text": "One simultaneous caption"},
            {"start": 0, "duration": 1000, "text": "Another simultaneous caption"},
        ])
        self.assertEqual(sample.caption_segments(captions, 1, 30, 3), [])

    def test_forward_only_wav_cropping_preserves_pcm_and_format(self):
        audio, frames = wav_bytes()
        with wave.open(ReadOnly(audio), "rb") as reader:
            for start in [0.5, 2.5]:
                clip = sample.crop_wav(reader, sample.Segment(0, start, start + 1, "test"))
                with wave.open(io.BytesIO(clip), "rb") as decoded:
                    self.assertEqual((decoded.getnchannels(), decoded.getsampwidth(),
                                      decoded.getframerate(), decoded.getnframes()), (2, 2, 100, 100))
                    self.assertEqual(decoded.readframes(100), frames[int(start * 400):int((start + 1) * 400)])
            with self.assertRaisesRegex(ValueError, "out of order"):
                sample.crop_wav(reader, sample.Segment(0, 0.5, 1.5, "test"))
            with self.assertRaisesRegex(ValueError, "outside"):
                sample.crop_wav(reader, sample.Segment(0, 3.5, 4.5, "test"))

    def test_truncated_wav_never_produces_a_training_pair(self):
        audio, _ = wav_bytes()
        with wave.open(ReadOnly(audio[:-40]), "rb") as reader:
            with self.assertRaisesRegex(ValueError, "ended within"):
                sample.crop_wav(reader, sample.Segment(0, 3, 4, "test"))

    def test_only_passing_pairs_train_with_the_original_reference(self):
        for hypothesis, dry_run, max_wer, expected in [
            ("hello world today", False, 0.15, (True, True)),
            ("Hello, WORLD today!", True, 0.15, (True, False)),
            ("entirely different sentence", False, 0.15, (False, False)),
            ("", False, 1, (False, False)),
            ("hello world tomorrow", False, 1 / 3, (True, True)),
        ]:
            with self.subTest(hypothesis=hypothesis, dry_run=dry_run):
                calls = []

                def handler(path, headers, body):
                    calls.append((path, multipart_fields(headers, body)))
                    return json_reply({"text": hypothesis} if path.endswith("transcriptions")
                                      else {"run_id": "test", "inference_reloaded": True})

                with http_server(handler) as url, contextlib.redirect_stdout(io.StringIO()) as output:
                    args = sample.parse_args(["--server-url", url + "/v1", "--steps", "2",
                                              "--learning-rate", "0.00002"])
                    args.dry_run, args.max_wer = dry_run, max_wer
                    audio, _ = wav_bytes()
                    reference = "Hello, world today!"
                    self.assertEqual(sample.check_and_train(audio, reference, args, {}), expected)
                self.assertEqual(len(calls), 1 + expected[1])
                if expected[1]:
                    self.assertEqual(calls[1][1]["transcription"].decode(), reference)
                    self.assertEqual(calls[0][1]["file"], calls[1][1]["file"])
                    self.assertEqual(calls[1][1]["file"], audio)
                    self.assertEqual(calls[1][1]["steps"], b"2")
                    self.assertEqual(calls[1][1]["learning_rate"], b"2e-05")
                quality = json.loads(output.getvalue().splitlines()[0])
                self.assertEqual(quality["quality_passed"], expected[0])

    def test_training_failure_stops_without_retry(self):
        for failure in [TimeoutError("timeout"), RuntimeError("HTTP 503"),
                        {"inference_reloaded": False}]:
            with self.subTest(failure=failure):
                with patch.object(sample, "request_json", side_effect=[{"text": "hello world today"}, failure]) as request:
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        with self.assertRaisesRegex(RuntimeError, "accepted run may continue") as raised:
                            sample.check_and_train(b"wav", "hello world today", sample.parse_args([]), {})
                    if isinstance(failure, Exception):
                        self.assertIn(str(failure), str(raised.exception))
                    self.assertEqual(request.call_count, 2)
                    self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["event"], "training_error")

    def test_http_error_includes_server_detail(self):
        with http_server(lambda *args: json_reply({"detail": "training unavailable"}, 503)) as url:
            with self.assertRaisesRegex(RuntimeError, "training unavailable"):
                sample.request_json(url, fields={"response_format": "json"}, audio=b"wav")

    def test_metadata_get_retries_temporary_500_and_identifies_service(self):
        calls = []

        def handler(path, headers, body):
            calls.append(path)
            return json_reply({"error": "Unexpected error."}, 500) if len(calls) < 3 else json_reply({"rows": []})

        with http_server(handler) as url, patch.object(sample.time, "sleep") as sleep:
            with contextlib.redirect_stderr(io.StringIO()) as output:
                self.assertEqual(sample.request_json(url + "/rows"), {"rows": []})
        self.assertEqual(calls, ["/rows"] * 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        self.assertIn(f"GET {url}/rows: HTTP 500", output.getvalue())

    def test_metadata_get_stops_after_bounded_retries(self):
        with http_server(lambda *args: json_reply({"error": "Unexpected error."}, 500)) as url:
            with patch.object(sample.time, "sleep") as sleep, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, f"GET {url}/rows: HTTP 500"):
                    sample.request_json(url + "/rows")
        self.assertEqual(sleep.call_count, 3)

    def test_training_post_never_retries_http_500(self):
        calls = []

        def handler(path, headers, body):
            calls.append(path)
            return json_reply({"error": "training failed"}, 500)

        with http_server(handler) as url, patch.object(sample.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, f"POST {url}/audio/training: HTTP 500"):
                sample.request_json(url + "/audio/training", fields={"transcription": "hello"}, audio=b"wav")
        self.assertEqual(calls, ["/audio/training"])
        sleep.assert_not_called()

    def test_pagination_filters_language_shard_and_truncated_transcripts(self):
        pages = [
            {"rows": [{"row": {"id": "wrong", "lang": "en", "shard": "0000"}}], "num_rows_total": 3},
            {"rows": [{"row": {"id": "truncated", "lang": "de", "shard": "0000"},
                       "truncated_cells": ["transcript"]}], "num_rows_total": 3},
            {"rows": [{"row": {"id": "matched", "lang": "de", "shard": "0000"}}], "num_rows_total": 3},
        ]
        with patch.object(sample, "request_json", side_effect=pages) as request:
            with contextlib.redirect_stdout(io.StringIO()):
                rows = list(sample.stream_recordings("de", ["0000"], 0, 10))
        self.assertEqual(rows, [{"id": "matched", "lang": "de", "shard": "0000"}])
        self.assertEqual([parse_qs(urlparse(call.args[0]).query)["offset"] for call in request.call_args_list],
                         [["0"], ["1"], ["2"]])
        self.assertTrue(all(parse_qs(urlparse(call.args[0]).query)["config"] == ["preview"]
                            for call in request.call_args_list))

    def test_run_streams_http_wav_and_caps_dry_run_checks(self):
        calls = []
        audio, _ = wav_bytes()

        def handler(path, headers, body):
            calls.append(path)
            if path.startswith("/rows"):
                return json_reply({"num_rows_total": 1, "rows": [{"row": {
                    "id": "test", "lang": "de", "shard": "0000",
                    "audio": [{"src": url + "/audio.wav", "type": "audio/wav"}],
                    "transcript": json.dumps([{"start": i * 1000, "duration": 1000,
                                               "text": "Hello world today"} for i in range(4)]),
                }}]})
            if path == "/audio.wav":
                return 200, "audio/wav", audio
            return json_reply({"text": "Hello world today"})

        with http_server(handler) as url, patch.object(sample, "VIEWER", url):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                sample.run(sample.parse_args(["--server-url", url + "/v1", "--dry-run", "--max-segments", "2"]))
        self.assertEqual(calls[-3:], ["/audio.wav"] + ["/v1/audio/transcriptions"] * 2)
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1]), {
            "event": "summary", "checked": 2, "accepted": 2, "trained": 0, "dry_run": True,
        })

    def test_script_runs_with_site_packages_disabled(self):
        script = Path(sample.__file__).resolve()
        result = subprocess.run([sys.executable, "-I", "-S", str(script), "--help"],
                                capture_output=True, text=True, check=True)
        self.assertIn("standard library only", result.stdout)


if __name__ == "__main__":
    unittest.main()
