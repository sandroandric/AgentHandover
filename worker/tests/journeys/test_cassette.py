"""The Ollama cassette: keys, replay through the real call sites, record,
byte-identical replay, misses, and fault injection.

``test_record_against_real_ollama`` runs only with ``AH_RECORD=1`` and a
reachable Ollama (model from ``AH_RECORD_MODEL``, else the first one
installed); everything else runs offline.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agenthandover_worker import behavioral_synthesizer, model_profiles
from agenthandover_worker import scene_annotator, sop_generator, task_segmenter
from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningConfig
from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig
from journeys.cassette import (
    Cassette,
    encode_response,
    load_entries,
    open_cassette,
    request_key,
)

SELFTEST = "cassette_selftest"

# A valid 1x1 white PNG, the image attached to the annotation call.
PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4"
    "AAAAAElFTkSuQmCC"
)


def call_sites(host: str, tmp_path: Path) -> dict:
    """One request through each production Ollama call site, pointed at ``host``.

    The hand-written ``cassette_selftest.jsonl`` holds exactly these requests.
    """
    reasoner = LLMReasoner(ReasoningConfig(model="qwen3.5:4b", ollama_host=host))
    kb = VectorKB(tmp_path / "vectors.db", VectorKBConfig(ollama_host=host))
    return {
        "annotate": lambda: scene_annotator._call_ollama_vlm(
            model="gemma4:e2b",
            prompt="Describe the visible app and the user's action.",
            image_base64=PNG_1X1_B64,
            host=host,
            num_predict=300,
        )[0],
        "sop": lambda: sop_generator._call_ollama(
            model="gemma4:e4b",
            prompt="Write the SOP as JSON for: open the daily news report.",
            host=host,
            num_predict=2000,
            think=False,
            format_json=True,
        )[0],
        "synthesis": lambda: behavioral_synthesizer._call_ollama_text(
            model="gemma4:e4b",
            prompt="Summarise why the user opens the daily news report.",
            host=host,
            num_predict=500,
            think=False,
        )[0],
        "reasoning": lambda: reasoner._call_ollama(
            "Is this the same task? Answer yes or no.",
            "You compare workflows.",
            num_predict=50,
            think=False,
        )[0],
        "kb_embed": lambda: kb.compute_embeddings(["open report", "send email"]),
        "segment_embed": lambda: task_segmenter._compute_embeddings(
            ["open report"], model="nomic-embed-text", host=host
        ),
        "version": lambda: model_profiles.check_ollama_version(host),
    }


def _post(host: str, path: str, body: dict) -> tuple[int, bytes]:
    req = urllib.request.Request(
        host + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _get(host: str, path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(host + path, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


CHAT_BODY = {
    "model": "gemma4:e2b",
    "messages": [
        {"role": "user", "content": "What is on screen?", "images": [PNG_1X1_B64]}
    ],
    "stream": False,
    "options": {"num_predict": 100},
}


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


class TestKeys:
    BASE = {
        "model": "m",
        "prompt": "p",
        "stream": False,
        "options": {"num_predict": 10, "num_ctx": 8192},
        "images": [PNG_1X1_B64],
    }

    def test_option_order_and_transport_fields_do_not_matter(self):
        reordered = dict(self.BASE, options={"num_ctx": 8192, "num_predict": 10})
        reordered["keep_alive"] = "5m"
        reordered["stream"] = True
        assert request_key("/api/generate", reordered) == request_key(
            "/api/generate", self.BASE
        )

    def test_every_input_that_changes_the_answer_changes_the_key(self):
        base = request_key("/api/generate", self.BASE)
        variants = [
            dict(self.BASE, model="m2"),
            dict(self.BASE, prompt="p2"),
            dict(self.BASE, options={"num_predict": 11, "num_ctx": 8192}),
            dict(self.BASE, images=[PNG_1X1_B64[:-8] + "AAAAAA=="]),
            dict(self.BASE, system="s"),
            dict(self.BASE, think=True),
        ]
        keys = {request_key("/api/generate", v) for v in variants}
        assert base not in keys and len(keys) == len(variants)
        assert request_key("/api/chat", self.BASE) != base

    def test_images_are_keyed_by_their_bytes(self):
        from journeys.cassette import normalise_request

        material = normalise_request(self.BASE)
        assert len(material["images"][0]) == 64  # sha256 hex, not the base64
        chat = normalise_request(CHAT_BODY)
        assert chat["messages"][0]["images"] == material["images"]
        assert CHAT_BODY["messages"][0]["images"] == [PNG_1X1_B64]  # not mutated


# ---------------------------------------------------------------------------
# Replay through the production call sites
# ---------------------------------------------------------------------------


def _expected_answers() -> dict:
    entries = load_entries(open_cassette(SELFTEST).path)
    by_path: dict[str, list[dict]] = {}
    for entry in entries.values():
        by_path.setdefault(entry["path"], []).append(entry["response"])
    return by_path


def test_selftest_cassette_is_hand_written_and_valid():
    entries = load_entries(open_cassette(SELFTEST).path)
    assert {e["source"] for e in entries.values()} == {"hand-written"}
    paths = {e["path"] for e in entries.values()}
    assert paths == {
        "/api/generate", "/api/chat", "/api/embed", "/api/tags", "/api/version",
    }
    for entry in entries.values():
        if entry["path"] in ("/api/generate", "/api/chat"):
            for field in ("prompt_eval_count", "eval_count", "eval_duration",
                          "total_duration"):
                assert field in entry["response"], (entry["key"], field)


def test_replay_through_every_production_call_site(tmp_path):
    with open_cassette(SELFTEST, record=False) as cassette:
        answers = {name: call() for name, call in
                   call_sites(cassette.host, tmp_path).items()}

    generated = {r["response"] for r in _expected_answers()["/api/generate"]}
    assert answers["annotate"] in generated
    assert json.loads(answers["sop"])["title"] == "Open the daily news report"
    assert answers["synthesis"] in generated
    assert answers["reasoning"] == "yes"
    assert len(answers["kb_embed"]) == 2 and all(answers["kb_embed"])
    assert len(answers["segment_embed"]) == 1 and answers["segment_embed"][0]
    assert answers["version"] == "0.30.6"
    # Every answer came from the cassette and carries Ollama's timing.
    assert len(cassette.served) == 7
    assert cassette.model_seconds() > 0


def test_replay_chat_and_tags(tmp_path):
    with open_cassette(SELFTEST, record=False) as cassette:
        status, body = _post(cassette.host, "/api/chat", CHAT_BODY)
        assert status == 200
        assert json.loads(body)["message"]["role"] == "assistant"
        status, body = _get(cassette.host, "/api/tags")
        assert status == 200
        assert [m["name"] for m in json.loads(body)["models"]]


def test_replay_miss_fails_the_test_with_the_key(tmp_path, capsys):
    cassette = open_cassette(SELFTEST, record=False)
    body = {"model": "gemma4:e2b", "prompt": "never recorded", "stream": False}
    key = request_key("/api/generate", body)
    with pytest.raises(AssertionError, match=key):
        with cassette:
            status, answer = _post(cassette.host, "/api/generate", body)
            assert status == 500 and key in json.loads(answer)["error"]
            # The production call site surfaces the miss as ConnectionError.
            with pytest.raises(ConnectionError):
                scene_annotator._call_ollama_vlm(
                    model="gemma4:e2b", prompt="never recorded", host=cassette.host
                )
    assert key in capsys.readouterr().err
    assert len(cassette.misses) == 2


def test_streaming_requests_are_refused(tmp_path):
    with open_cassette(SELFTEST, record=False) as cassette:
        status, body = _post(
            cassette.host, "/api/generate", {"model": "m", "prompt": "p"}
        )
    assert status == 400 and "stream" in json.loads(body)["error"]
    assert cassette.misses == []


def test_missing_cassette_in_replay_mode_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="AH_RECORD=1"):
        Cassette(tmp_path / "nope.jsonl", record=False)


def test_tampered_or_duplicate_entries_are_rejected(tmp_path):
    lines = open_cassette(SELFTEST).path.read_text().splitlines()
    dup = tmp_path / "dup.jsonl"
    dup.write_text(lines[0] + "\n" + lines[0] + "\n")
    with pytest.raises(ValueError, match="duplicate key"):
        load_entries(dup)

    entry = json.loads(lines[0])
    entry["request"]["model"] = "edited-by-hand"
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text(json.dumps(entry) + "\n")
    with pytest.raises(ValueError, match="does not match"):
        load_entries(tampered)


# ---------------------------------------------------------------------------
# Record mode (against a stand-in upstream) and byte-identical replay
# ---------------------------------------------------------------------------


class _FakeOllama:
    """A stand-in upstream that counts requests; not a cassette."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_GET(self):
                outer.requests.append((self.path, {}))
                if self.path == "/api/tags":
                    self._send(200, {"models": [{"name": "gemma4:e2b"}]})
                else:
                    self._send(200, {"version": "0.30.6"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                if body.get("model") == "missing":
                    self._send(404, {"error": "model 'missing' not found"})
                    return
                # Pretty-printed with spaces, unlike the cassette's encoding.
                self._send(200, {
                    "model": body["model"],
                    "response": f"answer to {body['prompt']}",
                    "done": True,
                    "total_duration": 2_000_000_000,
                    "prompt_eval_count": 12,
                    "eval_count": 4,
                    "eval_duration": 1_500_000_000,
                }, indent=2)

            def _send(self, status, obj, indent=None):
                data = json.dumps(obj, indent=indent).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.host = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._thread.is_alive():
            self.server.shutdown()
            self.server.server_close()


@pytest.fixture
def fake_ollama():
    fake = _FakeOllama()
    yield fake
    fake.close()


def test_record_then_replay_is_byte_identical(tmp_path, fake_ollama):
    path = tmp_path / "journey.jsonl"
    body = {"model": "gemma4:e2b", "prompt": "hello", "stream": False,
            "options": {"num_predict": 5}}

    with Cassette(path, record=True, upstream=fake_ollama.host) as rec:
        recorded = [_post(rec.host, "/api/generate", body),
                    _get(rec.host, "/api/version")]
        # A second identical request is served from the cassette, not proxied.
        again = _post(rec.host, "/api/generate", body)
        # Upstream errors pass through and are not recorded.
        status, _ = _post(rec.host, "/api/generate", dict(body, model="missing"))
        assert status == 404
    assert again == recorded[0]
    assert len(fake_ollama.requests) == 3
    assert len(rec.recorded) == 2

    entries = load_entries(path)
    assert len(entries) == 2
    generate = entries[request_key("/api/generate", body)]
    assert generate["source"] == "recorded"
    assert generate["response"]["eval_duration"] == 1_500_000_000
    assert recorded[0] == (200, encode_response(generate["response"]))

    fake_ollama.close()  # replay must not need the upstream
    with Cassette(path, record=False) as replay:
        replayed = [_post(replay.host, "/api/generate", body),
                    _get(replay.host, "/api/version")]
    assert replayed == recorded
    assert replay.model_seconds() == 2.0


def test_record_mode_appends_only_new_entries(tmp_path, fake_ollama):
    path = tmp_path / "journey.jsonl"
    first = {"model": "m", "prompt": "one", "stream": False}
    second = {"model": "m", "prompt": "two", "stream": False}
    with Cassette(path, record=True, upstream=fake_ollama.host) as rec:
        _post(rec.host, "/api/generate", first)
    with Cassette(path, record=True, upstream=fake_ollama.host) as rec:
        _post(rec.host, "/api/generate", first)
        _post(rec.host, "/api/generate", second)
    assert [b["prompt"] for _, b in fake_ollama.requests] == ["one", "two"]
    assert len(path.read_text().splitlines()) == 2


def test_record_mode_is_the_default_when_ah_record_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("AH_RECORD", "1")
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")
    cassette = Cassette(tmp_path / "new.jsonl")
    assert cassette.record and cassette.upstream == "http://127.0.0.1:9"
    monkeypatch.delenv("AH_RECORD")
    assert not open_cassette(SELFTEST).record


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


def test_down_answers_503_for_n_requests_then_recovers(tmp_path):
    with open_cassette(SELFTEST, record=False) as cassette:
        status, body = _get(cassette.host, "/__cassette/down?n=2")
        assert status == 200 and json.loads(body)["down"] == 2
        for _ in range(2):
            with pytest.raises(ConnectionError):
                call_sites(cassette.host, tmp_path)["annotate"]()
        assert call_sites(cassette.host, tmp_path)["annotate"]()
    assert cassette.misses == []


def test_stop_and_start_is_a_real_refused_connection(tmp_path):
    with open_cassette(SELFTEST, record=False) as cassette:
        port = cassette.port
        cassette.stop()
        with pytest.raises(ConnectionError, match="Connection refused"):
            call_sites(cassette.host, tmp_path)["annotate"]()
        cassette.start()
        assert cassette.port == port
        assert call_sites(cassette.host, tmp_path)["annotate"]()


def test_slow_delays_the_next_n_answers(tmp_path):
    with open_cassette(SELFTEST, record=False) as cassette:
        _get(cassette.host, "/__cassette/slow?seconds=0.3&n=1")
        start = time.monotonic()
        call_sites(cassette.host, tmp_path)["version"]()
        assert time.monotonic() - start >= 0.3
        start = time.monotonic()
        call_sites(cassette.host, tmp_path)["version"]()
        assert time.monotonic() - start < 0.3

        _get(cassette.host, "/__cassette/slow?seconds=0.2")
        _get(cassette.host, "/__cassette/down?n=5")
        status, body = _get(cassette.host, "/__cassette/reset")
        assert json.loads(body) == {
            "down": 0, "slow_seconds": 0.0, "slow_remaining": None, "slow_path": None,
        }
        assert _get(cassette.host, "/__cassette/slow")[0] == 400
        assert _get(cassette.host, "/__cassette/nope")[0] == 404


def test_slow_with_a_path_delays_only_requests_to_that_path(tmp_path):
    with open_cassette(SELFTEST, record=False) as cassette:
        _get(cassette.host, "/__cassette/slow?seconds=0.3&n=1&path=/api/generate")
        start = time.monotonic()
        call_sites(cassette.host, tmp_path)["version"]()
        assert time.monotonic() - start < 0.3
        assert cassette.fault_state()["slow_remaining"] == 1
        start = time.monotonic()
        call_sites(cassette.host, tmp_path)["annotate"]()
        assert time.monotonic() - start >= 0.3
        assert [path for path, _request in cassette.delayed] == ["/api/generate"]
        assert cassette.fault_state() == {
            "down": 0, "slow_seconds": 0.0, "slow_remaining": None, "slow_path": None,
        }


# ---------------------------------------------------------------------------
# Real Ollama (test Mac only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("AH_RECORD") != "1",
    reason="records against a real Ollama only with AH_RECORD=1",
)
def test_record_against_real_ollama(tmp_path):
    upstream = Cassette(tmp_path / "probe.jsonl", record=True).upstream
    try:
        with urllib.request.urlopen(upstream + "/api/tags", timeout=5) as resp:
            installed = [m["name"] for m in json.loads(resp.read())["models"]]
    except OSError as exc:
        pytest.fail(f"AH_RECORD=1 but Ollama at {upstream} is unreachable: {exc}")
    model = os.environ.get("AH_RECORD_MODEL") or (installed or [None])[0]
    if model is None:
        pytest.fail(f"AH_RECORD=1 but Ollama at {upstream} has no models")

    body = {
        "model": model,
        "prompt": "Reply with the single word: ready",
        "stream": False,
        "think": False,
        "options": {"num_predict": 20, "temperature": 0},
    }
    path = tmp_path / "real.jsonl"
    with Cassette(path, record=True) as rec:
        recorded = [_post(rec.host, "/api/generate", body),
                    _get(rec.host, "/api/version"),
                    _get(rec.host, "/api/tags")]
    assert [status for status, _ in recorded] == [200, 200, 200]
    generate = load_entries(path)[request_key("/api/generate", body)]
    for field in ("prompt_eval_count", "eval_count", "eval_duration"):
        assert field in generate["response"]

    with Cassette(path, record=False) as replay:
        replayed = [_post(replay.host, "/api/generate", body),
                    _get(replay.host, "/api/version"),
                    _get(replay.host, "/api/tags")]
    assert replayed == recorded


# ---------------------------------------------------------------------------
# Production wiring: every Ollama client main() builds honours ollama_host
# ---------------------------------------------------------------------------


def test_main_points_the_llm_reasoner_at_the_configured_host(
    tmp_path, tmp_db_path, monkeypatch
):
    """main() used to build ``LLMReasoner(ReasoningConfig())``, so reasoner
    calls went to localhost:11434 whatever ``[vlm] ollama_host`` said, and a
    journey's cassette would not see them."""
    import logging
    import signal

    from agenthandover_worker import llm_reasoning
    from agenthandover_worker import main as worker_main

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config_dir = (
        home / "Library" / "Application Support" / "agenthandover"
        if worker_main._platform.system() == "Darwin"
        else home / ".config" / "agenthandover"
    )
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        '[vlm]\nollama_host = "http://127.0.0.1:1"\n'
    )

    class Built(Exception):
        pass

    built: list = []

    class CapturingReasoner:
        def __init__(self, config=None, vlm_queue=None):
            built.append(config)
            raise Built

    monkeypatch.setattr(llm_reasoning, "LLMReasoner", CapturingReasoner)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    root = logging.getLogger()
    root_handlers, root_level = list(root.handlers), root.level
    try:
        with pytest.raises(Built):
            worker_main.main([
                "--db-path", str(tmp_db_path),
                "--sops-dir", str(tmp_path / "sops"),
                "--knowledge-dir", str(tmp_path / "kb"),
                "--log-level", "ERROR",
            ])
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for handler in list(root.handlers):
            if handler not in root_handlers:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(root_level)

    assert built[0].ollama_host == "http://127.0.0.1:1"
