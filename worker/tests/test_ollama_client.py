"""The one Ollama client: exact request bytes, error classes, the reply's
counters, and the counters reaching ``worker-status.json``.

``urllib.request.urlopen`` is replaced; nothing talks to a real Ollama.
The replay through a real HTTP server is in ``journeys/test_cassette.py``.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from daemon_db import create_daemon_db

from agenthandover_worker import (
    behavioral_synthesizer,
    model_profiles,
    scene_annotator,
    sop_generator,
    task_segmenter,
)
from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningConfig
from agenthandover_worker.ollama import (
    OllamaBadResponse,
    OllamaClient,
    OllamaError,
    OllamaUnavailable,
    Response,
)
from agenthandover_worker.ops_telemetry import COMPUTE, MODEL_USAGE
from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig

HOST = "http://localhost:11434"
PNG_1X1_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4"
    "AAAAAElFTkSuQmCC"
)

REPLY = {
    "model": "m",
    "response": '{"title": "t", "steps": []}',
    "done": True,
    "done_reason": "stop",
    "total_duration": 3_000_000_000,
    "load_duration": 500_000_000,
    "prompt_eval_count": 1234,
    "prompt_eval_duration": 1_000_000_000,
    "eval_count": 56,
    "eval_duration": 1_500_000_000,
}


class _Urlopen:
    """Stands in for ``urllib.request.urlopen``: records the raw request
    (method, URL, body bytes, timeout) and answers from a queue."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent: list[tuple[str, str, bytes | None, float | None]] = []

    def __call__(self, req, timeout=None):
        self.sent.append((req.get_method(), req.full_url, req.data, timeout))
        reply = self.replies.pop(0) if self.replies else REPLY
        if isinstance(reply, BaseException):
            raise reply
        resp = MagicMock()
        resp.read.return_value = (
            reply if isinstance(reply, bytes) else json.dumps(reply).encode("utf-8")
        )
        resp.__enter__ = lambda s: s
        resp.__exit__ = lambda s, *a: False
        return resp


@pytest.fixture(autouse=True)
def _fresh_usage():
    MODEL_USAGE.reset()
    yield
    MODEL_USAGE.reset()


# ---------------------------------------------------------------------------
# Request bytes: identical to the pre-client call sites (cassette keys and
# Ollama's prompt cache depend on them).  The expected strings were captured
# from the hand-rolled clients before they were replaced.
# ---------------------------------------------------------------------------


def _call_sites(tmp_path: Path) -> list[tuple[str, callable, str]]:
    img = tmp_path / "frame.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 10)
    import base64

    img_b64 = base64.b64encode(img.read_bytes()).decode("ascii")
    return [
        (
            "annotation (image, schema, keep_alive, profile options)",
            lambda: scene_annotator._call_ollama_vlm(
                model="m", prompt="p", image_path=img, system="sys", think=False,
                format={"type": "object"}, keep_alive="30m", num_ctx=4096,
                extra_options={"temperature": 0.0, "top_k": 1}, timeout=42.0,
            ),
            '{"model": "m", "prompt": "p", "stream": false, "think": false, '
            '"options": {"num_predict": 1500, "num_ctx": 4096, "temperature": 0.0, '
            '"top_k": 1}, "system": "sys", "format": {"type": "object"}, '
            '"keep_alive": "30m", "images": ["' + img_b64 + '"]}',
        ),
        (
            "annotation (base64 image, defaults)",
            lambda: scene_annotator._call_ollama_vlm(
                model="m", prompt="p", image_base64=PNG_1X1_B64, num_predict=300,
            ),
            '{"model": "m", "prompt": "p", "stream": false, "think": false, '
            '"options": {"num_predict": 300, "num_ctx": 8192}, '
            '"images": ["' + PNG_1X1_B64 + '"]}',
        ),
        (
            "preload",
            lambda: scene_annotator.SceneAnnotator(scene_annotator.AnnotationConfig(
                model="qwen3.5:2b", timeout_seconds=77.0,
            )).preload(),
            '{"model": "qwen3.5:2b", "prompt": "", "stream": false, "think": false, '
            '"options": {"num_predict": 1, "num_ctx": 4096}, "keep_alive": "30m"}',
        ),
        (
            "sop (json format, thinking off)",
            lambda: sop_generator._call_ollama(
                model="m", prompt="p", num_predict=2000, think=False, format_json=True,
            ),
            '{"model": "m", "prompt": "p", "stream": false, "think": false, '
            '"options": {"num_predict": 2000, "num_ctx": 24576, "temperature": 0.3}, '
            '"format": "json"}',
        ),
        (
            "sop (thinking on drops the grammar)",
            lambda: sop_generator._call_ollama(
                model="m", prompt="p", system="s", think="high", format_json=True,
                extra_options={"top_p": 0.9},
            ),
            '{"model": "m", "prompt": "p", "stream": false, "think": "high", '
            '"options": {"num_predict": 6000, "num_ctx": 24576, "temperature": 0.3, '
            '"top_p": 0.9}, "system": "s"}',
        ),
        (
            "synthesis",
            lambda: behavioral_synthesizer._call_ollama_text(
                model="m", prompt="p", system="s", extra_options={"top_k": 3},
            ),
            '{"model": "m", "prompt": "p", "stream": false, "think": true, '
            '"options": {"num_predict": 3000, "top_k": 3}, "system": "s"}',
        ),
        (
            "reasoning",
            lambda: LLMReasoner(ReasoningConfig(model="q"))._call_ollama(
                "p", "sys", num_predict=50, think=False,
            ),
            '{"model": "q", "prompt": "p", "stream": false, "think": false, '
            '"options": {"num_predict": 50, "num_ctx": 16384, "temperature": 0.3}, '
            '"system": "sys"}',
        ),
        (
            "segmenter embeddings",
            lambda: task_segmenter._compute_embeddings(
                ["a", "b"], model="nomic-embed-text",
            ),
            '{"model": "nomic-embed-text", "input": '
            '["clustering: a", "clustering: b"]}',
        ),
    ]


def test_request_bytes_are_unchanged(tmp_path):
    for name, call, expected in _call_sites(tmp_path):
        opener = _Urlopen(
            {"embeddings": [[1.0], [2.0]]} if "embeddings" in name else REPLY
        )
        with patch("urllib.request.urlopen", opener):
            call()
        assert len(opener.sent) == 1, name
        method, url, data, _ = opener.sent[0]
        assert method == "POST", name
        assert data.decode("utf-8") == expected, name


def test_vector_kb_batches_and_keeps_the_body(tmp_path):
    opener = _Urlopen({"embeddings": [[1.0], [2.0]]}, {"embeddings": [[3.0]]})
    kb = VectorKB(create_daemon_db(tmp_path / "v.db"), VectorKBConfig(batch_size=2))
    with patch("urllib.request.urlopen", opener):
        vectors = kb.compute_embeddings(["a", "b", "c"])
    assert vectors == [[1.0], [2.0], [3.0]]
    assert [data for _, _, data, _ in opener.sent] == [
        b'{"model": "nomic-embed-text", "input": '
        b'["search_document: a", "search_document: b"]}',
        b'{"model": "nomic-embed-text", "input": ["search_document: c"]}',
    ]
    assert opener.sent[0][3] == kb.config.timeout_seconds


def test_get_endpoints():
    opener = _Urlopen(
        {"version": "0.30.6"},
        {"models": [{"name": "gemma4:e2b"}, {"model": "qwen3.5:2b"}, "junk"]},
    )
    with patch("urllib.request.urlopen", opener):
        assert model_profiles.check_ollama_version(HOST) == "0.30.6"
        assert OllamaClient(HOST).tags() == ["gemma4:e2b", "qwen3.5:2b"]
    assert [(m, u, d, t) for m, u, d, t in opener.sent] == [
        ("GET", f"{HOST}/api/version", None, 5),
        ("GET", f"{HOST}/api/tags", None, 3.0),
    ]


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def test_generate_returns_the_counters():
    with patch("urllib.request.urlopen", _Urlopen(REPLY)):
        response = OllamaClient(HOST).generate("m", "p", think=False)
    assert isinstance(response, Response)
    assert response.text == REPLY["response"]
    assert response.prompt_eval_count == 1234
    assert response.eval_count == 56
    assert response.prompt_eval_duration == 1_000_000_000
    assert response.eval_duration == 1_500_000_000
    assert response.load_duration == 500_000_000
    assert response.total_duration == 3_000_000_000
    assert response.done_reason == "stop"
    assert response.elapsed >= 0


def test_generate_omits_unset_fields():
    opener = _Urlopen(REPLY)
    with patch("urllib.request.urlopen", opener):
        OllamaClient(HOST).generate("m", "p", system="", images=[])
    assert opener.sent[0][2] == b'{"model": "m", "prompt": "p", "stream": false}'


def test_truncated_output_is_logged(caplog):
    with patch("urllib.request.urlopen", _Urlopen(dict(REPLY, done_reason="length"))):
        OllamaClient(HOST).generate("m", "p", options={"num_predict": 9})
    assert "hit num_predict=9" in caplog.text
    assert MODEL_USAGE.snapshot()["m"]["truncated_calls"] == 1


def test_embed_prefix_and_padding(caplog):
    opener = _Urlopen({"embeddings": [[1.0]]})
    with patch("urllib.request.urlopen", opener):
        vectors = OllamaClient(HOST).embed(
            "e", ["a", "b"], prefix="search_document: ",
        )
    assert opener.sent[0][2] == (
        b'{"model": "e", "input": ["search_document: a", "search_document: b"]}'
    )
    assert vectors == [[1.0], []]
    assert "expected 2, got 1" in caplog.text
    assert OllamaClient(HOST).embed("e", []) == []


def test_embed_text_request_bytes_are_unchanged():
    """Text-only calls send the bytes they sent before multimodal inputs
    and ``dimensions`` existed (the journeys' cassette keys hash them)."""
    opener = _Urlopen({"embeddings": [[1.0], [2.0]]}, {"embeddings": [[3.0]]})
    with patch("urllib.request.urlopen", opener):
        vectors = OllamaClient(HOST).embed(
            "nomic-embed-text", ["a", "b", "c"], prefix="clustering: ", batch_size=2,
        )
        OllamaClient(HOST).embed("nomic-embed-text", ["plain"])
    assert [sent[2] for sent in opener.sent] == [
        b'{"model": "nomic-embed-text", "input": ["clustering: a", "clustering: b"]}',
        b'{"model": "nomic-embed-text", "input": ["clustering: c"]}',
        b'{"model": "nomic-embed-text", "input": ["plain"]}',
    ]
    # Without ``dimensions`` the vectors come back as Ollama sent them.
    assert vectors == [[1.0], [2.0], [3.0]]


def test_embed_multimodal_request_shape():
    """Image + text items: the prefix goes on the text, image bytes are
    base64-encoded (a base64 str passes through), ``dimensions`` is a
    top-level field, and the truncated vectors come back unit length."""
    import base64

    png = base64.b64decode(PNG_1X1_B64)
    opener = _Urlopen({"embeddings": [[3.0, 4.0], [0.0, 2.0], [1.0, 0.0], [0.0, 0.0]]})
    with patch("urllib.request.urlopen", opener):
        vectors = OllamaClient(HOST).embed(
            "embeddinggemma-2:440m",
            [
                {"text": "Safari — Stripe payouts", "image": png},
                {"image": PNG_1X1_B64},
                {"text": "only text"},
                "a plain string",
            ],
            prefix="title: none | text: ",
            dimensions=256,
        )
    assert json.loads(opener.sent[0][2]) == {
        "model": "embeddinggemma-2:440m",
        "input": [
            {"text": "title: none | text: Safari — Stripe payouts", "image": PNG_1X1_B64},
            {"image": PNG_1X1_B64},
            {"text": "title: none | text: only text"},
            "title: none | text: a plain string",
        ],
        "dimensions": 256,
    }
    assert vectors == [[0.6, 0.8], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]


@pytest.mark.parametrize("item", [{}, {"txt": "x"}, {"text": "x", "audio": b""}, 3])
def test_embed_rejects_a_malformed_item(item):
    with patch("urllib.request.urlopen", _Urlopen({"embeddings": [[1.0]]})) as opener:
        with pytest.raises(ValueError, match="an embed input is"):
            OllamaClient(HOST).embed("e", [item])
    assert opener.sent == []


def test_has_model():
    client = OllamaClient(HOST)
    with patch("urllib.request.urlopen", _Urlopen({"models": [{"name": "gemma4:latest"}]})):
        assert client.has_model("gemma4") is True
    with patch("urllib.request.urlopen", _Urlopen({"models": [{"name": "other:1b"}]})):
        assert client.has_model("gemma4:e2b") is False
    refused = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
    with patch("urllib.request.urlopen", _Urlopen(refused)):
        assert client.has_model("gemma4:e2b") is False


# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(f"{HOST}/api/generate", code, "err", {}, io.BytesIO(b"{}"))


def test_unload_and_ps():
    """``unload`` sends ``keep_alive: 0`` with an empty prompt (Ollama's
    unload request, not counted as a model call); ``ps`` lists the loaded
    models with their sizes."""
    loaded = {"models": [
        {"name": "qwen3.5:4b", "size": 3_400_000_000, "size_vram": 3_400_000_000},
        "junk",
    ]}
    opener = _Urlopen({"model": "qwen3.5:4b", "done": True, "done_reason": "unload"}, loaded)
    with patch("urllib.request.urlopen", opener):
        OllamaClient(HOST).unload("qwen3.5:4b")
        assert OllamaClient(HOST).ps() == [loaded["models"][0]]
    assert opener.sent == [
        ("POST", f"{HOST}/api/generate",
         b'{"model": "qwen3.5:4b", "prompt": "", "stream": false, "keep_alive": 0}', 30.0),
        ("GET", f"{HOST}/api/ps", None, 5.0),
    ]
    assert MODEL_USAGE.snapshot() == {}


def test_a_failed_load_is_unavailable_with_ollamas_reason():
    """Ollama's 500 for a load that timed out is a transient deferral, and
    its message says why."""
    body = b'{"error":"timed out waiting for llama-server to start"}'
    error = urllib.error.HTTPError(f"{HOST}/api/generate", 500, "Internal Server Error", {}, io.BytesIO(body))
    with patch("urllib.request.urlopen", _Urlopen(error)):
        with pytest.raises(OllamaUnavailable, match="timed out waiting for llama-server to start"):
            OllamaClient(HOST).generate("gemma4:12b-it-qat", "p")


@pytest.mark.parametrize("error, expected", [
    (urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
     OllamaUnavailable),
    (_http_error(404), OllamaUnavailable),
    (_http_error(500), OllamaUnavailable),
    (_http_error(503), OllamaUnavailable),
    (ConnectionResetError(), OllamaUnavailable),
    (urllib.error.URLError(TimeoutError("timed out")), TimeoutError),
    (TimeoutError("timed out"), TimeoutError),
    (_http_error(400), OllamaBadResponse),
])
def test_errors_are_classified(error, expected):
    with patch("urllib.request.urlopen", _Urlopen(error)):
        with pytest.raises(expected) as raised:
            OllamaClient(HOST).generate("m", "p")
    assert isinstance(raised.value, OllamaError)
    if expected is OllamaUnavailable:
        assert issubclass(expected, ConnectionError)  # callers defer on it
    assert MODEL_USAGE.snapshot() == {}


@pytest.mark.parametrize("error", [
    _http_error(400),
    urllib.error.URLError(TimeoutError("timed out")),
    urllib.error.URLError(ConnectionRefusedError(61, "Connection refused")),
])
def test_embedding_callers_degrade_on_every_ollama_error(tmp_path, error):
    """vector_kb degrades (no vector, no results) on any client error, as it
    did when its own client re-raised every failure as ConnectionError."""
    kb = VectorKB(create_daemon_db(tmp_path / "v.db"), VectorKBConfig())
    with patch("urllib.request.urlopen", _Urlopen(error)):
        assert kb.upsert("sop", "s1", "open the report") is False
    with patch("urllib.request.urlopen", _Urlopen(error)):
        assert kb.search("report") == []


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]", b"\xff\xfe"])
def test_a_reply_that_is_not_an_object_is_a_bad_response(body):
    with patch("urllib.request.urlopen", _Urlopen(body)):
        with pytest.raises(OllamaBadResponse):
            OllamaClient(HOST).generate("m", "p")


def test_embed_errors_share_the_classes():
    with patch("urllib.request.urlopen", _Urlopen(_http_error(404))):
        with pytest.raises(OllamaUnavailable):
            task_segmenter._compute_embeddings(["a"], model="nomic-embed-text")


# ---------------------------------------------------------------------------
# Usage counters -> worker-status.json
# ---------------------------------------------------------------------------


def test_every_reply_is_counted_per_model():
    with patch("urllib.request.urlopen", _Urlopen(REPLY, REPLY, {
        "embeddings": [[1.0]], "prompt_eval_count": 3, "total_duration": 10_000_000,
    })):
        client = OllamaClient(HOST)
        client.generate("m", "p")
        client.generate("m", "p")
        client.embed("e", ["x"])
    usage = MODEL_USAGE.snapshot()
    assert usage["m"]["generate_calls"] == 2
    assert usage["m"]["prompt_eval_count"] == 2468
    assert usage["m"]["eval_count"] == 112
    assert usage["m"]["prompt_eval_seconds"] == 2.0
    assert usage["m"]["eval_seconds"] == 3.0
    assert usage["m"]["load_seconds"] == 1.0
    assert usage["m"]["total_seconds"] == 6.0
    assert usage["e"]["embed_calls"] == 1
    assert usage["e"]["prompt_eval_count"] == 3
    assert usage["e"]["eval_count"] == 0


def test_eval_count_reaches_worker_status(tmp_path):
    """A reply's ``eval_count`` travels Response -> MODEL_USAGE ->
    ``worker-status.json`` (what the app and the lead's scorecards read)."""
    from agenthandover_worker.main import _write_worker_status

    with patch("urllib.request.urlopen", _Urlopen(REPLY)):
        scene_annotator._call_ollama_vlm(
            model="gemma4:e2b", prompt="p", image_base64=PNG_1X1_B64,
        )
    with patch("agenthandover_worker.main._status_dir", return_value=tmp_path):
        _write_worker_status(
            started_at="2026-09-22T10:00:00Z",
            events_processed_today=1,
            sops_generated=0,
            last_pipeline_duration_ms=None,
            consecutive_errors=0,
            vlm_available=True,
        )
    status = json.loads((tmp_path / "worker-status.json").read_text())
    calls = status["model_calls"]["gemma4:e2b"]
    assert calls["eval_count"] == REPLY["eval_count"] == 56
    assert calls["prompt_eval_count"] == 1234
    assert calls["generate_calls"] == 1


# ---------------------------------------------------------------------------
# Model seconds per call type (WP-C1) -> worker-status.json "compute"
# ---------------------------------------------------------------------------

_CALL_TYPE_OF_SITE = {
    "annotation (image, schema, keep_alive, profile options)": "annotation",
    "annotation (base64 image, defaults)": "annotation",
    "preload": "preload",
    "sop (json format, thinking off)": "sop",
    "sop (thinking on drops the grammar)": "sop",
    "synthesis": "synthesis",
    "reasoning": "reasoning",
    "segmenter embeddings": "embed",
}


def test_every_call_site_is_counted_under_its_call_type(tmp_path):
    for name, call, _ in _call_sites(tmp_path):
        COMPUTE.reset()
        reply = (
            {"embeddings": [[1.0], [2.0]], "total_duration": 250_000_000}
            if "embeddings" in name else REPLY
        )
        with patch("urllib.request.urlopen", _Urlopen(reply)):
            call()
        snap = COMPUTE.snapshot()
        expected = _CALL_TYPE_OF_SITE[name]
        assert snap["model_calls_by_call_type"] == {expected: 1}, name
        assert snap["model_seconds_by_call_type"] == {
            expected: reply["total_duration"] / 1e9
        }, name
    COMPUTE.reset()


def test_compute_reaches_worker_status(tmp_path):
    from agenthandover_worker.main import _write_worker_status

    COMPUTE.reset()
    with patch("urllib.request.urlopen", _Urlopen(REPLY)):
        scene_annotator._call_ollama_vlm(model="m", prompt="p")
        sop_generator._call_ollama(model="m", prompt="p", think=False)
    with patch("agenthandover_worker.main._status_dir", return_value=tmp_path):
        _write_worker_status(
            started_at="2026-09-22T10:00:00Z",
            events_processed_today=0,
            sops_generated=0,
            last_pipeline_duration_ms=None,
            consecutive_errors=0,
            vlm_available=True,
        )
    COMPUTE.reset()
    compute = json.loads((tmp_path / "worker-status.json").read_text())["compute"]
    assert compute["model_seconds_by_call_type"] == {"annotation": 3.0, "sop": 3.0}
    assert compute["model_calls_by_call_type"] == {"annotation": 1, "sop": 1}
    assert compute["model_seconds"] == 6.0
    assert compute["last_hour"]["model_seconds_by_call_type"] == {
        "annotation": 3.0, "sop": 3.0,
    }
