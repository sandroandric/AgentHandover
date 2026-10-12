"""The one Ollama HTTP client.

Every model call the worker makes (annotation, frame diffs, SOP
generation, behavioral synthesis, reasoning, embeddings, the tags probe,
the version check, and the loaded-model list and unloads of
``model_scheduler``) goes through :class:`OllamaClient`.  Callers build
the request fields; this module owns the transport, the error
classification and the usage counters.

Failures are classified so callers can tell "try again later" from
"this call failed"; all of them derive from :class:`OllamaError`:

* :class:`OllamaUnavailable` (a ``ConnectionError``): Ollama is
  unreachable (refused, reset, DNS), the model is not pulled (HTTP 404),
  or Ollama answered 5xx (a model that fails to load or runs out of
  memory).  Retrying later can succeed.
* :class:`OllamaTimeout` (a ``TimeoutError``): no reply within the
  timeout (a cold model load).
* :class:`OllamaBadResponse` (a ``RuntimeError``): any other HTTP error,
  or a reply that is not a JSON object.

Every successful generate or embed call is counted in
``ops_telemetry.MODEL_USAGE`` (tokens and Ollama's own durations, per
model) and in ``ops_telemetry.COMPUTE`` (model seconds per ``call_type``:
annotation, diff, sop, synthesis, reasoning, embed, preload or other),
both of which the worker publishes in ``worker-status.json``.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from agenthandover_worker import ops_telemetry
from agenthandover_worker.inference_runner import model_wait

logger = logging.getLogger(__name__)

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(Exception):
    """Base of every failure :class:`OllamaClient` raises."""


class OllamaUnavailable(OllamaError, ConnectionError):
    """Ollama could not serve the call: unreachable (refused, reset, DNS),
    the model is not pulled (HTTP 404), or it answered 5xx.  Retrying later
    can succeed, so callers defer the work instead of discarding it."""


class OllamaTimeout(OllamaError, TimeoutError):
    """No reply within the timeout (often a cold model load)."""


class OllamaBadResponse(OllamaError, RuntimeError):
    """Ollama answered, but with an HTTP error other than 404/5xx or with
    a body that is not a JSON object.  Retrying the same call will not help."""


@dataclass(frozen=True)
class Response:
    """One ``/api/generate`` reply.

    Durations are Ollama's own, in nanoseconds (``None`` when Ollama did
    not report them); ``elapsed`` is the wall-clock seconds of the HTTP
    call as seen by the worker.
    """

    text: str
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_duration: int | None = None
    load_duration: int | None = None
    total_duration: int | None = None
    done_reason: str | None = None
    elapsed: float = 0.0
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_reply(cls, reply: dict, elapsed: float) -> "Response":
        return cls(
            text=reply.get("response", "") or "",
            prompt_eval_count=reply.get("prompt_eval_count"),
            eval_count=reply.get("eval_count"),
            prompt_eval_duration=reply.get("prompt_eval_duration"),
            eval_duration=reply.get("eval_duration"),
            load_duration=reply.get("load_duration"),
            total_duration=reply.get("total_duration"),
            done_reason=reply.get("done_reason"),
            elapsed=elapsed,
            raw=reply,
        )


class OllamaClient:
    """HTTP client for one Ollama server.

    Request bodies are built field by field in a fixed order and only
    carry the fields a caller sets, so the same call always produces the
    same bytes (recorded cassettes key on them).
    """

    def __init__(self, host: str = DEFAULT_HOST) -> None:
        self.host = host

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        images: list[str] | None = None,
        options: dict | None = None,
        format: dict | str | None = None,  # noqa: A002 - Ollama's field name
        think: bool | str | None = None,
        keep_alive: str | None = None,
        timeout: float | None = None,
        call_type: str = "other",
    ) -> Response:
        """POST ``/api/generate`` (non-streaming) and return the reply.

        Args:
            system: system prompt; omitted when empty.
            images: base64-encoded images.
            options: Ollama ``options`` (num_predict, num_ctx, sampling).
            format: a JSON schema (grammar-constrained output), ``"json"``,
                or None for free text.
            think: False, True, "low", "medium" or "high"; a top-level
                field, never inside ``options``.  None omits it.
            keep_alive: how long Ollama keeps the model loaded after the
                call; None omits it (Ollama's default is 5m).
            timeout: seconds to wait for the reply; None waits forever.
            call_type: what the call is for, in the compute counters.
        """
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        if think is not None:
            payload["think"] = think
        if options is not None:
            payload["options"] = options
        if system:
            payload["system"] = system
        if format is not None:
            payload["format"] = format
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if images:
            payload["images"] = list(images)

        start = time.monotonic()
        reply = self._request("POST", "/api/generate", payload, timeout)
        response = Response.from_reply(reply, time.monotonic() - start)

        logger.debug(
            "ollama generate model=%s prompt_eval_count=%s eval_count=%s "
            "done_reason=%s",
            model, response.prompt_eval_count, response.eval_count,
            response.done_reason,
        )
        if response.done_reason == "length":
            logger.warning(
                "Ollama output for %s hit num_predict=%s and was truncated",
                model, (options or {}).get("num_predict"),
            )
        ops_telemetry.MODEL_USAGE.record(model, "generate", response)
        ops_telemetry.COMPUTE.record_call(call_type, response)
        return response

    def preload(
        self,
        model: str,
        keep_alive: str,
        *,
        options: dict | None = None,
        timeout: float | None = None,
    ) -> Response:
        """Load *model* and keep it loaded for *keep_alive*.

        An empty prompt makes Ollama load the model and return.  Pass the
        same ``num_ctx`` in *options* as the real calls: a different
        context size forces a reload.
        """
        return self.generate(
            model, "", options=options, think=False,
            keep_alive=keep_alive, timeout=timeout, call_type="preload",
        )

    def unload(self, model: str, timeout: float | None = 30.0) -> None:
        """Ask Ollama to unload *model* now (an empty prompt with
        ``keep_alive: 0``).  A model that is not loaded is left alone:
        Ollama answers without loading it."""
        self._request(
            "POST", "/api/generate",
            {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout,
        )

    def ps(self, timeout: float | None = 5.0) -> list[dict]:
        """GET ``/api/ps``: the loaded models, each a dict with at least
        ``name`` and ``size`` (bytes, weights plus context)."""
        reply = self._request("GET", "/api/ps", None, timeout)
        return [m for m in reply.get("models") or [] if isinstance(m, dict)]

    def embed(
        self,
        model: str,
        texts: list[str | dict],
        *,
        prefix: str = "",
        dimensions: int | None = None,
        batch_size: int | None = None,
        timeout: float | None = None,
    ) -> list[list[float]]:
        """POST ``/api/embed``: one vector per input, in order.

        Each input is a text, or a multimodal item ``{"text"?: str,
        "image"?: bytes | str}`` with at least one of the two (an image as
        raw bytes or already base64-encoded; Ollama >= 0.40).  *prefix* is
        prepended to every text (task prefixes such as
        ``"search_document: "``).  *dimensions* asks Ollama to truncate each
        vector to that many dimensions (Matryoshka models); the truncated
        vectors are L2-normalised here, so their cosine is their dot
        product whether or not the server normalises them.  *batch_size*
        splits the inputs over several requests; None sends them in one.
        A reply with fewer vectors than inputs is padded with empty vectors.

        For text-only inputs without *dimensions* the request is exactly
        ``{"model", "input": [prefix + text, ...]}``.
        """
        if not texts:
            return []
        inputs = [_embed_input(item, prefix) for item in texts]
        step = batch_size or len(inputs)
        vectors: list[list[float]] = []
        for start in range(0, len(inputs), step):
            batch = inputs[start : start + step]
            payload: dict[str, Any] = {"model": model, "input": batch}
            if dimensions is not None:
                payload["dimensions"] = dimensions
            began = time.monotonic()
            reply = self._request("POST", "/api/embed", payload, timeout)
            embeddings = list(reply.get("embeddings") or [])[: len(batch)]
            if len(embeddings) != len(batch):
                logger.warning(
                    "Embedding count mismatch from %s: expected %d, got %d",
                    model, len(batch), len(embeddings),
                )
                embeddings.extend([] for _ in range(len(batch) - len(embeddings)))
            if dimensions is not None:
                embeddings = [_unit(v) for v in embeddings]
            vectors.extend(embeddings)
            response = Response.from_reply(reply, time.monotonic() - began)
            ops_telemetry.MODEL_USAGE.record(model, "embed", response)
            ops_telemetry.COMPUTE.record_call("embed", response)
        return vectors

    def tags(self, timeout: float | None = 3.0) -> list[str]:
        """GET ``/api/tags``: the names of the pulled models."""
        reply = self._request("GET", "/api/tags", None, timeout)
        names: list[str] = []
        for entry in reply.get("models") or []:
            if not isinstance(entry, dict):
                continue
            for key in ("name", "model"):
                name = str(entry.get(key) or "")
                if name and name not in names:
                    names.append(name)
        return names

    def has_model(self, model: str, timeout: float | None = 3.0) -> bool:
        """True when Ollama answers and has *model* pulled (a bare name
        means its ``:latest`` tag)."""
        wanted = model if ":" in model else f"{model}:latest"
        try:
            names = self.tags(timeout=timeout)
        except Exception:
            return False
        return wanted in names or model in names

    def version(self, timeout: float | None = 5.0) -> str:
        """GET ``/api/version``: the server's version string."""
        return str(self._request("GET", "/api/version", None, timeout).get("version", ""))

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _request(
        self, method: str, path: str, payload: dict | None, timeout: float | None,
    ) -> dict:
        """Send one request and return the decoded JSON object."""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data is not None else {},
            method=method,
        )
        model = (payload or {}).get("model", "?")
        try:
            # The worker's shared state is free for the control thread
            # while this thread waits for Ollama (inference_runner).
            with model_wait(f"{method} {path} {model}"):
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise OllamaUnavailable(
                    f"model {model!r} not available at {self.host} ({path})"
                ) from exc
            if exc.code >= 500:
                # Ollama answers 5xx while a model fails to load ("timed out
                # waiting for llama-server to start") or runs out of memory:
                # the server is there but cannot serve now.
                raise OllamaUnavailable(
                    f"Ollama HTTP {exc.code} at {self.host}{path}: "
                    f"{_error_detail(exc) or exc.reason}"
                ) from exc
            raise OllamaBadResponse(
                f"Ollama HTTP {exc.code} at {self.host}{path}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise OllamaTimeout(
                    f"Ollama timed out after {timeout:.0f}s ({path})"
                    if timeout is not None else f"Ollama timed out ({path})"
                ) from exc
            raise OllamaUnavailable(f"Ollama not reachable at {self.host}: {exc}") from exc
        except ConnectionError as exc:  # reset/refused while reading the reply
            raise OllamaUnavailable(f"Ollama connection lost at {self.host}: {exc}") from exc
        except TimeoutError as exc:  # the socket timed out while reading
            raise OllamaTimeout(
                f"Ollama timed out after {timeout:.0f}s ({path})"
                if timeout is not None else f"Ollama timed out ({path})"
            ) from exc

        try:
            reply = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaBadResponse(
                f"Ollama at {self.host}{path} sent a reply that is not JSON: {exc}"
            ) from exc
        if not isinstance(reply, dict):
            raise OllamaBadResponse(
                f"Ollama at {self.host}{path} sent {type(reply).__name__}, not an object"
            )
        return reply


def _embed_input(item: str | dict, prefix: str) -> str | dict:
    """One ``/api/embed`` input: a text with *prefix*, or a multimodal item
    with *prefix* on its text and its image base64-encoded."""
    if isinstance(item, str):
        return prefix + item
    if not isinstance(item, dict) or not item or set(item) - {"text", "image"}:
        raise ValueError(
            f"an embed input is a str or a dict of 'text' and/or 'image', not {item!r:.80}"
        )
    out: dict[str, str] = {}
    if "text" in item:
        out["text"] = prefix + item["text"]
    if "image" in item:
        image = item["image"]
        out["image"] = (
            base64.b64encode(image).decode("ascii")
            if isinstance(image, (bytes, bytearray)) else image
        )
    return out


def _unit(vector: list[float]) -> list[float]:
    """*vector* scaled to length 1 (an empty or zero vector unchanged)."""
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """The ``error`` field of an Ollama error body, or ""."""
    try:
        body = json.loads(exc.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    return str(body.get("error") or "") if isinstance(body, dict) else ""
