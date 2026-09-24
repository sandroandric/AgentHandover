"""Ollama cassette: record model answers once, replay them in journeys.

A tiny HTTP server on a random localhost port that speaks the part of the
Ollama API the worker uses (``/api/generate``, ``/api/chat``, ``/api/embed``,
``/api/tags``, ``/api/version``, ``/api/ps``).  Journeys point every ``ollama_host`` at
``cassette.host``; no production code changes.

Replay (default): answers come from ``cassettes/<journey>.jsonl``.  A request
with no entry is answered with HTTP 500, the key is printed, and the miss is
kept in ``cassette.misses``; leaving the ``with`` block then fails the test.

Record (``AH_RECORD=1``): requests with no entry are proxied to a real Ollama
(``AH_RECORD_UPSTREAM``, else ``OLLAMA_HOST``, else ``http://localhost:11434``)
and the full JSON response is appended to the cassette.  Entries already in
the file are replayed, so recording is incremental.  A request the upstream
cannot be reached for is a miss, as in replay: pointing ``AH_RECORD_UPSTREAM``
at a closed port lists every request the cassette lacks without recording
any.

Both modes serve the stored response with the same serialisation, so a
recorded answer and its replay are byte-identical.

Key: sha256 over the path and the request body with ``stream`` and
``keep_alive`` removed and every image replaced by the sha256 of its bytes.
JSON is dumped with sorted keys, so option order does not matter.  An
unload (an empty prompt with ``keep_alive: 0``) therefore keys as an empty
prompt, and ``/api/ps`` (a GET without a body) by its path alone: a cassette
holds the one loaded-model list recorded at the journey's first call.

Fault injection (control endpoints, not counted as API requests):
  ``/__cassette/down?n=N``        answer the next N API requests with HTTP 503
  ``/__cassette/slow?seconds=S[&n=N][&path=P]``  delay API answers by S
                                  seconds (the next N, or all until reset;
                                  with ``path``, only requests to that path)
  ``/__cassette/reset``           clear both
For a true refused connection, call ``stop()`` and later ``start()``: the
server comes back on the same port.

Pruning (cassettes are append-only, so a re-recording leaves the entries
nobody asks for any more): run every journey in replay with
``AH_CASSETTE_LOG=<file>``, which logs the keys each cassette served and
missed and every test's outcome, then
``python -m journeys.cassette prune --used-keys <file>`` (from
``worker/tests``) drops the entries no journey requested.  It keeps an
entry whose key a journey's source names (a skip marker checking for a
recording), and leaves a cassette alone when a test that opened it did
not pass (failed, errored, xfailed), when a skipped test's reason names
it (a skip marker waiting on a recording), when it missed a request,
when no test opened it, and always for the self-test cassette, whose
entries are hand-written.
"""

from __future__ import annotations

import base64
import binascii
import argparse
import copy
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

CASSETTE_DIR = Path(__file__).parent / "cassettes"

API_PATHS = frozenset(
    {"/api/generate", "/api/chat", "/api/embed", "/api/tags", "/api/version", "/api/ps"}
)
_STREAMING_PATHS = frozenset({"/api/generate", "/api/chat"})
_IGNORED_FIELDS = ("stream", "keep_alive")


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _image_digest(image: str) -> str:
    try:
        raw = base64.b64decode(image, validate=True)
    except (binascii.Error, ValueError):
        raw = image.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalise_request(body: dict) -> dict:
    """The request as stored and keyed: images hashed, transport fields dropped."""
    material = {k: v for k, v in body.items() if k not in _IGNORED_FIELDS}
    if "images" in material:
        material["images"] = [_image_digest(img) for img in material["images"]]
    if "messages" in material:
        messages = copy.deepcopy(material["messages"])
        for message in messages:
            if isinstance(message, dict) and "images" in message:
                message["images"] = [_image_digest(img) for img in message["images"]]
        material["messages"] = messages
    return material


def key_of(path: str, material: dict) -> str:
    """sha256 of (path, normalised request) with sorted keys."""
    blob = json.dumps(
        {"path": path, "request": material},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def request_key(path: str, body: dict) -> str:
    return key_of(path, normalise_request(body))


def encode_response(response: dict) -> bytes:
    """The one serialisation used for every answer, recorded or replayed."""
    return json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def load_entries(path: Path) -> dict[str, dict]:
    """Read a cassette file; every stored key must match its stored request."""
    entries: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            expected = key_of(entry["path"], entry["request"])
            if entry["key"] != expected:
                raise ValueError(
                    f"{path}:{lineno}: stored key {entry['key']} does not match "
                    f"its request (expected {expected})"
                )
            if expected in entries:
                raise ValueError(f"{path}:{lineno}: duplicate key {expected}")
            entries[expected] = entry
    return entries


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class Cassette:
    """Replay (or record) Ollama answers on ``self.host``.

    Use as a context manager; leaving it stops the server and raises
    ``AssertionError`` naming every replay miss.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        record: bool | None = None,
        upstream: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.record = (
            os.environ.get("AH_RECORD") == "1" if record is None else record
        )
        self.upstream = (
            upstream
            or os.environ.get("AH_RECORD_UPSTREAM")
            or os.environ.get("OLLAMA_HOST")
            or "http://localhost:11434"
        ).rstrip("/")
        if self.upstream and "://" not in self.upstream:
            self.upstream = f"http://{self.upstream}"

        if self.path.exists():
            self.entries = load_entries(self.path)
        elif self.record:
            self.entries = {}
        else:
            raise FileNotFoundError(
                f"cassette {self.path} does not exist; record it with AH_RECORD=1"
            )

        self.served: list[dict] = []
        self.misses: list[str] = []
        self.recorded: list[str] = []
        self._down = 0
        self._slow_seconds = 0.0
        self._slow_remaining: int | None = None
        self._slow_path: str | None = None
        # (path, request body) of every API request answered late.
        self.delayed: list[tuple[str, dict]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    # -- lifecycle ---------------------------------------------------------

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "Cassette":
        handler = type("_Handler", (_Handler,), {"cassette": self})
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="ollama-cassette",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        assert self._thread is not None
        self._thread.join()
        self._server = None
        self._thread = None
        with self._lock:
            log_use({
                "cassette": str(self.path.resolve()),
                "test": current_test(),
                "served": sorted({entry["key"] for entry in self.served}),
                "misses": sorted(set(self.misses)),
            })

    def __enter__(self) -> "Cassette":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
        if exc_type is None:
            self.assert_no_misses()

    def assert_no_misses(self) -> None:
        if self.misses:
            raise AssertionError(
                f"cassette {self.path} has no entry for "
                f"{len(self.misses)} request(s): {', '.join(self.misses)}; "
                "re-record with AH_RECORD=1"
            )

    # -- accounting --------------------------------------------------------

    def model_seconds(self) -> float:
        """Sum of Ollama's ``total_duration`` over every answer served."""
        return sum(
            entry["response"].get("total_duration", 0) for entry in self.served
        ) / 1e9

    # -- faults ------------------------------------------------------------

    def set_down(self, n: int) -> None:
        with self._lock:
            self._down = n

    def set_slow(
        self, seconds: float, n: int | None = None, path: str | None = None,
    ) -> None:
        with self._lock:
            self._slow_seconds = seconds
            self._slow_remaining = n
            self._slow_path = path

    def reset_faults(self) -> None:
        with self._lock:
            self._down = 0
            self._slow_seconds = 0.0
            self._slow_remaining = None
            self._slow_path = None

    def _take_faults(self, path: str) -> tuple[bool, float]:
        """Consume one request's worth of fault budget: (down, delay)."""
        with self._lock:
            if self._down > 0:
                self._down -= 1
                return True, 0.0
            if self._slow_path is not None and path != self._slow_path:
                return False, 0.0
            delay = self._slow_seconds
            if delay and self._slow_remaining is not None:
                self._slow_remaining -= 1
                if self._slow_remaining <= 0:
                    self._slow_seconds = 0.0
                    self._slow_remaining = None
                    self._slow_path = None
            return False, delay

    def fault_state(self) -> dict:
        with self._lock:
            return {
                "down": self._down,
                "slow_seconds": self._slow_seconds,
                "slow_remaining": self._slow_remaining,
                "slow_path": self._slow_path,
            }

    # -- answering ---------------------------------------------------------

    def answer(self, method: str, path: str, raw_body: bytes) -> tuple[int, bytes]:
        """Return (status, body) for one API request."""
        body = json.loads(raw_body) if raw_body else {}
        if path in _STREAMING_PATHS and body.get("stream", True) is not False:
            return 400, _error(
                'cassette serves only "stream": false requests to ' + path
            )
        material = normalise_request(body)
        key = key_of(path, material)

        with self._lock:
            entry = self.entries.get(key)
        if entry is not None:
            with self._lock:
                self.served.append(entry)
            return 200, encode_response(entry["response"])

        if not self.record:
            with self._lock:
                self.misses.append(key)
            print(
                f"CASSETTE MISS {key} {path} model={body.get('model', '')!r} "
                f"cassette={self.path}",
                file=sys.stderr,
            )
            return 500, _error(f"cassette miss: no entry for key {key} ({path})")

        return self._record(method, path, raw_body, key, material)

    def _record(
        self, method: str, path: str, raw_body: bytes, key: str, material: dict
    ) -> tuple[int, bytes]:
        req = urllib.request.Request(
            self.upstream + path,
            data=raw_body if method == "POST" else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                response = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Ollama's own error answer (e.g. a model it does not have):
            # passed through, not recorded.
            return exc.code, exc.read()
        except urllib.error.URLError as exc:
            self._unrecorded(key, path, material, f"upstream unreachable: {exc.reason}")
            return 502, _error(f"cassette upstream {self.upstream} unreachable: {exc}")

        entry = {
            "key": key,
            "path": path,
            "request": material,
            "response": response,
            "source": "recorded",
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with self._lock:
            if key not in self.entries:
                self.entries[key] = entry
                self.recorded.append(key)
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            entry = self.entries[key]
            self.served.append(entry)
        return 200, encode_response(entry["response"])


    def _unrecorded(self, key: str, path: str, material: dict, why: str) -> None:
        """A request with no entry and no upstream to answer it: a miss,
        as in replay (the test fails naming it)."""
        with self._lock:
            self.misses.append(key)
        print(
            f"CASSETTE MISS {key} {path} model={material.get('model', '')!r} "
            f"cassette={self.path} (record: {why})",
            file=sys.stderr,
        )


def _error(message: str) -> bytes:
    return encode_response({"error": message})


class _Handler(BaseHTTPRequestHandler):
    cassette: Cassette
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def _handle(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length) if length else b""
        url = urlsplit(self.path)

        if url.path.startswith("/__cassette/"):
            status, body = self._control(url.path, parse_qs(url.query))
        elif url.path in API_PATHS:
            down, delay = self.cassette._take_faults(url.path)
            if down:
                status, body = 503, _error("cassette: simulated Ollama outage")
            else:
                if delay:
                    time.sleep(delay)
                    try:
                        request = json.loads(raw_body) if raw_body else {}
                    except json.JSONDecodeError:
                        request = {}
                    with self.cassette._lock:
                        self.cassette.delayed.append((url.path, request))
                try:
                    status, body = self.cassette.answer(method, url.path, raw_body)
                except json.JSONDecodeError as exc:
                    status, body = 400, _error(f"cassette: request is not JSON: {exc}")
        else:
            status, body = 404, _error(f"cassette: unknown path {url.path}")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _control(self, path: str, query: dict) -> tuple[int, bytes]:
        cassette = self.cassette
        try:
            if path == "/__cassette/down":
                cassette.set_down(int(query.get("n", ["1"])[0]))
            elif path == "/__cassette/slow":
                n = query.get("n")
                only = query.get("path")
                cassette.set_slow(
                    float(query["seconds"][0]), int(n[0]) if n else None,
                    only[0] if only else None,
                )
            elif path == "/__cassette/reset":
                cassette.reset_faults()
            else:
                return 404, _error(f"cassette: unknown control {path}")
        except (KeyError, ValueError) as exc:
            return 400, _error(f"cassette: bad control query: {exc}")
        return 200, encode_response(cassette.fault_state())


def open_cassette(journey: str, *, record: bool | None = None) -> Cassette:
    """The cassette for ``journey`` at ``cassettes/<journey>.jsonl``, not started."""
    return Cassette(CASSETTE_DIR / f"{journey}.jsonl", record=record)


# ---------------------------------------------------------------------------
# Usage log and pruning
# ---------------------------------------------------------------------------

USAGE_LOG_ENV = "AH_CASSETTE_LOG"
# Hand-written entries that exercise every endpoint (test_cassette.py).
_NEVER_PRUNED = frozenset({"cassette_selftest"})


def current_test() -> str:
    """The running test's node id (pytest's ``PYTEST_CURRENT_TEST``
    without its phase), empty outside pytest."""
    return os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]


def log_use(record: dict) -> None:
    """Append *record* to the usage log named by ``AH_CASSETTE_LOG``, if set."""
    path = os.environ.get(USAGE_LOG_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def log_test_report(report) -> None:
    """Log one pytest phase report (``pytest_runtest_logreport``): the
    test's outcome and, for a skip or an xfail, its reason."""
    if not os.environ.get(USAGE_LOG_ENV):
        return
    xfail = hasattr(report, "wasxfail")
    reason = ""
    if xfail:
        reason = str(report.wasxfail)
    elif report.skipped:
        longrepr = report.longrepr
        reason = str(longrepr[2] if isinstance(longrepr, tuple) else longrepr)
    log_use({"test": report.nodeid, "when": report.when,
             "outcome": "xfailed" if xfail else report.outcome, "reason": reason})


_KEY_RE = re.compile(r"\b[0-9a-f]{64}\b")


def keys_named_in_journeys(journeys_dir: Path = Path(__file__).parent) -> set[str]:
    """Every cassette key the journeys' sources name: the skip markers'
    ``*_MISSING_KEYS`` and the entries a test inspects."""
    return {key for source in journeys_dir.glob("*.py")
            for key in _KEY_RE.findall(source.read_text(encoding="utf-8"))}


def plan_prune(records: list[dict], cassette_dir: Path = CASSETTE_DIR,
               named_keys: set[str] | None = None) -> dict[str, dict]:
    """Per cassette in *cassette_dir* (by stem): ``before`` and ``keep``
    (the keys to keep, None to leave the file untouched) and ``why``.
    *named_keys* (default :func:`keys_named_in_journeys`) are always kept."""
    if named_keys is None:
        named_keys = keys_named_in_journeys()
    passed: dict[str, bool] = {}
    reasons: list[str] = []
    used: dict[Path, set[str]] = {}
    missed: set[Path] = set()
    openers: dict[Path, set[str]] = {}
    for record in records:
        if "outcome" in record:
            ok = record["outcome"] == "passed"
            passed[record["test"]] = passed.get(record["test"], True) and ok
            if record["reason"]:
                reasons.append(record["reason"])
        elif "cassette" in record:
            path = Path(record["cassette"])
            used.setdefault(path, set()).update(record["served"])
            openers.setdefault(path, set()).add(record["test"])
            if record["misses"]:
                missed.add(path)
    if not passed:
        raise ValueError("the usage log holds no test outcome: run the journeys "
                         f"with {USAGE_LOG_ENV} set first")

    plan: dict[str, dict] = {}
    for path in sorted(cassette_dir.glob("*.jsonl")):
        name, resolved = path.stem, path.resolve()
        before = len(load_entries(path))
        named = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
        not_passed = sorted(t for t in openers.get(resolved, ())
                            if not t or not passed.get(t, False))
        why = ""
        if name in _NEVER_PRUNED:
            why = "hand-written self-test entries"
        elif resolved not in used:
            why = "no test opened it"
        elif resolved in missed:
            why = "a request missed it"
        elif not_passed:
            why = f"opened by a test that did not pass: {', '.join(not_passed)}"
        elif any(named.search(reason) for reason in reasons):
            why = "a skipped or xfailed test's reason names it"
        plan[name] = {"before": before,
                      "keep": None if why else used[resolved] | named_keys, "why": why}
    return plan


def prune(records: list[dict], cassette_dir: Path = CASSETTE_DIR,
          named_keys: set[str] | None = None, *, dry_run: bool = False) -> dict[str, dict]:
    """Apply :func:`plan_prune`: rewrite each prunable cassette with only
    its used (and named) entries, lines kept byte for byte in their order.
    Returns the plan with ``after`` set."""
    plan = plan_prune(records, cassette_dir, named_keys)
    for name, item in plan.items():
        path = cassette_dir / f"{name}.jsonl"
        keep = item["keep"]
        if keep is None:
            item["after"] = item["before"]
            continue
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        kept = [line for line in lines
                if line.strip() and json.loads(line)["key"] in keep]
        item["after"] = len(kept)
        if not dry_run and len(kept) != item["before"]:
            path.write_text("".join(kept), encoding="utf-8")
    return plan


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m journeys.cassette")
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("prune", help="drop the entries no journey requested")
    cmd.add_argument("--used-keys", required=True, type=Path,
                     help=f"the usage log a full journeys run wrote ({USAGE_LOG_ENV})")
    cmd.add_argument("--dry-run", action="store_true", help="report, change nothing")
    args = parser.parse_args(argv)
    records = [json.loads(line) for line in
               args.used_keys.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        plan = prune(records, dry_run=args.dry_run)
    except ValueError as exc:
        print(f"prune: {exc}", file=sys.stderr)
        return 2
    for name, item in plan.items():
        note = f"  (left alone: {item['why']})" if item["why"] else ""
        print(f"{name}.jsonl: {item['before']} -> {item['after']}{note}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
