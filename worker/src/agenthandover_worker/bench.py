"""Model bake-off: replay a journey cassette's prompts against live models.

``agenthandover bench --tier ID --cassette PATH`` (the CLI runs
``python -m agenthandover_worker.bench``) sends every text prompt recorded in
a journey cassette to the models of a tier on a live Ollama and reports, per
role, tokens per second and a golden-diff score: how closely the live answer
matches the answer recorded in the cassette.  So the next model decision
for the tier table (``model_profiles.TIERS``) is measured, not assumed.

Roles: a request whose ``format`` is the scene-annotation schema (it has an
``app`` property) is an *annotation* call and goes to the tier's
``annotation_model``; every other text prompt (Skill generation, synthesis,
reasoning, frame diffs) goes to its ``sop_model``.  Requests carrying images
(the cassette keeps only their hashes) and ``/api/chat`` requests cannot be
re-sent and are counted as skipped; so are model preloads (empty prompts).

``--annotators a,b,c`` also sends every annotation prompt to each candidate
and reports per-field agreement with the recorded answer for the fields the
compact annotation schema extracts (``app``, ``location``,
``key_text.urls``, ``visible_items``), alongside tok/s.

The report is JSON (``--out``), printed as a table too.  The bench never
writes to the worker's data; it only reads the cassette and calls Ollama.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agenthandover_worker.model_profiles import TIERS, get_tier
from agenthandover_worker.ollama import OllamaClient, OllamaError

# The fields the compact annotation schema extracts (WP-C3b); a field the
# golden answer does not carry is not scored.
ANNOTATION_FIELDS = ("app", "location", "key_text.urls", "visible_items")


@dataclass
class Prompt:
    """One replayable request from a cassette."""

    key: str
    role: str  # "annotation" | "sop"
    request: dict
    golden: str


@dataclass
class CallResult:
    key: str
    model: str
    tokens_per_second: float | None
    eval_count: int | None
    seconds: float
    golden_score: float
    fields: dict[str, bool] = field(default_factory=dict)
    error: str = ""


def load_prompts(cassette: Path) -> tuple[list[Prompt], dict[str, int]]:
    """The replayable prompts of *cassette*, and counts of what was skipped."""
    prompts: list[Prompt] = []
    skipped = {"chat": 0, "images": 0, "preload": 0, "other": 0}
    for line in cassette.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        path, request = entry.get("path"), entry.get("request") or {}
        if path == "/api/chat":
            skipped["chat"] += 1
            continue
        if path != "/api/generate":
            skipped["other"] += 1
            continue
        if request.get("images"):
            skipped["images"] += 1
            continue
        if not request.get("prompt"):
            skipped["preload"] += 1
            continue
        fmt = request.get("format")
        is_annotation = isinstance(fmt, dict) and "app" in (fmt.get("properties") or {})
        prompts.append(Prompt(
            key=entry["key"],
            role="annotation" if is_annotation else "sop",
            request=request,
            golden=str((entry.get("response") or {}).get("response", "")),
        ))
    return prompts, skipped


def _as_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            out.update(_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: value}


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if isinstance(value, list):
        return sorted(json.dumps(_norm(v), sort_keys=True) for v in value)
    return value


def golden_score(live: str, golden: str) -> float:
    """1.0 for the recorded answer.  JSON objects score the share of the
    golden answer's leaf fields the live answer matches (strings compared
    case- and whitespace-insensitively, lists as sets); other text scores
    difflib's similarity ratio."""
    live_obj, gold_obj = _as_json(live), _as_json(golden)
    if isinstance(gold_obj, dict):
        if not isinstance(live_obj, dict):
            return 0.0
        gold_leaves, live_leaves = _leaves(gold_obj), _leaves(live_obj)
        if not gold_leaves:
            return 1.0 if not live_leaves else 0.0
        hits = sum(
            1 for k, v in gold_leaves.items()
            if k in live_leaves and _norm(live_leaves[k]) == _norm(v)
        )
        return hits / len(gold_leaves)
    return difflib.SequenceMatcher(None, live, golden).ratio()


def field_agreement(live: str, golden: str) -> dict[str, bool]:
    """Per annotation field: does the live answer match the recorded one?"""
    live_leaves = _leaves(_as_json(live) or {})
    gold_leaves = _leaves(_as_json(golden) or {})
    return {
        f: f in live_leaves and _norm(live_leaves[f]) == _norm(gold_leaves[f])
        for f in ANNOTATION_FIELDS if f in gold_leaves
    }


def run_prompt(client: OllamaClient, prompt: Prompt, model: str, timeout: float) -> CallResult:
    req = prompt.request
    began = time.monotonic()
    try:
        response = client.generate(
            model,
            req["prompt"],
            system=req.get("system"),
            options=req.get("options"),
            format=req.get("format"),
            think=req.get("think"),
            timeout=timeout,
        )
    except OllamaError as exc:
        return CallResult(prompt.key, model, None, None, time.monotonic() - began, 0.0,
                          error=str(exc))
    tps = None
    if response.eval_count and response.eval_duration:
        tps = response.eval_count / (response.eval_duration / 1e9)
    result = CallResult(
        key=prompt.key, model=model, tokens_per_second=tps,
        eval_count=response.eval_count, seconds=response.elapsed,
        golden_score=golden_score(response.text, prompt.golden),
    )
    if prompt.role == "annotation":
        result.fields = field_agreement(response.text, prompt.golden)
    return result


def summarise(results: list[CallResult]) -> dict:
    """Mean tok/s, mean golden score, per-field accuracy, error count."""
    ok = [r for r in results if not r.error]
    rates = [r.tokens_per_second for r in ok if r.tokens_per_second]
    fields: dict[str, list[bool]] = {}
    for r in ok:
        for name, hit in r.fields.items():
            fields.setdefault(name, []).append(hit)
    return {
        "calls": len(results),
        "errors": len(results) - len(ok),
        "tokens_per_second": sum(rates) / len(rates) if rates else None,
        "golden_score": sum(r.golden_score for r in ok) / len(ok) if ok else None,
        "seconds_per_call": sum(r.seconds for r in ok) / len(ok) if ok else None,
        "field_accuracy": {k: sum(v) / len(v) for k, v in fields.items()},
    }


def bench(
    tier_id: str,
    cassette: Path,
    *,
    host: str,
    annotators: list[str] | None = None,
    limit: int | None = None,
    timeout: float = 600.0,
) -> dict:
    tier = get_tier(tier_id)
    if tier is None:
        raise ValueError(
            f"unknown tier {tier_id!r}; one of " + ", ".join(t.id for t in TIERS)
        )
    prompts, skipped = load_prompts(cassette)
    by_role = {
        role: [p for p in prompts if p.role == role][:limit]
        for role in ("annotation", "sop")
    }
    client = OllamaClient(host)
    report: dict = {
        "tier": asdict(tier),
        "cassette": str(cassette),
        "host": host,
        "ollama_version": client.version(),
        "skipped": skipped,
        "roles": {},
        "annotators": {},
    }
    for role, model in (("annotation", tier.annotation_model), ("sop", tier.sop_model)):
        results = [run_prompt(client, p, model, timeout) for p in by_role[role]]
        report["roles"][role] = {
            "model": model, **summarise(results), "calls_detail": [asdict(r) for r in results],
        }
    for candidate in annotators or []:
        results = [run_prompt(client, p, candidate, timeout) for p in by_role["annotation"]]
        report["annotators"][candidate] = {
            **summarise(results), "calls_detail": [asdict(r) for r in results],
        }
    return report


def format_table(report: dict) -> str:
    def fmt(v: float | None, spec: str) -> str:
        return "-" if v is None else format(v, spec)

    lines = [
        f"tier {report['tier']['id']} on Ollama {report['ollama_version']} "
        f"({report['host']}); skipped {report['skipped']}",
        f"{'role / model':44} {'calls':>5} {'err':>4} {'tok/s':>7} {'golden':>7} {'s/call':>7}",
    ]
    rows = [(f"{role}: {r['model']}", r) for role, r in report["roles"].items()]
    rows += [(f"candidate: {m}", r) for m, r in report["annotators"].items()]
    for name, r in rows:
        lines.append(
            f"{name:44} {r['calls']:>5} {r['errors']:>4} "
            f"{fmt(r['tokens_per_second'], '.1f'):>7} {fmt(r['golden_score'], '.3f'):>7} "
            f"{fmt(r['seconds_per_call'], '.1f'):>7}"
        )
        for f, acc in r["field_accuracy"].items():
            lines.append(f"    {f:40} {acc:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agenthandover bench",
        description="Replay a journey cassette's prompts against a tier's live models.",
    )
    parser.add_argument("--tier", required=True, choices=[t.id for t in TIERS])
    parser.add_argument("--cassette", required=True, type=Path,
                        help="a journey cassette (worker/tests/journeys/cassettes/*.jsonl)")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--annotators", default="",
                        help="comma-separated candidate annotation models")
    parser.add_argument("--limit", type=int, default=None,
                        help="at most this many prompts per role")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per call")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    if not args.cassette.is_file():
        print(f"no cassette at {args.cassette}", file=sys.stderr)
        return 2
    try:
        report = bench(
            args.tier, args.cassette, host=args.host,
            annotators=[m for m in args.annotators.split(",") if m.strip()],
            limit=args.limit, timeout=args.timeout,
        )
    except OllamaError as exc:
        print(f"Ollama at {args.host} is not answering: {exc}", file=sys.stderr)
        return 1
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
