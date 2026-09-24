"""``agenthandover bench``: replay cassette prompts against a tier's models.

The "live" Ollama here is a replaying cassette server holding the same
entries, so a prompt sent to its recorded model gets the recorded answer
back (golden score 1.0) and a prompt sent to any other model is a miss
(an Ollama error, counted per candidate).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agenthandover_worker import bench
from journeys.cassette import Cassette, normalise_request, request_key

ANN_FORMAT = {"type": "object", "properties": {"app": {"type": "string"}}}
GOLDEN_ANN = json.dumps({
    "app": "Google Chrome", "location": "https://news.ycombinator.com/",
    "key_text": {"urls": ["https://news.ycombinator.com/item?id=1"], "names": []},
    "visible_items": ["Show HN: a thing", "Ask HN: another"],
})
GOLDEN_SOP = json.dumps({"title": "Daily news digest", "steps": ["open HN", "copy titles"]})


def _entry(path: str, request: dict, response: dict) -> str:
    return json.dumps({
        "key": request_key(path, request), "path": path,
        "request": normalise_request(request),
        "response": response, "source": "hand-written",
    })


@pytest.fixture()
def cassette_file(tmp_path: Path) -> Path:
    timing = {"eval_count": 100, "eval_duration": 2_000_000_000, "done": True}
    lines = [
        _entry("/api/generate",
               {"model": "qwen3.5:4b", "prompt": "annotate frame 1", "think": False,
                "options": {"num_predict": 300}, "format": ANN_FORMAT},
               {"model": "qwen3.5:4b", "response": GOLDEN_ANN, **timing}),
        _entry("/api/generate",
               {"model": "qwen3.5:4b", "prompt": "write the skill", "think": False,
                "system": "You write Skills."},
               {"model": "qwen3.5:4b", "response": GOLDEN_SOP, **timing}),
        _entry("/api/generate",
               {"model": "qwen3.5:4b", "prompt": "", "think": False},
               {"model": "qwen3.5:4b", "response": "", "done": True}),
        _entry("/api/generate",
               {"model": "qwen3.5:4b", "prompt": "what is on screen", "images": ["aGVsbG8="]},
               {"model": "qwen3.5:4b", "response": "{}", **timing}),
        _entry("/api/chat",
               {"model": "qwen3.5:4b", "messages": [{"role": "user", "content": "hi"}]},
               {"model": "qwen3.5:4b", "message": {"content": "hi"}}),
        _entry("/api/version", {}, {"version": "0.34.2"}),
    ]
    path = tmp_path / "focus.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_prompts_are_split_by_role_and_unreplayable_ones_skipped(cassette_file):
    prompts, skipped = bench.load_prompts(cassette_file)
    assert [(p.role, p.request["prompt"]) for p in prompts] == [
        ("annotation", "annotate frame 1"), ("sop", "write the skill"),
    ]
    assert skipped == {"chat": 1, "images": 1, "preload": 1, "other": 1}


def test_bench_reports_tok_per_s_and_golden_score(cassette_file, tmp_path):
    # The recommended tier runs qwen3.5:4b for both roles, the cassette's model.
    with Cassette(cassette_file, record=False) as live:
        report = bench.bench(
            "recommended", cassette_file, host=live.host, annotators=["qwen3.5:2b"],
        )
        assert len(live.misses) == 1  # the candidate's call: expected here
        live.misses.clear()
    ann, sop = report["roles"]["annotation"], report["roles"]["sop"]
    assert report["ollama_version"] == "0.34.2"
    assert (ann["model"], ann["calls"], ann["errors"]) == ("qwen3.5:4b", 1, 0)
    assert ann["tokens_per_second"] == pytest.approx(50.0)
    assert ann["golden_score"] == 1.0
    assert ann["field_accuracy"] == {
        "app": 1.0, "location": 1.0, "key_text.urls": 1.0, "visible_items": 1.0,
    }
    assert (sop["model"], sop["golden_score"]) == ("qwen3.5:4b", 1.0)
    # A candidate the cassette never answered for fails every call.
    candidate = report["annotators"]["qwen3.5:2b"]
    assert (candidate["calls"], candidate["errors"]) == (1, 1)
    assert "tier recommended" in bench.format_table(report)


def test_golden_score_compares_json_fields_and_text():
    gold = json.dumps({"app": "Notes", "key_text": {"urls": ["a", "b"]}})
    assert bench.golden_score(gold, gold) == 1.0
    reordered = json.dumps({"key_text": {"urls": ["b", "a"]}, "app": " notes "})
    assert bench.golden_score(reordered, gold) == 1.0
    assert bench.golden_score(json.dumps({"app": "Slack"}), gold) == 0.0
    assert bench.golden_score("not json", gold) == 0.0
    assert 0.0 < bench.golden_score("hello world", "hello there") < 1.0


def test_field_agreement_scores_only_fields_the_golden_has():
    gold = json.dumps({"app": "Notes", "location": "x"})
    live = json.dumps({"app": "notes", "location": "y", "visible_items": ["z"]})
    assert bench.field_agreement(live, gold) == {"app": True, "location": False}


def test_cli_writes_the_report(cassette_file, tmp_path, capsys):
    out = tmp_path / "bench" / "recommended.json"
    with Cassette(cassette_file, record=False) as live:
        rc = bench.main([
            "--tier", "recommended", "--cassette", str(cassette_file),
            "--host", live.host, "--out", str(out),
        ])
    assert rc == 0
    assert json.loads(out.read_text())["tier"]["id"] == "recommended"
    assert "annotation: qwen3.5:4b" in capsys.readouterr().out
    assert bench.main(["--tier", "standard", "--cassette", str(tmp_path / "nope")]) == 2
