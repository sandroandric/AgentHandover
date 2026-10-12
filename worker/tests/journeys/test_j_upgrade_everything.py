"""J-upgrade-everything: an upgrader who chose "Upgrade everything" moves
from nomic-embed-text to EmbeddingGemma 2 without a moment of mixed or
broken search.

The worker's own pieces (``KnowledgeBase``, ``VectorKB``,
``EmbeddingUpgrade``, the ``stages/upgrade`` job, ``SessionLinker``,
``TaskSegmenter``, ``ProcedureMatcher``, ``find_matching_sop``) run on a
daemon database with Ollama answered by recorded cassettes.

Skills (``search_threshold`` cassette, recorded for both models): the eight
procedures of ``test_search_threshold`` are approved and embedded with
nomic; the memory offer says "everything".  The re-embed runs three
procedures per job; Ollama goes down mid-way; the worker restarts.  Then:

* during the re-embed every search ranks only nomic vectors, exactly as
  before it started;
* after the outage and the restart it resumes: each procedure is
  re-embedded once;
* it switches: every row is EmbeddingGemma's, config.toml says so, the
  status is done;
* Skill search, the procedure matcher and SOP dedup then work on
  EmbeddingGemma's thresholds (the pins of ``test_search_threshold``,
  through the switched objects).

Everything (``upgrade_everything`` cassette): a database with one vector
of each source type (procedure, daily summary, profile, preserved text,
an annotation's frame vector) re-embeds and switches through the real
Ollama checks (``/api/version``, ``/api/tags``); the task segmenter then
reuses the re-embedded frame vector without asking Ollama.  Its requests
are listed by :func:`_everything_requests`; until they are recorded the
test skips naming them.

Record: ``AH_RECORD=1 AH_RECORD_UPSTREAM=http://<ollama> pytest <this file>``
(the Ollama there must serve nomic-embed-text and embeddinggemma-2:440m).
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agenthandover_worker.config import Config, EmbeddingConfig
from agenthandover_worker.embedding_upgrade import TARGET_MODEL, EmbeddingUpgrade
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.main import _embed_procedure, _procedure_text
from agenthandover_worker.model_profiles import (
    EMBEDDING_MODEL,
    EMBEDDING_PROFILES,
    MEMORY_OFFER_FILE,
)
from agenthandover_worker.procedure_matcher import ProcedureMatcher
from agenthandover_worker.session_linker import SessionLinker
from agenthandover_worker.sop_dedup import find_matching_sop
from agenthandover_worker.stages import daily
from agenthandover_worker.stages import upgrade as upgrade_stage
from agenthandover_worker.task_segmenter import (
    AnnotatedFrame,
    SegmenterConfig,
    TaskSegment,
    TaskSegmenter,
    frame_text,
)
from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig
from daemon_db import create_daemon_db, insert_event
from journeys.cassette import CASSETTE_DIR, load_entries, open_cassette, request_key
from journeys.test_search_threshold import LABELS, PROCEDURES, QUESTIONS, REDRAFTS, UNRELATED

GEMMA = EMBEDDING_PROFILES[TARGET_MODEL]
NOMIC = EMBEDDING_PROFILES[EMBEDDING_MODEL]

CONFIG = """[vlm]
annotation_model = "qwen3.5:4b"

[embedding]
model = "nomic-embed-text"
"""


def _install(tmp_path: Path) -> SimpleNamespace:
    db = create_daemon_db(tmp_path / "events.db")
    kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=db)
    kb.ensure_structure()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / MEMORY_OFFER_FILE).write_text(json.dumps({"choice": "everything"}))
    config_file = tmp_path / "config.toml"
    config_file.write_text(CONFIG)
    return SimpleNamespace(db=db, kb=kb, data_dir=data_dir, config_file=config_file)


def _upgrade(world, host: str, **kwargs) -> EmbeddingUpgrade:
    return EmbeddingUpgrade(
        world.db, knowledge_base=world.kb, data_dir=world.data_dir,
        config_file=world.config_file, ollama_host=host, model=EMBEDDING_MODEL,
        # One text per request: the requests the cassettes hold.
        batch_size=1, **kwargs,
    )


def _ctx(world, vkb: VectorKB, upgrade: EmbeddingUpgrade) -> SimpleNamespace:
    """The worker context's fields the upgrade job and the switch use."""
    return SimpleNamespace(
        cfg=Config(path=world.config_file, embedding=EmbeddingConfig(model=EMBEDDING_MODEL)),
        db_path=world.db,
        vector_kb=vkb,
        session_linker=SessionLinker(world.kb, vector_kb=vkb),
        task_segmenter=TaskSegmenter(SegmenterConfig(embedding_model=EMBEDDING_MODEL), vector_kb=vkb),
        procedure_matcher=ProcedureMatcher(kb=world.kb, vector_kb=vkb),
        procedure_writer=None,
        embedding_upgrade=upgrade,
    )


def _models(db: Path) -> set[str]:
    conn = sqlite3.connect(db)
    try:
        return {m for (m,) in conn.execute("SELECT model FROM vector_store")}
    finally:
        conn.close()


def _rankings(vkb: VectorKB) -> list[list[tuple[str, float, str]]]:
    out = []
    for _, question in QUESTIONS:
        results = vkb.search(question, top_k=len(PROCEDURES), source_types=["procedure"])
        out.append([(r.source_id, r.score, r.model) for r in results])
    return out


# ---------------------------------------------------------------------------
# Skills: replayed from the search_threshold recordings
# ---------------------------------------------------------------------------


def test_skills_switch_to_embeddinggemma_without_mixed_ranking(tmp_path: Path) -> None:
    world = _install(tmp_path)
    for procedure in PROCEDURES:
        world.kb.save_procedure(dict(procedure))
        # The recorded requests are of exactly this text.
        assert _procedure_text(world.kb.get_procedure(procedure["id"])) == _procedure_text(procedure)

    with open_cassette("search_threshold") as tape:
        vkb = VectorKB(world.db, VectorKBConfig(ollama_host=tape.host, embedding_model=EMBEDDING_MODEL))
        for procedure in PROCEDURES:
            _embed_procedure(vkb, world.kb.get_procedure(procedure["id"]))
        before = _rankings(vkb)
        assert all(len(r) == len(PROCEDURES) for r in before)

        # The search_threshold recordings hold no /api/version or /api/tags:
        # the Ollama checks run in test_upgrade_everything_re_embeds_every_source_type.
        upgrade = _upgrade(world, tape.host, rows_per_job=3, preflight=lambda: None)
        ctx = _ctx(world, vkb, upgrade)
        tick = SimpleNamespace(embedding_upgrade=None)

        upgrade_stage.run(ctx, tick)
        assert tick.embedding_upgrade == {
            "state": "running", "done": 3, "total": 8, "dropped": 0, "error": None,
        }
        # Mid-way: the same nomic rankings, nothing of the new model ranked.
        assert _rankings(vkb) == before
        assert _models(world.db) == {EMBEDDING_MODEL}

        # Ollama goes away for a request: the job backs off, nothing lost.
        tape.set_down(1)
        upgrade_stage.run(ctx, tick)
        assert tick.embedding_upgrade["state"] == "running"
        assert "outage" in tick.embedding_upgrade["error"]
        assert tick.embedding_upgrade["done"] == 3
        assert _rankings(vkb) == before

        # The worker restarts: a new process resumes from the staged rows.
        upgrade.close()
        upgrade = ctx.embedding_upgrade = _upgrade(world, tape.host, rows_per_job=3, preflight=lambda: None)
        for _ in range(4):
            upgrade_stage.run(ctx, tick)
            if tick.embedding_upgrade["state"] == "done":
                break
        assert tick.embedding_upgrade == {
            "state": "done", "done": 8, "total": 8, "dropped": 0, "error": None,
        }
        gemma_texts = [
            e["request"]["input"][0] for e in tape.served
            if e["path"] == "/api/embed" and e["request"]["model"] == TARGET_MODEL
        ]
        assert sorted(gemma_texts) == sorted(
            GEMMA.doc_prefix + _procedure_text(p) for p in PROCEDURES
        ), "each procedure re-embedded exactly once"

        # Switched: rows, objects and config.toml.
        assert _models(world.db) == {TARGET_MODEL}
        assert vkb.config.embedding_model == TARGET_MODEL
        assert ctx.cfg.embedding.model == TARGET_MODEL
        assert world.config_file.read_text() == CONFIG.replace(EMBEDDING_MODEL, TARGET_MODEL)
        thresholds = vkb.thresholds
        assert thresholds == GEMMA.thresholds

        # Skill search on EmbeddingGemma's floor.
        after = _rankings(vkb)
        firsts = 0
        for (want, question), ranked in zip(QUESTIONS, after):
            assert {model for _, _, model in ranked} == {TARGET_MODEL}
            ids = [sid for sid, _, _ in ranked]
            assert want in ids[:2], question
            assert dict((sid, s) for sid, s, _ in ranked)[want] >= thresholds.min_score
            firsts += ids[0] == want
        assert firsts >= len(QUESTIONS) - 1

        # The matcher: no task label reaches another procedure, most reach
        # their own, through the context's matcher.
        matched = 0
        for want, label in LABELS:
            semantic = vkb.search(label, top_k=5, source_types=["procedure"], min_score=thresholds.matcher)
            assert {r.source_id for r in semantic} <= {want}, label
            found = ctx.procedure_matcher.match_segment(
                TaskSegment(segment_id=want, cluster_id=0, task_label=label),
            )
            matched += bool(semantic) and found[0][0] == want
        assert matched >= 0.75 * len(LABELS)
        for text in UNRELATED:
            assert vkb.search(text, source_types=["procedure"], min_score=thresholds.matcher) == []

        # SOP dedup: a redraft finds its procedure.
        existing = [{"slug": p["id"], "title": p["title"]} for p in PROCEDURES]
        for want, text in REDRAFTS:
            title, _, description = text.partition(" | ")
            index = find_matching_sop({"title": title, "description": description}, existing, vector_kb=vkb)
            assert index is not None and existing[index]["slug"] == want, text

        # The linker and the segmenter use EmbeddingGemma's numbers.
        assert ctx.session_linker._vector_kb.thresholds.linker_cos == GEMMA.thresholds.linker_cos
        assert ctx.task_segmenter._similarity_threshold() == GEMMA.thresholds.segmenter
        vkb.close()
        upgrade.close()


# ---------------------------------------------------------------------------
# Every source type: the upgrade_everything recordings
# ---------------------------------------------------------------------------

EVERYTHING = "upgrade_everything"
DAY = "2026-10-11"
ANNOTATION = {
    "app": "Google Chrome",
    "location": "dashboard.stripe.com/payouts",
    "task_context": {"what_doing": "Reconciling the Stripe payout in the dashboard", "is_workflow": True},
}
PRESERVED = "Stripe Payouts: payout po_1Q2 of $4,210.55 arrived on Oct 10 at Chase checking"
QUESTION = QUESTIONS[0]


def _everything_world(tmp_path: Path) -> tuple[SimpleNamespace, list[tuple[str, str, str, str]]]:
    """A database holding one vector source of each type; returns it and
    (source type, source id, text, nomic prefix) of each."""
    world = _install(tmp_path)
    procedure = dict(PROCEDURES[0])
    world.kb.save_procedure(procedure)
    world.kb.save_daily_summary(DAY, {
        "task_count": 2, "active_hours": 3.5,
        "tasks": [
            {"intent": "Reconcile the Stripe payout"},
            {"intent": "Review the pull request for the billing service"},
        ],
    })
    world.kb.update_profile({"tools": {"Google Chrome": 42, "Slack": 17}})
    conn = sqlite3.connect(world.db)
    event_id = insert_event(conn, processed=1)
    conn.execute(
        "UPDATE events SET scene_annotation_json = ? WHERE id = ?",
        (json.dumps(ANNOTATION), event_id),
    )
    conn.execute(
        "INSERT INTO preserved_text (event_id, source, timestamp, text_content) "
        "VALUES (?, 'ocr', '2026-10-10T09:00:00Z', ?)",
        (event_id, PRESERVED),
    )
    conn.commit()
    conn.close()
    frame = AnnotatedFrame.from_event({"id": event_id, "scene_annotation_json": json.dumps(ANNOTATION)})
    sources = [
        ("procedure", procedure["id"], _procedure_text(world.kb.get_procedure(procedure["id"])), NOMIC.doc_prefix),
        ("daily_summary", DAY, daily.summary_text(
            DAY, 2, 3.5,
            ["Reconcile the Stripe payout", "Review the pull request for the billing service"],
        ), NOMIC.doc_prefix),
        ("profile", "user", daily._profile_text(world.kb.get_profile()), NOMIC.doc_prefix),
        ("preserved_text", f"{event_id}:ocr", PRESERVED, NOMIC.doc_prefix),
        ("annotation", event_id, frame_text(frame), NOMIC.cluster_prefix),
    ]
    return world, sources


def _everything_requests(sources) -> dict[str, str]:
    """{cassette key: what it is} of every request the journey makes."""
    requests = {
        request_key("/api/version", {}): "GET /api/version",
        request_key("/api/tags", {}): "GET /api/tags",
        request_key("/api/embed", {
            "model": TARGET_MODEL, "input": [GEMMA.query_prefix + QUESTION[1]],
        }): f"{TARGET_MODEL} query: {QUESTION[1]}",
    }
    for stype, _, text, nomic_prefix in sources:
        gemma_prefix = GEMMA.cluster_prefix if stype == "annotation" else GEMMA.doc_prefix
        requests[request_key("/api/embed", {
            "model": EMBEDDING_MODEL, "input": [nomic_prefix + text],
        })] = f"{EMBEDDING_MODEL} {stype}"
        requests[request_key("/api/embed", {
            "model": TARGET_MODEL, "input": [gemma_prefix + text],
        })] = f"{TARGET_MODEL} {stype}"
    return requests


def test_upgrade_everything_re_embeds_every_source_type(tmp_path: Path) -> None:
    world, sources = _everything_world(tmp_path)
    path = CASSETTE_DIR / f"{EVERYTHING}.jsonl"
    if os.environ.get("AH_RECORD") != "1":
        recorded = load_entries(path) if path.exists() else {}
        missing = {k: v for k, v in _everything_requests(sources).items() if k not in recorded}
        if missing:
            pytest.skip(
                f"cassette {EVERYTHING}.jsonl lacks {len(missing)} request(s) (record on "
                "the test Mac): " + "; ".join(f"{k} ({v})" for k, v in sorted(missing.items()))
            )

    with open_cassette(EVERYTHING) as tape:
        vkb = VectorKB(world.db, VectorKBConfig(ollama_host=tape.host, embedding_model=EMBEDDING_MODEL))
        for stype, sid, text, prefix in sources:
            assert vkb.upsert(stype, sid, text, prefix=prefix), stype
        upgrade = _upgrade(world, tape.host, rows_per_job=2)
        ctx = _ctx(world, vkb, upgrade)
        tick = SimpleNamespace(embedding_upgrade=None)
        for _ in range(5):
            upgrade_stage.run(ctx, tick)
            if tick.embedding_upgrade["state"] == "done":
                break
        assert tick.embedding_upgrade == {
            "state": "done", "done": 5, "total": 5, "dropped": 0, "error": None,
        }
        assert _models(world.db) == {TARGET_MODEL}
        assert f'model = "{TARGET_MODEL}"' in world.config_file.read_text()

        # Skill search on the switched store (the question's procedure
        # score is above the floor for both models, test_search_threshold).
        results = vkb.search(QUESTION[1], source_types=["procedure"], min_score=vkb.thresholds.min_score)
        assert [r.source_id for r in results] == [QUESTION[0]]

        # The segmenter reuses the re-embedded frame vector: no request.
        _, event_id, text, _ = sources[-1]
        frame = AnnotatedFrame.from_event({"id": event_id, "scene_annotation_json": json.dumps(ANNOTATION)})
        served = len(tape.served)
        assert ctx.task_segmenter._cached_embeddings([frame], [text])[0] is not None
        assert len(tape.served) == served
        vkb.close()
        upgrade.close()
