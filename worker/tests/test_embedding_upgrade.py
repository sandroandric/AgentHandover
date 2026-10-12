"""Regression pins for "Upgrade everything" (``embedding_upgrade``): the
re-embed's progress (the staging table as its cursor), the switch's
transaction, the config write, the in-process switch and the status
object.  The journey is ``journeys/test_j_upgrade_everything.py``.

Ollama is a fake here: its vectors are a hash of the prefixed text, so a
vector says which text and prefix made it.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agenthandover_worker import embedding_upgrade as eu
from agenthandover_worker import main as wm
from agenthandover_worker.config import Config, EmbeddingConfig
from agenthandover_worker.embedding_upgrade import (
    RETRY_SECONDS,
    STATE_KEY,
    TARGET_MODEL,
    EmbeddingUpgrade,
)
from agenthandover_worker.knowledge_base import KnowledgeBase
from agenthandover_worker.model_profiles import (
    EMBEDDING_MODEL,
    EMBEDDING_PROFILES,
    MEMORY_OFFER_FILE,
    set_embedding_model,
    switch_embedding_file,
)
from agenthandover_worker.ollama import OllamaUnavailable
from agenthandover_worker.session_linker import SessionLinker
from agenthandover_worker.stages import daily
from agenthandover_worker.stages import upgrade as upgrade_stage
from agenthandover_worker.task_segmenter import (
    AnnotatedFrame,
    SegmenterConfig,
    TaskSegmenter,
    frame_text,
)
from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig, _text_hash
from daemon_db import create_daemon_db, insert_event

GEMMA = EMBEDDING_PROFILES[TARGET_MODEL]
NOMIC = EMBEDDING_PROFILES[EMBEDDING_MODEL]

CONFIG = """# AgentHandover
[vlm]
annotation_model = "qwen3.5:4b"   # kept

[embedding]
# the text embedding model
model = "nomic-embed-text"
retention_days = 30

[memory]
enabled = true
"""


def _vector(text: str, dim: int = 768) -> list[float]:
    rng = random.Random(hashlib.sha256(text.encode("utf-8")).digest())
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


class FakeOllama:
    """``OllamaClient``'s embed/version/has_model."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, list[str]]] = []
        self.fail: Exception | None = None
        self.version_string = "0.40.3"
        self.models = {TARGET_MODEL, EMBEDDING_MODEL}

    def embed(self, model, texts, *, prefix="", dimensions=None, batch_size=None, timeout=None):
        if self.fail is not None:
            raise self.fail
        self.requests.append((model, prefix, list(texts)))
        return [_vector(model + prefix + t) for t in texts]

    def version(self, timeout=None) -> str:
        if self.fail is not None:
            raise self.fail
        return self.version_string

    def has_model(self, model, timeout=None) -> bool:
        return model in self.models

    def texts(self, model: str = TARGET_MODEL) -> list[str]:
        return [t for m, _, texts in self.requests if m == model for t in texts]


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


ANNOTATION = {
    "app": "Google Chrome",
    "location": "dashboard.stripe.com/payouts",
    "task_context": {"what_doing": "Reconciling the Stripe payout", "is_workflow": True},
}


@pytest.fixture()
def world(tmp_path: Path):
    """A nomic-populated database: one vector of each source type the
    worker embeds, built by the code that embeds them."""
    db = create_daemon_db(tmp_path / "events.db")
    kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=db)
    kb.ensure_structure()
    procedure = {
        "id": "reconcile-stripe-payout",
        "title": "Reconcile the Stripe payout",
        "description": "Match the payout against the bank deposit.",
        "steps": [{"action": "Open the Stripe dashboard"}, {"action": "Mark it reconciled"}],
    }
    kb.save_procedure(procedure)
    kb.save_daily_summary("2026-10-11", {
        "task_count": 2, "active_hours": 3.5,
        "tasks": [{"intent": "Reconcile payouts"}, {"intent": "Answer email"}],
    })
    kb.update_profile({"tools": {"Stripe": 4}})

    conn = sqlite3.connect(db)
    event_id = insert_event(conn, processed=1)
    conn.execute(
        "UPDATE events SET scene_annotation_json = ? WHERE id = ?",
        (json.dumps(ANNOTATION), event_id),
    )
    conn.execute(
        "INSERT INTO preserved_text (event_id, source, timestamp, text_content) "
        "VALUES (?, 'ocr', '2026-10-01T09:00:00Z', 'Payout 42 arrived on Oct 1')",
        (event_id,),
    )
    conn.commit()
    conn.close()

    fake = FakeOllama()
    vkb = VectorKB(db, VectorKBConfig(embedding_model=EMBEDDING_MODEL))
    frame = AnnotatedFrame.from_event({"id": event_id, "scene_annotation_json": json.dumps(ANNOTATION)})
    sources = [
        ("procedure", procedure["id"], wm._procedure_text(kb.get_procedure(procedure["id"])), NOMIC.doc_prefix),
        ("daily_summary", "2026-10-11",
         daily.summary_text("2026-10-11", 2, 3.5, ["Reconcile payouts", "Answer email"]), NOMIC.doc_prefix),
        ("profile", "user", daily._profile_text(kb.get_profile()), NOMIC.doc_prefix),
        ("preserved_text", f"{event_id}:ocr", "Payout 42 arrived on Oct 1", NOMIC.doc_prefix),
        ("annotation", event_id, frame_text(frame), NOMIC.cluster_prefix),
    ]
    for stype, sid, text, prefix in sources:
        vkb.upsert(stype, sid, text, embedding=_vector(EMBEDDING_MODEL + prefix + text), prefix=prefix)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_file = tmp_path / "config.toml"
    config_file.write_text(CONFIG, encoding="utf-8")
    yield SimpleNamespace(
        db=db, kb=kb, vkb=vkb, fake=fake, data_dir=data_dir, config_file=config_file,
        sources=sources, event_id=event_id, clock=Clock(), tmp_path=tmp_path,
    )
    vkb.close()


def _choose(world, choice: str = "everything") -> None:
    (world.data_dir / MEMORY_OFFER_FILE).write_text(
        json.dumps({"choice": choice, "wanted_visual_search": True}), encoding="utf-8",
    )


def _upgrade(world, **kwargs) -> EmbeddingUpgrade:
    kwargs.setdefault("rows_per_job", 2)
    return EmbeddingUpgrade(
        world.db, knowledge_base=world.kb, data_dir=world.data_dir,
        config_file=world.config_file, ollama_host="http://unused",
        model=EMBEDDING_MODEL, monotonic=world.clock, client=world.fake, **kwargs,
    )


def _rows(db: Path, table: str = "vector_store") -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            f"SELECT source_type, source_id, model, prefix, text_hash FROM {table} "
            "ORDER BY source_type"
        ).fetchall()
    finally:
        conn.close()


def _ctx(world, upgrade: EmbeddingUpgrade) -> SimpleNamespace:
    """The context fields the switch touches, with real components."""
    return SimpleNamespace(
        cfg=Config(embedding=EmbeddingConfig(model=EMBEDDING_MODEL)),
        db_path=world.db,
        vector_kb=world.vkb,
        session_linker=SessionLinker(world.kb, vector_kb=world.vkb),
        task_segmenter=TaskSegmenter(
            SegmenterConfig(embedding_model=EMBEDDING_MODEL), vector_kb=world.vkb,
        ),
        procedure_matcher=None,
        procedure_writer=None,
        embedding_upgrade=upgrade,
    )


def _run_to_switch(world, upgrade: EmbeddingUpgrade, limit: int = 20) -> int:
    for n in range(1, limit + 1):
        if upgrade.step():
            return n
    raise AssertionError("the re-embed never finished")


# ---------------------------------------------------------------------------
# The trigger and the status object
# ---------------------------------------------------------------------------


def test_nothing_happens_until_the_user_chooses_upgrade_everything(world) -> None:
    upgrade = _upgrade(world)
    assert not upgrade.wanted() and upgrade.status() is None
    assert upgrade.step() is False
    _choose(world, "memory_only")
    assert not upgrade.wanted() and upgrade.step() is False
    assert world.fake.requests == []
    assert all(model == EMBEDDING_MODEL for _, _, model, _, _ in _rows(world.db))


def test_a_new_install_on_the_new_model_never_upgrades(world) -> None:
    _choose(world)
    upgrade = _upgrade(world)
    upgrade.model = TARGET_MODEL
    assert not upgrade.wanted()
    assert upgrade.step() is False and world.fake.requests == []


def test_the_status_object_counts_the_rows(world) -> None:
    _choose(world)
    upgrade = _upgrade(world)
    assert upgrade.status() == {
        "state": "pending", "done": 0, "total": 5, "dropped": 0, "error": None,
    }
    upgrade.step()
    assert upgrade.status() == {
        "state": "running", "done": 2, "total": 5, "dropped": 0, "error": None,
    }


def test_the_job_is_submitted_at_p2_only_while_it_is_due(world) -> None:
    from agenthandover_worker.inference_runner import P2
    from agenthandover_worker.stages import Tick, control

    upgrade = _upgrade(world)
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(memory=SimpleNamespace(enabled=False)),
        memory_writer=None, focus_processor=None, v2_schema_ok=True, runner=None,
        embedding_upgrade=upgrade,
    )

    def jobs() -> dict:
        return {key: priority for key, priority, _ in control.submit_model_jobs(ctx, Tick())}

    assert "embedding-upgrade" not in jobs()
    _choose(world)
    assert jobs()["embedding-upgrade"] == P2


def test_worker_status_carries_the_upgrade(world) -> None:
    status = {"state": "running", "done": 2, "total": 5, "dropped": 0, "error": None}
    wm._write_worker_status(
        started_at="2026-10-12T00:00:00Z", events_processed_today=0, sops_generated=0,
        last_pipeline_duration_ms=None, consecutive_errors=0, vlm_available=True,
        embedding_upgrade=status,
    )
    written = json.loads((wm._status_dir() / "worker-status.json").read_text())
    assert written["embedding_upgrade"] == status
    wm._write_worker_status(
        started_at="2026-10-12T00:00:00Z", events_processed_today=0, sops_generated=0,
        last_pipeline_duration_ms=None, consecutive_errors=0, vlm_available=True,
    )
    assert "embedding_upgrade" not in json.loads(
        (wm._status_dir() / "worker-status.json").read_text()
    )


# ---------------------------------------------------------------------------
# The re-embed
# ---------------------------------------------------------------------------


def test_every_source_type_is_re_embedded_from_the_text_that_made_it(world) -> None:
    _choose(world)
    upgrade = _upgrade(world, rows_per_job=10)
    assert upgrade.step() is False
    staged = {r[0]: r for r in _rows(world.db, "vector_store_next")}
    assert set(staged) == {"procedure", "daily_summary", "profile", "preserved_text", "annotation"}
    for stype, sid, text, _ in world.sources:
        row = staged[stype]
        assert row[1:] == (
            sid, TARGET_MODEL,
            GEMMA.cluster_prefix if stype == "annotation" else GEMMA.doc_prefix,
            _text_hash(text),
        ), stype
    assert sorted(world.fake.texts()) == sorted(text for *_, text, _ in world.sources)


def test_search_never_mixes_models_during_the_re_embed(world) -> None:
    _choose(world)
    upgrade = _upgrade(world)
    upgrade.step()
    query = _vector(EMBEDDING_MODEL + NOMIC.query_prefix + "stripe payout")
    results = world.vkb.search_by_vector(query, top_k=10, min_score=-1.0)
    assert len(results) == 5
    assert {r.model for r in results} == {EMBEDDING_MODEL}
    # The new model's vectors are staged, never ranked.
    assert world.vkb.search_by_vector(_vector("x"), model=TARGET_MODEL) == []


def test_a_restart_resumes_where_the_re_embed_stopped(world) -> None:
    _choose(world)
    first = _upgrade(world)
    first.step()
    first.close()
    second = _upgrade(world)
    assert second.status()["done"] == 2
    _run_to_switch(world, second)
    # Each source was embedded once.
    assert len(world.fake.texts()) == 5


def test_a_source_changed_mid_way_is_re_embedded_again(world) -> None:
    _choose(world)
    upgrade = _upgrade(world, rows_per_job=10)
    upgrade.step()
    proc = world.kb.get_procedure("reconcile-stripe-payout")
    proc["title"] = "Reconcile the Stripe payout (weekly)"
    world.kb.save_procedure(proc)
    new_text = wm._procedure_text(world.kb.get_procedure("reconcile-stripe-payout"))
    world.vkb.upsert(
        "procedure", "reconcile-stripe-payout", new_text,
        embedding=_vector(EMBEDDING_MODEL + NOMIC.doc_prefix + new_text),
    )
    assert upgrade.status()["done"] == 5  # cached until the next job
    upgrade.step()
    assert world.fake.texts().count(new_text) == 1
    assert upgrade.step() is True
    row = [r for r in _rows(world.db) if r[0] == "procedure"][0]
    assert row[2:] == (TARGET_MODEL, GEMMA.doc_prefix, _text_hash(new_text))


def test_a_source_whose_text_is_gone_is_dropped_and_counted(world) -> None:
    _choose(world)
    conn = sqlite3.connect(world.db)
    conn.execute("DELETE FROM events WHERE id = ?", (world.event_id,))
    conn.commit()
    conn.close()
    upgrade = _upgrade(world)
    _run_to_switch(world, upgrade)
    assert {r[0] for r in _rows(world.db)} == {
        "procedure", "daily_summary", "profile", "preserved_text",
    }
    assert upgrade.status()["dropped"] == 1


def test_a_source_deleted_after_it_was_staged_stays_deleted(world) -> None:
    _choose(world)
    upgrade = _upgrade(world, rows_per_job=10)
    upgrade.step()
    world.vkb.delete_by_source("preserved_text", f"{world.event_id}:ocr")
    # Forgotten means forgotten: the staged vector goes with the next job,
    # even one that waits.
    upgrade._retry_at = world.clock.t + 60
    assert upgrade.step() is False
    assert "preserved_text" not in {r[0] for r in _rows(world.db, "vector_store_next")}
    world.clock.t += 60
    assert upgrade.step() is True
    assert "preserved_text" not in {r[0] for r in _rows(world.db)}
    assert _rows(world.db, "vector_store_next") == []


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_ollama_without_the_model_fails_clearly_and_keeps_nomic(world) -> None:
    _choose(world)
    world.fake.models = {EMBEDDING_MODEL}
    upgrade = _upgrade(world)
    assert upgrade.step() is False
    status = upgrade.status()
    assert status["state"] == "failed"
    assert status["error"] == (
        f"{TARGET_MODEL} is not downloaded: download it (ollama pull {TARGET_MODEL})"
    )
    assert world.fake.requests == []
    assert {r[2] for r in _rows(world.db)} == {EMBEDDING_MODEL}
    assert set_embedding_model(world.config_file.read_text(), EMBEDDING_MODEL) == CONFIG
    # Asked again later; once the model is there the re-embed starts.
    world.fake.models.add(TARGET_MODEL)
    assert upgrade.step() is False and world.fake.requests == []
    world.clock.t += RETRY_SECONDS
    upgrade.step()
    assert upgrade.status()["state"] == "running" and upgrade.status()["error"] is None


def test_an_old_ollama_fails_with_the_update_path(world) -> None:
    _choose(world)
    world.fake.version_string = "0.39.1"
    upgrade = _upgrade(world)
    upgrade.step()
    assert upgrade.status()["state"] == "failed"
    assert "needs Ollama 0.40.0 or newer (this Mac has 0.39.1): update Ollama" in (
        upgrade.status()["error"]
    )


def test_ollama_going_away_mid_way_backs_off_and_resumes(world) -> None:
    _choose(world)
    upgrade = _upgrade(world)
    upgrade.step()
    world.fake.fail = OllamaUnavailable("Ollama not reachable at http://unused")
    assert upgrade.step() is False
    status = upgrade.status()
    assert status["state"] == "running" and "not reachable" in status["error"]
    assert status["done"] == 2
    world.fake.fail = None
    assert upgrade.step() is False and len(world.fake.texts()) == 2  # backing off
    world.clock.t += 60
    _run_to_switch(world, upgrade)
    assert len(world.fake.texts()) == 5


def test_the_switch_waits_for_the_memory_guard(world) -> None:
    _choose(world)
    reason = ["the Mac is short of memory"]
    upgrade = _upgrade(world, blocker=lambda: reason[0])
    assert upgrade.step() is False and world.fake.requests == []
    assert upgrade.waiting == "the Mac is short of memory"
    reason[0] = None
    upgrade.step()
    assert len(world.fake.texts()) == 2


def test_a_config_that_cannot_be_edited_keeps_everything_on_nomic(world) -> None:
    _choose(world)
    world.config_file.write_text('embedding = { model = "nomic-embed-text" }\n')
    upgrade = _upgrade(world, rows_per_job=10)
    upgrade.step()
    assert upgrade.step() is False
    assert upgrade.status()["state"] == "failed"
    assert "cannot be edited safely" in upgrade.status()["error"]
    assert {r[2] for r in _rows(world.db)} == {EMBEDDING_MODEL}


def test_changing_the_answer_drops_the_staged_vectors(world) -> None:
    _choose(world)
    upgrade = _upgrade(world)
    upgrade.step()
    _choose(world, "memory_only")
    assert upgrade_stage.due(SimpleNamespace(embedding_upgrade=upgrade))
    upgrade.step()
    assert _rows(world.db, "vector_store_next") == []
    assert upgrade.status() is None
    assert not upgrade_stage.due(SimpleNamespace(embedding_upgrade=upgrade))


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------


def test_the_switch_replaces_every_row_in_one_step_and_writes_the_config(world) -> None:
    _choose(world)
    upgrade = _upgrade(world)
    ctx = _ctx(world, upgrade)
    tick = SimpleNamespace(embedding_upgrade=None)
    for _ in range(5):
        upgrade_stage.run(ctx, tick)
    assert {r[2] for r in _rows(world.db)} == {TARGET_MODEL}
    assert len(_rows(world.db)) == 5
    assert _rows(world.db, "vector_store_next") == []
    assert tick.embedding_upgrade == {
        "state": "done", "done": 5, "total": 5, "dropped": 0, "error": None,
    }
    # The worker's objects embed and search with the new model.
    assert ctx.cfg.embedding.model == TARGET_MODEL
    assert world.vkb.config.embedding_model == TARGET_MODEL
    assert world.vkb.thresholds == GEMMA.thresholds
    assert ctx.task_segmenter._profile.model == TARGET_MODEL
    assert ctx.task_segmenter._similarity_threshold() == GEMMA.thresholds.segmenter
    assert ctx.session_linker._vector_kb.thresholds.linker_cos == GEMMA.thresholds.linker_cos
    # The segmenter's cache is the re-embedded frame vector.
    _, sid, text, _ = world.sources[-1]
    frame = AnnotatedFrame.from_event({"id": sid, "scene_annotation_json": json.dumps(ANNOTATION)})
    cached = ctx.task_segmenter._cached_embeddings([frame], [text])
    assert cached[0] is not None
    # config.toml: only the model line changed, after a backup.
    assert world.config_file.read_text() == CONFIG.replace(
        'model = "nomic-embed-text"', f'model = "{TARGET_MODEL}"',
    )
    backups = list(world.tmp_path.glob("config.toml.*.bak"))
    assert [b.read_text() for b in backups] == [CONFIG]
    # Search now ranks the new model's rows.
    query = _vector(TARGET_MODEL + GEMMA.query_prefix + "stripe payout")
    results = world.vkb.search_by_vector(query, min_score=-1.0)
    assert len(results) == 5 and {r.model for r in results} == {TARGET_MODEL}
    # Done: no job any more.
    assert not upgrade_stage.due(ctx)


def test_a_restart_after_the_rows_switched_completes_the_switch(world) -> None:
    _choose(world)
    upgrade = _upgrade(world, rows_per_job=10)
    _run_to_switch(world, upgrade)
    # Interrupted before the config write.
    assert json.loads(sqlite3.connect(world.db).execute(
        "SELECT value FROM worker_state WHERE key = ?", (STATE_KEY,),
    ).fetchone()[0])["state"] == "switched"
    assert upgrade.status()["state"] == "running"
    assert "nomic-embed-text" in world.config_file.read_text()
    upgrade.close()

    restarted = _upgrade(world)
    ctx = _ctx(world, restarted)
    assert upgrade_stage.finish(ctx) is True
    assert ctx.vector_kb.config.embedding_model == TARGET_MODEL
    assert f'model = "{TARGET_MODEL}"' in world.config_file.read_text()
    assert restarted.status()["state"] == "done"
    assert upgrade_stage.finish(ctx) is False


def test_a_config_write_that_fails_is_retried(world, monkeypatch) -> None:
    _choose(world)
    upgrade = _upgrade(world, rows_per_job=10)
    ctx = _ctx(world, upgrade)
    _run_to_switch(world, upgrade)

    def refuse(*args, **kwargs):
        raise PermissionError("read-only")

    monkeypatch.setattr(eu, "switch_embedding_file", refuse)
    assert upgrade_stage.finish(ctx) is False
    assert upgrade.status()["state"] == "running"
    assert "read-only" in upgrade.status()["error"]
    assert upgrade_stage.due(ctx)
    monkeypatch.undo()
    upgrade_stage.run(ctx, SimpleNamespace(embedding_upgrade=None))
    assert upgrade.status()["state"] == "done"


def test_the_session_linker_forgets_the_old_models_intent_vectors(world) -> None:
    linker = SessionLinker(world.kb, vector_kb=world.vkb)
    linker._embeddings = {"reconcile payouts": [1.0, 0.0]}
    linker._not_embedded = {"x"}
    linker.use_vector_kb(world.vkb)
    assert linker._embeddings == {} and linker._not_embedded == set()


# ---------------------------------------------------------------------------
# The config write
# ---------------------------------------------------------------------------


def test_set_embedding_model_replaces_only_the_model_line() -> None:
    assert set_embedding_model(CONFIG, TARGET_MODEL) == CONFIG.replace(
        'model = "nomic-embed-text"', f'model = "{TARGET_MODEL}"',
    )
    assert set_embedding_model(CONFIG, EMBEDDING_MODEL) == CONFIG


def test_set_embedding_model_adds_the_key_or_the_section() -> None:
    text = '[embedding]\nretention_days = 30\n\n[memory]\nenabled = true\n'
    assert set_embedding_model(text, TARGET_MODEL) == (
        f'[embedding]\nmodel = "{TARGET_MODEL}"\nretention_days = 30\n\n[memory]\nenabled = true\n'
    )
    text = '[vlm]\nannotation_model = "qwen3.5:4b"'
    assert set_embedding_model(text, TARGET_MODEL) == (
        f'[vlm]\nannotation_model = "qwen3.5:4b"\n\n[embedding]\nmodel = "{TARGET_MODEL}"\n'
    )
    assert set_embedding_model("", TARGET_MODEL) == f'[embedding]\nmodel = "{TARGET_MODEL}"\n'


def test_set_embedding_model_refuses_what_it_cannot_edit_safely() -> None:
    assert set_embedding_model('embedding = { model = "x" }\n', TARGET_MODEL) is None
    assert set_embedding_model("[embedding\n", TARGET_MODEL) is None


def test_switch_embedding_file_writes_a_new_file_or_backs_up_the_old(tmp_path: Path) -> None:
    missing = tmp_path / "new" / "config.toml"
    assert switch_embedding_file(missing, TARGET_MODEL) is None
    assert missing.read_text() == f'[embedding]\nmodel = "{TARGET_MODEL}"\n'
    existing = tmp_path / "config.toml"
    existing.write_text(CONFIG)
    backup = switch_embedding_file(existing, TARGET_MODEL)
    assert backup is not None and backup.read_text() == CONFIG
    assert switch_embedding_file(existing, TARGET_MODEL) is None
    existing.write_text('embedding = { model = "x" }\n')
    with pytest.raises(ValueError):
        switch_embedding_file(existing, TARGET_MODEL)
