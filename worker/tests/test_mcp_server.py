"""Tests for the MCP server module.

Tests the server builder and tool logic without requiring the mcp
package — mocks the FastMCP decorator and tests the underlying
functions directly.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from daemon_db import create_daemon_db, kb_db


# ---------------------------------------------------------------------------
# Test the server's tool functions directly by extracting them
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_kb(tmp_path: Path):
    """Create a fake knowledge base with test procedures."""
    proc_dir = tmp_path / "procedures"
    proc_dir.mkdir(parents=True)

    procs = [
        {
            "id": "reddit-marketing",
            "title": "Reddit Community Marketing",
            "description": "Daily engagement",
            "lifecycle_state": "agent_ready",
            "confidence_avg": 0.89,
            "apps_involved": ["Chrome"],
            "steps": [{"action": "Open Reddit"}],
            "strategy": "Target high-signal posts",
            "voice_profile": {"formality": "casual"},
        },
        {
            "id": "deploy-staging",
            "title": "Deploy to Staging",
            "description": "CI/CD deploy",
            "lifecycle_state": "draft",
            "confidence_avg": 0.72,
            "apps_involved": ["Terminal"],
            "steps": [{"action": "Run tests"}],
        },
    ]

    for proc in procs:
        (proc_dir / f"{proc['id']}.json").write_text(
            json.dumps(proc), encoding="utf-8",
        )

    # Create a mock KB
    kb = MagicMock()
    kb.list_procedures.return_value = procs
    kb.get_procedure.side_effect = lambda slug: next(
        (p for p in procs if p["id"] == slug), None
    )
    kb.get_profile.return_value = {
        "tools": {"browser": "Chrome"},
        "writing_style": {"formality": "casual"},
    }
    return kb


class TestMCPServerBuild:
    """Test that the MCP server can be constructed (mocked)."""

    def test_build_requires_mcp_package(self):
        """If mcp is not importable, should raise ImportError."""
        with patch.dict("sys.modules", {"mcp": None, "mcp.server.fastmcp": None}):
            with pytest.raises(ImportError):
                from agenthandover_worker.mcp_server import build_mcp_server
                # Force re-import
                import importlib
                import agenthandover_worker.mcp_server as mod
                importlib.reload(mod)
                mod.build_mcp_server()


class TestMCPToolLogic:
    """Test the tool functions' logic using mocked KB."""

    def test_list_ready_filters_agent_ready(self, fake_kb):
        """Only agent_ready procedures should appear in ready list."""
        procs = fake_kb.list_procedures()
        ready = [p for p in procs if p.get("lifecycle_state") == "agent_ready"]
        assert len(ready) == 1
        assert ready[0]["id"] == "reddit-marketing"

    def test_list_all_returns_everything(self, fake_kb):
        procs = fake_kb.list_procedures()
        assert len(procs) == 2

    def test_get_procedure_found(self, fake_kb):
        proc = fake_kb.get_procedure("reddit-marketing")
        assert proc is not None
        assert proc["title"] == "Reddit Community Marketing"
        assert proc["voice_profile"]["formality"] == "casual"

    def test_get_procedure_not_found(self, fake_kb):
        proc = fake_kb.get_procedure("nonexistent")
        assert proc is None

    def test_search_fallback_keyword(self, fake_kb):
        """Without vector_kb, search should fall back to keyword matching."""
        procs = fake_kb.list_procedures()
        query = "reddit"
        matches = [
            p for p in procs
            if query in p.get("title", "").lower()
            or query in p.get("description", "").lower()
        ]
        assert len(matches) == 1
        assert matches[0]["id"] == "reddit-marketing"

    def test_get_profile(self, fake_kb):
        profile = fake_kb.get_profile()
        assert "tools" in profile
        assert "writing_style" in profile


class TestMCPResourceLogic:
    """Test resource rendering logic."""

    def test_procedures_index_format(self, fake_kb):
        procs = fake_kb.list_procedures()
        lines = ["# AgentHandover Procedures\n"]
        for proc in procs:
            state = proc.get("lifecycle_state", "observed")
            marker = "ready" if state == "agent_ready" else state
            lines.append(
                f"- **{proc.get('title', 'Untitled')}** "
                f"(`{proc.get('id', '?')}`) [{marker}]"
            )
        index = "\n".join(lines)
        assert "Reddit Community Marketing" in index
        assert "[ready]" in index
        assert "[draft]" in index

    def test_procedure_detail_markdown(self, fake_kb):
        proc = fake_kb.get_procedure("reddit-marketing")
        lines = [f"# {proc['title']}\n"]
        if proc.get("strategy"):
            lines.append(f"## Strategy\n{proc['strategy']}\n")
        vp = proc.get("voice_profile", {})
        if vp.get("formality"):
            lines.append(f"## Voice\nTone: {vp['formality']}\n")
        md = "\n".join(lines)
        assert "# Reddit Community Marketing" in md
        assert "Target high-signal" in md
        assert "casual" in md


# ---------------------------------------------------------------------------
# One path for the worker and the MCP server; search_skills says its mode
# ---------------------------------------------------------------------------


def test_mcp_db_path_matches_worker():
    """The MCP server reads the database and knowledge base the worker writes."""
    from agenthandover_worker import knowledge_base, mcp_server
    from agenthandover_worker import main as worker_main

    assert mcp_server._get_db_path() == worker_main.DEFAULT_DB_PATH
    assert mcp_server._get_db_path().name == "events.db"
    assert mcp_server._get_kb_root() == worker_main.KNOWLEDGE_BASE_DIR
    assert knowledge_base.DEFAULT_KNOWLEDGE_DIR == worker_main.KNOWLEDGE_BASE_DIR


def _fake_embed(self, texts, **_):
    """Texts about Reddit point one way, everything else another."""
    return [[1.0, 0.0] if "reddit" in t.lower() else [0.0, 1.0] for t in texts]


class TestSearchSkillsMode:

    @pytest.fixture
    def server(self, tmp_path: Path):
        from agenthandover_worker.knowledge_base import KnowledgeBase
        from agenthandover_worker.mcp_server import build_mcp_server

        # The worker's database: the procedures and the vector store.
        db = create_daemon_db(tmp_path / "events.db")
        kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=db)
        kb.ensure_structure()
        kb.save_procedure({
            "id": "reddit-marketing", "title": "Reddit Community Marketing",
            "description": "Daily engagement", "lifecycle_state": "agent_ready",
            "steps": [{"action": "Open Reddit"}],
        })
        with patch("agenthandover_worker.mcp_server._get_kb_root", return_value=tmp_path / "knowledge"), \
             patch("agenthandover_worker.mcp_server._get_db_path", return_value=db):
            srv = build_mcp_server()

            def search(query: str) -> dict:
                return json.loads(srv._tool_manager.get_tool("search_skills").fn(query))

            yield search, db

    @staticmethod
    def _embed_skill(db: Path) -> None:
        from agenthandover_worker.vector_kb import VectorKB

        with patch.object(VectorKB, "compute_embeddings", _fake_embed), VectorKB(db) as vkb:
            vkb.upsert("procedure", "reddit-marketing", "Reddit Community Marketing")

    def test_no_database_falls_back_and_warns(self, tmp_path, caplog):
        """Before the daemon created the database there is no Skill to
        find (procedures live in it) and no vector store."""
        from agenthandover_worker.mcp_server import build_mcp_server

        with patch("agenthandover_worker.mcp_server._get_kb_root", return_value=tmp_path / "knowledge"), \
             patch("agenthandover_worker.mcp_server._get_db_path", return_value=tmp_path / "none.db"):
            srv = build_mcp_server()
            with caplog.at_level("WARNING", logger="agenthandover_worker.mcp_server"):
                result = json.loads(srv._tool_manager.get_tool("search_skills").fn("reddit"))
        assert result["mode"] == "keyword_fallback"
        assert "no worker database" in result["reason"]
        assert result["results"] == []
        assert "Semantic search unavailable" in caplog.text
        assert not (tmp_path / "none.db").exists()

    def test_no_embedded_skill_falls_back(self, server):
        from agenthandover_worker.vector_kb import VectorKB

        search, db = server
        VectorKB(db).close()
        result = search("reddit")
        assert result["mode"] == "keyword_fallback"
        assert result["reason"] == "no Skill has been embedded yet"

    def test_semantic_when_the_skill_is_embedded(self, server):
        from agenthandover_worker.vector_kb import VectorKB

        search, db = server
        self._embed_skill(db)
        with patch.object(VectorKB, "compute_embeddings", _fake_embed):
            result = search("post on the subreddit")
        assert result["mode"] == "semantic"
        assert result["reason"] == ""
        assert result["results"] == [
            {"slug": "reddit-marketing", "title": "Reddit Community Marketing", "score": 1.0},
        ]

    def test_a_running_session_follows_an_embedding_model_switch(self, server):
        """"Upgrade everything" switches [embedding] model while an MCP
        session runs: the next search opens the store with the new model
        (searching with the old one would match no row)."""
        from agenthandover_worker import config as config_mod
        from agenthandover_worker.vector_kb import VectorKB

        search, db = server
        self._embed_skill(db)
        real = config_mod.load_config()
        models: list[str] = []

        def embed(self, texts, **_):
            models.append(self.config.embedding_model)
            return _fake_embed(self, texts)

        with patch.object(VectorKB, "compute_embeddings", embed):
            search("reddit")
            switched = dataclasses.replace(
                real, embedding=dataclasses.replace(real.embedding, model="embeddinggemma-2:440m"),
            )
            with patch.object(config_mod, "load_config", return_value=switched):
                search("reddit")
        assert models == ["nomic-embed-text", "embeddinggemma-2:440m"]

    def test_query_prefix_and_floor_are_the_embedding_profiles(self, server):
        """nomic-embed-text (the default ``[embedding] model``): the query
        embeds with ``search_query: `` and results stop at its 0.3."""
        from agenthandover_worker.vector_kb import VectorKB

        search, db = server
        self._embed_skill(db)
        prefixes: list[str] = []
        searches: list[dict] = []

        def embed(self, texts, *, prefix=None):
            prefixes.append(prefix)
            return _fake_embed(self, texts, prefix=prefix)

        real_search = VectorKB.search_by_vector

        def search_by_vector(self, vector, **kwargs):
            searches.append(kwargs)
            return real_search(self, vector, **kwargs)

        with patch.object(VectorKB, "compute_embeddings", embed), \
             patch.object(VectorKB, "search_by_vector", search_by_vector):
            assert search("post on the subreddit")["mode"] == "semantic"
        assert prefixes == ["search_query: "]
        assert searches == [
            {"top_k": 5, "source_types": ["procedure"], "min_score": 0.3},
        ]

    def test_embedding_model_down_falls_back_and_warns(self, server, caplog):
        from agenthandover_worker.ollama import OllamaUnavailable
        from agenthandover_worker.vector_kb import VectorKB

        search, db = server
        self._embed_skill(db)

        def down(self, texts, **_):
            raise OllamaUnavailable("connection refused")

        with patch.object(VectorKB, "compute_embeddings", down), \
             caplog.at_level("WARNING", logger="agenthandover_worker.mcp_server"):
            result = search("reddit")
        assert result["mode"] == "keyword_fallback"
        assert "embedding model unavailable" in result["reason"]
        assert "embedding model unavailable" in caplog.text
        assert [r["slug"] for r in result["results"]] == ["reddit-marketing"]

    def test_embedding_timeout_is_short_and_not_retried_every_call(self, server):
        """Seen on a real Mac: Ollama was busy, and each of four searches in
        one Claude Code turn waited 60 s for /api/embed before falling back."""
        from agenthandover_worker import mcp_server
        from agenthandover_worker.ollama import OllamaTimeout
        from agenthandover_worker.vector_kb import VectorKB

        search, db = server
        self._embed_skill(db)
        timeouts: list[float] = []

        def busy(self, texts, **_):
            timeouts.append(self.config.timeout_seconds)
            raise OllamaTimeout("Ollama timed out after 8s (/api/embed)")

        with patch.object(VectorKB, "compute_embeddings", busy):
            first = search("reddit")
            second = search("reddit")
        assert timeouts == [mcp_server.QUERY_EMBED_TIMEOUT_S]
        assert first["mode"] == second["mode"] == "keyword_fallback"
        assert "retrying after" in second["reason"]

        with patch.object(VectorKB, "compute_embeddings", _fake_embed), \
             patch.object(mcp_server.time, "monotonic",
                          return_value=mcp_server.time.monotonic() + mcp_server.EMBED_RETRY_AFTER_S + 1):
            assert search("reddit")["mode"] == "semantic"


class TestKeywordFallback:
    """What an agent types is a sentence, not a title substring."""

    @pytest.fixture
    def search(self, tmp_path: Path):
        from agenthandover_worker.knowledge_base import KnowledgeBase
        from agenthandover_worker.mcp_server import build_mcp_server

        kb = KnowledgeBase(root=tmp_path / "knowledge", db_path=kb_db(tmp_path / "knowledge"))
        kb.ensure_structure()
        kb.save_procedure({
            "id": "x-replyu", "title": "x replyu", "short_title": "Reply to Health Thread",
            "description": "", "tags": ["communication"],
            "steps": [{"action": "Navigate to the X feed and locate the post"},
                      {"action": "Click the 'Reply' button"}],
        })
        kb.save_procedure({
            "id": "expense-report", "title": "Submit expense report",
            "description": "Monthly receipts", "steps": [{"action": "Open Expensify"}],
        })
        # No Skill is embedded in the database: search falls back to keywords.
        with patch("agenthandover_worker.mcp_server._get_kb_root", return_value=tmp_path / "knowledge"), \
             patch("agenthandover_worker.mcp_server._get_db_path", return_value=kb.db_path):
            srv = build_mcp_server()
            yield lambda q: json.loads(srv._tool_manager.get_tool("search_skills").fn(q))["results"]

    def test_sentence_query_finds_the_skill(self, search):
        results = search("reply to a post on X")
        assert [r["slug"] for r in results] == ["x-replyu"]
        assert results[0]["score"] == 0.75  # "reply" in the title, "post" in a step

    def test_title_words_rank_above_step_words(self, search):
        assert [r["slug"] for r in search("expense report")] == ["expense-report"]
        assert search("receipts")[0]["score"] == 0.5

    def test_no_overlap_returns_nothing(self, search):
        assert search("deploy kubernetes") == []


class TestCommandLine:
    """``agenthandover-mcp`` parses its arguments before it builds a server:
    ``--help`` and ``--version`` answer and exit, an unknown argument is an
    error, and nothing starts a server by accident."""

    def _main(self, argv):
        from agenthandover_worker import mcp_server

        with patch.object(mcp_server, "build_mcp_server") as build:
            mcp_server.main(argv)
        return build

    def test_help_prints_usage_and_starts_nothing(self, capsys):
        with pytest.raises(SystemExit) as exit_:
            self._main(["--help"])
        assert exit_.value.code == 0
        out = capsys.readouterr().out
        assert out.startswith("usage: agenthandover-mcp")
        assert "--sse" in out

    def test_version(self, capsys):
        from agenthandover_worker import __version__

        with pytest.raises(SystemExit) as exit_:
            self._main(["--version"])
        assert exit_.value.code == 0
        assert capsys.readouterr().out.strip() == f"agenthandover-mcp {__version__}"

    def test_unknown_argument_exits_2_with_a_message(self, capsys):
        with pytest.raises(SystemExit) as exit_:
            self._main(["--port", "9000"])
        assert exit_.value.code == 2
        assert "unrecognized arguments: --port 9000" in capsys.readouterr().err

    def test_default_transport_is_stdio(self):
        build = self._main([])
        build.return_value.run.assert_called_once_with(transport="stdio")

    def test_sse_flag(self):
        build = self._main(["--sse"])
        build.return_value.run.assert_called_once_with(transport="sse")
