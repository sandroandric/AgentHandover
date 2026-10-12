"""The embedding profiles table (WP-E0).

nomic-embed-text's row must carry exactly the numbers today's readers hold
(WP-E2 rewires them to the table; until then a drift on either side would
silently change behaviour when it does).  EmbeddingGemma 2's row has its
model card's prefixes, 256 dimensions and no thresholds until WP-E3
calibrates them.
"""

from __future__ import annotations

import inspect

from agenthandover_worker import model_profiles as mp


def test_nomic_profile_carries_todays_reader_numbers():
    from agenthandover_worker import mcp_server, query_api, session_linker, sop_dedup, vector_kb
    from agenthandover_worker.procedure_matcher import ProcedureMatcher
    from agenthandover_worker.task_segmenter import SegmenterConfig

    profile = mp.EMBEDDING_PROFILES["nomic-embed-text"]
    assert profile.model == mp.EMBEDDING_MODEL == "nomic-embed-text"
    assert profile.dim == vector_kb.VectorKBConfig().embedding_dim == 768
    assert profile.doc_prefix == vector_kb.DOCUMENT_PREFIX == "search_document: "
    assert profile.query_prefix == vector_kb.QUERY_PREFIX == "search_query: "
    assert profile.cluster_prefix == vector_kb.CLUSTERING_PREFIX == "clustering: "

    t = profile.thresholds
    assert t is not None
    assert t.segmenter == SegmenterConfig().similarity_threshold == 0.83
    assert t.linker_cos == session_linker._COSINE_THRESHOLD == 0.89
    assert t.linker_band == session_linker._LLM_BAND["cosine"] == 0.08
    assert t.dedup == sop_dedup._DEFAULT_THRESHOLD == 0.70
    assert t.dedup_family == sop_dedup._FAMILY_THRESHOLD == 0.60
    matcher_default = inspect.signature(ProcedureMatcher).parameters["match_threshold"].default
    assert t.matcher == matcher_default == 0.50
    # Skill search's default floor is an inline literal in both readers.
    assert t.min_score == 0.3
    assert "min_score=0.3," in inspect.getsource(mcp_server)
    assert 'body.get("min_score", 0.3)' in inspect.getsource(query_api)


def test_memory_profile_is_embeddinggemma_2_at_256_dims_uncalibrated():
    profile = mp.EMBEDDING_PROFILES[mp.MEMORY_MODEL]
    assert mp.MEMORY_MODEL == "embeddinggemma-2:440m"
    assert profile.model == mp.MEMORY_MODEL
    assert profile.dim == 256
    assert profile.doc_prefix == "title: none | text: "
    assert profile.query_prefix == "task: search result | query: "
    assert profile.cluster_prefix == "task: clustering | query: "
    assert profile.thresholds is None
    assert mp.MODEL_SIZES_GB[mp.MEMORY_MODEL] == 0.71


def test_the_profiles_are_keyed_by_their_model():
    assert set(mp.EMBEDDING_PROFILES) == {mp.EMBEDDING_MODEL, mp.MEMORY_MODEL}
    for model, profile in mp.EMBEDDING_PROFILES.items():
        assert profile.model == model


def test_tiers_document_names_the_memory_model_and_its_ollama_minimum():
    doc = mp.tiers_document()
    assert doc["memory_model"] == mp.MEMORY_MODEL == "embeddinggemma-2:440m"
    assert doc["minimum_ollama_version_for_memory"] == mp.MINIMUM_OLLAMA_VERSION_FOR_MEMORY == "0.40.0"
    assert doc["model_sizes_gb"][mp.MEMORY_MODEL] == 0.71
    # Memory is not a tier model: no tier pulls it until Memory is chosen.
    assert all(mp.MEMORY_MODEL not in tier.models for tier in mp.TIERS)
    assert mp.version_at_least("0.40.3", mp.MINIMUM_OLLAMA_VERSION_FOR_MEMORY)
    assert not mp.version_at_least("0.34.2", mp.MINIMUM_OLLAMA_VERSION_FOR_MEMORY)
