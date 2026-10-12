"""The embedding profiles table (WP-E0, readers rewired in WP-E2).

nomic-embed-text's row carries exactly the numbers v0.5's readers held
(task_segmenter, session_linker, sop_dedup, procedure_matcher, mcp_server,
query_api, vector_kb's prefixes): every reader now takes them from the row,
so these literals are what keeps nomic's behaviour byte-identical.
EmbeddingGemma 2's row has its model card's prefixes, 256 dimensions
(asked of Ollama with ``dimensions``) and no thresholds until WP-E3
calibrates them.
"""

from __future__ import annotations

import inspect

from agenthandover_worker import model_profiles as mp


def test_nomic_profile_carries_the_v05_reader_numbers():
    profile = mp.EMBEDDING_PROFILES["nomic-embed-text"]
    assert profile.model == mp.EMBEDDING_MODEL == "nomic-embed-text"
    assert profile.dim == 768
    assert profile.dimensions is None  # requests carry no "dimensions"
    assert profile.doc_prefix == "search_document: "
    assert profile.query_prefix == "search_query: "
    assert profile.cluster_prefix == "clustering: "
    assert profile.thresholds == mp.EmbeddingThresholds(
        segmenter=0.83,
        linker_cos=0.89,
        linker_band=0.08,
        dedup=0.70,
        matcher=0.50,
        min_score=0.3,
    )


def test_readers_hold_no_copy_of_a_profile_number():
    """The v0.5 copies are gone: a reader that kept one would not follow
    the configured model."""
    from agenthandover_worker import mcp_server, query_api, session_linker, vector_kb
    from agenthandover_worker.task_segmenter import SegmenterConfig

    for name in ("DOCUMENT_PREFIX", "QUERY_PREFIX", "CLUSTERING_PREFIX"):
        assert not hasattr(vector_kb, name)
    assert not hasattr(vector_kb.VectorKBConfig(), "embedding_dim")
    for name in ("_COSINE_THRESHOLD", "_LLM_BAND", "_THRESHOLDS"):
        assert not hasattr(session_linker, name)
    assert SegmenterConfig().similarity_threshold is None
    assert "min_score=0.3" not in inspect.getsource(mcp_server)
    assert "0.3)" not in inspect.getsource(query_api.QueryAPIHandler._handle_semantic_search)


def test_a_model_without_a_row_is_embedded_as_nomic_was():
    profile = mp.embedding_profile("mxbai-embed-large")
    nomic = mp.EMBEDDING_PROFILES[mp.EMBEDDING_MODEL]
    assert profile.model == "mxbai-embed-large"
    assert (profile.doc_prefix, profile.query_prefix, profile.cluster_prefix) == (
        nomic.doc_prefix, nomic.query_prefix, nomic.cluster_prefix,
    )
    assert profile.thresholds == nomic.thresholds
    assert profile.dimensions is None
    assert mp.embedding_profile(mp.MEMORY_MODEL) is mp.EMBEDDING_PROFILES[mp.MEMORY_MODEL]


def test_memory_profile_is_embeddinggemma_2_at_256_dims_uncalibrated():
    profile = mp.EMBEDDING_PROFILES[mp.MEMORY_MODEL]
    assert mp.MEMORY_MODEL == "embeddinggemma-2:440m"
    assert profile.model == mp.MEMORY_MODEL
    assert profile.dim == 256
    assert profile.doc_prefix == "title: none | text: "
    assert profile.query_prefix == "task: search result | query: "
    assert profile.cluster_prefix == "task: clustering | query: "
    assert profile.thresholds is None
    assert profile.truncated and profile.dimensions == 256
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
