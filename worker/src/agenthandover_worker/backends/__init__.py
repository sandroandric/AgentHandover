"""VLM inference backends (Ollama, OpenAI-compatible, Anthropic, Google).

Backend classes are imported lazily by ``vlm_worker.VLMWorker`` to avoid
pulling in optional SDK dependencies at module level.
"""
