"""VLM inference backends package.

Re-exports shared utilities. Backend classes are imported lazily
to avoid pulling in heavy optional dependencies at module level.
"""

from agenthandover_worker.backends._json_parser import extract_json

__all__ = ["extract_json"]
