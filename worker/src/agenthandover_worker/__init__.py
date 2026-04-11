"""AgentHandover apprentice worker — episode building, semantic translation, SOP induction."""


def _read_package_version() -> str:
    """Read the installed package version from the dist-info metadata.

    Returns ``"unknown"`` if metadata can't be loaded — e.g. running from
    a source tree without ``pip install -e``, or if the dist-info is
    corrupted. Wrapping in ``try/except Exception`` is deliberate: this
    runs at package import time, and ANY failure here would block every
    submodule of ``agenthandover_worker`` from loading. A display-only
    version string is never worth blocking startup.

    Reading from metadata eliminates the "forgot to bump the hardcoded
    string" class of bug — the single source of truth is
    ``worker/pyproject.toml``. Caught during v0.2.3 → v0.2.4
    investigation: ``__version__`` had been stuck at "0.1.0" since
    v0.2.0, and ``_WORKER_VERSION`` had been stuck at "0.2.0" in three
    other files (``main.py``, ``procedure_schema.py``, ``query_api.py``)
    across three releases.
    """
    try:
        from importlib.metadata import version
        return version("agenthandover-worker")
    except Exception:
        return "unknown"


__version__ = _read_package_version()
