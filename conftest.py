"""Repository-root pytest configuration.

Applies the sandboxed ``HOME`` (see ``worker/tests/home_sandbox.py``) to
every test root listed in ``pyproject.toml``, not only ``worker/tests``.
"""

from home_sandbox import (  # noqa: F401  (pytest discovers these by name)
    pytest_configure,
    pytest_unconfigure,
    real_home_untouched,
)
