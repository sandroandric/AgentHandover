"""Shared LLM reasoning utility for AgentHandover.

Provides a single ``LLMReasoner`` class that all pipeline modules use
for targeted Qwen calls.  Wraps Ollama with:

- **Budget enforcement** via ``VLMFallbackQueue.can_dispatch()``
- **Response parsing** (strip thinking tags, extract JSON fences)
- **Provenance tracking** (prompt hash, model, timestamp)
- **Abstention detection** (``INSUFFICIENT_EVIDENCE`` keyword)

Three response modes:
- ``reason_json()``  → parsed dict
- ``reason_text()``  → plain string
- ``reason_yesno()`` → True / False / None

All methods return ``ReasoningResult``; callers check ``.success`` and
``.abstained`` before using ``.value``.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from agenthandover_worker.json_extract import clean_model_text, extract_json
from agenthandover_worker.model_profiles import STANDARD_TIER
from agenthandover_worker.ollama import OllamaClient

if TYPE_CHECKING:
    from agenthandover_worker.vlm_queue import VLMFallbackQueue

logger = logging.getLogger(__name__)


_ABSTENTION_MARKER = "INSUFFICIENT_EVIDENCE"

_MAX_USER_DATA_LENGTH = 500


def sanitize_user_data(text: str, max_length: int = _MAX_USER_DATA_LENGTH) -> str:
    """Sanitize user-controlled data before embedding in LLM prompts.

    Truncates to ``max_length``, strips control characters, and wraps
    in delimiters so the LLM can distinguish user data from instructions.
    """
    if not isinstance(text, str):
        text = str(text)
    # Strip control characters except newline/tab
    cleaned = "".join(
        ch for ch in text
        if ch in ("\n", "\t") or (ord(ch) >= 32)
    )
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    return cleaned


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ReasoningConfig:
    """Configuration for the LLM reasoner."""

    model: str = STANDARD_TIER.sop_model  # main() passes the configured model
    ollama_host: str = "http://localhost:11434"
    num_predict: int = 4000
    timeout: float = 1800.0
    think: bool = True


@dataclass
class ReasoningResult:
    """Result of an LLM reasoning call."""

    value: Any = None           # dict | str | bool | None
    success: bool = False
    abstained: bool = False
    error: str | None = None
    model: str = ""
    prompt_hash: str = ""
    elapsed_seconds: float = 0.0
    generated_at: str = ""
    confidence: float | None = None


# ---------------------------------------------------------------------------
# LLMReasoner
# ---------------------------------------------------------------------------


class LLMReasoner:
    """Shared LLM reasoning interface for all pipeline modules.

    Args:
        config: Model, host, timeout, and prediction settings.
        vlm_queue: Optional budget queue.  When provided, each call
            checks ``can_dispatch()`` before invoking Ollama and
            records compute time via ``record_completion()`` after.
    """

    def __init__(
        self,
        config: ReasoningConfig | None = None,
        vlm_queue: "VLMFallbackQueue | None" = None,
    ) -> None:
        self.config = config or ReasoningConfig()
        self._queue = vlm_queue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason_json(
        self,
        prompt: str,
        system: str = "",
        caller: str = "",
        think: bool | None = None,
    ) -> ReasoningResult:
        """Call Qwen and parse the response as a JSON dict.

        Returns ``ReasoningResult`` with ``value`` as a dict on success,
        ``None`` on failure, or ``abstained=True`` if the model says
        ``INSUFFICIENT_EVIDENCE``.

        Args:
            think: Override for thinking mode.  Defaults to ``config.think``
                (True).  JSON responses benefit from thinking.
        """
        result = self._execute(prompt, system, caller, think=think)
        if not result.success:
            return result

        raw_text = result.value
        if isinstance(raw_text, str) and _ABSTENTION_MARKER in raw_text:
            result.abstained = True
            result.value = None
            return result

        parsed = self._parse_json(raw_text)
        if parsed is None:
            result.success = False
            result.error = "Failed to parse JSON from response"
            result.value = None
            return result

        result.value = parsed
        return result

    def reason_text(
        self,
        prompt: str,
        system: str = "",
        caller: str = "",
        think: bool | None = None,
    ) -> ReasoningResult:
        """Call Qwen and return the response as plain text.

        Strips thinking tags but does no JSON parsing.

        Args:
            think: Override for thinking mode.  Defaults to ``False``
                for short-answer text calls to avoid exhausting tokens
                on internal reasoning.
        """
        if think is None:
            think = False
        result = self._execute(prompt, system, caller, think=think)
        if not result.success:
            return result

        raw_text = result.value
        if isinstance(raw_text, str) and _ABSTENTION_MARKER in raw_text:
            result.abstained = True
            result.value = None
            return result

        # Clean up: strip whitespace and surrounding quotes
        if isinstance(raw_text, str):
            text = raw_text.strip().strip('"').strip("'").strip()
            result.value = text

        return result

    def reason_yesno(
        self,
        prompt: str,
        system: str = "",
        caller: str = "",
        think: bool | None = None,
    ) -> ReasoningResult:
        """Call Qwen and parse the response as a boolean yes/no.

        Returns ``True`` for YES, ``False`` for NO, ``None`` if
        ambiguous or abstained.

        Args:
            think: Override for thinking mode.  Defaults to ``False``
                for yes/no calls to avoid exhausting tokens on reasoning.
        """
        if think is None:
            think = False
        result = self._execute(prompt, system, caller, think=think)
        if not result.success:
            return result

        raw_text = result.value
        if isinstance(raw_text, str) and _ABSTENTION_MARKER in raw_text:
            result.abstained = True
            result.value = None
            return result

        parsed = self._parse_yesno(raw_text)
        result.value = parsed
        if parsed is None:
            # Ambiguous — not a failure, just indeterminate
            result.success = True
        return result

    # ------------------------------------------------------------------
    # Provenance helper (for callers to attach to generated fields)
    # ------------------------------------------------------------------

    def make_provenance(
        self,
        result: ReasoningResult,
        caller: str = "",
        evidence_refs: list[str] | None = None,
    ) -> dict:
        """Build a provenance dict from a reasoning result."""
        return {
            "source": "llm_reasoning",
            "model": result.model,
            "prompt_hash": result.prompt_hash,
            "generated_at": result.generated_at,
            "elapsed_seconds": result.elapsed_seconds,
            "confidence": result.confidence,
            "caller": caller,
            "evidence_refs": evidence_refs or [],
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute(
        self,
        prompt: str,
        system: str,
        caller: str,
        think: bool | None = None,
    ) -> ReasoningResult:
        """Budget-check, call Ollama, build base ReasoningResult."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        now_iso = datetime.now(timezone.utc).isoformat()

        base = ReasoningResult(
            model=self.config.model,
            prompt_hash=prompt_hash,
            generated_at=now_iso,
        )

        # Budget check
        if self._queue is not None and not self._queue.can_dispatch():
            base.error = "Over daily VLM budget"
            logger.debug("LLM reasoning skipped (%s): over budget", caller)
            return base

        # Call Ollama
        try:
            raw_text, elapsed = self._call_ollama(prompt, system, think=think)
        except ConnectionError as exc:
            base.error = f"Ollama connection failed: {exc}"
            logger.warning("LLM reasoning failed (%s): %s", caller, exc)
            return base
        except Exception as exc:
            base.error = f"Ollama call failed: {exc}"
            logger.warning("LLM reasoning failed (%s): %s", caller, exc)
            return base

        base.success = True
        base.elapsed_seconds = elapsed
        base.value = raw_text

        # Record compute time on budget queue
        if self._queue is not None:
            try:
                job_id = f"reasoning-{uuid.uuid4().hex[:8]}"
                self._queue.record_completion(
                    job_id=job_id,
                    compute_minutes=elapsed / 60.0,
                    result={"caller": caller, "prompt_hash": prompt_hash},
                )
            except Exception:
                logger.debug("Failed to record compute time", exc_info=True)

        return base

    def _call_ollama(
        self,
        prompt: str,
        system: str,
        num_predict: int | None = None,
        think: bool | None = None,
    ) -> tuple[str, float]:
        """Call Ollama /api/generate. This is the mock boundary for tests.

        Returns the answer without ``<think>`` blocks and the call's
        wall-clock seconds.
        """
        use_think = think if think is not None else self.config.think
        response = OllamaClient(self.config.ollama_host).generate(
            self.config.model,
            prompt,
            system=system,
            options={
                "num_predict": num_predict or self.config.num_predict,
                "num_ctx": 16384,
                "temperature": 0.3,
            },
            think=use_think,
            timeout=self.config.timeout,
            call_type="reasoning",
        )
        return clean_model_text(response.text), response.elapsed

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str | Any) -> dict | None:
        """Extract a JSON dict from raw text."""
        if not isinstance(raw, str):
            return None
        try:
            return extract_json(raw)
        except ValueError:
            return None

    @staticmethod
    def _parse_yesno(raw: str | Any) -> bool | None:
        """Parse a yes/no response. Returns True, False, or None."""
        if not isinstance(raw, str):
            return None

        text = raw.strip().lower()

        # Check first word
        first_word = text.split()[0] if text.split() else ""
        first_word = first_word.strip(".,!:;")

        if first_word in ("yes", "true", "correct", "affirmative"):
            return True
        if first_word in ("no", "false", "incorrect", "negative"):
            return False

        # Check if YES or NO appears as a whole word in the opening
        import re as _re
        if _re.search(r"\byes\b", text[:30]):
            return True
        if _re.search(r"\bno\b", text[:30]):
            return False

        return None
