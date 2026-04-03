"""Behavioral synthesizer — second-pass VLM analysis for strategy extraction.

Runs AFTER initial SOP generation to extract higher-level patterns:
- Strategy: the overall approach behind the mechanical steps
- Selection criteria: what the user engages with vs skips
- Content templates: structural patterns in user-produced output
- Decision branches: conditions that determine different paths
- Guardrails: behavioral boundaries consistently respected
- Timing: duration, phases, and think points

This module uses a separate VLM prompt focused on behavioral patterns,
not step discovery.  It takes the already-generated steps as context
so the VLM can focus on the WHY, not the WHAT.

Trigger: only when a procedure accumulates 3+ observations.
Re-trigger: every 3 additional observations after last synthesis.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BehavioralInsights:
    """Results of behavioral synthesis for a procedure."""

    strategy: str = ""
    selection_criteria: list[dict] = field(default_factory=list)
    content_templates: list[dict] = field(default_factory=list)
    decision_branches: list[dict] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

BEHAVIORAL_SYSTEM_PROMPT = """\
You are a behavioral analyst examining recorded work sessions. \
You analyze patterns in HOW and WHY a person works, not just WHAT they click. \
Your output is structured JSON describing the strategy, decision logic, and \
behavioral patterns behind observed workflows.

Respond with ONLY valid JSON. No markdown fences, no commentary."""


BEHAVIORAL_SYNTHESIS_PROMPT = """\
You are analyzing {observation_count} recorded sessions of the same workflow.

WORKFLOW TITLE: {title}
WORKFLOW DESCRIPTION: {description}

The mechanical steps have already been identified:
{steps_summary}

{daily_context}
{session_context}
{continuity_context}
{evidence_context}

Now analyze the STRATEGY behind these steps.  Respond with a JSON object:

{{
  "strategy": "<high-level description of the user's approach and goals — \
not a list of steps, but the reasoning and intent behind them>",

  "selection_criteria": [
    {{
      "criterion": "<what determines which items get attention vs are skipped>",
      "examples": ["<concrete examples from the sessions>"],
      "confidence": 0.0
    }}
  ],

  "content_templates": [
    {{
      "template": "<structural pattern in user-produced output, with {{{{variable}}}} \
placeholders for parts that change>",
      "variables": ["<list of variable names in the template>"],
      "examples": ["<1-2 concrete examples of this template being used>"]
    }}
  ],

  "decision_branches": [
    {{
      "condition": "<what condition determines which path the user takes>",
      "action": "<what the user does when this condition is true>",
      "observed_count": 0
    }}
  ],

  "guardrails": [
    "<behavioral boundary: something the user NEVER does or always avoids>"
  ],

  "timing": {{
    "avg_duration_minutes": 0,
    "phases": [
      {{
        "name": "<phase name, e.g. 'research', 'compose', 'review'>",
        "typical_duration_minutes": 0,
        "description": "<what happens in this phase>"
      }}
    ],
    "think_points": ["<moments where the user pauses to think or decide>"]
  }},

  "confidence": 0.0
}}

Rules:
- "strategy" should be 2-4 sentences explaining the APPROACH, not the steps
- "selection_criteria" should explain WHAT the user engages with vs skips
- "content_templates" should capture the STRUCTURE of text the user produces
- "decision_branches" should explain branch conditions from observed behavior
- "guardrails" should list things the user consistently AVOIDS doing
- "timing" should reflect observed durations and work phases
- "confidence" should be 0.0-1.0 reflecting how well the sessions support analysis
- If you cannot determine something, use empty arrays or null values
- Base everything on OBSERVED patterns, do not speculate
{voice_guidance}

Respond with ONLY the JSON object."""


# ---------------------------------------------------------------------------
# VLM client (reused pattern from sop_generator)
# ---------------------------------------------------------------------------

def _call_ollama_text(
    *,
    model: str,
    prompt: str,
    host: str = "http://localhost:11434",
    num_predict: int = 12000,
    system: str = "",
    timeout: float = 600.0,
    think: bool | str = True,
    extra_options: dict | None = None,
) -> tuple[str, float]:
    """Call Ollama's /api/generate for text-only synthesis."""
    import urllib.request
    import urllib.error

    url = f"{host}/api/generate"

    options: dict = {"num_predict": num_predict}
    if extra_options:
        options.update(extra_options)

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": options,
    }
    if system:
        payload["system"] = system

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Ollama not reachable at {host}: {exc}") from exc

    elapsed = time.monotonic() - start
    return result.get("response", ""), elapsed


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_json_response(raw: str) -> dict | None:
    """Parse a VLM JSON response, handling fences, thinking tags, and truncation."""
    text = _THINK_RE.sub("", raw).strip()
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    if not text:
        return None

    # Find the first { to start of JSON
    start = text.find("{")
    if start < 0:
        return None
    text = text[start:]

    # Try direct parse
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    # Try repairing truncated JSON by closing open brackets
    repaired = text
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")

    # Strip trailing partial content (after last complete value)
    # Find the last comma or complete value before truncation
    for trim_char in [",", '"', "}", "]"]:
        last_pos = repaired.rfind(trim_char)
        if last_pos > 0:
            candidate = repaired[:last_pos + 1]
            # Close open brackets/braces
            suffix = "]" * max(0, candidate.count("[") - candidate.count("]"))
            suffix += "}" * max(0, candidate.count("{") - candidate.count("}"))
            try:
                data = json.loads(candidate + suffix)
                if isinstance(data, dict):
                    logger.debug("Repaired truncated behavioral JSON (%d open braces closed)", open_braces)
                    return data
            except json.JSONDecodeError:
                continue

    return None


# ---------------------------------------------------------------------------
# BehavioralSynthesizer
# ---------------------------------------------------------------------------


@dataclass
class SynthesizerConfig:
    """Configuration for the behavioral synthesizer."""

    model: str = "qwen3.5:4b"
    ollama_host: str = "http://localhost:11434"
    num_predict: int = 8000
    timeout: float = 600.0
    min_observations: int = 3
    re_synthesis_interval: int = 3


class BehavioralSynthesizer:
    """Extract behavioral patterns from accumulated workflow observations.

    Runs a second-pass VLM analysis AFTER initial SOP generation.
    Takes the same demonstrations plus accumulated daily summaries
    and session links to extract strategy, decision patterns,
    content templates, and guardrails.
    """

    def __init__(
        self,
        config: SynthesizerConfig | None = None,
        vlm_queue: Any | None = None,
    ) -> None:
        self.config = config or SynthesizerConfig()
        self._vlm_queue = vlm_queue

    def synthesize(
        self,
        slug: str,
        procedure: dict,
        observations: list[list[dict]],
        daily_summaries: list[dict] | None = None,
        linked_tasks: list[dict] | None = None,
        continuity_spans: list[dict] | None = None,
        force: bool = False,
    ) -> BehavioralInsights:
        """Run behavioral synthesis for a procedure.

        Args:
            slug: Procedure identifier.
            procedure: The current v3 procedure dict.
            observations: All demonstrations (list of timelines).
            daily_summaries: Optional daily summaries for timing/recurrence.
            linked_tasks: Optional cross-day task links.
            continuity_spans: Optional continuity spans for interruption patterns.
            force: If True, bypass the min_observations check. Used for
                focus recordings where a single demonstration should still
                get behavioral analysis. Does NOT bypass the budget check.

        Returns:
            BehavioralInsights with extracted patterns.
        """
        if not force and len(observations) < self.config.min_observations:
            logger.debug(
                "Skipping synthesis for '%s': %d observations < %d minimum",
                slug, len(observations), self.config.min_observations,
            )
            return BehavioralInsights()

        # Budget check — avoid unbounded VLM spending
        if self._vlm_queue is not None and not self._vlm_queue.can_dispatch():
            logger.debug(
                "Skipping synthesis for '%s': over daily VLM budget", slug,
            )
            return BehavioralInsights()

        # Build the synthesis prompt
        prompt = self._build_prompt(
            procedure, observations,
            daily_summaries=daily_summaries,
            linked_tasks=linked_tasks,
            continuity_spans=continuity_spans,
        )

        # Call VLM with model-specific profile
        from agenthandover_worker.model_profiles import get_profile
        profile = get_profile(self.config.model)
        try:
            raw_response, elapsed = _call_ollama_text(
                model=self.config.model,
                prompt=prompt,
                host=self.config.ollama_host,
                num_predict=12000,  # high budget — thinking consumes ~3-4K tokens
                system=profile.sop_system or BEHAVIORAL_SYSTEM_PROMPT,
                timeout=self.config.timeout,
                think=profile.sop_think if profile.sop_think else True,
                extra_options={
                    k: v for k, v in profile.sop_options().items()
                    if k not in ("num_predict",)
                },
            )
        except ConnectionError as exc:
            logger.warning("Behavioral synthesis VLM failed for '%s': %s", slug, exc)
            return BehavioralInsights()
        except Exception as exc:
            logger.warning("Behavioral synthesis failed for '%s': %s", slug, exc)
            return BehavioralInsights()

        # Record compute time on budget queue
        if self._vlm_queue is not None:
            try:
                import uuid as _uuid
                self._vlm_queue.record_completion(
                    job_id=f"synthesis-{_uuid.uuid4().hex[:8]}",
                    compute_minutes=elapsed / 60.0,
                    result={"caller": "behavioral_synthesizer", "slug": slug},
                )
            except Exception:
                logger.debug("Failed to record synthesis compute time", exc_info=True)

        # Parse response
        parsed = _parse_json_response(raw_response)
        if parsed is None:
            logger.warning(
                "Behavioral synthesis: invalid JSON for '%s' (response length=%d, first 200 chars: %s)",
                slug, len(raw_response), raw_response[:200],
            )
            return BehavioralInsights()

        insights = self._parse_insights(parsed)

        logger.info(
            "Behavioral synthesis for '%s': strategy=%s, "
            "%d criteria, %d templates, %d branches, %d guardrails "
            "(%.1fs VLM, confidence=%.2f)",
            slug,
            bool(insights.strategy),
            len(insights.selection_criteria),
            len(insights.content_templates),
            len(insights.decision_branches),
            len(insights.guardrails),
            elapsed,
            insights.confidence,
        )

        return insights

    def should_synthesize(self, procedure: dict) -> bool:
        """Check whether a procedure needs (re-)synthesis.

        Returns True if:
        - Total observations >= min_observations AND never synthesized, OR
        - Observations since last synthesis >= re_synthesis_interval
        """
        evidence = procedure.get("evidence", {})
        total_obs = evidence.get("total_observations", 0)

        if total_obs < self.config.min_observations:
            return False

        last_synth = procedure.get("last_synthesized")
        if last_synth is None:
            return True

        obs_at_last = procedure.get("_obs_at_last_synthesis", 0)
        return (total_obs - obs_at_last) >= self.config.re_synthesis_interval

    def merge_insights_into_procedure(
        self,
        procedure: dict,
        insights: BehavioralInsights,
    ) -> dict:
        """Merge synthesis results into a procedure dict.

        Does NOT mutate the input.  Returns a new dict.
        """
        proc = dict(procedure)

        if insights.strategy:
            proc["strategy"] = insights.strategy
        if insights.selection_criteria:
            proc["selection_criteria"] = insights.selection_criteria
        if insights.content_templates:
            proc["content_templates"] = insights.content_templates
        if insights.guardrails:
            constraints = dict(proc.get("constraints", {}))
            existing = constraints.get("guardrails", [])
            merged = list(existing)
            for g in insights.guardrails:
                if g not in merged:
                    merged.append(g)
            constraints["guardrails"] = merged
            proc["constraints"] = constraints
        if insights.decision_branches:
            proc["branches"] = insights.decision_branches
        if insights.timing:
            proc["workflow_rhythm"] = insights.timing
        if insights.confidence > 0:
            proc["behavioral_confidence"] = round(insights.confidence, 4)

        proc["last_synthesized"] = datetime.now(timezone.utc).isoformat()
        proc["_obs_at_last_synthesis"] = proc.get("evidence", {}).get(
            "total_observations", 0
        )

        return proc

    # ------------------------------------------------------------------
    # Individual extraction methods (for targeted re-extraction)
    # ------------------------------------------------------------------

    def extract_strategy(self, procedure: dict, observations: list[list[dict]]) -> str:
        """Extract just the strategy string from observations."""
        insights = self.synthesize(
            procedure.get("id", "unknown"), procedure, observations,
        )
        return insights.strategy

    def extract_selection_criteria(
        self, procedure: dict, observations: list[list[dict]],
    ) -> list[dict]:
        """Extract selection criteria from observations."""
        insights = self.synthesize(
            procedure.get("id", "unknown"), procedure, observations,
        )
        return insights.selection_criteria

    def extract_content_templates(
        self, procedure: dict, observations: list[list[dict]],
    ) -> list[dict]:
        """Extract content templates from observations."""
        insights = self.synthesize(
            procedure.get("id", "unknown"), procedure, observations,
        )
        return insights.content_templates

    def infer_guardrails(
        self, procedure: dict, observations: list[list[dict]],
    ) -> list[str]:
        """Infer guardrails from observations."""
        insights = self.synthesize(
            procedure.get("id", "unknown"), procedure, observations,
        )
        return insights.guardrails

    def extract_timing(
        self,
        daily_summaries: list[dict] | None = None,
        linked_tasks: list[dict] | None = None,
    ) -> dict:
        """Extract timing patterns from daily summaries and linked tasks.

        Pure data extraction — no VLM call needed.
        """
        timing: dict[str, Any] = {
            "avg_duration_minutes": None,
            "phases": [],
            "think_points": [],
        }

        if daily_summaries:
            durations = []
            for summary in daily_summaries:
                for task in summary.get("tasks", []):
                    dur = task.get("duration_minutes", 0)
                    if dur > 0:
                        durations.append(dur)
            if durations:
                timing["avg_duration_minutes"] = round(
                    sum(durations) / len(durations), 1
                )

        if linked_tasks:
            total_dur = []
            for link in linked_tasks:
                dur = link.get("total_duration_minutes", 0)
                if dur > 0:
                    total_dur.append(dur)
            if total_dur and timing["avg_duration_minutes"] is None:
                timing["avg_duration_minutes"] = round(
                    sum(total_dur) / len(total_dur), 1
                )

        return timing

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        procedure: dict,
        observations: list[list[dict]],
        daily_summaries: list[dict] | None = None,
        linked_tasks: list[dict] | None = None,
        continuity_spans: list[dict] | None = None,
    ) -> str:
        """Build the behavioral synthesis VLM prompt."""
        title = procedure.get("title", "Unknown Workflow")
        description = procedure.get("description", "")

        # Format steps summary
        steps = procedure.get("steps", [])
        steps_lines = []
        for s in steps:
            action = s.get("action", s.get("step", ""))
            app = s.get("app", s.get("parameters", {}).get("app", ""))
            conf = s.get("confidence", 0.0)
            steps_lines.append(f"  {s.get('step_id', '?')}. {action}")
            if app:
                steps_lines[-1] += f" (in {app})"
            if conf > 0:
                steps_lines[-1] += f" [confidence: {conf:.0%}]"
        steps_summary = "\n".join(steps_lines) if steps_lines else "(no steps)"

        # Format daily context
        daily_context = ""
        if daily_summaries:
            daily_lines = ["DAILY CONTEXT (from daily summaries):"]
            for ds in daily_summaries[-5:]:  # Last 5 days
                date = ds.get("date", "?")
                hours = ds.get("active_hours", 0)
                tasks = ds.get("tasks", [])
                matching = [
                    t for t in tasks
                    if t.get("matched_procedure") == procedure.get("id")
                    or t.get("intent", "").lower() in title.lower()
                ]
                if matching:
                    for t in matching:
                        daily_lines.append(
                            f"  {date}: {t.get('intent', '?')} "
                            f"({t.get('duration_minutes', 0):.0f} min)"
                        )
            if len(daily_lines) > 1:
                daily_context = "\n".join(daily_lines)

        # Format session context
        session_context = ""
        if linked_tasks:
            session_lines = ["CROSS-DAY SESSION CONTEXT:"]
            for lt in linked_tasks[:5]:
                intent = lt.get("intent", "?")
                span = lt.get("span_days", 0)
                total = lt.get("total_duration_minutes", 0)
                sessions = lt.get("sessions", [])
                session_lines.append(
                    f"  - {intent}: {len(sessions)} sessions over {span} days, "
                    f"total {total:.0f} min"
                )
            if len(session_lines) > 1:
                session_context = "\n".join(session_lines)

        # Format continuity context
        continuity_context = ""
        if continuity_spans:
            cont_lines = ["CONTINUITY CONTEXT (interruption patterns):"]
            for span in continuity_spans[:5]:
                goal = span.get("goal_summary", "?")
                interruptions = span.get("interruption_count", 0)
                duration = span.get("total_duration_seconds", 0) / 60.0
                cont_lines.append(
                    f"  - {goal}: {interruptions} interruptions, "
                    f"{duration:.0f} min total"
                )
            if len(cont_lines) > 1:
                continuity_context = "\n".join(cont_lines)

        # Format extracted evidence (from evidence_extractor pre-expiry extraction)
        evidence_context = ""
        extracted = procedure.get("evidence", {}).get("extracted_evidence", {})
        if extracted:
            ev_lines = ["EXTRACTED EVIDENCE (from raw observation analysis):"]

            # Content produced by user (clipboard, text inputs)
            content = extracted.get("content_produced", [])
            if content:
                ev_lines.append("  CONTENT PRODUCED BY USER:")
                for item in content[:10]:
                    ctype = item.get("type", "?")
                    preview = item.get("value_preview", item.get("content_types", ""))
                    ev_lines.append(f"    - [{ctype}] {preview}")

            # Selection/engagement signals
            signals = extracted.get("selection_signals", [])
            if signals:
                ev_lines.append("  ENGAGEMENT SIGNALS (what user spent time on vs skipped):")
                for sig in signals[:10]:
                    loc = sig.get("location", "?")
                    dwell = sig.get("avg_dwell_seconds", 0)
                    engagement = sig.get("engagement", "?")
                    ev_lines.append(
                        f"    - {loc}: {dwell:.0f}s avg dwell ({engagement} engagement)"
                    )

            # URL patterns
            urls = extracted.get("url_patterns", [])
            if urls:
                ev_lines.append("  URL PATTERNS:")
                for u in urls[:8]:
                    ev_lines.append(
                        f"    - {u.get('url', '?')} (visited {u.get('visit_count', 0)}x)"
                    )

            # Timing patterns
            timing = extracted.get("timing_patterns", {})
            if timing:
                total_dur = timing.get("total_duration_seconds", 0)
                pauses = timing.get("significant_pauses", 0)
                avg_gap = timing.get("avg_gap_seconds", 0)
                ev_lines.append(
                    f"  TIMING: {total_dur / 60:.0f} min total, "
                    f"{pauses} significant pauses, {avg_gap:.0f}s avg gap"
                )

            if len(ev_lines) > 1:
                evidence_context = "\n".join(ev_lines)

        # Build voice guidance from existing voice_profile
        voice_guidance = ""
        vp = procedure.get("voice_profile", {})
        if vp and vp.get("formality"):
            parts = [
                f"- The user's writing style is {vp['formality']}"
                f" (formality score: {vp.get('formality_score', 0):.1f})",
            ]
            if vp.get("avg_sentence_length"):
                parts.append(
                    f"- Average sentence length: {vp['avg_sentence_length']:.0f} words"
                )
            if vp.get("uses_emoji"):
                parts.append("- The user uses emoji/emoticons in their writing")
            parts.append(
                "- content_templates should MATCH this style — "
                "if the user writes casually, templates should be casual"
            )
            voice_guidance = "\nUSER WRITING STYLE:\n" + "\n".join(parts)

        return BEHAVIORAL_SYNTHESIS_PROMPT.format(
            observation_count=len(observations),
            title=title,
            description=description,
            steps_summary=steps_summary,
            daily_context=daily_context or "(no daily context available)",
            session_context=session_context or "(no cross-day context available)",
            continuity_context=continuity_context or "(no continuity context available)",
            evidence_context=evidence_context or "(no extracted evidence available)",
            voice_guidance=voice_guidance,
        )

    @staticmethod
    def _parse_insights(data: dict) -> BehavioralInsights:
        """Parse VLM JSON response into BehavioralInsights."""
        return BehavioralInsights(
            strategy=data.get("strategy", ""),
            selection_criteria=[
                sc for sc in data.get("selection_criteria", [])
                if isinstance(sc, dict) and sc.get("criterion")
            ],
            content_templates=[
                ct for ct in data.get("content_templates", [])
                if isinstance(ct, dict) and ct.get("template")
            ],
            decision_branches=[
                br for br in data.get("decision_branches", [])
                if isinstance(br, dict) and br.get("condition")
            ],
            guardrails=[
                g for g in data.get("guardrails", [])
                if isinstance(g, str) and g.strip()
            ],
            timing=data.get("timing", {}),
            confidence=float(data.get("confidence", 0.0)),
        )
