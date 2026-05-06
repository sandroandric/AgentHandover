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

    goal: str = ""
    strategy: str = ""
    selection_criteria: list[dict] = field(default_factory=list)
    content_templates: list[dict] = field(default_factory=list)
    decision_branches: list[dict] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    confidence: float = 0.0


class EmptyInsightsError(ValueError):
    """Raised when ``_parse_insights`` receives a JSON object that has no
    substantive behavioral content (no goal AND no strategy AND no
    selection_criteria AND no guardrails).

    The synthesizer treats this as a recoverable failure and retries the
    VLM call once before giving up.  Distinguished from a malformed-JSON
    failure (which surfaces as ``parsed is None`` upstream) so the retry
    feedback can be different.
    """


# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

BEHAVIORAL_SYSTEM_PROMPT = """\
You are analyzing the work of a SPECIFIC INDIVIDUAL — not workflows in \
general.  When a USER PROFILE is provided, treat it as authoritative \
context about who this person is, what tools they use, and how they work.  \
Reason about WHY they work this way given their role and stack, not just \
WHAT patterns emerge from clicks.

Your output is structured JSON describing the strategy, decision logic, \
and behavioral patterns behind observed workflows — grounded in the \
user's actual identity and tooling when known.

Respond with ONLY valid JSON. No markdown fences, no commentary."""


BEHAVIORAL_SYNTHESIS_PROMPT = """\
You are analyzing {observation_count} recorded sessions of the same workflow.

WORKFLOW TITLE: {title}
WORKFLOW DESCRIPTION: {description}

{user_context}

The mechanical steps have already been identified:
{steps_summary}

{timeline_evidence}
{daily_context}
{session_context}
{continuity_context}
{evidence_context}

Now analyze the STRATEGY behind these steps.  Respond with a JSON object:

{{
  "goal": "<ONE concrete sentence: what is the user ultimately producing, \
FOR WHOM, and with WHAT TRIGGER/CADENCE? Must include: the artifact produced \
(e.g. 'email to self', 'Slack update to team'), the recipient (specific \
person, team, or channel if visible), and the trigger (daily, weekly, on \
event X, etc.). Example of concrete: 'Sends a daily digest email of top 10 \
Hacker News stories to themselves at sandro@sandric.co to stay updated on \
tech news.' Example of BAD abstract: 'research-to-communication cycle'. \
If you cannot determine the concrete goal, say so explicitly: \
'Unclear — recipient/cadence not visible in observations'>",

  "strategy": "<2-4 sentences elaborating HOW the user achieves the goal — \
the approach, ordering, and reasoning. This is a SUPPORTING explanation of \
the goal above, not a restatement. Must be grounded in specific evidence \
from the observations — quote visible text (emails, subjects, URLs, counts) \
when present.>",

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
- "goal" is THE MOST IMPORTANT FIELD. It must be CONCRETE — naming the \
artifact, the recipient, and the trigger. If any of those three are visible \
in the observations (specific email addresses, subject lines, counts, \
usernames, channels, schedules), you MUST incorporate them verbatim. \
Generic phrases like "research cycle", "communication workflow", \
"information management" are NOT acceptable goals — they describe shape, \
not intent.
- "strategy" should be 2-4 sentences explaining HOW the user achieves the \
goal, not restating the goal itself
- Use VERBATIM TEXT from OCR/annotations where it appears (recipient emails, \
subject lines, URLs, item counts, selected text). Don't generalize "an email \
address" when the observations show "sandro@sandric.co".
- "selection_criteria" should explain WHAT the user engages with vs skips
- "content_templates" should capture the STRUCTURE of text the user produces
- "decision_branches" should explain branch conditions from observed behavior
- "guardrails" should list things the user consistently AVOIDS doing
- "timing" should reflect observed durations and work phases
- "confidence" should be 0.0-1.0 reflecting how well the sessions support analysis
- If you cannot determine something, use empty arrays or null values
- Base everything on OBSERVED patterns, do not speculate
- When a USER PROFILE is provided, your strategy and guardrails MUST \
reflect who this user actually is — their role, primary tools, and \
accounts.  Don't describe a generic "user" when the profile tells you \
this is e.g. a designer, a developer, or a founder.  Ground every \
inference in their actual stack and working patterns.
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
        knowledge_base: Any | None = None,
    ) -> None:
        self.config = config or SynthesizerConfig()
        self._vlm_queue = vlm_queue
        # KB is used to inject user profile (tools/accounts/working hours)
        # into the synthesis prompt so the model reasons about WHO the user
        # is, not an abstract "user".  Optional — falls back to observation-
        # derived signals or a generic framing when KB/profile is empty.
        self._kb = knowledge_base

    def synthesize(
        self,
        slug: str,
        procedure: dict,
        observations: list[list[dict]],
        daily_summaries: list[dict] | None = None,
        linked_tasks: list[dict] | None = None,
        continuity_spans: list[dict] | None = None,
        force: bool = False,
        _retry_count: int = 0,
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
            # Allow one retry for transient parse failures.  After a single
            # retry we give up to avoid runaway loops on systemically broken
            # model output.
            if _retry_count < 1:
                logger.info(
                    "Retrying behavioral synthesis for '%s' after parse failure", slug,
                )
                return self.synthesize(
                    slug, procedure, observations,
                    daily_summaries=daily_summaries,
                    linked_tasks=linked_tasks,
                    continuity_spans=continuity_spans,
                    force=force,
                    _retry_count=_retry_count + 1,
                )
            return BehavioralInsights()

        try:
            insights = self._parse_insights(parsed)
        except EmptyInsightsError as exc:
            logger.warning(
                "Behavioral synthesis returned empty insights for '%s': %s "
                "(response length=%d)",
                slug, exc, len(raw_response),
            )
            if _retry_count < 1:
                logger.info(
                    "Retrying behavioral synthesis for '%s' after empty insights", slug,
                )
                return self.synthesize(
                    slug, procedure, observations,
                    daily_summaries=daily_summaries,
                    linked_tasks=linked_tasks,
                    continuity_spans=continuity_spans,
                    force=force,
                    _retry_count=_retry_count + 1,
                )
            return BehavioralInsights()

        logger.info(
            "Behavioral synthesis for '%s': goal=%s, strategy=%s, "
            "%d criteria, %d templates, %d branches, %d guardrails "
            "(%.1fs VLM, confidence=%.2f, retries=%d)",
            slug,
            bool(insights.goal),
            bool(insights.strategy),
            len(insights.selection_criteria),
            len(insights.content_templates),
            len(insights.decision_branches),
            len(insights.guardrails),
            elapsed,
            insights.confidence,
            _retry_count,
        )
        if insights.goal:
            logger.info(
                "Behavioral synthesis for '%s' extracted goal: %s",
                slug, insights.goal[:240],
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

        ``last_synthesized`` is only set when at least one substantive field
        was extracted (goal, strategy, selection_criteria, guardrails, or
        decision_branches).  Previously this timestamp was set
        unconditionally — even when ``insights`` was the all-defaults
        ``BehavioralInsights()`` returned on parse failure — which produced
        the v0.2.x false-positive "synthesis succeeded" signal on Skills
        like marketing-stats-email.
        """
        proc = dict(procedure)

        had_substantive_extraction = bool(
            insights.goal
            or insights.strategy
            or insights.selection_criteria
            or insights.guardrails
            or insights.decision_branches
        )

        if insights.goal:
            proc["goal"] = insights.goal
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

        if had_substantive_extraction:
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

        # Format user profile context (who the user is)
        user_context = self._format_user_context(observations)

        # Format per-frame timeline evidence — the actual verbatim text the
        # frames contain (email addresses, typed text, URLs, values).  This
        # is the data the model needs to write a CONCRETE goal.  Without it,
        # the prompt asks for verbatim text but never shows any.
        timeline_evidence = self._format_timeline_evidence(observations)

        return BEHAVIORAL_SYNTHESIS_PROMPT.format(
            observation_count=len(observations),
            title=title,
            description=description,
            user_context=user_context or "(no user profile yet — first-time user; infer from observations only)",
            steps_summary=steps_summary,
            timeline_evidence=timeline_evidence or "(no per-frame evidence available)",
            daily_context=daily_context or "(no daily context available)",
            session_context=session_context or "(no cross-day context available)",
            continuity_context=continuity_context or "(no continuity context available)",
            evidence_context=evidence_context or "(no extracted evidence available)",
            voice_guidance=voice_guidance,
        )

    @staticmethod
    def _format_timeline_evidence(observations: list[list[dict]]) -> str:
        """Format per-frame evidence so the model can quote verbatim text.

        ``observations`` is a list of timelines (one per recorded session).
        Each timeline is a list of frame dicts.  This formatter extracts
        the rich fields populated by the focus_processor's
        ``_build_pre_analysis_obs`` (action, app, location, email_addresses,
        urls, typed_text, visible_values, plus copy/paste content_preview
        when present) and renders them as a numbered evidence list the
        synthesizer prompt can quote from.

        Without this, the prompt asks the model to write a concrete goal
        with verbatim emails/URLs/counts but never actually surfaces any
        of that data — the model has to guess from the steps_summary
        alone, and rightly returns the explicit "Unclear" fallback.
        """
        if not observations:
            return ""

        lines: list[str] = ["TIMELINE EVIDENCE (verbatim text from each frame — quote these directly when filling goal/strategy):"]
        max_frames_total = 60  # cap to keep prompt size reasonable
        rendered = 0

        for sess_idx, timeline in enumerate(observations):
            if not timeline:
                continue
            if len(observations) > 1:
                lines.append(f"\n[Session {sess_idx + 1}]")
            for f_idx, frame in enumerate(timeline):
                if rendered >= max_frames_total:
                    lines.append("  ... (more frames truncated)")
                    break
                if not isinstance(frame, dict):
                    continue
                parts: list[str] = []
                app = frame.get("app", "") or ""
                loc = frame.get("location", "") or ""
                if app:
                    parts.append(f"app={app}")
                if loc:
                    parts.append(f"loc={loc}")
                action = frame.get("action", "") or ""
                if action:
                    parts.append(f"doing={action[:140]}")

                # The verbatim grounding fields (added by OCR injection +
                # _build_pre_analysis_obs).  These are what the model needs
                # to write a concrete goal.
                emails = frame.get("email_addresses") or []
                if emails:
                    parts.append(f"emails={list(emails)[:6]}")
                names = frame.get("names") or []
                if names:
                    parts.append(f"names={list(names)[:6]}")
                urls = frame.get("urls") or []
                if urls:
                    parts.append(f"urls={list(urls)[:4]}")
                typed = frame.get("typed_text") or ""
                if typed:
                    parts.append(f'typed="{typed[:140]}"')
                selected = frame.get("selected_text") or ""
                if selected:
                    parts.append(f'selected="{selected[:140]}"')
                # active_element is a STRONG signal for what the user is
                # currently focused on (e.g. "Send button" tells the
                # model the user just sent the email)
                active = frame.get("active_element") or ""
                if active:
                    parts.append(f'focus="{active[:120]}"')
                values = frame.get("visible_values") or []
                if values:
                    parts.append(f"values={list(values)[:12]}")
                headings = frame.get("headings") or []
                if headings:
                    parts.append(f"headings={list(headings)[:6]}")
                # Compose-specific structured fields (recipient, subject,
                # body) — these are the highest-signal verbatim text
                compose = frame.get("compose")
                if isinstance(compose, dict) and compose:
                    parts.append(f"compose={compose}")
                # Copy/paste content_preview (set by focus_processor
                # when a clipboard event is attached to the frame)
                clip = frame.get("clipboard_preview") or ""
                if clip:
                    parts.append(f'COPIED="{clip[:200]}"')

                if parts:
                    lines.append(f"  frame {f_idx + 1}: " + " | ".join(parts))
                    rendered += 1

            if rendered >= max_frames_total:
                break

        return "\n".join(lines) if rendered > 0 else ""

    # ------------------------------------------------------------------
    # User context formatting
    # ------------------------------------------------------------------

    def _format_user_context(self, observations: list[list[dict]]) -> str:
        """Format a USER PROFILE block describing who the user is.

        Order of preference:
        1. KB profile (rich: tools, accounts, working_hours, comm_style) —
           used when the profile has been built from accumulated daily
           summaries.
        2. Cold-start fallback: derive primary apps from the current
           observation batch so even the very first synthesis has some
           "who" context to reason about.
        3. Empty string — caller substitutes a generic placeholder.
        """
        # 1) KB profile
        profile = self._load_kb_profile()
        if profile and self._profile_has_signal(profile):
            return self._format_kb_profile(profile)

        # 2) Cold-start: derive signals from observations
        cold_start = self._format_cold_start_context(observations)
        if cold_start:
            return cold_start

        return ""

    def _load_kb_profile(self) -> dict | None:
        """Load the user profile from the KB, if available."""
        if self._kb is None:
            return None
        try:
            return self._kb.get_profile()
        except Exception:
            logger.debug("Failed to load KB profile for synthesizer", exc_info=True)
            return None

    @staticmethod
    def _profile_has_signal(profile: dict) -> bool:
        """True if the profile has enough data to be worth injecting."""
        if not isinstance(profile, dict):
            return False
        tools = profile.get("tools") or {}
        accounts = profile.get("accounts") or []
        working_hours = profile.get("working_hours") or {}
        # At minimum need either a primary app or an identifiable account
        has_apps = bool(tools.get("primary_apps")) or bool(
            tools.get("browser") or tools.get("editor") or tools.get("terminal")
        )
        return bool(has_apps or accounts or working_hours)

    @staticmethod
    def _format_kb_profile(profile: dict) -> str:
        """Render the KB profile as a USER PROFILE block for the prompt."""
        lines = ["USER PROFILE (who this user is):"]

        tools = profile.get("tools") or {}
        primary_apps = tools.get("primary_apps") or []
        if primary_apps:
            top = primary_apps[:6]
            app_strs = [
                f"{a.get('app', '?')} ({a.get('total_minutes', 0):.0f} min)"
                for a in top
            ]
            lines.append("  Primary apps: " + ", ".join(app_strs))

        stack_parts = []
        if tools.get("browser"):
            stack_parts.append(f"browser={tools['browser']}")
        if tools.get("editor"):
            stack_parts.append(f"editor={tools['editor']}")
        if tools.get("terminal"):
            stack_parts.append(f"terminal={tools['terminal']}")
        if stack_parts:
            lines.append("  Stack: " + ", ".join(stack_parts))

        accounts = profile.get("accounts") or []
        if accounts:
            account_strs = [
                f"{a.get('service', '?')} ({a.get('frequency', '?')})"
                for a in accounts[:8]
            ]
            lines.append("  Accounts used: " + ", ".join(account_strs))

        wh = profile.get("working_hours") or {}
        wh_parts = []
        if wh.get("typical_start") and wh.get("typical_end"):
            wh_parts.append(f"{wh['typical_start']}-{wh['typical_end']}")
        if wh.get("avg_active_hours"):
            wh_parts.append(f"~{wh['avg_active_hours']}h/day active")
        if wh.get("weekend_active"):
            wh_parts.append("weekends active")
        if wh_parts:
            lines.append("  Working hours: " + ", ".join(wh_parts))

        cs = profile.get("communication_style") or {}
        channels = cs.get("primary_channels") or []
        if channels:
            lines.append("  Primary comm channels: " + ", ".join(channels[:4]))

        ws = profile.get("writing_style") or {}
        if ws.get("formality"):
            lines.append(
                f"  Writing style: {ws['formality']}"
                + (f" (confidence {ws.get('confidence', 0):.2f})" if ws.get("confidence") else "")
            )

        # Inferred role hint — cheap heuristic so the model gets a starting
        # frame.  The LLM can override this with its own reasoning.
        role_hint = BehavioralSynthesizer._infer_role_hint(tools, accounts)
        if role_hint:
            lines.append(f"  Likely role (heuristic): {role_hint}")

        return "\n".join(lines) if len(lines) > 1 else ""

    @staticmethod
    def _infer_role_hint(tools: dict, accounts: list) -> str:
        """Lightweight role hint from tools + accounts.

        Intentionally conservative — only returns a hint when the
        signals are strong.  The VLM is expected to refine or reject
        this.
        """
        account_services = {
            a.get("service", "").lower() for a in accounts if isinstance(a, dict)
        }
        primary_apps = {
            (a.get("app", "") or "").lower()
            for a in (tools.get("primary_apps") or [])
            if isinstance(a, dict)
        }
        all_signals = account_services | primary_apps
        editor = (tools.get("editor") or "").lower()

        dev_signals = {"github", "vercel", "aws"}
        design_signals = {"figma", "sketch"}
        pm_signals = {"jira", "linear", "notion"}

        has_dev = bool(dev_signals & all_signals) or bool(editor)
        has_design = bool(design_signals & all_signals) or "figma" in primary_apps
        has_pm = bool(pm_signals & all_signals)

        if has_dev and not has_design:
            return "software developer / engineer"
        if has_design and not has_dev:
            return "designer"
        if has_dev and has_design and has_pm:
            return "founder / generalist builder"
        if has_pm and not has_dev and not has_design:
            return "product / project manager"
        return ""

    @staticmethod
    def _format_cold_start_context(observations: list[list[dict]]) -> str:
        """Cold-start fallback: derive apps from current observation batch.

        When there's no KB profile yet (first-time user, first synthesis),
        we still want the VLM to see which apps the user actually uses in
        this workflow.  We aggregate app names from the observation events
        so the model can at least reason about the tooling.
        """
        from collections import Counter

        app_counter: Counter = Counter()
        for timeline in observations or []:
            for event in timeline or []:
                if not isinstance(event, dict):
                    continue
                app = (
                    event.get("app")
                    or event.get("app_name")
                    or event.get("application")
                    or (event.get("context") or {}).get("app")
                )
                if isinstance(app, str) and app.strip():
                    app_counter[app.strip()] += 1

        if not app_counter:
            return ""

        top_apps = [app for app, _ in app_counter.most_common(6)]
        lines = [
            "USER PROFILE (first-time user — derived from this workflow only):",
            "  Observed apps in this workflow: " + ", ".join(top_apps),
            "  Note: no historical profile exists yet.  Reason from what you see "
            "in this workflow, and be conservative about generalizing to the "
            "user's broader habits.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_insights(data: dict) -> BehavioralInsights:
        """Parse VLM JSON response into BehavioralInsights.

        Raises ``EmptyInsightsError`` when the parsed JSON has no
        substantive content — no goal, no strategy, no selection_criteria,
        and no guardrails.  This prevents the v0.2.x silent-failure mode
        where the VLM returned ``{"goal": "", "strategy": "", "guardrails": []}``,
        the parser accepted it, ``last_synthesized`` was set, and the
        procedure shipped to ``agent_ready`` with empty behavioral data.
        Decision_branches and content_templates alone are not enough to
        consider synthesis successful — they're optional, while goal or
        strategy is required for an actionable Skill.
        """
        if not isinstance(data, dict):
            raise EmptyInsightsError(
                f"VLM response is not a JSON object (got {type(data).__name__})"
            )

        goal = str(data.get("goal", "") or "").strip()
        strategy = str(data.get("strategy", "") or "").strip()
        selection_criteria = [
            sc for sc in data.get("selection_criteria", [])
            if isinstance(sc, dict) and sc.get("criterion")
        ]
        content_templates = [
            ct for ct in data.get("content_templates", [])
            if isinstance(ct, dict) and ct.get("template")
        ]
        decision_branches = [
            br for br in data.get("decision_branches", [])
            if isinstance(br, dict) and br.get("condition")
        ]
        guardrails = [
            g for g in data.get("guardrails", [])
            if isinstance(g, str) and g.strip()
        ]
        timing = data.get("timing", {}) if isinstance(data.get("timing"), dict) else {}
        confidence = 0.0
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        # Require at least one substantive field for the synthesis to be
        # considered successful.  goal OR strategy is the minimum bar.
        if not goal and not strategy and not selection_criteria and not guardrails:
            raise EmptyInsightsError(
                "synthesis returned no goal, strategy, criteria, or guardrails"
            )

        return BehavioralInsights(
            goal=goal,
            strategy=strategy,
            selection_criteria=selection_criteria,
            content_templates=content_templates,
            decision_branches=decision_branches,
            guardrails=guardrails,
            timing=timing,
            confidence=confidence,
        )
