"""Variant detector — Needleman-Wunsch semantic alignment and variant detection.

Aligns workflow demonstrations step-by-step using dynamic programming,
detects divergent variants across multiple observations, and extracts
parameterised fields.  No VLM calls — everything from pre-computed data.
"""
from __future__ import annotations

import logging
import re
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from agenthandover_worker.sop_dedup import _url_to_domain
from agenthandover_worker.task_segmenter import _cosine_similarity

logger = logging.getLogger(__name__)

# Compiled regex patterns for parameter type inference
_RE_URL = re.compile(r"https?://")
_RE_EMAIL = re.compile(r"[^@]+@[^@]+\.[^@]+")
_RE_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_RE_DATE = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}")
_RE_FILEPATH_UNIX = re.compile(r"[/~][\w/.-]+")
_RE_FILEPATH_WIN = re.compile(r"[A-Z]:\\")

_PARAM_FIELDS = ("input", "target", "location")

#: The most steps one demonstration brings to alignment, and the most
#: entries the canonical list holds: the passive SOP prompt carries one line
#: per canonical step next to its sampled frames, and aligning two
#: demonstrations costs len(a) x len(b) similarity computations.
MAX_STEPS = 60


@dataclass
class AlignedStep:
    """A single position in a Needleman-Wunsch alignment."""
    position: int
    action_a: str | None = None
    action_b: str | None = None
    similarity: float = 0.0
    is_match: bool = False
    is_insertion: bool = False
    parameters_differ: list[str] = field(default_factory=list)
    step_a: dict | None = None
    step_b: dict | None = None
    # Positions of ``step_a`` in sequence A and ``step_b`` in sequence B
    # (None on the side that has a gap).
    index_a: int | None = None
    index_b: int | None = None


@dataclass
class WorkflowVariant:
    """A group of demonstrations that share a common divergence pattern."""
    variant_id: str
    context: dict = field(default_factory=dict)
    divergent_steps: list[dict] = field(default_factory=list)
    fixed_steps: list[dict] = field(default_factory=list)
    demo_indices: list[int] = field(default_factory=list)


@dataclass
class ParameterExtraction:
    """A field that varies across demonstrations at a given step position."""
    name: str
    type: str = "text"
    values_seen: list[str] = field(default_factory=list)
    step_positions: list[int] = field(default_factory=list)
    confidence: float = 0.0


def _self_align(steps: list[dict]) -> list[AlignedStep]:
    """Build a trivial perfect self-alignment for a step list."""
    return [
        AlignedStep(
            position=k, action_a=s.get("action", ""),
            action_b=s.get("action", ""), similarity=1.0,
            is_match=True, step_a=s, step_b=s, index_a=k, index_b=k,
        )
        for k, s in enumerate(steps)
    ]


def _evenly(n: int, cap: int) -> list[int]:
    """At most *cap* indices of ``range(n)``, evenly spaced, first and last kept."""
    if n <= cap:
        return list(range(n))
    if cap <= 1:
        return list(range(cap))
    return sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})


def _step_key(step: dict) -> tuple[str, ...]:
    return tuple(str(step.get(k) or "").strip().lower() for k in ("action", "app", "location"))


def merge_steps(
    steps: list[dict], embeddings: list[list[float] | None] | None = None,
) -> tuple[list[dict], list[list[float] | None] | None]:
    """*steps* with each run of consecutive steps sharing ``(action, app,
    location)`` merged into its first, then evenly capped at
    :data:`MAX_STEPS`.  *embeddings* (one per step, None where a step has
    none) are kept in step with them."""
    keep: list[int] = []
    for i, step in enumerate(steps):
        if not keep or _step_key(steps[keep[-1]]) != _step_key(step):
            keep.append(i)
    keep = [keep[i] for i in _evenly(len(keep), MAX_STEPS)]
    merged = [steps[i] for i in keep]
    if embeddings is None:
        return merged, None
    return merged, [embeddings[i] if i < len(embeddings) else None for i in keep]


def frame_step(annotation: object) -> dict | None:
    """The alignment step of one annotated frame: what the user was doing,
    where, and on what (the list entry they picked, else the control they
    were on).  None for a frame without an annotation."""
    from agenthandover_worker.scene_annotator import as_compact

    if not isinstance(annotation, dict):
        return None
    ann = as_compact(annotation)
    chosen = next(
        (str(item.get("label") or "")
         for item in ann.get("visible_items") or []
         if isinstance(item, dict) and item.get("chosen")),
        "",
    )
    return {
        "action": (ann.get("task_context") or {}).get("what_doing", ""),
        "app": ann.get("app", ""),
        "location": ann.get("location", ""),
        "target": chosen or str((ann.get("ui_state") or {}).get("active_element") or ""),
        "input": "",
    }


class VariantDetector:
    """Detect workflow variants and extract parameters from multiple demos.

    Uses Needleman-Wunsch global alignment with composite similarity
    (embedding cosine + app match + domain overlap) to align workflow steps,
    then groups demonstrations by divergence patterns.
    """

    def __init__(self, gap_penalty: float = -0.3, match_threshold: float = 0.60) -> None:
        self._gap_penalty = gap_penalty
        self._match_threshold = match_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def semantic_align(
        self, steps_a: list[dict], steps_b: list[dict],
        embeddings_a: list[list[float] | None] | None = None,
        embeddings_b: list[list[float] | None] | None = None,
    ) -> list[AlignedStep]:
        """Align two step sequences using Needleman-Wunsch dynamic programming.

        A step pair is compared on its embeddings when both steps have one,
        else on the word overlap of their actions."""
        m, n = len(steps_a), len(steps_b)
        if m == 0 and n == 0:
            return []
        if m == 0:
            return [AlignedStep(position=j, action_b=steps_b[j].get("action", ""),
                                is_insertion=True, step_b=steps_b[j], index_b=j)
                    for j in range(n)]
        if n == 0:
            return [AlignedStep(position=i, action_a=steps_a[i].get("action", ""),
                                is_insertion=True, step_a=steps_a[i], index_a=i)
                    for i in range(m)]

        gap = self._gap_penalty

        # 1. Build & initialise scoring matrix
        S = [[0.0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            S[i][0] = i * gap
        for j in range(1, n + 1):
            S[0][j] = j * gap

        # 2. Fill
        for i in range(1, m + 1):
            ea = embeddings_a[i - 1] if embeddings_a and i - 1 < len(embeddings_a) else None
            for j in range(1, n + 1):
                eb = embeddings_b[j - 1] if embeddings_b and j - 1 < len(embeddings_b) else None
                sim = self._composite_similarity(steps_a[i - 1], steps_b[j - 1], ea, eb)
                S[i][j] = max(S[i - 1][j - 1] + sim, S[i - 1][j] + gap, S[i][j - 1] + gap)

        # 3. Traceback
        aligned: list[AlignedStep] = []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                ea = embeddings_a[i - 1] if embeddings_a and i - 1 < len(embeddings_a) else None
                eb = embeddings_b[j - 1] if embeddings_b and j - 1 < len(embeddings_b) else None
                sim = self._composite_similarity(steps_a[i - 1], steps_b[j - 1], ea, eb)
                if S[i][j] == S[i - 1][j - 1] + sim:
                    aligned.append(AlignedStep(
                        position=0, action_a=steps_a[i - 1].get("action", ""),
                        action_b=steps_b[j - 1].get("action", ""), similarity=sim,
                        is_match=sim >= self._match_threshold,
                        parameters_differ=self._differing_params(steps_a[i - 1], steps_b[j - 1]),
                        step_a=steps_a[i - 1], step_b=steps_b[j - 1],
                        index_a=i - 1, index_b=j - 1,
                    ))
                    i -= 1; j -= 1
                    continue
            if i > 0 and (j == 0 or S[i - 1][j] + gap == S[i][j]):
                aligned.append(AlignedStep(position=0, action_a=steps_a[i - 1].get("action", ""),
                                           is_insertion=True, step_a=steps_a[i - 1],
                                           index_a=i - 1))
                i -= 1
            else:
                aligned.append(AlignedStep(position=0, action_b=steps_b[j - 1].get("action", ""),
                                           is_insertion=True, step_b=steps_b[j - 1],
                                           index_b=j - 1))
                j -= 1

        aligned.reverse()
        for idx, step in enumerate(aligned):
            step.position = idx
        return aligned

    def analyse(
        self, label: str, timelines: list[list[dict]],
        embeddings: list[list[list[float] | None]] | None = None,
    ) -> dict:
        """Passive discovery's analysis of one workflow's demonstrations.

        *timelines*: per demonstration, its frames (``{"annotation": ...}``);
        *embeddings*: per demonstration, the task segmenter's vector for
        each frame (None where it has none).  Each timeline becomes its
        frames' steps, merged and capped (:func:`merge_steps`); the demos
        are aligned once (:meth:`align_demos`) and that alignment feeds
        variants, parameters and the canonical workflow.

        Returns ``{"alignments", "variants", "parameters", "normalized",
        "demo_steps"}``, or ``{}`` when fewer than two demos have steps.
        """
        demo_steps: list[list[dict]] = []
        demo_embs: list[list[list[float] | None] | None] = []
        for d, timeline in enumerate(timelines):
            frame_embs = embeddings[d] if embeddings and d < len(embeddings) else None
            steps: list[dict] = []
            embs: list[list[float] | None] = []
            for i, frame in enumerate(timeline):
                step = frame_step(frame.get("annotation"))
                if step is None:
                    continue
                steps.append(step)
                embs.append(frame_embs[i] if frame_embs and i < len(frame_embs) else None)
            steps, merged_embs = merge_steps(steps, embs)
            if steps:
                demo_steps.append(steps)
                demo_embs.append(merged_embs)
        if len(demo_steps) < 2:
            return {}
        alignments = self.align_demos(demo_steps, demo_embs)
        variants = self.detect_variants(label, demo_steps, alignments=alignments)
        return {
            "alignments": alignments,
            "variants": variants,
            "parameters": self.extract_parameters(demo_steps, alignments),
            "normalized": self.normalize_workflow(demo_steps, variants, alignments),
            "demo_steps": demo_steps,
        }

    def align_demos(
        self, demos: list[list[dict]],
        embeddings: list[list[list[float] | None] | None] | None = None,
    ) -> list[list[AlignedStep]]:
        """Every demo aligned against the first (the first against itself):
        one Needleman-Wunsch pass per other demo, shared by
        :meth:`detect_variants`, :meth:`extract_parameters` and
        :meth:`normalize_workflow`.  *embeddings*: per demo, one vector (or
        None) per step."""
        if not demos:
            return []
        emb = embeddings or [None] * len(demos)
        return [_self_align(demos[0])] + [
            self.semantic_align(demos[0], demo, emb[0], emb[i])
            for i, demo in enumerate(demos[1:], start=1)
        ]

    def detect_variants(
        self, slug: str, demos: list[list[dict]],
        reference_steps: list[dict] | None = None,
        embeddings: list[list[list[float] | None]] | None = None,
        alignments: list[list[AlignedStep]] | None = None,
    ) -> list[WorkflowVariant]:
        """Detect workflow variants across multiple demonstrations.

        *alignments* (from :meth:`align_demos`: every demo against the
        first) are used as given instead of aligning again."""
        if len(demos) < 2:
            return []

        # Pick reference
        if alignments is not None:
            ref_idx = 0
        elif reference_steps is not None:
            ref, ref_idx, ref_emb = reference_steps, -1, None
        else:
            ref_idx = max(range(len(demos)), key=lambda i: len(demos[i]))
            ref = demos[ref_idx]
            ref_emb = embeddings[ref_idx] if embeddings else None

        # Align every demo against reference
        if alignments is None:
            alignments = []
            for d_idx, demo in enumerate(demos):
                if d_idx == ref_idx:
                    alignments.append(_self_align(ref))
                    continue
                demo_emb = embeddings[d_idx] if embeddings else None
                alignments.append(self.semantic_align(ref, demo, ref_emb, demo_emb))

        max_len = max(len(a) for a in alignments) if alignments else 0

        # Collect per-position actions
        pos_actions: dict[int, list[str | None]] = defaultdict(list)
        pos_steps: dict[int, list[dict | None]] = defaultdict(list)
        for a in alignments:
            for pos in range(max_len):
                if pos < len(a):
                    s = a[pos]
                    act = s.action_b if s.action_b is not None else s.action_a
                    pos_actions[pos].append(act)
                    pos_steps[pos].append(s.step_b if s.step_b is not None else s.step_a)
                else:
                    pos_actions[pos].append(None)
                    pos_steps[pos].append(None)

        # Classify positions
        fixed_pos, div_pos = [], []
        for pos in range(max_len):
            normed = {(a or "").strip().lower() for a in pos_actions[pos]}
            (fixed_pos if len(normed) <= 1 else div_pos).append(pos)

        if not div_pos:
            return []

        # Signature per demo → group by divergence pattern
        sigs: dict[int, tuple[str, ...]] = {}
        for d_idx in range(len(demos)):
            parts = []
            for pos in div_pos:
                acts = pos_actions[pos]
                parts.append((acts[d_idx] or "").strip().lower() if d_idx < len(acts) else "")
            sigs[d_idx] = tuple(parts)

        groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for d_idx, sig in sigs.items():
            groups[sig].append(d_idx)

        # Build variants
        variants: list[WorkflowVariant] = []
        for sig, demo_indices in groups.items():
            div_steps = []
            for pi, pos in enumerate(div_pos):
                first = demo_indices[0]
                step_data = pos_steps[pos][first] if first < len(pos_steps[pos]) else None
                div_steps.append({"position": pos, "action": sig[pi], "step": step_data})

            fix_steps = []
            for pos in fixed_pos:
                acts = pos_actions[pos]
                fix_steps.append({
                    "position": pos,
                    "action": (acts[0] or "").strip() if acts else "",
                    "step": pos_steps[pos][0] if pos_steps[pos] else None,
                })

            context = {}
            da = [d["action"] for d in div_steps if d["action"]]
            if da:
                context["distinguishing_actions"] = da

            variants.append(WorkflowVariant(
                variant_id=uuid.uuid4().hex[:12], context=context,
                divergent_steps=div_steps, fixed_steps=fix_steps,
                demo_indices=demo_indices,
            ))

        logger.debug("detect_variants slug=%s demos=%d variants=%d", slug, len(demos), len(variants))
        return variants

    def extract_parameters(
        self, demos: list[list[dict]],
        alignments: list[list[AlignedStep]] | None = None,
    ) -> list[ParameterExtraction]:
        """Extract parameterised fields from aligned demonstrations."""
        if len(demos) < 2:
            return []

        if alignments is None:
            alignments = self.align_demos(demos)
        if len(alignments) != len(demos):
            raise ValueError(
                f"{len(alignments)} alignments for {len(demos)} demos: pass one per "
                "demo, the first aligned against itself (see align_demos)"
            )

        max_len = max(len(a) for a in alignments) if alignments else 0
        num_demos = len(demos)
        parameters: list[ParameterExtraction] = []

        for pos in range(max_len):
            field_vals: dict[str, list[str]] = defaultdict(list)
            for a in alignments:
                if pos < len(a):
                    sdict = a[pos].step_b if a[pos].step_b is not None else a[pos].step_a
                    for f in _PARAM_FIELDS:
                        field_vals[f].append(str((sdict or {}).get(f, "") or ""))
                else:
                    for f in _PARAM_FIELDS:
                        field_vals[f].append("")

            for f in _PARAM_FIELDS:
                non_empty = [v for v in field_vals[f] if v.strip()]
                distinct = set(non_empty)
                if len(non_empty) < 2 or len(distinct) < 2:
                    continue
                parameters.append(ParameterExtraction(
                    name=f, type=self._infer_type(non_empty),
                    values_seen=sorted(distinct), step_positions=[pos],
                    confidence=len(non_empty) / num_demos,
                ))

        # Merge same-name parameters across positions
        merged: dict[str, ParameterExtraction] = {}
        for p in parameters:
            if p.name in merged:
                ex = merged[p.name]
                ex.step_positions.extend(p.step_positions)
                for v in p.values_seen:
                    if v not in ex.values_seen:
                        ex.values_seen.append(v)
                ex.values_seen.sort()
                ex.confidence = max(ex.confidence, p.confidence)
            else:
                merged[p.name] = p
        return list(merged.values())

    def normalize_workflow(
        self, demos: list[list[dict]], variants: list[WorkflowVariant],
        alignments: list[list[AlignedStep]] | None = None,
    ) -> dict:
        """Produce a canonical workflow from multiple demonstrations
        (*alignments* from :meth:`align_demos`, computed here when absent).

        Returns ``{"canonical_steps": [...], "branches": [...], "parameters": [...]}``
        with at most :data:`MAX_STEPS` canonical steps, evenly spaced over
        the workflow (each keeps its alignment ``position``).
        """
        if not demos:
            return {"canonical_steps": [], "branches": [], "parameters": []}
        if len(demos) == 1:
            canonical = [{"position": k, "action": s.get("action", ""), "confidence": 1.0,
                          "alternatives": [], "step": s} for k, s in enumerate(demos[0])]
            return {"canonical_steps": canonical, "branches": [], "parameters": []}

        if alignments is None:
            alignments = self.align_demos(demos)
        max_len = max(len(a) for a in alignments) if alignments else 0
        total = len(demos)

        canonical_steps: list[dict] = []
        branches: list[dict] = []

        for pos in range(max_len):
            counter: Counter[str] = Counter()
            rep_steps: dict[str, dict | None] = {}
            for a in alignments:
                if pos < len(a):
                    s = a[pos]
                    act = s.action_b if s.action_b is not None else s.action_a
                    normed = (act or "").strip()
                    counter[normed] += 1
                    if normed not in rep_steps:
                        rep_steps[normed] = s.step_b if s.step_b is not None else s.step_a
            if not counter:
                continue

            mc = counter.most_common()
            canon_act, canon_cnt = mc[0]
            alts = [{"action": a, "observation_count": c, "step": rep_steps.get(a)}
                    for a, c in mc[1:]]
            canonical_steps.append({
                "position": pos, "action": canon_act,
                "confidence": canon_cnt / total, "alternatives": alts,
                "step": rep_steps.get(canon_act),
            })
            if alts:
                branches.append({"position": pos, "canonical_action": canon_act,
                                 "alternatives": alts})

        canonical_steps = [canonical_steps[i] for i in _evenly(len(canonical_steps), MAX_STEPS)]
        kept = {c["position"] for c in canonical_steps}
        branches = [b for b in branches if b["position"] in kept]
        params = self.extract_parameters(demos, alignments)
        param_dicts = [{"name": p.name, "type": p.type, "values_seen": p.values_seen,
                        "step_positions": p.step_positions, "confidence": p.confidence}
                       for p in params]
        return {"canonical_steps": canonical_steps, "branches": branches, "parameters": param_dicts}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _composite_similarity(self, step_a: dict, step_b: dict,
                              emb_a: list[float] | None, emb_b: list[float] | None) -> float:
        """Weighted composite similarity: 0.5 semantic + 0.3 app + 0.2 domain."""
        if emb_a is not None and emb_b is not None:
            sem = 0.5 * _cosine_similarity(emb_a, emb_b)
        else:
            sem = 0.5 * self._normalized_string_sim(step_a.get("action", ""), step_b.get("action", ""))

        app_a = (step_a.get("app", "") or "").lower()
        app_b = (step_b.get("app", "") or "").lower()
        app_score = 0.3 if (app_a and app_b and app_a == app_b) else 0.0

        loc_a, loc_b = step_a.get("location", "") or "", step_b.get("location", "") or ""
        return sem + app_score + 0.2 * self._domain_overlap(loc_a, loc_b)

    @staticmethod
    def _normalized_string_sim(a: str, b: str) -> float:
        """Jaccard similarity over lowercased word tokens."""
        ta, tb = set(a.lower().split()), set(b.lower().split())
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    @staticmethod
    def _domain_overlap(loc_a: str, loc_b: str) -> float:
        """1.0 same domain, 0.0 different, 0.5 both empty (neutral)."""
        da, db = _url_to_domain(loc_a), _url_to_domain(loc_b)
        if not da and not db:
            return 0.5
        return 1.0 if da == db else 0.0

    @staticmethod
    def _differing_params(step_a: dict, step_b: dict) -> list[str]:
        """Return field names that differ between two aligned steps."""
        diffs: list[str] = []
        for f in _PARAM_FIELDS:
            va = str(step_a.get(f, "") or "").strip()
            vb = str(step_b.get(f, "") or "").strip()
            if va and vb and va != vb:
                diffs.append(f)
        return diffs

    @staticmethod
    def _infer_type(values: list[str]) -> str:
        """Infer parameter type from observed values using regex heuristics."""
        for v in values:
            v = v.strip()
            if not v:
                continue
            if _RE_URL.match(v):
                return "url"
            if _RE_EMAIL.match(v):
                return "email"
            if _RE_NUMBER.match(v):
                return "number"
            if _RE_DATE.match(v):
                return "date"
            if _RE_FILEPATH_UNIX.match(v) or _RE_FILEPATH_WIN.match(v):
                return "filepath"
        return "text"
