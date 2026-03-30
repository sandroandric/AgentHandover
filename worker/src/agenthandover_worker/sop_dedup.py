"""SOP Deduplication — structural fingerprint matching and merging.

Prevents duplicate SOPs from accumulating when users record the same
workflow multiple times.  Uses a structural fingerprint (apps, URL
domains, action verbs) with Jaccard similarity to detect when a newly
generated SOP matches an existing one, and merges them instead of
creating a duplicate.

When an ``LLMReasoner`` is provided, step-level merge conflicts (where
two versions of a step differ textually) are resolved by the LLM
instead of the simple "keep more steps" heuristic.

Design:
- **Deterministic**: no embedding model needed — pure set operations
- **Conservative**: threshold 0.7 avoids false merges on similar-but-different tasks
- **Transparent**: fingerprint components are human-readable

Integration points:
- Called after SOP generation (``sop_generator.py``) but before writing
- Uses the cumulative SOP registry (``sop-registry.json``) as the source
  of known SOPs
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from agenthandover_worker.llm_reasoning import LLMReasoner

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Fingerprint computation
# ------------------------------------------------------------------


def compute_fingerprint(sop: dict) -> dict:
    """Build a structural fingerprint from an SOP template.

    Components:
    - ``apps``: sorted list of normalized app names
    - ``domains``: sorted list of URL domains extracted from steps/preconditions
    - ``action_verbs``: sorted list of action verb types (first word of each step)

    All components are lowercased and deduplicated.
    """
    apps = _extract_apps(sop)
    domains = _extract_domains(sop)
    action_verbs = _extract_action_verbs(sop)

    return {
        "apps": sorted(apps),
        "domains": sorted(domains),
        "action_verbs": sorted(action_verbs),
        "variant_family": sop.get("variant_family"),
    }


def _extract_apps(sop: dict) -> set[str]:
    """Extract normalized app names from the SOP."""
    apps: set[str] = set()

    for app in sop.get("apps_involved", []):
        apps.add(_normalize_app(app))

    for step in sop.get("steps", []):
        params = step.get("parameters", {})
        if isinstance(params, dict) and params.get("app"):
            apps.add(_normalize_app(params["app"]))

    return apps


def _normalize_app(app: str) -> str:
    """Normalize app name for comparison.

    Strips PID suffixes, bundle IDs, and arrow notation:
    - "Visual Studio Code in pid:1234:Visual Studio Code" -> "visual studio code"
    - "com.chrome.Chrome" -> "chrome"
    - "VS Code → Terminal" -> "vs code"
    """
    app = app.lower().strip()

    # Strip " in pid:NNNN:..." suffix
    pid_match = re.search(r"\s+in\s+pid:\d+:", app)
    if pid_match:
        app = app[:pid_match.start()]

    # Strip bundle ID prefix (com.xxx.AppName -> appname)
    if app.startswith("com.") or app.startswith("org."):
        parts = app.split(".")
        app = parts[-1] if parts else app

    # Strip arrow notation ("vs code → terminal" -> "vs code")
    if "→" in app or "->" in app:
        app = re.split(r"\s*[→>]\s*", app)[0].strip()
    elif " → " in app:
        app = app.split(" → ")[0].strip()

    return app.strip()


def _extract_domains(sop: dict) -> set[str]:
    """Extract URL domains from steps and preconditions."""
    domains: set[str] = set()

    # From step locations
    for step in sop.get("steps", []):
        params = step.get("parameters", {})
        if isinstance(params, dict):
            location = params.get("location", "")
            domain = _url_to_domain(location)
            if domain:
                domains.add(domain)

        target = step.get("target", "")
        domain = _url_to_domain(target)
        if domain:
            domains.add(domain)

    # From preconditions
    for pre in sop.get("preconditions", []):
        if isinstance(pre, str) and pre.startswith("url_open:"):
            url = pre[len("url_open:"):]
            domain = _url_to_domain(url)
            if domain:
                domains.add(domain)

    return domains


def _url_to_domain(text: str) -> str:
    """Extract domain from a URL string, or return empty."""
    text = text.strip()
    if not text:
        return ""
    if not text.startswith(("http://", "https://")):
        # Check if it looks like a bare domain
        if "." in text and "/" not in text.split(".")[0]:
            text = "https://" + text
        else:
            return ""
    try:
        parsed = urlparse(text)
        host = parsed.hostname or ""
        # Strip www. prefix
        if host.startswith("www."):
            host = host[4:]
        return host.lower()
    except Exception:
        return ""


def _extract_action_verbs(sop: dict) -> set[str]:
    """Extract normalized action verbs from SOP steps.

    Takes the first word of each step description, lowercased.
    Maps common synonyms to canonical forms.
    """
    _VERB_MAP = {
        "navigate": "open",
        "go": "open",
        "visit": "open",
        "browse": "open",
        "launch": "open",
        "type": "enter",
        "input": "enter",
        "fill": "enter",
        "write": "enter",
        "press": "click",
        "tap": "click",
        "hit": "click",
        "submit": "click",
        "check": "verify",
        "confirm": "verify",
        "validate": "verify",
        "inspect": "review",
        "examine": "review",
        "look": "review",
        "read": "review",
        "choose": "select",
        "pick": "select",
        "filter": "select",
        "wait": "wait",
        "pause": "wait",
    }

    verbs: set[str] = set()
    for step in sop.get("steps", []):
        action = step.get("step", step.get("action", ""))
        if not action:
            continue
        # First word, lowercased
        first_word = action.strip().split()[0].lower() if action.strip() else ""
        if first_word:
            canonical = _VERB_MAP.get(first_word, first_word)
            verbs.add(canonical)

    return verbs


# ------------------------------------------------------------------
# Similarity computation
# ------------------------------------------------------------------


def fingerprint_similarity(fp1: dict, fp2: dict) -> float:
    """Compute weighted Jaccard similarity between two fingerprints.

    Weights:
    - apps: 0.35 (strongest signal — same apps = likely same task)
    - domains: 0.35 (same websites = likely same task)
    - action_verbs: 0.30 (same action types = structural match)

    Returns a float in [0.0, 1.0].
    """
    apps_sim = _jaccard(set(fp1.get("apps", [])), set(fp2.get("apps", [])))
    domains_sim = _jaccard(set(fp1.get("domains", [])), set(fp2.get("domains", [])))
    verbs_sim = _jaccard(set(fp1.get("action_verbs", [])), set(fp2.get("action_verbs", [])))

    # If both have no domains (e.g. desktop-only workflows), redistribute
    # domain weight to apps and verbs equally
    if not fp1.get("domains") and not fp2.get("domains"):
        return apps_sim * 0.55 + verbs_sim * 0.45

    return apps_sim * 0.35 + domains_sim * 0.35 + verbs_sim * 0.30


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
    if not a and not b:
        return 1.0  # Both empty = identical (no signal)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ------------------------------------------------------------------
# Matching
# ------------------------------------------------------------------

_DEFAULT_THRESHOLD = 0.70


def find_matching_sop(
    new_sop: dict,
    existing_sops: list[dict],
    threshold: float = _DEFAULT_THRESHOLD,
    vector_kb=None,
) -> int | None:
    """Find an existing SOP that matches the new one.

    Tries semantic similarity via vector_kb first (catches "deploy staging"
    = "push to stage"), then falls back to structural fingerprints.

    Args:
        new_sop: The newly generated SOP template.
        existing_sops: List of previously generated SOP templates.
        threshold: Minimum similarity score to consider a match (0.0–1.0).
        vector_kb: Optional VectorKB for semantic matching.

    Returns:
        Index into ``existing_sops`` of the best match, or ``None`` if no
        match exceeds the threshold.
    """
    if not existing_sops:
        return None

    # Try semantic match first via vector KB
    if vector_kb is not None:
        try:
            title = new_sop.get("title", "")
            desc = new_sop.get("description", "")
            query = f"{title} | {desc}" if desc else title
            if query:
                results = vector_kb.search(
                    query,
                    top_k=3,
                    source_types=["procedure"],
                    min_score=threshold,
                )
                if results:
                    # Map vector result (slug) back to index in existing_sops
                    slug_to_idx = {
                        s.get("slug", ""): i for i, s in enumerate(existing_sops)
                    }
                    for r in results:
                        if r.source_id in slug_to_idx:
                            logger.info(
                                "SOP dedup (semantic): match found "
                                "(score=%.2f, '%s' matches '%s')",
                                r.score,
                                new_sop.get("slug", "?"),
                                r.source_id,
                            )
                            return slug_to_idx[r.source_id]
        except Exception:
            pass  # fall through to fingerprint matching

    new_fp = compute_fingerprint(new_sop)

    best_idx: int | None = None
    best_score = 0.0

    for i, existing in enumerate(existing_sops):
        existing_fp = existing.get("_fingerprint")
        if existing_fp is None:
            existing_fp = compute_fingerprint(existing)

        score = fingerprint_similarity(new_fp, existing_fp)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_score >= threshold and best_idx is not None:
        logger.info(
            "SOP dedup: match found (score=%.2f, slug='%s' matches '%s')",
            best_score,
            new_sop.get("slug", "?"),
            existing_sops[best_idx].get("slug", "?"),
        )
        return best_idx

    return None


# ------------------------------------------------------------------
# Merging
# ------------------------------------------------------------------


def _resolve_conflict_with_llm(
    reasoner: "LLMReasoner",
    step_a: dict,
    step_b: dict,
) -> dict | None:
    """Ask the LLM to resolve a conflict between two step versions.

    Returns:
    - ``step_a`` if the LLM says keep A
    - ``step_b`` if the LLM says keep B
    - ``step_a`` with ``step_b`` added to alternatives if keep both
    - ``None`` on failure / abstention
    """
    action_a = step_a.get("step", step_a.get("action", ""))
    action_b = step_b.get("step", step_b.get("action", ""))

    prompt = (
        f"Two versions of a procedure step differ. "
        f"Version A: '{action_a}'. Version B: '{action_b}'. "
        f"Which is better, or should both be kept as alternatives? "
        f'Respond with JSON: {{"keep": "A" or "B" or "both", '
        f'"reason": "..."}}. '
        f"If insufficient context, respond with INSUFFICIENT_EVIDENCE."
    )

    try:
        result = reasoner.reason_json(
            prompt=prompt,
            caller="sop_dedup._resolve_conflict_with_llm",
        )
        if not result.success or result.abstained or not result.value:
            return None

        keep = result.value.get("keep", "").upper()
        if keep == "A":
            return dict(step_a)
        elif keep == "B":
            return dict(step_b)
        elif keep == "BOTH":
            merged_step = dict(step_a)
            alternatives = list(merged_step.get("alternatives", []))
            alternatives.append(step_b)
            merged_step["alternatives"] = alternatives
            return merged_step
    except Exception:
        logger.debug("LLM step conflict resolution failed", exc_info=True)

    return None


def merge_sops(
    existing: dict,
    new_sop: dict,
    evidence_normalizer=None,
    llm_reasoner: "LLMReasoner | None" = None,
) -> dict:
    """Merge a new SOP into an existing one.

    Strategy:
    - ``slug``: keep existing (stable identity)
    - ``title``: keep latest (VLM may improve wording)
    - ``task_description``: keep latest
    - ``steps``: keep the version with more steps; if tied, keep latest
    - ``variables``: union by name (add new variables, update existing)
    - ``episode_count``: accumulate
    - ``confidence_avg``: will be recomputed by caller
    - ``apps_involved``: union
    - ``preconditions``: union (deduped)
    - ``execution_overview``: keep latest
    - ``source``: keep existing
    - ``_timeline``: keep latest (most recent recording)
    """
    if evidence_normalizer is not None:
        try:
            # evidence_normalizer.merge_with_evidence expects v3 procedure
            # dicts, but merge_sops operates on SOP templates.  Only use
            # evidence-based merge when the existing dict looks like a v3
            # procedure (has "schema_version" and "id").
            if existing.get("schema_version") and existing.get("id"):
                return evidence_normalizer.merge_with_evidence(
                    existing, new_sop.get("steps", []),
                )
        except Exception:
            logger.debug("Evidence-based merge failed, using default", exc_info=True)
    # ... existing logic continues as fallback ...

    merged = dict(existing)  # Shallow copy

    # Accumulate episode count
    merged["episode_count"] = (
        existing.get("episode_count", 0) + new_sop.get("episode_count", 0)
    )

    # Keep latest title and description (VLM may improve)
    merged["title"] = new_sop.get("title", existing.get("title", ""))
    if new_sop.get("task_description"):
        merged["task_description"] = new_sop["task_description"]

    # Steps: keep the version with more steps; if tied, keep latest.
    # When step counts are close (within 1) and an LLM reasoner is
    # available, attempt per-step conflict resolution instead.
    existing_steps = existing.get("steps", [])
    new_steps = new_sop.get("steps", [])
    step_diff = abs(len(existing_steps) - len(new_steps))

    if step_diff <= 1 and llm_reasoner is not None and existing_steps and new_steps:
        # Attempt LLM-based step-by-step conflict resolution
        min_len = min(len(existing_steps), len(new_steps))
        resolved_steps: list[dict] = []
        used_llm = False

        for i in range(min_len):
            s_a = existing_steps[i]
            s_b = new_steps[i]
            action_a = (s_a.get("step", s_a.get("action", ""))).strip().lower()
            action_b = (s_b.get("step", s_b.get("action", ""))).strip().lower()

            if action_a != action_b:
                # Actual conflict — ask LLM
                resolved = _resolve_conflict_with_llm(llm_reasoner, s_a, s_b)
                if resolved is not None:
                    resolved_steps.append(resolved)
                    used_llm = True
                else:
                    # LLM failed — fall back to keeping new step
                    resolved_steps.append(s_b)
            else:
                # No conflict — keep new (latest)
                resolved_steps.append(s_b)

        # Append any extra steps from the longer list
        if len(existing_steps) > min_len:
            resolved_steps.extend(existing_steps[min_len:])
        elif len(new_steps) > min_len:
            resolved_steps.extend(new_steps[min_len:])

        if used_llm:
            merged["steps"] = resolved_steps
        elif len(new_steps) >= len(existing_steps):
            merged["steps"] = new_steps
        # else: keep existing steps
    elif len(new_steps) >= len(existing_steps):
        merged["steps"] = new_steps
    # else: keep existing steps (more comprehensive)

    # Variables: union by name
    merged["variables"] = _merge_variables(
        existing.get("variables", []),
        new_sop.get("variables", []),
    )

    # Apps: union
    existing_apps = set(existing.get("apps_involved", []))
    new_apps = set(new_sop.get("apps_involved", []))
    merged["apps_involved"] = sorted(existing_apps | new_apps)

    # Preconditions: union, deduped
    existing_pre = existing.get("preconditions", [])
    new_pre = new_sop.get("preconditions", [])
    seen_pre: set[str] = set()
    merged_pre: list[str] = []
    for p in existing_pre + new_pre:
        if p not in seen_pre:
            seen_pre.add(p)
            merged_pre.append(p)
    merged["preconditions"] = merged_pre

    # Execution overview: keep latest if present
    new_eo = new_sop.get("execution_overview")
    if new_eo and isinstance(new_eo, dict) and new_eo:
        merged["execution_overview"] = new_eo

    # Timeline: keep latest
    if new_sop.get("_timeline"):
        merged["_timeline"] = new_sop["_timeline"]

    # Store updated fingerprint
    merged["_fingerprint"] = compute_fingerprint(merged)

    # Mark that this was a merge
    merged["_merge_count"] = existing.get("_merge_count", 0) + 1

    logger.info(
        "SOP dedup: merged '%s' into '%s' (episodes: %d, merge #%d)",
        new_sop.get("slug", "?"),
        merged["slug"],
        merged["episode_count"],
        merged["_merge_count"],
    )

    return merged


_FAMILY_THRESHOLD = 0.60


def detect_procedure_family(
    sops: list[dict],
    threshold: float = _FAMILY_THRESHOLD,
) -> list[dict]:
    """Group SOPs with similarity > threshold (but < merge threshold) into variant families.

    Returns list of family dicts: {"family_id", "canonical_slug", "variant_slugs", "shared_apps"}
    """
    if len(sops) < 2:
        return []

    # Compute all fingerprints
    fps = [(sop.get("slug", f"sop-{i}"), compute_fingerprint(sop)) for i, sop in enumerate(sops)]

    # Build adjacency list for family grouping
    edges: dict[str, set[str]] = {slug: set() for slug, _ in fps}
    for i, (slug_a, fp_a) in enumerate(fps):
        for j, (slug_b, fp_b) in enumerate(fps):
            if i >= j:
                continue
            sim = fingerprint_similarity(fp_a, fp_b)
            if sim >= threshold and sim < 0.70:  # family range: [0.60, 0.70)
                edges[slug_a].add(slug_b)
                edges[slug_b].add(slug_a)

    # BFS to find connected components
    visited: set[str] = set()
    families: list[dict] = []

    for slug, _ in fps:
        if slug in visited or not edges[slug]:
            continue
        component: list[str] = []
        queue = [slug]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in edges[current]:
                if neighbor not in visited:
                    queue.append(neighbor)

        if len(component) >= 2:
            # Find canonical: highest episode_count
            sop_map = {s.get("slug", ""): s for s in sops}
            canonical = max(component, key=lambda s: sop_map.get(s, {}).get("episode_count", 0))
            variants = [s for s in component if s != canonical]
            shared = set(sop_map.get(canonical, {}).get("apps_involved", []))
            for v in variants:
                shared &= set(sop_map.get(v, {}).get("apps_involved", []))
            families.append({
                "family_id": f"family-{canonical}",
                "canonical_slug": canonical,
                "variant_slugs": variants,
                "shared_apps": sorted(shared),
            })

    return families


def _merge_variables(
    existing_vars: list[dict],
    new_vars: list[dict],
) -> list[dict]:
    """Merge variable lists by name, keeping the richer definition.

    If both have a variable with the same name, keep the one with
    more non-empty fields (better example/description).
    """
    by_name: dict[str, dict] = {}

    for var in existing_vars:
        name = var.get("name", "")
        if name:
            by_name[name] = var

    for var in new_vars:
        name = var.get("name", "")
        if not name:
            continue
        if name in by_name:
            # Keep the richer definition
            existing_richness = sum(1 for v in by_name[name].values() if v)
            new_richness = sum(1 for v in var.values() if v)
            if new_richness > existing_richness:
                by_name[name] = var
        else:
            by_name[name] = var

    return list(by_name.values())


# ------------------------------------------------------------------
# Registry (cumulative SOP cache)
# ------------------------------------------------------------------

_REGISTRY_FILE = "sop-registry.json"


def load_registry(state_dir: Path) -> list[dict]:
    """Load the cumulative SOP registry from disk.

    The registry stores all known SOP templates (without ``_timeline``
    to save space) keyed by slug.  Used for dedup matching.
    """
    registry_path = state_dir / _REGISTRY_FILE
    if not registry_path.is_file():
        return []
    try:
        with open(registry_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        logger.debug("Could not read SOP registry", exc_info=True)
    return []


def save_registry(state_dir: Path, sops: list[dict]) -> None:
    """Save the cumulative SOP registry to disk.

    Strips ``_timeline`` from each SOP to save space (timelines can be
    large with DOM nodes).
    """
    from agenthandover_worker.exporter import AtomicWriter

    # Strip heavyweight fields before persisting
    clean = []
    for sop in sops:
        entry = {k: v for k, v in sop.items() if k != "_timeline"}
        # Ensure fingerprint is stored
        if "_fingerprint" not in entry:
            entry["_fingerprint"] = compute_fingerprint(entry)
        clean.append(entry)

    registry_path = state_dir / _REGISTRY_FILE
    state_dir.mkdir(parents=True, exist_ok=True)
    AtomicWriter.write(registry_path, json.dumps(clean, indent=2, default=str))


def deduplicate_templates(
    new_sops: list[dict],
    state_dir: Path,
    threshold: float = _DEFAULT_THRESHOLD,
    evidence_normalizer=None,
    llm_reasoner: "LLMReasoner | None" = None,
) -> list[dict]:
    """Deduplicate a batch of new SOPs against the cumulative registry.

    For each new SOP:
    1. Check if it matches an existing SOP in the registry
    2. If yes: merge into existing (accumulate episodes, update steps)
    3. If no: add as new entry

    Updates the registry on disk after processing.

    Args:
        new_sops: Newly generated SOP templates.
        state_dir: Directory for the SOP registry file.
        threshold: Minimum similarity score for matching.
        evidence_normalizer: Optional ``EvidenceNormalizer`` instance.
            When provided, ``merge_sops`` uses semantic alignment
            instead of the simple "keep more steps" strategy.

    Returns the final list of SOP templates to export (merged + new).
    """
    registry = load_registry(state_dir)
    output: list[dict] = []

    for new_sop in new_sops:
        match_idx = find_matching_sop(new_sop, registry, threshold)

        if match_idx is not None:
            # Merge into existing (with evidence-based merge if available)
            merged = merge_sops(
                registry[match_idx], new_sop,
                evidence_normalizer=evidence_normalizer,
                llm_reasoner=llm_reasoner,
            )
            registry[match_idx] = merged
            output.append(merged)
        else:
            # New SOP — add to registry
            new_sop["_fingerprint"] = compute_fingerprint(new_sop)
            registry.append(new_sop)
            output.append(new_sop)

    # Persist updated registry
    save_registry(state_dir, registry)

    return output
