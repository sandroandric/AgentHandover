"""CSS Rot Filter — strip randomized CSS classes, prefer stable selectors.

Implements section 9.3 of the AgentHandover spec: never store unstable selectors.
Auto-generated CSS class names from frameworks like Emotion, styled-components,
CSS Modules, and Vue scoped styles are detected and stripped.  Stable selectors
(ARIA labels, visible text, roles, test IDs) are promoted instead.

Priority order for stable selectors (lower rank = more stable):
  0. aria_label     — ARIA-label / accessible name
  1. visible_text   — Visible innerText (normalized)
  2. role_heading   — Role + relative position to stable headings
  3. test_id        — data-testid (if stable — not randomized)
  4. semantic_class  — Meaningful CSS class (e.g. btn, header, nav)
  5. vision_bbox    — Vision bounding box fallback (least stable)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns matching auto-generated CSS class names
# ---------------------------------------------------------------------------

ROT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^css-[a-z0-9]+$"),                              # Emotion
    re.compile(r"^sc-[a-zA-Z0-9]+$"),                            # styled-components
    re.compile(r"^[a-zA-Z]+_[a-zA-Z]+__[a-z0-9]+$"),            # CSS Modules
    re.compile(
        r"^[a-z]{1,4}-(?=[a-z0-9]*[0-9])[a-z0-9]{4,12}$"
    ),                                                           # Generic hash
    re.compile(r"^data-v-[a-f0-9]+$"),                           # Vue scoped
]

# ---------------------------------------------------------------------------
# Stable selector types in priority order (lower index = more stable)
# ---------------------------------------------------------------------------

STABLE_SELECTOR_PRIORITY: list[str] = [
    "aria_label",
    "visible_text",
    "role_heading",
    "test_id",
    "semantic_class",
    "vision_bbox",
]


class CSSRotFilter:
    """Filters CSS rot from DOM selectors and promotes stable alternatives."""

    def is_css_rot(self, class_name: str) -> bool:
        """Check if a CSS class name is auto-generated rot.

        Returns ``True`` when *class_name* matches any of the known
        auto-generated patterns (Emotion, styled-components, CSS Modules,
        generic hashes, Vue scoped attributes).
        """
        if not class_name:
            return False

        for pattern in ROT_PATTERNS:
            if pattern.match(class_name):
                return True
        return False

    def strip_rot_classes(self, class_list: list[str]) -> list[str]:
        """Remove auto-generated classes from *class_list*, keep semantic ones.

        Returns a new list containing only classes that are NOT detected as
        CSS rot.  Preserves the original ordering of the kept classes.
        """
        return [cls for cls in class_list if not self.is_css_rot(cls)]

    def clean_selector(self, selector: str) -> str:
        """Remove CSS rot class references from a full CSS selector string.

        Strips ``.css-abc123``, ``.sc-bdfBjE``, etc. from selectors like
        ``div.container.css-1a2b3c > button.sc-xyz123`` while preserving
        structural and semantic parts.

        If cleaning a segment removes all classes, the bare tag name is kept.
        The method preserves combinator tokens (``>``, ``+``, ``~``, `` ``).
        """
        if not selector:
            return ""

        # Split on combinators while keeping them as tokens
        # Matches: ">", "+", "~", or whitespace (descendant combinator)
        tokens = re.split(r"(\s*[>+~]\s*|\s+)", selector)

        cleaned_tokens: list[str] = []
        for token in tokens:
            stripped = token.strip()
            # Combinators and empty strings pass through
            if not stripped or stripped in (">", "+", "~"):
                cleaned_tokens.append(token)
                continue

            # Clean individual selector segment (e.g. "div.container.css-abc123")
            cleaned_tokens.append(self._clean_segment(token))

        result = "".join(cleaned_tokens)
        # Collapse multiple spaces
        result = re.sub(r" {2,}", " ", result).strip()
        return result

    def _clean_segment(self, segment: str) -> str:
        """Clean a single selector segment by removing rot classes.

        Handles segments like ``div.container.css-1a2b3c#myid``.
        Parses out the tag, id, classes, and attribute selectors, then
        reassembles without the rot classes.
        """
        # Extract attribute selectors first to avoid confusion
        attr_selectors: list[str] = []
        attr_pattern = re.compile(r"\[.*?\]")
        attrs = attr_pattern.findall(segment)
        attr_selectors.extend(attrs)
        segment_no_attrs = attr_pattern.sub("", segment)

        # Split into parts: tag, #id, .class
        # Extract tag name (everything before the first . or #)
        parts_match = re.match(r"^([a-zA-Z][a-zA-Z0-9-]*)?", segment_no_attrs)
        tag = parts_match.group(1) if parts_match and parts_match.group(1) else ""

        # Extract ID
        id_match = re.search(r"#([a-zA-Z0-9_-]+)", segment_no_attrs)
        element_id = id_match.group(0) if id_match else ""

        # Extract classes
        classes = re.findall(r"\.([a-zA-Z0-9_-]+)", segment_no_attrs)
        clean_classes = self.strip_rot_classes(classes)

        # Reassemble
        result = tag
        if element_id:
            result += element_id
        for cls in clean_classes:
            result += f".{cls}"
        for attr in attr_selectors:
            result += attr

        return result if result else tag or segment

    def rank_selector_stability(self, selector_type: str) -> int:
        """Return stability rank for a selector type (lower = more stable).

        Types not in the priority list receive ``len(STABLE_SELECTOR_PRIORITY)``
        (worse than any known type).
        """
        try:
            return STABLE_SELECTOR_PRIORITY.index(selector_type)
        except ValueError:
            return len(STABLE_SELECTOR_PRIORITY)

    def best_stable_selector(self, candidates: list[dict]) -> dict | None:
        """From a list of selector candidates, pick the most stable one.

        Each candidate dict must contain at least::

            {"type": "aria_label", "value": "Submit", "selector": "[aria-label='Submit']"}

        Returns the candidate with the lowest stability rank, or ``None``
        if *candidates* is empty.  When two candidates share the same rank,
        the first one in the list wins (preserving caller ordering).
        """
        if not candidates:
            return None

        best: dict | None = None
        best_rank = len(STABLE_SELECTOR_PRIORITY) + 1

        for candidate in candidates:
            ctype = candidate.get("type", "")
            rank = self.rank_selector_stability(ctype)
            if rank < best_rank:
                best_rank = rank
                best = candidate

        return best
