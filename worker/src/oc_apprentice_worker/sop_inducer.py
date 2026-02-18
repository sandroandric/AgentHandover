"""SOP Inducer — mine repeated subgraphs from episodes to produce SOP templates.

Implements section 10.1 of the OpenMimic spec.  Uses PrefixSpan to discover
frequent sequential patterns across episodes, then abstracts variable slots
where values differ across instances.

The inducer takes lists of semantic step dicts (from ``SemanticStep.to_sop_step()``)
grouped by episode and produces SOP template dicts ready for formatting.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime

from prefixspan import PrefixSpan

logger = logging.getLogger(__name__)


class SOPInducer:
    """Mine frequent patterns from episodes and produce SOP templates.

    Parameters
    ----------
    min_support:
        Minimum fraction of episodes a pattern must appear in (0.0-1.0).
    min_pattern_length:
        Minimum number of steps in a pattern to be considered.
    vlm_worker:
        Optional VLMWorker instance for VLM-assisted variable classification.
        When provided and available, VLM is used to classify variables with
        higher accuracy than heuristics alone.
    vlm_confidence_threshold:
        Minimum VLM confidence to trust its classification (0.0-1.0).
        Below this threshold, falls back to heuristic classification.
    """

    def __init__(
        self,
        min_support: float = 0.3,
        min_pattern_length: int = 3,
        vlm_worker: object | None = None,
        vlm_confidence_threshold: float = 0.7,
    ):
        self.min_support = min_support
        self.min_pattern_length = min_pattern_length
        self._vlm_worker = vlm_worker
        self._vlm_confidence_threshold = vlm_confidence_threshold

    def induce(self, episodes: list[list[dict]]) -> list[dict]:
        """Mine frequent patterns from episodes and produce SOP templates.

        Args:
            episodes: list of episodes, each episode is a list of semantic steps
                (dicts from SemanticStep.to_sop_step())

        Returns:
            list of SOP template dicts, each containing:
            {
                "slug": "task_name",
                "title": "Human-Readable Task Title",
                "steps": [{"step": "click", "target": "Submit button", ...}],
                "variables": [{"name": "customer_name", "type": "string", "example": "..."}],
                "confidence_avg": 0.87,
                "episode_count": 5,
                "apps_involved": ["Chrome", "Excel"],
            }
        """
        if not episodes:
            return []

        # Filter out empty episodes
        non_empty = [ep for ep in episodes if ep]
        if not non_empty:
            return []

        # Build encoding tables
        encoded, code_to_signature, signature_to_steps = self._encode_steps(non_empty)

        # Mine frequent patterns
        raw_patterns = self._mine_patterns(encoded, len(non_empty))
        if not raw_patterns:
            return []

        results: list[dict] = []
        for support_count, pattern_codes in raw_patterns:
            # Decode pattern codes back to step signatures
            pattern_steps = []
            for code in pattern_codes:
                sig = code_to_signature.get(code)
                if sig is None:
                    continue
                # Take a representative step for this signature
                candidates = signature_to_steps.get(sig, [])
                if candidates:
                    pattern_steps.append(candidates[0].copy())

            if len(pattern_steps) < self.min_pattern_length:
                continue

            # Single-pass scan: collect instances, apps, and exceptions together
            all_instances, apps, exceptions = self._scan_episodes_for_pattern(
                non_empty, pattern_codes, code_to_signature, signature_to_steps
            )

            # Abstract variables
            variables = self._abstract_variables(pattern_steps, all_instances)

            # Compute average confidence
            confidence_avg = self._compute_avg_confidence(all_instances)

            # Generate slug and title
            slug = self._generate_slug(pattern_steps)
            title = self._generate_title(pattern_steps, apps)

            # Detect preconditions and postconditions from collected instances
            preconditions = self._detect_preconditions(all_instances)
            postconditions = self._detect_postconditions(all_instances)

            results.append({
                "slug": slug,
                "title": title,
                "steps": pattern_steps,
                "variables": variables,
                "confidence_avg": round(confidence_avg, 4),
                "episode_count": support_count,
                "apps_involved": sorted(set(apps)),
                "preconditions": preconditions,
                "postconditions": postconditions,
                "exceptions_seen": exceptions,
            })

        return results

    def _encode_steps(
        self, episodes: list[list[dict]]
    ) -> tuple[list[list[int]], dict[int, str], dict[str, list[dict]]]:
        """Encode semantic steps as integer sequences for PrefixSpan.

        Each unique step signature (intent + target) maps to a unique integer.

        Returns:
            (encoded_sequences, code_to_signature, signature_to_steps)
        """
        signature_to_code: dict[str, int] = {}
        code_to_signature: dict[int, str] = {}
        signature_to_steps: dict[str, list[dict]] = defaultdict(list)
        next_code = 0

        encoded: list[list[int]] = []

        for episode in episodes:
            seq: list[int] = []
            for step in episode:
                sig = self._step_signature(step)
                if sig not in signature_to_code:
                    signature_to_code[sig] = next_code
                    code_to_signature[next_code] = sig
                    next_code += 1

                code = signature_to_code[sig]
                seq.append(code)
                signature_to_steps[sig].append(step)

            encoded.append(seq)

        return encoded, code_to_signature, signature_to_steps

    def _step_signature(self, step: dict) -> str:
        """Create a canonical signature for a step based on intent and target.

        The signature normalizes the target to lowercase for matching,
        so that "Submit Button" and "submit button" are treated as the same
        step type.
        """
        intent = step.get("step", "unknown")
        target = step.get("target", "")
        # Normalize target: lowercase, strip extra whitespace
        normalized_target = " ".join(target.lower().split())
        return f"{intent}::{normalized_target}"

    def _mine_patterns(self, encoded: list[list[int]], episode_count: int) -> list:
        """Run PrefixSpan with minimum support.

        Returns list of (support_count, pattern) tuples where pattern is a list
        of integer codes.
        """
        abs_support = max(2, int(self.min_support * episode_count))

        # Safety cap: stratified subsample if input is too large
        avg_steps = sum(len(ep) for ep in encoded) / max(len(encoded), 1)
        data_size = episode_count * avg_steps
        mining_input = encoded
        if data_size > 50000:
            logger.warning(
                "PrefixSpan input too large (%.0f), stratified subsampling to ~50000",
                data_size,
            )
            target_episodes = int(50000 / max(avg_steps, 1))
            mining_input = self._stratified_sample(encoded, target_episodes)

        ps = PrefixSpan(mining_input)
        # Mine frequent patterns with minimum length constraint
        raw = ps.frequent(abs_support)

        # Filter by minimum pattern length and collect up to 1000 patterns
        patterns = []
        for count, pat in raw:
            if len(pat) >= self.min_pattern_length:
                patterns.append((count, pat))
                if len(patterns) >= 1000:
                    break

        # Sort by support count descending, then pattern length descending
        patterns.sort(key=lambda x: (x[0], len(x[1])), reverse=True)

        return patterns

    @staticmethod
    def _stratified_sample(
        encoded: list[list[int]], target_count: int
    ) -> list[list[int]]:
        """Stratified sampling by episode length to preserve pattern diversity.

        Buckets episodes into short/medium/long by length, then samples
        proportionally from each bucket.  This avoids bias against rare
        patterns that only appear in unusually short or long episodes.
        """
        if target_count >= len(encoded):
            return encoded

        # Bucket by length: short (<= p25), medium (p25-p75), long (> p75)
        lengths = sorted(len(ep) for ep in encoded)
        p25 = lengths[len(lengths) // 4]
        p75 = lengths[3 * len(lengths) // 4]

        buckets: dict[str, list[list[int]]] = {"short": [], "medium": [], "long": []}
        for ep in encoded:
            ep_len = len(ep)
            if ep_len <= p25:
                buckets["short"].append(ep)
            elif ep_len > p75:
                buckets["long"].append(ep)
            else:
                buckets["medium"].append(ep)

        # Sample proportionally from each bucket
        result: list[list[int]] = []
        total = len(encoded)
        for bucket_name, bucket_eps in buckets.items():
            if not bucket_eps:
                continue
            proportion = len(bucket_eps) / total
            bucket_target = max(1, round(target_count * proportion))
            bucket_target = min(bucket_target, len(bucket_eps))
            result.extend(random.sample(bucket_eps, bucket_target))

        # If rounding left us short or over, adjust
        if len(result) > target_count:
            result = random.sample(result, target_count)

        return result

    def _scan_episodes_for_pattern(
        self,
        episodes: list[list[dict]],
        pattern_codes: list[int],
        code_to_signature: dict[int, str],
        signature_to_steps: dict[str, list[dict]],
    ) -> tuple[list[list[dict]], list[str], list[str]]:
        """Single-pass scan: collect instances, apps, and exceptions together.

        Previously these were three separate scans (_collect_instances,
        _extract_apps, _detect_exceptions), each computing ep_sigs and
        calling _find_subsequence independently.  This combined version
        does one pass over episodes per pattern.

        Returns:
            (all_instances, apps_sorted, exceptions)
        """
        pattern_sigs = [code_to_signature[c] for c in pattern_codes]
        instances: list[list[dict]] = []
        apps: set[str] = set()
        exceptions: list[str] = []
        error_indicators = {"cancel", "error", "undo", "revert", "discard", "close"}
        seen_exceptions: set[str] = set()

        for episode in episodes:
            ep_sigs = [self._step_signature(s) for s in episode]
            matches = self._find_subsequence(ep_sigs, pattern_sigs)
            if not matches:
                continue

            # Collect instances
            for match_indices in matches:
                instance = [episode[i] for i in match_indices]
                instances.append(instance)

            # Extract apps and exceptions from the matching episode (one scan)
            for step in episode:
                # Apps
                params = step.get("parameters", {})
                if isinstance(params, dict):
                    app = params.get("app_id") or params.get("app")
                    if app:
                        apps.add(app)
                pre_state = step.get("pre_state", {})
                if isinstance(pre_state, dict):
                    app = pre_state.get("app_id") or pre_state.get("app")
                    if app:
                        apps.add(app)

                # Exceptions
                intent = step.get("step", "").lower()
                target = step.get("target", "").lower()
                for indicator in error_indicators:
                    if indicator in intent or indicator in target:
                        desc = f"{intent}:{target}" if target else intent
                        if desc not in seen_exceptions:
                            seen_exceptions.add(desc)
                            exceptions.append(desc)

        return instances, sorted(apps), exceptions

    def _find_subsequence(
        self, sequence: list[str], pattern: list[str]
    ) -> list[list[int]]:
        """Find all contiguous subsequence matches of pattern in sequence.

        Returns list of index lists, one per match.
        """
        results: list[list[int]] = []
        pat_len = len(pattern)
        seq_len = len(sequence)

        for start in range(seq_len - pat_len + 1):
            if sequence[start:start + pat_len] == pattern:
                results.append(list(range(start, start + pat_len)))

        return results

    def _abstract_variables(
        self, pattern_steps: list[dict], all_instances: list[list[dict]]
    ) -> list[dict]:
        """Detect variable slots where values differ across instances.

        Variable Abstraction Rules:
        - Named entities (customer names, IDs) -> parameterize
        - Numeric ranges -> detect min/max
        - Enum values -> build choice set
        - Timestamps -> normalize to ${today}, ${date}
        - File paths -> extract stem/extension
        """
        if len(all_instances) < 2:
            return []

        variables: list[dict] = []
        seen_var_names: set[str] = set()

        for step_idx in range(len(pattern_steps)):
            # Collect all values at this position across instances
            step_values: list[dict] = []
            for instance in all_instances:
                if step_idx < len(instance):
                    step_values.append(instance[step_idx])

            if len(step_values) < 2:
                continue

            # Check target field for variability
            targets = [s.get("target", "") for s in step_values]
            if len(set(targets)) > 1:
                var = self._classify_variable(
                    f"step_{step_idx + 1}_target",
                    targets,
                    seen_var_names,
                )
                if var:
                    variables.append(var)
                    seen_var_names.add(var["name"])

            # Check parameter values for variability
            all_param_keys: set[str] = set()
            for sv in step_values:
                params = sv.get("parameters", {})
                if isinstance(params, dict):
                    all_param_keys.update(params.keys())

            for pkey in sorted(all_param_keys):
                param_values = []
                for sv in step_values:
                    params = sv.get("parameters", {})
                    if isinstance(params, dict) and pkey in params:
                        param_values.append(params[pkey])

                if len(param_values) >= 2 and len(set(str(v) for v in param_values)) > 1:
                    var = self._classify_variable(
                        f"step_{step_idx + 1}_{pkey}",
                        param_values,
                        seen_var_names,
                    )
                    if var:
                        variables.append(var)
                        seen_var_names.add(var["name"])

        return variables

    def _classify_variable(
        self, base_name: str, values: list, seen_names: set[str]
    ) -> dict | None:
        """Classify a variable based on its observed values.

        If a VLM worker is available and has budget, attempts VLM-assisted
        classification first. Falls back to heuristics if VLM is unavailable,
        returns low confidence, or classifies as "constant" (skip variable).

        Returns a variable dict or None if the values are too uniform or
        classified as constant by VLM.
        """
        str_values = [str(v) for v in values]
        unique_values = list(set(str_values))

        if len(unique_values) <= 1:
            return None

        # Ensure unique name
        name = base_name
        counter = 1
        while name in seen_names:
            name = f"{base_name}_{counter}"
            counter += 1

        # Attempt VLM-assisted classification if available
        if self._vlm_worker is not None:
            try:
                vlm_result = self._vlm_worker.classify_variable(
                    step_context=base_name,
                    param_name=name,
                    values=str_values[:20],
                )
                if vlm_result is not None:
                    confidence = vlm_result.get("confidence", 0.0)
                    if confidence >= self._vlm_confidence_threshold:
                        classification = vlm_result.get("classification", "variable")
                        if classification == "constant":
                            logger.debug(
                                "VLM classified %s as constant (confidence=%.2f) — skipping",
                                name,
                                confidence,
                            )
                            return None

                        # VLM says it's a variable — use its type classification
                        var_type = vlm_result.get("var_type", "string")
                        result: dict = {
                            "name": name,
                            "type": var_type,
                            "example": str(values[0]),
                            "vlm_classified": True,
                        }
                        # Add type-specific fields
                        if var_type == "number":
                            numeric_values = []
                            for v in str_values:
                                try:
                                    numeric_values.append(float(v))
                                except (ValueError, TypeError):
                                    pass
                            if numeric_values:
                                result["min"] = min(numeric_values)
                                result["max"] = max(numeric_values)
                        elif var_type == "enum" and len(unique_values) <= 10:
                            result["choices"] = sorted(unique_values)

                        logger.debug(
                            "VLM classified %s as %s/%s (confidence=%.2f)",
                            name,
                            classification,
                            var_type,
                            confidence,
                        )
                        return result
                    else:
                        logger.debug(
                            "VLM confidence too low for %s (%.2f < %.2f), falling back to heuristics",
                            name,
                            confidence,
                            self._vlm_confidence_threshold,
                        )
            except Exception:
                logger.debug(
                    "VLM classification failed for %s, falling back to heuristics",
                    name,
                    exc_info=True,
                )

        # Heuristic classification fallback
        return self._classify_variable_heuristic(name, str_values, unique_values, values)

    def _classify_variable_heuristic(
        self,
        name: str,
        str_values: list[str],
        unique_values: list[str],
        raw_values: list,
    ) -> dict | None:
        """Heuristic-only variable classification (original logic)."""
        # Check if all values are numeric
        numeric_values: list[float] = []
        all_numeric = True
        for v in str_values:
            try:
                numeric_values.append(float(v))
            except (ValueError, TypeError):
                all_numeric = False
                break

        if all_numeric and numeric_values:
            return {
                "name": name,
                "type": "number",
                "example": str(raw_values[0]),
                "min": min(numeric_values),
                "max": max(numeric_values),
            }

        # Check for timestamp-like values
        if self._looks_like_timestamp(str_values):
            return {
                "name": name,
                "type": "date",
                "example": str(raw_values[0]),
            }

        # Check for file paths
        if self._looks_like_filepath(str_values):
            return {
                "name": name,
                "type": "filepath",
                "example": str(raw_values[0]),
            }

        # If few unique values, treat as enum
        if len(unique_values) <= 10:
            return {
                "name": name,
                "type": "enum",
                "example": str(raw_values[0]),
                "choices": sorted(unique_values),
            }

        # Default to string
        return {
            "name": name,
            "type": "string",
            "example": str(raw_values[0]),
        }

    def _looks_like_timestamp(self, values: list[str]) -> bool:
        """Heuristic check if values look like timestamps or dates."""
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # ISO date
            r"\d{2}/\d{2}/\d{4}",  # US date
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}",  # ISO datetime
        ]
        matches = 0
        for v in values:
            for pat in date_patterns:
                if re.search(pat, v):
                    matches += 1
                    break
        return matches >= len(values) * 0.8

    def _looks_like_filepath(self, values: list[str]) -> bool:
        """Heuristic check if values look like file paths."""
        matches = 0
        for v in values:
            if "/" in v or "\\" in v:
                # Check for common file path patterns
                if re.search(r"[/\\]\w+\.\w+$", v) or v.startswith(("/", "~", "C:\\")):
                    matches += 1
        return matches >= len(values) * 0.8


    def _compute_avg_confidence(self, instances: list[list[dict]]) -> float:
        """Compute average confidence across all steps in all instances."""
        total = 0.0
        count = 0
        for instance in instances:
            for step in instance:
                conf = step.get("confidence", 0.0)
                if isinstance(conf, (int, float)):
                    total += conf
                    count += 1
        return total / count if count > 0 else 0.0

    def _detect_preconditions(self, instances: list[list[dict]]) -> list[str]:
        """Detect what apps/URLs must be open before the SOP starts.

        Examines the first step of each instance to identify common
        preconditions (app must be open, URL must be navigated to).
        """
        preconditions: list[str] = []
        app_counts: Counter[str] = Counter()
        url_counts: Counter[str] = Counter()

        for instance in instances:
            if not instance:
                continue
            first_step = instance[0]
            pre_state = first_step.get("pre_state", {})
            if isinstance(pre_state, dict):
                app = pre_state.get("app_id") or pre_state.get("app")
                if app:
                    app_counts[app] += 1
                url = pre_state.get("url")
                if url:
                    url_counts[url] += 1

        # If an app appears in >= 80% of instances, it's a precondition
        threshold = max(1, len(instances) * 0.8)
        for app, count in app_counts.most_common():
            if count >= threshold:
                preconditions.append(f"app_open:{app}")

        for url, count in url_counts.most_common():
            if count >= threshold:
                preconditions.append(f"url_open:{url}")

        return preconditions

    def _detect_postconditions(self, instances: list[list[dict]]) -> list[str]:
        """Detect the final state after the SOP completes.

        Examines the last step of each instance to identify common
        postconditions (file saved, email sent, navigation completed).
        """
        postconditions: list[str] = []
        final_intents: Counter[str] = Counter()
        final_targets: Counter[str] = Counter()

        for instance in instances:
            if not instance:
                continue
            last_step = instance[-1]
            intent = last_step.get("step", "")
            target = last_step.get("target", "")
            if intent:
                final_intents[intent] += 1
            if target:
                final_targets[target] += 1

        # Common final actions indicate postconditions
        threshold = max(1, len(instances) * 0.5)
        for intent, count in final_intents.most_common():
            if count >= threshold:
                postconditions.append(f"final_action:{intent}")

        for target, count in final_targets.most_common(3):
            if count >= threshold:
                postconditions.append(f"final_target:{target}")

        return postconditions


    def _generate_slug(self, steps: list[dict]) -> str:
        """Generate a URL-safe slug from the pattern's first few steps.

        Takes the first 3 steps' intents and targets, combines them into
        a readable slug like "click_submit_type_email_click_send".
        Appends a short hash suffix for collision resistance.
        """
        parts: list[str] = []
        all_step_parts: list[str] = []
        for step in steps[:3]:
            intent = step.get("step", "action")
            target = step.get("target", "")
            # Extract first meaningful word from target
            target_words = re.findall(r"[a-zA-Z]+", target)
            target_word = target_words[0].lower() if target_words else ""

            if target_word:
                parts.append(f"{intent}_{target_word}")
            else:
                parts.append(intent)

        # Build hash from the full step sequence for uniqueness
        for step in steps:
            all_step_parts.append(f"{step.get('step', '')}:{step.get('target', '')}")
        hash_suffix = hashlib.sha256(
            "|".join(all_step_parts).encode("utf-8")
        ).hexdigest()[:6]

        raw = "_".join(parts)

        # Normalize: remove non-ASCII, replace spaces/special chars with underscore
        slug = unicodedata.normalize("NFKD", raw)
        slug = re.sub(r"[^\w\s-]", "", slug).strip().lower()
        slug = re.sub(r"[\s_]+", "_", slug)
        slug = slug[:73]  # Cap length leaving room for hash suffix

        slug = f"{slug}_{hash_suffix}"
        return slug

    def _generate_title(self, steps: list[dict], apps: list[str]) -> str:
        """Generate a human-readable title for the SOP.

        Format: "Verb Object in App" based on the pattern's dominant actions.
        """
        if not steps:
            return "Untitled SOP"

        # Count intents
        intent_counts = Counter(s.get("step", "action") for s in steps)
        dominant_intent = intent_counts.most_common(1)[0][0]

        # Get a meaningful target from the first step with a target
        target = ""
        for step in steps:
            t = step.get("target", "")
            if t:
                target = t
                break

        # Build title
        # Capitalize intent
        verb = dominant_intent.capitalize()

        # Shorten target if too long
        if len(target) > 40:
            target = target[:37] + "..."

        if target and apps:
            title = f"{verb} {target} in {', '.join(apps[:2])}"
        elif target:
            title = f"{verb} {target}"
        elif apps:
            title = f"{verb} workflow in {', '.join(apps[:2])}"
        else:
            title = f"{verb} workflow"

        return title
