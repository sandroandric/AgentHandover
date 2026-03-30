"""Pure-function metric computations for the AgentHandover evaluation harness.

All functions use only the Python standard library (no numpy).
Every function returns a dict or float.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _comb2(n: int) -> int:
    """Return C(n, 2) = n*(n-1)/2, the number of ways to choose 2 from n."""
    if n < 2:
        return 0
    return n * (n - 1) // 2


# ---------------------------------------------------------------------------
# 1. precision / recall / F1 from raw counts
# ---------------------------------------------------------------------------

def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    """Compute precision, recall, and F1 from raw true-positive / false-positive / false-negative counts.

    Returns:
        dict with keys ``"precision"``, ``"recall"``, ``"f1"``.
        Each value is 0.0 when the denominator would be zero.
    """
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# 2. Binary classification metrics
# ---------------------------------------------------------------------------

def classification_metrics(predicted: list[bool], actual: list[bool]) -> dict:
    """Compute binary classification metrics from parallel boolean lists.

    Args:
        predicted: list of predicted labels (True = positive).
        actual:    list of ground-truth labels (True = positive).

    Returns:
        dict with keys ``"precision"``, ``"recall"``, ``"f1"``,
        ``"accuracy"``, ``"tp"``, ``"fp"``, ``"fn"``, ``"tn"``.

    Raises:
        ValueError: if the two lists differ in length.
    """
    if len(predicted) != len(actual):
        raise ValueError(
            f"predicted and actual must have the same length "
            f"({len(predicted)} != {len(actual)})"
        )

    tp = fp = fn = tn = 0
    for p, a in zip(predicted, actual):
        if p and a:
            tp += 1
        elif p and not a:
            fp += 1
        elif not p and a:
            fn += 1
        else:
            tn += 1

    prf = precision_recall_f1(tp, fp, fn)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ---------------------------------------------------------------------------
# 3. Boundary overlap (Jaccard)
# ---------------------------------------------------------------------------

def boundary_overlap_jaccard(
    predicted_boundaries: list[set[str]],
    truth_boundaries: list[set[str]],
) -> float:
    """Average best-match Jaccard overlap between truth and predicted boundaries.

    For each truth boundary, the predicted boundary with the highest Jaccard
    similarity is selected.  The function returns the mean of those best-match
    Jaccard values across all truth boundaries.

    Args:
        predicted_boundaries: list of sets, each containing element identifiers
            that belong to one predicted segment.
        truth_boundaries: list of sets, each containing element identifiers
            that belong to one ground-truth segment.

    Returns:
        Mean best-match Jaccard in [0.0, 1.0].
        Returns 0.0 when *truth_boundaries* is empty.
    """
    if not truth_boundaries:
        return 0.0

    total_jaccard = 0.0
    for truth_set in truth_boundaries:
        best_jaccard = 0.0
        for pred_set in predicted_boundaries:
            intersection = len(truth_set & pred_set)
            union = len(truth_set | pred_set)
            jaccard = _safe_div(intersection, union)
            if jaccard > best_jaccard:
                best_jaccard = jaccard
        total_jaccard += best_jaccard

    return total_jaccard / len(truth_boundaries)


# ---------------------------------------------------------------------------
# 4. Adjusted Rand Index
# ---------------------------------------------------------------------------

def adjusted_rand_index(
    predicted_labels: list[int],
    truth_labels: list[int],
) -> float:
    """Compute the Adjusted Rand Index (ARI) for two clusterings.

    Uses the combinatorial (contingency-table) formula:

        ARI = (RI - Expected_RI) / (max_RI - Expected_RI)

    where the index is computed from the sum of C(n_ij, 2) values in the
    contingency table versus the row/column marginal sums.

    Args:
        predicted_labels: cluster assignment for each item (integers).
        truth_labels:     ground-truth cluster assignment (integers).

    Returns:
        ARI value in [-1, 1].  Returns 0.0 when the denominator is zero
        (e.g. all items in a single cluster in both partitions).

    Raises:
        ValueError: if the two lists differ in length.
    """
    if len(predicted_labels) != len(truth_labels):
        raise ValueError(
            f"predicted_labels and truth_labels must have the same length "
            f"({len(predicted_labels)} != {len(truth_labels)})"
        )

    n = len(predicted_labels)
    if n == 0:
        return 0.0

    # Build contingency table as a Counter of (pred_label, truth_label) pairs.
    contingency: Counter[tuple[int, int]] = Counter()
    for p, t in zip(predicted_labels, truth_labels):
        contingency[(p, t)] += 1

    # Row sums (predicted cluster sizes) and column sums (truth cluster sizes).
    row_sums: Counter[int] = Counter()
    col_sums: Counter[int] = Counter()
    for (p, t), count in contingency.items():
        row_sums[p] += count
        col_sums[t] += count

    # Sum of C(n_ij, 2) over all cells.
    sum_comb_nij = sum(_comb2(v) for v in contingency.values())

    # Sum of C(a_i, 2) over row sums.
    sum_comb_a = sum(_comb2(v) for v in row_sums.values())

    # Sum of C(b_j, 2) over column sums.
    sum_comb_b = sum(_comb2(v) for v in col_sums.values())

    # Total pairs.
    total_comb = _comb2(n)

    # Expected index.
    expected = _safe_div(sum_comb_a * sum_comb_b, total_comb)

    # Max index.
    max_index = 0.5 * (sum_comb_a + sum_comb_b)

    numerator = sum_comb_nij - expected
    denominator = max_index - expected

    return _safe_div(numerator, denominator)


# ---------------------------------------------------------------------------
# 5. Cluster purity
# ---------------------------------------------------------------------------

def cluster_purity(
    predicted_labels: list[int],
    truth_labels: list[int],
) -> float:
    """Compute cluster purity: fraction of items whose predicted cluster
    agrees with the majority truth class in that cluster, weighted by
    cluster size.

    Purity = (1/N) * sum_k max_j |c_k AND t_j|

    where c_k is the set of items in predicted cluster k and t_j is the
    set of items with truth label j.

    Args:
        predicted_labels: cluster assignment for each item.
        truth_labels:     ground-truth class for each item.

    Returns:
        Purity in [0.0, 1.0].  Returns 0.0 when the input is empty.

    Raises:
        ValueError: if the two lists differ in length.
    """
    if len(predicted_labels) != len(truth_labels):
        raise ValueError(
            f"predicted_labels and truth_labels must have the same length "
            f"({len(predicted_labels)} != {len(truth_labels)})"
        )

    n = len(predicted_labels)
    if n == 0:
        return 0.0

    # Group truth labels by predicted cluster.
    clusters: dict[int, Counter[int]] = {}
    for p, t in zip(predicted_labels, truth_labels):
        if p not in clusters:
            clusters[p] = Counter()
        clusters[p][t] += 1

    correct = sum(counter.most_common(1)[0][1] for counter in clusters.values())
    return correct / n


# ---------------------------------------------------------------------------
# 6. Family precision / recall for merge decisions
# ---------------------------------------------------------------------------

def family_precision_recall(
    predicted_merges: list[tuple[str, str]],
    truth_merges: list[tuple[str, str]],
) -> dict:
    """Precision, recall, and F1 for merge decisions treated as sets of
    unordered pairs.

    Each merge is a pair of identifiers.  Order within the pair does not
    matter (i.e. (A, B) == (B, A)).

    Args:
        predicted_merges: list of predicted merge pairs.
        truth_merges:     list of ground-truth merge pairs.

    Returns:
        dict with keys ``"precision"``, ``"recall"``, ``"f1"``,
        ``"tp"``, ``"fp"``, ``"fn"``.
    """
    pred_set = {frozenset(pair) for pair in predicted_merges}
    truth_set = {frozenset(pair) for pair in truth_merges}

    tp = len(pred_set & truth_set)
    fp = len(pred_set - truth_set)
    fn = len(truth_set - pred_set)

    prf = precision_recall_f1(tp, fp, fn)
    return {
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# 7. Noise-frame drop accuracy
# ---------------------------------------------------------------------------

def noise_drop_accuracy(
    predicted_dropped: set[str],
    truth_noise_ids: set[str],
    total_ids: set[str],
) -> dict:
    """Accuracy, precision, and recall for noise-frame dropping.

    A "positive" is a frame identified as noise (and therefore dropped).

    Args:
        predicted_dropped: IDs the system chose to drop.
        truth_noise_ids:   IDs that are truly noise.
        total_ids:         full set of all frame IDs in the evaluation set.

    Returns:
        dict with keys ``"accuracy"``, ``"noise_precision"``,
        ``"noise_recall"``.
    """
    tp = len(predicted_dropped & truth_noise_ids)
    fp = len(predicted_dropped - truth_noise_ids)
    fn = len(truth_noise_ids - predicted_dropped)
    tn = len(total_ids - predicted_dropped - truth_noise_ids)

    accuracy = _safe_div(tp + tn, len(total_ids)) if total_ids else 0.0
    noise_precision = _safe_div(tp, tp + fp)
    noise_recall = _safe_div(tp, tp + fn)

    return {
        "accuracy": accuracy,
        "noise_precision": noise_precision,
        "noise_recall": noise_recall,
    }


# ---------------------------------------------------------------------------
# 8. Multiclass macro-averaged F1
# ---------------------------------------------------------------------------

def multiclass_macro_f1(predicted: list[str], actual: list[str]) -> dict:
    """Compute macro-averaged F1 across all classes.

    For each class that appears in *actual*, binary precision/recall/F1 is
    computed (one-vs-rest).  The macro F1 is the unweighted average of the
    per-class F1 scores.

    Args:
        predicted: List of predicted class labels (strings).
        actual:    List of actual class labels (strings).

    Returns:
        ``{"macro_f1": float, "per_class": {class: {"precision", "recall", "f1"}}}``

    Raises:
        ValueError: if the two lists differ in length.
    """
    if len(predicted) != len(actual):
        raise ValueError(
            f"predicted and actual must have the same length "
            f"({len(predicted)} != {len(actual)})"
        )

    # Discover all classes from ground truth
    classes = sorted(set(actual))

    per_class: dict[str, dict[str, float]] = {}

    for cls in classes:
        tp = fp = fn = 0
        for p, a in zip(predicted, actual):
            if p == cls and a == cls:
                tp += 1
            elif p == cls and a != cls:
                fp += 1
            elif p != cls and a == cls:
                fn += 1

        prf = precision_recall_f1(tp, fp, fn)
        per_class[cls] = {
            "precision": round(prf["precision"], 4),
            "recall": round(prf["recall"], 4),
            "f1": round(prf["f1"], 4),
        }

    if classes:
        macro_f1 = sum(pc["f1"] for pc in per_class.values()) / len(classes)
    else:
        macro_f1 = 0.0

    return {
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }
