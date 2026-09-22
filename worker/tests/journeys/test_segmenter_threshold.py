"""The segmenter's similarity threshold against real nomic-embed-text vectors.

Twenty annotated frames from eight tasks, several of which share a site
(two tasks on Expensify, two on GitHub) or an app.  The frames go through
``TaskSegmenter.segment`` with Ollama answered by the recorded cassette, so
the vectors are the model's real output for the ``clustering:``-prefixed
texts the segmenter sends.

Measured on this set (greedy centroid clustering): the prefix lifts every
similarity by about 0.09.  Unprefixed at 0.75 the set gave 11 clusters with
no two tasks merged; prefixed, 0.75 merges seven tasks into one cluster and
0.83 gives the same 11 clusters with no merge.  Three tasks (deploy, issue,
news) still split in two at 0.83, as they did before the prefix: a split
costs a demonstration, a merge produces a chimera Skill, so the pin is
"no cluster holds two tasks".  The threshold is to be re-baselined on the
three-week passive fixture when it exists.

Record: ``AH_RECORD=1 AH_RECORD_UPSTREAM=http://<ollama> pytest <this file>``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from agenthandover_worker.task_segmenter import (
    SegmenterConfig,
    TaskSegmenter,
    _cluster_frames,
)
from journeys.cassette import open_cassette

JOURNEY = "segmenter_threshold"

# (task, app, what_doing, location)
FRAMES = [
    ("expense", "Google Chrome", "Filing expense report", "https://expensify.com/reports"),
    ("expense", "Google Chrome", "Attaching receipt image to expense", "https://expensify.com/report/R123"),
    ("expense", "Google Chrome", "Entering amount and category for a lunch expense", "https://expensify.com/report/R123"),
    ("expense", "Google Chrome", "Submitting the monthly expense report for approval", "https://expensify.com/reports"),
    ("exp_settings", "Google Chrome", "Updating bank account details in profile settings", "https://expensify.com/settings/wallet"),
    ("exp_settings", "Google Chrome", "Changing default currency in account preferences", "https://expensify.com/settings/preferences"),
    ("invoice", "Google Chrome", "Uploading invoice PDF to shared folder", "https://www.dropbox.com/home/Invoices"),
    ("invoice", "Google Chrome", "Renaming invoice file with client name and month", "https://www.dropbox.com/home/Invoices"),
    ("invoice", "Google Chrome", "Sharing invoice folder link with the accountant", "https://www.dropbox.com/home/Invoices"),
    ("pr", "Google Chrome", "Reviewing pull request diff", "https://github.com/org/repo/pull/42"),
    ("pr", "Google Chrome", "Leaving a review comment on a changed line", "https://github.com/org/repo/pull/42/files"),
    ("pr", "Google Chrome", "Approving the pull request", "https://github.com/org/repo/pull/42"),
    ("issue", "Google Chrome", "Triaging new GitHub issues and adding labels", "https://github.com/org/repo/issues"),
    ("issue", "Google Chrome", "Assigning an issue to a teammate", "https://github.com/org/repo/issues/77"),
    ("news", "Google Chrome", "Reading daily news headlines", "https://news.ycombinator.com"),
    ("news", "Google Chrome", "Opening an article about AI chips", "https://news.ycombinator.com/item?id=1"),
    ("deploy", "Terminal", "Running deploy script for staging", ""),
    ("deploy", "Terminal", "Checking deployment logs for errors", ""),
    ("mail", "Mail", "Replying to a customer email about a refund", ""),
    ("mail", "Mail", "Writing an email reply to support ticket", ""),
]


def _events() -> list[dict]:
    start = datetime(2026, 3, 10, 9, 0, tzinfo=timezone.utc)
    events = []
    for i, (_task, app, what_doing, location) in enumerate(FRAMES):
        annotation = {
            "app": app,
            "location": location,
            "task_context": {"what_doing": what_doing, "is_workflow": True},
        }
        events.append({
            "id": f"f{i:02d}",
            "timestamp": (start + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "scene_annotation_json": json.dumps(annotation),
            "annotation_status": "completed",
        })
    return events


def _partition(groups) -> set[frozenset[str]]:
    """Clusters as sets of task labels, to compare with the true tasks."""
    return {frozenset(FRAMES[i][0] for i in members) for members in groups}


def _segment():
    with open_cassette(JOURNEY) as tape:
        segmenter = TaskSegmenter(config=SegmenterConfig(ollama_host=tape.host))
        return segmenter.segment(_events())


def test_default_threshold_never_merges_two_tasks() -> None:
    result = _segment()

    by_cluster: dict[int, list[int]] = {}
    for seg in result.segments:
        for frame in seg.frames:
            by_cluster.setdefault(seg.cluster_id, []).append(int(frame.event_id[1:]))

    assert sum(len(m) for m in by_cluster.values()) == len(FRAMES)
    merged = [labels for labels in _partition(by_cluster.values()) if len(labels) > 1]
    assert merged == []
    # The two longest tasks stay whole.
    clusters_of = {
        task: {cid for cid, members in by_cluster.items()
               if any(FRAMES[i][0] == task for i in members)}
        for task in ("expense", "invoice")
    }
    assert all(len(cids) == 1 for cids in clusters_of.values()), clusters_of


def test_old_threshold_merges_tasks_under_the_prefix() -> None:
    """Why the default moved from 0.75: prefixed vectors sit closer together."""
    result = _segment()
    frames = sorted(
        (f for seg in result.segments for f in seg.frames),
        key=lambda f: f.event_id,
    )
    assert all(f.embedding for f in frames)
    merged = [
        labels for labels in _partition(_cluster_frames(frames, 0.75).values())
        if len(labels) > 1
    ]
    assert merged, "0.75 no longer merges distinct tasks; re-check the default"
