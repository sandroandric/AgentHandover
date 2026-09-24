"""Regression pins for divergence branches (WP-M5): where an instance of a
Skill leaves its steps, the hypotheses that records, and the branch two
agreeing instances promote.  The end-to-end path (observations attached to
a Skill through the observation log) is pinned in ``test_workflow_mining.py``
(``TestSkillsAndObservations``)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from agenthandover_worker import branch_mining as bm
from agenthandover_worker.ledger import derive_knowledge
from agenthandover_worker.task_segmenter import AnnotatedFrame

STRIPE = "https://dashboard.stripe.com/payouts"
PAYOUT = "https://dashboard.stripe.com/payouts/po_1QxAbCdEfGh23Jk"
SHEET = "https://docs.google.com/spreadsheets/d/1pQx7RkTz0Lm4aVn8sWc2YbE5dJh3uFg/edit"
SLACK = "https://app.slack.com/client/T04ACME9/C05FINANCE"
MAIL = "https://mail.google.com/mail/u/0/#inbox"

PROC = {
    "id": "payout",
    "steps": [
        {"action": "Open the payouts list", "app": "Google Chrome", "location": STRIPE},
        {"action": "Open today's payout", "app": "Google Chrome", "location": PAYOUT},
        {"action": "Enter the amount in the sheet", "app": "Google Chrome", "location": SHEET},
        {"action": "Post in #finance", "app": "Google Chrome", "location": SLACK},
    ],
}


def _frames(prefix: str, places: list[tuple[str, dict]]) -> list[AnnotatedFrame]:
    t0 = datetime(2026, 9, 1, 9, tzinfo=timezone.utc)
    out = []
    for i, (loc, extra) in enumerate(places):
        ann = {
            "app": "Google Chrome", "location": loc,
            "task_context": {"what_doing": extra.get("what", f"working on {loc}"), "is_workflow": True},
            "visible_items": [{"label": label} for label in extra.get("items", [])],
            "_screen_text": extra.get("text", ""),
        }
        out.append(AnnotatedFrame.from_event({
            "id": f"{prefix}-{i}",
            "timestamp": (t0 + timedelta(minutes=i)).isoformat(),
            "scene_annotation_json": json.dumps(ann),
        }))
    return out


def _proc() -> dict:
    return json.loads(json.dumps(PROC))


FAILED = {"items": ["Payout failed", "Retry"], "text": "Payout failed\nBank declined"}


def _diverging(prefix: str, screen: dict = FAILED) -> list[AnnotatedFrame]:
    """The routine, but on a failed payout the user emails the bank first."""
    return _frames(prefix, [
        (STRIPE, {}), (PAYOUT, screen),
        (MAIL, {"what": "Emailing the bank about the failed payout"}),
        (SHEET, {}), (SLACK, {}),
    ])


class TestFindDivergences:
    def test_the_routine_as_written_does_not_diverge(self) -> None:
        frames = _frames("r", [(STRIPE, {}), (PAYOUT, {}), (PAYOUT, {}), (SHEET, {}), (SLACK, {})])
        assert bm.find_divergences(_proc(), frames) == []

    def test_returning_to_a_step_place_is_not_a_divergence(self) -> None:
        frames = _frames("r", [(STRIPE, {}), (PAYOUT, {}), (SHEET, {}), (PAYOUT, {}), (SHEET, {}), (SLACK, {})])
        assert bm.find_divergences(_proc(), frames) == []

    def test_a_new_place_before_a_step(self) -> None:
        frames = _diverging("d")
        [div] = bm.find_divergences(_proc(), frames)
        assert div["step_index"] == 2  # before "Enter the amount in the sheet"
        assert [f.event_id for f in div["alternative"]] == ["d-2"]
        # The screen the user decided on: the frame before the new place.
        assert div["condition_frame"].event_id == "d-1"

    def test_a_skipped_step(self) -> None:
        frames = _frames("s", [(STRIPE, {}), (PAYOUT, {}), (SLACK, {})])
        [div] = bm.find_divergences(_proc(), frames)
        assert div["step_index"] == 2 and div["alternative"] == []
        assert div["condition_frame"].event_id == "s-1"

    def test_a_new_place_after_the_last_step(self) -> None:
        frames = _frames("a", [(STRIPE, {}), (PAYOUT, {}), (SHEET, {}), (SLACK, {}), (MAIL, {})])
        [div] = bm.find_divergences(_proc(), frames)
        assert div["step_index"] == 4

    def test_steps_without_a_place_cannot_be_compared(self) -> None:
        proc = {"steps": [{"action": "Think it over"}]}
        assert bm.find_divergences(proc, _diverging("d")) == []


class TestHypothesesAndBranches:
    def test_one_instance_is_a_hypothesis_with_its_evidence(self) -> None:
        proc = _proc()
        assert bm.record_divergences(proc, "obs-1", _diverging("d1"))
        [hyp] = proc["branch_hypotheses"]
        assert hyp["step_index"] == 2
        assert hyp["condition_state"] == {
            "ocr_snippet": "Payout failed\nBank declined",
            "url": PAYOUT,
            "visible_items": ["Payout failed", "Retry"],
        }
        assert hyp["alternative_steps"] == [{
            "action": "Emailing the bank about the failed payout",
            "app": "Google Chrome", "location": MAIL,
        }]
        assert hyp["evidence"] == [
            {"kind": "instance", "id": "obs-1"},
            {"kind": "frame", "id": "d1-1"},
            {"kind": "frame", "id": "d1-2"},
        ]
        assert proc["branches"] == []

    def test_two_agreeing_instances_promote_a_branch(self) -> None:
        proc = _proc()
        bm.record_divergences(proc, "obs-1", _diverging("d1"))
        other = {"items": ["Payout failed", "Contact support"], "text": "Payout failed\nInsufficient funds"}
        assert bm.record_divergences(proc, "obs-2", _diverging("d2", other))
        [branch] = proc["branches"]
        assert branch["source"] == "divergence"
        assert branch["step_index"] == 2
        assert branch["condition"] == f'the screen shows "Payout failed" ({PAYOUT})'
        assert branch["action"] == "before step 3: Emailing the bank about the failed payout"
        assert {e["id"] for e in branch["evidence"] if e["kind"] == "instance"} == {"obs-1", "obs-2"}
        # The ledger's decisions slot now points at the branch.
        assert derive_knowledge(proc)["decisions"]["value_ref"] == "branches"

    def test_a_third_agreeing_instance_adds_its_evidence(self) -> None:
        proc = _proc()
        for n in range(3):
            bm.record_divergences(proc, f"obs-{n}", _diverging(f"d{n}"))
        [branch] = proc["branches"]
        assert {e["id"] for e in branch["evidence"] if e["kind"] == "instance"} == {"obs-0", "obs-1", "obs-2"}

    def test_screens_that_share_nothing_do_not_promote(self) -> None:
        proc = _proc()
        bm.record_divergences(proc, "obs-1", _diverging("d1"))
        bm.record_divergences(proc, "obs-2", _diverging("d2", {"items": ["Paid"], "text": "All good"}))
        assert len(proc["branch_hypotheses"]) == 2
        assert proc["branches"] == []

    def test_different_detours_do_not_agree(self) -> None:
        proc = _proc()
        bm.record_divergences(proc, "obs-1", _diverging("d1"))
        skip = _frames("s", [(STRIPE, {}), (PAYOUT, FAILED), (SLACK, {})])
        bm.record_divergences(proc, "obs-2", skip)
        assert proc["branches"] == []

    def test_an_instance_is_recorded_once(self) -> None:
        proc = _proc()
        assert bm.record_divergences(proc, "obs-1", _diverging("d1"))
        assert not bm.record_divergences(proc, "obs-1", _diverging("d1"))
        assert len(proc["branch_hypotheses"]) == 1

    def test_hypotheses_are_capped(self) -> None:
        proc = _proc()
        for n in range(bm.MAX_HYPOTHESES + 5):
            bm.record_divergences(proc, f"obs-{n}", _diverging(f"d{n}", {"items": [f"only {n}"]}))
        assert len(proc["branch_hypotheses"]) == bm.MAX_HYPOTHESES
        assert bm._instance(proc["branch_hypotheses"][0]) == "obs-5"


# The daily-news routine of the J-ledger fixtures (``focus_dailynews*``):
# read the front page, open each story worth sharing, note it, go back,
# post the digest.  Each run opens other stories on the same sites.
HN = "https://news.ycombinator.com/"
SLACK_TITLE = "daily-news (Channel) - Example Co - Slack"
PAYWALL = {"text": "research.example.com\nSubscribe to continue reading\nSubscribe | Sign in"}
GUIDE = "https://research.example.com/on-device-vision-guide"
DAILY = {
    "id": "daily-news-digest",
    "steps": [
        {"action": f"Open {loc or app}", "app": app, "location": loc}
        for app, loc in [
            ("Google Chrome", HN), ("Google Chrome", "https://blog.example.org/sqlite-wal"),
            ("Notes", ""), ("Google Chrome", HN), ("Google Chrome", GUIDE), ("Notes", ""),
            ("Google Chrome", GUIDE), ("Google Chrome", HN),
            ("Google Chrome", "https://github.com/example/cassette"), ("Notes", ""),
            ("Google Chrome", SLACK_TITLE),
        ]
    ],
}


def _daily() -> dict:
    return json.loads(json.dumps(DAILY))


def _run(prefix: str, places: list[tuple[str, str, dict]]) -> list[AnnotatedFrame]:
    t0 = datetime(2026, 9, 17, 8, tzinfo=timezone.utc)
    return [
        AnnotatedFrame.from_event({
            "id": f"{prefix}-{i}",
            "timestamp": (t0 + timedelta(minutes=i)).isoformat(),
            "scene_annotation_json": json.dumps({
                "app": app, "location": loc, "_screen_text": extra.get("text", ""),
                "task_context": {"what_doing": "", "is_workflow": True},
            }),
        })
        for i, (app, loc, extra) in enumerate(places)
    ]


def _paywall_run(prefix: str, story: str) -> list[AnnotatedFrame]:
    """A morning whose second story is paywalled: read through the archive."""
    archive = f"https://archive.example.net/?url=https://research.example.com/articles/{story}"
    return _run(prefix, [
        ("Google Chrome", HN, {}), ("Google Chrome", f"https://blog.example.org/{prefix}", {}),
        ("Notes", "", {}), ("Google Chrome", HN, {}),
        ("Google Chrome", f"https://research.example.com/articles/{story}", PAYWALL),
        ("Google Chrome", archive, {}), ("Notes", "", {}), ("Google Chrome", archive, {}),
        ("Google Chrome", HN, {}), ("Google Chrome", f"https://github.com/example/{prefix}", {}),
        ("Notes", "", {}), ("Google Chrome", SLACK_TITLE, {}),
    ])


def _skip_run(prefix: str) -> list[AnnotatedFrame]:
    """A morning that glances at a video and leaves the second story out."""
    return _run(prefix, [
        ("Google Chrome", HN, {}), ("Google Chrome", "https://blog.example.org/planner", {}),
        ("Notes", "", {}), ("Google Chrome", "https://blog.example.org/planner", {}),
        ("Google Chrome", HN, {}), ("Google Chrome", "https://video.example.com/watch/db", {}),
        ("Google Chrome", HN, {}), ("Google Chrome", "https://github.com/example/snap", {}),
        ("Notes", "", {}), ("Google Chrome", SLACK_TITLE, {}),
    ])


class TestOneDetourIsOneDivergence:
    def test_another_page_of_a_steps_site_is_the_step(self) -> None:
        frames = _frames("r", [(STRIPE, {}), (STRIPE + "/po_other", {}), (SHEET, {}), (SLACK, {})])
        assert bm.find_divergences(_proc(), frames) == []

    def test_the_paywall_detour_is_one_divergence(self) -> None:
        [div] = bm.find_divergences(_daily(), _paywall_run("d", "48213"))
        # It replaces "note the story, back to the story" (steps 6-7) with
        # the archived copy, the note and the archived copy again; the
        # paywalled article is the "open the story" step.
        assert (div["step_index"], div["rejoin_index"]) == (5, 7)
        assert [f.event_id for f in div["alternative"]] == ["d-5", "d-6", "d-7"]
        # Decided on the paywalled story.
        assert div["condition_frame"].event_id == "d-4"

    def test_the_left_out_story_is_one_divergence(self) -> None:
        [div] = bm.find_divergences(_daily(), _skip_run("s"))
        # The video replaces the second story's steps (open, note, back).
        assert (div["step_index"], div["rejoin_index"]) == (4, 7)
        assert [f.location for f in div["alternative"]] == ["https://video.example.com/watch/db"]
        assert div["condition_frame"].location == HN

    def test_two_paywall_mornings_make_one_branch_the_skip_a_hypothesis(self) -> None:
        proc = _daily()
        bm.record_divergences(proc, "skip", _skip_run("s"))
        bm.record_divergences(proc, "d1", _paywall_run("d1", "48213"))
        bm.record_divergences(proc, "d2", _paywall_run("d2", "51877"))
        [branch] = proc["branches"]
        assert branch["kind"] == "detour"
        assert (branch["step_index"], branch["rejoin_index"]) == (5, 7)
        assert "Subscribe to continue reading" in branch["condition"]
        assert branch["action"].startswith("instead of steps 6-7: go to https://archive.example.net/")
        assert {e["id"] for e in branch["evidence"] if e["kind"] == "instance"} == {"d1", "d2"}
        [skip] = [h for h in proc["branch_hypotheses"] if bm._instance(h) == "skip"]
        assert (skip["step_index"], skip["rejoin_index"]) == (4, 7)

    def test_a_late_return_to_a_detour_is_another_divergence(self) -> None:
        frames = _frames("m", [(STRIPE, {}), (MAIL, {}), (PAYOUT, {}), (SHEET, {}), (SLACK, {}), (MAIL, {})])
        divs = bm.find_divergences(_proc(), frames)
        assert [(d["step_index"], d["rejoin_index"]) for d in divs] == [(1, 1), (4, 4)]

    def test_skipped_steps_in_a_row_are_one_skip(self) -> None:
        frames = _frames("s", [(STRIPE, {"text": "Nothing today"}), (SLACK, {})])
        [div] = bm.find_divergences(_proc(), frames)
        assert (div["step_index"], div["rejoin_index"], div["alternative"]) == (1, 3, [])
        bm.record_divergences(proc := _proc(), "obs-1", frames)
        [hyp] = proc["branch_hypotheses"]
        assert hyp["kind"] == "skip"
