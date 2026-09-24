"""Regression pins for WP-L2: visible-versus-chosen list items on the
annotation and the selection criteria learned by contrasting chosen with
skipped entries across instances.

The journey proof is J-ledger's selection-criteria assertion
(``journeys/test_j_ledger.py``), which waits on the two extra recorded
runs of the daily-news routine.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from agenthandover_worker.behavioral_synthesizer import (
    BehavioralSynthesizer,
    SynthesizerConfig,
    contrast_instances,
    phrase_criteria,
)
from agenthandover_worker.contrast import (
    MAX_VISIBLE_ITEMS,
    criteria_hypotheses,
    ground_visible_items,
    item_attrs,
    page_rows,
)
from agenthandover_worker.focus_processor import FocusProcessor, recorded_observations
from agenthandover_worker.ledger import derive_knowledge
from agenthandover_worker.scene_annotator import AnnotationConfig, SceneAnnotator
from agenthandover_worker.skill_render import render_body

from test_text_first_annotation import _Recorder, _answer, _event

# ---------------------------------------------------------------------------
# Attributes of an entry
# ---------------------------------------------------------------------------


class TestItemAttrs:
    def test_counts_age_and_price(self):
        assert item_attrs("182 points by pg 3 hours ago | 45 comments") == {
            "age_minutes": 180.0, "point": 182.0, "comment": 45.0,
        }
        assert item_attrs("$1,299.00 laptop, 1.2k views, 40% off") == {
            "price": 1299.0, "view": 1200.0, "%": 40.0,
        }
        assert item_attrs("posted an hour ago") == {"age_minutes": 60.0}

    def test_dates_and_durations_are_not_counts(self):
        assert item_attrs("12 Sep 2026, 5 minutes read") == {}
        assert item_attrs("SQLite's WAL mode, explained") == {}


# ---------------------------------------------------------------------------
# visible_items of a frame
# ---------------------------------------------------------------------------

_DIGEST = "\n".join([
    "The quiet death of the RSS reader",
    "Designing a scheduler that respects laptop batteries",
    "SQLite's WAL mode, explained with diagrams",
    "Ask HN: How do you keep a daily reading habit?",
])

# The extension's anchor set: the clicked link, then its row and table.
_CLICKED_ROW = [
    {"tag": "a", "name": "SQLite's WAL mode, explained with diagrams",
     "text": "SQLite's WAL mode, explained with diagrams",
     "selector": "table > tr > td > a", "interacted": True},
    {"tag": "td", "selector": "table > tr > td",
     "text": "3. SQLite's WAL mode, explained with diagrams (sqlite.org)"},
    {"tag": "tr", "selector": "table > tr",
     "text": "3. SQLite's WAL mode, explained with diagrams (sqlite.org)"},
]

_OCR = "\n".join([
    "Hacker News  new | past | comments",
    "1. The quiet death of the RSS reader",
    "95 points by ada 7 hours ago | 20 comments",
    "2. Designing a scheduler that respects laptop batteries",
    "12 points by bob 1 hour ago | discuss",
    "3. SQLite's WAL mode, explained with diagrams",
    "310 points by cy 2 hours ago | 88 comments",
    "4. Ask HN: How do you keep a daily reading habit?",
])


class TestPageRows:
    def test_the_clicked_row_is_chosen_among_the_digest_rows(self):
        rows = page_rows({"nodes": _CLICKED_ROW, "digest": _DIGEST})
        assert [r["label"] for r in rows] == _DIGEST.splitlines()
        assert [r["chosen"] for r in rows] == [False, False, True, False]
        assert [r["attrs"]["position"] for r in rows] == [1.0, 2.0, 3.0, 4.0]

    def test_a_row_ancestor_identifies_the_pick_when_the_element_has_no_text(self):
        nodes = [{"tag": "span", "selector": "tr > td > span", "interacted": True},
                 *_CLICKED_ROW[1:]]
        rows = page_rows({"nodes": nodes, "digest": _DIGEST})
        assert [r["label"] for r in rows if r["chosen"]] == [
            "SQLite's WAL mode, explained with diagrams",
        ]

    def test_no_click_or_a_click_outside_the_list_is_not_a_pick(self):
        unclicked = [{**n, "interacted": False} for n in _CLICKED_ROW]
        assert page_rows({"nodes": unclicked, "digest": _DIGEST}) == []
        button = [{"tag": "button", "name": "Submit", "selector": "form > button",
                   "interacted": True}]
        assert page_rows({"nodes": button, "digest": _DIGEST}) == []
        # The earlier extension's tree has no digest.
        assert page_rows({"nodes": _CLICKED_ROW, "digest": ""}) == []

    def test_a_long_list_keeps_the_pick_and_the_first_others(self):
        digest = "\n".join(f"Story number {i} about things" for i in range(1, 31))
        nodes = [{"tag": "a", "name": "Story number 25 about things", "interacted": True,
                  "selector": "a"}]
        rows = page_rows({"nodes": nodes, "digest": digest})
        assert len(rows) == MAX_VISIBLE_ITEMS
        assert rows[-1] == {"label": "Story number 25 about things",
                            "attrs": {"position": 25.0}, "chosen": True}


class TestGroundVisibleItems:
    def test_dom_rows_win_and_read_their_numbers_from_the_ocr(self):
        model = [{"label": "RSS reader", "chosen": True}]
        items = ground_visible_items(model, {"nodes": _CLICKED_ROW, "digest": _DIGEST}, _OCR)
        by_label = {i["label"]: i for i in items}
        wal = by_label["SQLite's WAL mode, explained with diagrams"]
        assert wal["chosen"] is True
        assert wal["attrs"] == {"point": 310.0, "age_minutes": 120.0, "comment": 88.0,
                                "position": 3.0}
        assert by_label["Designing a scheduler that respects laptop batteries"]["attrs"] == {
            "point": 12.0, "age_minutes": 60.0, "position": 2.0,
        }
        # The last row has no subtext line: nothing is borrowed from elsewhere.
        assert by_label["Ask HN: How do you keep a daily reading habit?"]["attrs"] == {
            "position": 4.0,
        }

    def test_without_a_dom_pick_the_models_entries_stand_with_ocr_numbers(self):
        model = [{"label": "SQLite's WAL mode", "chosen": True},
                 {"label": "The quiet death of the RSS reader", "chosen": False},
                 {"label": "", "chosen": False}, "junk"]
        items = ground_visible_items(model, None, _OCR)
        assert items == [
            {"label": "SQLite's WAL mode", "chosen": True,
             "attrs": {"point": 310.0, "age_minutes": 120.0, "comment": 88.0}},
            {"label": "The quiet death of the RSS reader", "chosen": False,
             "attrs": {"point": 95.0, "age_minutes": 420.0, "comment": 20.0}},
        ]


class TestAnnotatorFillsVisibleItems:
    def test_a_clicked_list_row_becomes_the_stored_items_and_the_prompt_is_unchanged(
        self, tmp_path,
    ):
        page = {"url": "https://news.ycombinator.com/", "nodes": _CLICKED_ROW,
                "digest": _DIGEST}
        rec = _Recorder(_answer())
        with patch("urllib.request.urlopen", rec):
            with_digest = SceneAnnotator(AnnotationConfig()).annotate_event(
                _event(tmp_path, event_id="a"), page=page,
            )
        rec_plain = _Recorder(_answer())
        with patch("urllib.request.urlopen", rec_plain):
            SceneAnnotator(AnnotationConfig()).annotate_event(
                _event(tmp_path, event_id="b"), page={**page, "digest": ""},
            )
        # The digest is not shown to the model: the request is identical.
        assert rec.bodies[0]["prompt"] == rec_plain.bodies[0]["prompt"]
        items = with_digest.annotation["visible_items"]
        assert [i["label"] for i in items] == _DIGEST.splitlines()
        assert [i["label"] for i in items if i["chosen"]] == [
            "SQLite's WAL mode, explained with diagrams",
        ]

    def test_the_models_items_get_attrs_without_a_page(self, tmp_path):
        rec = _Recorder(_answer())
        with patch("urllib.request.urlopen", rec):
            result = SceneAnnotator(AnnotationConfig()).annotate_event(_event(tmp_path))
        assert result.annotation["visible_items"] == [
            {"label": "SQLite's WAL mode", "attrs": {}, "chosen": True},
        ]


# ---------------------------------------------------------------------------
# Criteria across instances
# ---------------------------------------------------------------------------


def _list(entries: list[tuple[str, dict, bool]]) -> list[dict]:
    return [{"label": label, "attrs": attrs, "chosen": chosen} for label, attrs, chosen in entries]


def _hn_run(iid: str, picked: str, stories: dict[str, dict]) -> dict:
    return {"id": iid, "lists": [
        _list([(label, attrs, label == picked) for label, attrs in stories.items()]),
    ]}


_RUNS = [
    _hn_run("sess-1", "Postgres 17 released", {
        "Postgres 17 released": {"point": 410, "age_minutes": 120},
        "My cat's startup": {"point": 12, "age_minutes": 60},
        "Old news about Perl": {"point": 30, "age_minutes": 2880},
    }),
    _hn_run("sess-2", "Rust in the kernel", {
        "Rust in the kernel": {"point": 250, "age_minutes": 300},
        "Show HN: a toy": {"point": 8, "age_minutes": 30},
        "Why I quit": {"point": 90, "age_minutes": 600},
    }),
    _hn_run("sess-3", "SQLite 4 announced", {
        "SQLite 4 announced": {"point": 300, "age_minutes": 60},
        "Ask HN: hiring?": {"point": 40, "age_minutes": 200},
        "Ten tabs I keep open": {"point": 5, "age_minutes": 10},
    }),
]


class TestCriteriaHypotheses:
    def test_a_threshold_that_separates_picks_from_skips_cites_every_run(self):
        [best, *_] = criteria_hypotheses(_RUNS)
        assert best["criterion"] == "have at least 250 points"
        assert (best["attribute"], best["op"], best["value"]) == ("point", ">=", 250)
        assert best["support"] == 3 and best["counter"] == 0
        assert best["evidence"] == [{"kind": "instance", "id": f"sess-{i}"} for i in (1, 2, 3)]
        assert best["status"] == "hypothesis"
        assert best["examples"] == [
            "Postgres 17 released", "Rust in the kernel", "SQLite 4 announced",
        ]

    def test_age_reads_as_at_most_old(self):
        runs = [
            _hn_run(f"s{i}", "x", {"x": {"age_minutes": 60 * i},
                                   "y": {"age_minutes": 2880},
                                   "z": {"age_minutes": 5000}})
            for i in (1, 2)
        ]
        [age] = criteria_hypotheses(runs)
        assert age["criterion"] == "are at most 2 hours old"
        assert age["counter"] == 0

    def test_a_keyword_the_picks_share_and_the_skips_do_not(self):
        runs = [
            _hn_run("s1", "Show HN: postgres tuner", {
                "Show HN: postgres tuner": {}, "Celebrity gossip": {}, "Stock tips": {},
            }),
            _hn_run("s2", "Postgres internals", {
                "Postgres internals": {}, "Weather today": {}, "Cooking pasta": {},
            }),
        ]
        [keyword] = criteria_hypotheses(runs)
        assert keyword["criterion"] == 'mention "postgres"'
        assert keyword["support"] == 2 and keyword["counter"] == 0

    def test_one_pick_or_runs_without_an_id_teach_nothing(self):
        assert criteria_hypotheses(_RUNS[:1]) == []
        assert criteria_hypotheses([{**run, "id": ""} for run in _RUNS]) == []

    def test_a_list_seen_in_many_frames_counts_once_and_later_picks_are_not_skips(self):
        stories = {"A": {"point": 200}, "B": {"point": 150}, "C": {"point": 3}}
        frames = [
            _list([(k, v, k == "A") for k, v in stories.items()]),
            _list([(k, v, k == "A") for k, v in stories.items()]),
            _list([(k, v, k == "B") for k, v in stories.items()]),
        ]
        [points] = criteria_hypotheses([{"id": "s1", "lists": frames}])
        # Two distinct picks; "B" was picked, so it is never a skip
        # that "have at least 150 points" fails to explain.
        assert points["criterion"] == "have at least 150 points"
        assert (points["support"], points["counter"]) == (2, 0)

    def test_a_criterion_that_admits_more_skips_than_it_rejects_is_dropped(self):
        runs = [
            _hn_run(f"s{i}", "x", {"x": {"point": 50}, "y": {"point": 60},
                                   "z": {"point": 10}})
            for i in (1, 2)
        ]
        # "At least 50" admits 60 as often as it rejects 10, and
        # "at most 50" the other way round.
        assert criteria_hypotheses(runs) == []


# ---------------------------------------------------------------------------
# The synthesizer words the criteria and never invents them
# ---------------------------------------------------------------------------


def _frame(entries: list[tuple[str, dict, bool]]) -> dict:
    ann = {
        "app": "Google Chrome", "location": "https://news.ycombinator.com/",
        "task_context": {"what_doing": "Picking a story", "is_workflow": True},
        "visible_items": _list(entries),
    }
    return FocusProcessor._build_pre_analysis_obs({"annotation": ann})


def _observations() -> list[list[dict]]:
    return [
        [_frame([(label, attrs, chosen) for label, attrs, chosen in (
            (item["label"], item["attrs"], item["chosen"]) for item in run["lists"][0]
        )])]
        for run in _RUNS
    ]


_ANSWER = {
    "goal": "Reads the top Hacker News story each morning",
    "strategy": "Opens Hacker News and reads the most upvoted story.",
    "selection_criteria": [
        {"id": "C1", "criterion": "Stories with at least 250 points", "confidence": 0.9},
        {"criterion": "Stories about databases", "confidence": 0.8},
    ],
    "guardrails": [],
}


class TestSynthesizerWordsCriteria:
    def test_the_pick_reaches_the_observation_with_its_attributes(self):
        [frame] = _observations()[0]
        assert frame["visible_items"] == [
            "Postgres 17 released (chosen)", "My cat's startup", "Old news about Perl",
        ]
        assert frame["list_choice"][0] == {
            "label": "Postgres 17 released", "attrs": {"point": 410, "age_minutes": 120},
            "chosen": True,
        }

    def test_the_prompt_lists_the_hypotheses_only_when_there_are_some(self):
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        proc = {"id": "hn", "title": "HN", "steps": []}
        plain = synth._build_prompt(proc, _observations())
        assert "SELECTION CRITERIA HYPOTHESES" not in plain
        hypotheses = criteria_hypotheses(contrast_instances(_observations(), ["a", "b", "c"]))
        shown = synth._build_prompt(proc, _observations(), hypotheses=hypotheses)
        assert "C1: chosen entries have at least 250 points (in 3 choices; 0 skipped" in shown
        assert "Do not add criteria that are not listed here." in shown
        # Only the block is added.
        assert shown.replace(shown[shown.index("\n\nSELECTION CRITERIA"):shown.index(
            "Do not add criteria that are not listed here.") + len(
            "Do not add criteria that are not listed here.")], "") == plain

    def test_synthesis_keeps_only_computed_criteria_worded_by_the_model(self):
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        with patch(
            "agenthandover_worker.behavioral_synthesizer._call_ollama_text",
            return_value=(json.dumps(_ANSWER), 1.0),
        ) as call:
            insights = synth.synthesize(
                "hn", {"id": "hn", "title": "HN", "steps": []}, _observations(),
                instance_ids=["sess-1", "sess-2", "sess-3"],
            )
        assert "SELECTION CRITERIA HYPOTHESES" in call.call_args.kwargs["prompt"]
        [criterion] = insights.selection_criteria
        assert criterion["criterion"] == "Stories with at least 250 points"
        assert criterion["basis"] == "have at least 250 points"
        assert criterion["status"] == "hypothesis"
        assert criterion["confidence"] == 1.0

    def test_without_evidence_the_models_criteria_are_dropped(self):
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        with patch(
            "agenthandover_worker.behavioral_synthesizer._call_ollama_text",
            return_value=(json.dumps(_ANSWER), 1.0),
        ) as call:
            insights = synth.synthesize("hn", {"id": "hn", "steps": []}, _observations())
        assert "SELECTION CRITERIA HYPOTHESES" not in call.call_args.kwargs["prompt"]
        assert insights.selection_criteria == []
        assert insights.goal

    def test_unworded_hypotheses_keep_their_computed_wording(self):
        hypotheses = criteria_hypotheses(_RUNS)
        [first, *_] = phrase_criteria(hypotheses, [{"criterion": "made up"}])
        assert first["criterion"] == "entries that have at least 250 points"

    def test_merge_marks_decisions_as_a_contrast_hypothesis_citing_three_runs(self):
        """The unit-level pin of J-ledger's selection-criteria assertion."""
        synth = BehavioralSynthesizer(SynthesizerConfig(min_observations=1))
        with patch(
            "agenthandover_worker.behavioral_synthesizer._call_ollama_text",
            return_value=(json.dumps(_ANSWER), 1.0),
        ):
            insights = synth.synthesize(
                "hn", {"id": "hn", "steps": []}, _observations(),
                instance_ids=["sess-1", "sess-2", "sess-3"],
            )
        before = {"id": "hn", "steps": [{"step_id": 1, "action": "Open HN"}],
                  "evidence": {"observations": [{"session_id": f"sess-{i}"} for i in (1, 2, 3)]},
                  "knowledge": {}}
        proc = synth.merge_insights_into_procedure(before, insights)
        assert before["knowledge"] == {}  # the input is not mutated
        decisions = derive_knowledge(proc)["decisions"]
        assert decisions["status"] == "hypothesis"
        assert decisions["source"] == "contrast"
        assert decisions["value_ref"] == "selection_criteria"
        assert {e["id"] for e in decisions["evidence"] if e["kind"] == "instance"} == {
            "sess-1", "sess-2", "sess-3",
        }
        # The packet says the criterion is not confirmed.
        skill = render_body({**proc, "title": "HN"})
        assert ("- Choose by: Stories with at least 250 points (unconfirmed: seen in 3 "
                "choices, 0 skipped entries also matched; confirm with the user before "
                "relying on it)") in skill


class _SessionDB:
    def __init__(self, sessions: dict[str, list[dict]]):
        self.sessions = sessions

    def get_focus_session_events(self, session_id):
        return self.sessions.get(session_id, [])


class TestRecordedObservations:
    def test_focus_sessions_become_rich_timelines_under_their_ids(self):
        ann = {"app": "Chrome", "task_context": {"what_doing": "Picking"},
               "visible_items": _list([("A", {"point": 5}, True), ("B", {}, False)])}
        db = _SessionDB({"s1": [{"scene_annotation_json": json.dumps(ann)},
                                {"scene_annotation_json": ""}]})
        proc = {"steps": [{"step_id": 1}], "evidence": {"observations": [
            {"session_id": "s1"}, {"session_id": "gone"}, {"obs_id": 7},
        ]}}
        observations, ids = recorded_observations(db, proc)
        assert ids == ["s1", "", ""]
        assert observations[0][0]["list_choice"][0]["label"] == "A"
        # Records without events fall back to the steps and cite nothing.
        assert observations[1] == observations[2] == [{"step_id": 1}]
