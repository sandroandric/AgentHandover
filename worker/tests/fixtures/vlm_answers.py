"""SOP templates built by the real generator from stored model answers.

Test fixtures that need an SOP template or a procedure must come from
``sop_generator._vlm_sop_to_template`` run on an answer in the shape
``FOCUS_SOP_PROMPT`` requests, never from a hand-written step dict.  A
hand-written step shape drifted from the generator once (the generator
writes app/location/input/verify under ``parameters``; the fixtures put
them at the top level), which hid the field loss in ``sop_to_procedure``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from agenthandover_worker.sop_generator import _vlm_sop_to_template

ANSWERS_DIR = Path(__file__).parent / "vlm_sop_answers"


def load_answer(name: str) -> dict:
    """Return the stored model answer ``vlm_sop_answers/<name>.json``."""
    return json.loads((ANSWERS_DIR / f"{name}.json").read_text(encoding="utf-8"))


def generated_sop_template(
    name: str,
    *,
    mode: str = "focus",
    answer: dict | None = None,
) -> dict:
    """Run the generator's answer-to-template step on a stored answer.

    ``answer`` overrides the stored answer (pass a modified copy of
    ``load_answer(name)`` to test a degraded model answer).
    """
    raw = copy.deepcopy(answer) if answer is not None else load_answer(name)
    return _vlm_sop_to_template(raw, mode=mode, confidence=0.9)
