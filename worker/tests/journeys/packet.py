"""Read a rendered Skill packet the way an agent does: YAML frontmatter,
then markdown sections, the steps as numbered blocks."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

GOLDEN_DIR = Path(__file__).parent / "golden"
_STEP_HEAD = re.compile(r"^(\d+)\. \[([^\]]+)\] (.*)$")


@dataclass
class PacketStep:
    number: int
    step_id: str
    intent: str
    fields: dict[str, str]
    text: str


@dataclass
class Packet:
    frontmatter: dict
    body: str

    def section(self, heading: str) -> str:
        match = re.search(
            rf"^## {re.escape(heading)}\n(.*?)(?=^## |\n---\n\*Generated|\Z)",
            self.body, re.S | re.M,
        )
        return match.group(1).strip() if match else ""

    def steps(self) -> list[PacketStep]:
        steps: list[PacketStep] = []
        for line in self.section("Steps").splitlines():
            head = _STEP_HEAD.match(line)
            if head:
                steps.append(PacketStep(
                    number=int(head.group(1)), step_id=head.group(2),
                    intent=head.group(3), fields={}, text=line,
                ))
                continue
            if not steps:
                continue
            steps[-1].text += "\n" + line
            label, sep, value = line.strip().partition(": ")
            if sep and label:
                steps[-1].fields[label] = value
        return steps


def assert_matches_golden(name: str, text: str) -> None:
    """``text`` equals ``golden/<name>``.  With ``AH_UPDATE_GOLDEN=1`` the
    golden is rewritten instead (review the diff before committing it)."""
    path = GOLDEN_DIR / name
    if os.environ.get("AH_UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    assert path.is_file(), f"no golden {path}; write it with AH_UPDATE_GOLDEN=1 and review it"
    assert text == path.read_text(encoding="utf-8"), (
        f"packet differs from {path}; if the change is intended, rewrite it "
        "with AH_UPDATE_GOLDEN=1 and review the diff"
    )


def parse_packet(text: str) -> Packet:
    """Split ``---`` YAML frontmatter from the body; the YAML must parse."""
    if not text.startswith("---\n"):
        return Packet(frontmatter={}, body=text)
    end = text.index("\n---\n", 4)
    return Packet(frontmatter=yaml.safe_load(text[4:end]) or {}, body=text[end + 5:])
