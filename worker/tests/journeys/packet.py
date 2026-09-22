"""Read a rendered Skill packet the way an agent does: YAML frontmatter,
then markdown sections, the steps as numbered blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

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


def parse_packet(text: str) -> Packet:
    """Split ``---`` YAML frontmatter from the body; the YAML must parse."""
    if not text.startswith("---\n"):
        return Packet(frontmatter={}, body=text)
    end = text.index("\n---\n", 4)
    return Packet(frontmatter=yaml.safe_load(text[4:end]) or {}, body=text[end + 5:])
