#!/usr/bin/env python3
"""The product version, kept identical in every file that carries it.

    python3 scripts/version.py check        print the version, or list the
                                            files that disagree and exit 1
    python3 scripts/version.py set 0.5.0    stamp 0.5.0 into every file

The CLI, daemon, worker, app and extension each read their version from
their own manifest (Cargo workspace, worker/pyproject.toml, the app's
Info.plist, the extension's manifest). `agenthandover doctor` compares the
CLI's version with the worker's, and build-pkg.sh names the package after
it, so they must agree. A version is MAJOR.MINOR.PATCH only: the app's
CFBundleShortVersionString and Chrome's manifest "version" accept dotted
integers only, and Python normalises "0.5.0-dev" to "0.5.0.dev0", which
would never equal the CLI's "0.5.0-dev".

The Homebrew cask is stamped at release time together with its sha256, so
it is not listed here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

_CARGO_CRATES = ("cli", "common", "daemon", "storage")

# (path relative to the repo root, pattern whose group "v" is the version,
#  how many matches the file must have). Every match is read and stamped.
FIELDS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    (
        "Cargo.toml",
        re.compile(r'(?ms)^\[workspace\.package\]\n(?:(?!^\[).)*?^version = "(?P<v>[^"]*)"'),
        1,
    ),
    (
        "Cargo.lock",
        re.compile(
            r'(?m)^name = "agenthandover-(?:' + "|".join(_CARGO_CRATES) + r')"\nversion = "(?P<v>[^"]*)"'
        ),
        len(_CARGO_CRATES),
    ),
    (
        "worker/pyproject.toml",
        re.compile(r'(?ms)^\[project\]\n(?:(?!^\[).)*?^version = "(?P<v>[^"]*)"'),
        1,
    ),
    (
        "app/AgentHandoverApp/Sources/AgentHandoverApp/Info.plist",
        re.compile(
            r"<key>CFBundle(?:Short)?Version(?:String)?</key>\s*<string>(?P<v>[^<]*)</string>"
        ),
        2,
    ),
    ("extension/manifest.json", re.compile(r'(?m)^  "version": "(?P<v>[^"]*)"'), 1),
    ("extension/package.json", re.compile(r'(?m)^  "version": "(?P<v>[^"]*)"'), 1),
    (
        "extension/package-lock.json",
        # The lockfile's own version and its root package's (packages -> "").
        re.compile(r'(?m)^( *)"name": "agenthandover-extension",\n\1"version": "(?P<v>[^"]*)"'),
        2,
    ),
)


def read_versions(root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """Each file's version strings, in file order. Raises ValueError when a
    file does not have the expected number of version fields (the file's
    layout changed and this table must follow)."""
    found: dict[str, list[str]] = {}
    for rel, pattern, count in FIELDS:
        text = (root / rel).read_text(encoding="utf-8")
        values = [m.group("v") for m in pattern.finditer(text)]
        if len(values) != count:
            raise ValueError(f"{rel}: expected {count} version field(s), found {len(values)}")
        found[rel] = values
    return found


def check(root: Path = REPO_ROOT) -> str:
    """The one version every file carries. Raises ValueError naming the
    disagreeing files otherwise."""
    found = read_versions(root)
    distinct = {v for values in found.values() for v in values}
    if len(distinct) != 1:
        lines = [f"  {rel}: {', '.join(values)}" for rel, values in found.items()]
        raise ValueError("versions disagree:\n" + "\n".join(lines))
    return distinct.pop()


def stamp(version: str, root: Path = REPO_ROOT) -> None:
    """Write ``version`` into every version field."""
    if not VERSION_RE.match(version):
        raise ValueError(f"{version!r} is not MAJOR.MINOR.PATCH")
    read_versions(root)  # every file has its fields before any is written

    def _replace(m: re.Match[str]) -> str:
        start, end = m.span("v")
        return m.group(0)[: start - m.start()] + version + m.group(0)[end - m.start():]

    for rel, pattern, _count in FIELDS:
        path = root / rel
        path.write_text(pattern.sub(_replace, path.read_text(encoding="utf-8")), encoding="utf-8")


def main(argv: list[str]) -> int:
    try:
        if argv == ["check"]:
            print(check())
            return 0
        if len(argv) == 2 and argv[0] == "set":
            stamp(argv[1])
            print(check())
            return 0
    except (ValueError, OSError) as exc:
        print(f"version.py: {exc}", file=sys.stderr)
        return 1
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
