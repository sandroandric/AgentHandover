"""One renderer for every Skill packet an agent receives.

Every export target (Claude Code skills and slash commands, Hermes,
OpenClaw, the generic filesystem export, the MCP server and Codex's
``AGENTS.md``) renders a v3 procedure through :func:`render_packet`.
Targets differ only in their frontmatter; the body is the same text for
all of them, so an agent reads the same instructions whichever way it
was connected.

The packet is built from three pure functions:

- :func:`frontmatter` — the per-target metadata dict,
- :func:`body_sections` — the markdown sections of the body,
- :func:`serialize_frontmatter` — YAML via ``yaml.safe_dump`` so titles
  and descriptions with colons, quotes or newlines stay valid YAML.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal

import yaml

from agenthandover_worker import __version__

Target = Literal[
    "claude-skill",
    "claude-command",
    "hermes",
    "openclaw",
    "generic",
    "mcp",
    "codex",
]

TARGETS: tuple[str, ...] = (
    "claude-skill",
    "claude-command",
    "hermes",
    "openclaw",
    "generic",
    "mcp",
    "codex",
)

# ---- Tool derivation -------------------------------------------------------
#
# ``allowed-tools`` is derived from the apps the steps run in.  A tool is
# granted only when a step's app implies it; ``Bash`` is granted only for
# terminal apps, never as a blanket default.

_BROWSER_APPS = frozenset({
    "chrome", "google chrome", "firefox", "safari", "brave", "edge",
    "microsoft edge", "arc", "opera", "vivaldi", "chromium",
})
_TERMINAL_APPS = frozenset({
    "terminal", "iterm", "iterm2", "warp", "alacritty", "kitty", "hyper",
    "ghostty", "wezterm",
})
_EDITOR_APPS = frozenset({
    "vs code", "vscode", "visual studio code", "code", "cursor", "intellij",
    "intellij idea", "pycharm", "webstorm", "neovim", "vim", "emacs",
    "sublime text", "xcode", "android studio", "zed", "textedit",
})
_FILESYSTEM_APPS = frozenset({"finder", "file explorer", "nautilus", "files"})

_TOOL_ORDER = ("Bash", "Read", "Write", "Edit", "Grep", "Glob", "WebFetch")

# Step fields rendered on their own labelled lines; every other key in a
# step's ``parameters`` is rendered verbatim after them.
_STEP_FIELDS = ("app", "location", "target", "input", "verify")
_SKIPPED_PARAMS = frozenset(_STEP_FIELDS) | {"app_id"}


def _tools_for_app(app: str) -> set[str]:
    name = app.lower().strip()
    if name in _TERMINAL_APPS or any(t in name for t in ("terminal", "shell", "console")):
        return {"Bash"}
    if name in _EDITOR_APPS or name.endswith(" ide"):
        return {"Read", "Edit", "Grep", "Glob"}
    if name in _BROWSER_APPS or "browser" in name:
        return {"WebFetch"}
    if name in _FILESYSTEM_APPS:
        return {"Read", "Write", "Glob"}
    return set()


def allowed_tools(proc: dict) -> list[str]:
    """Claude Code tools the procedure's apps call for, in a stable order.

    ``Read`` is always granted (reading is how an agent checks its work);
    everything else comes from the apps named on the steps and in
    ``apps_involved``.
    """
    apps = [a for a in proc.get("apps_involved") or [] if isinstance(a, str)]
    apps += [_field(s, "app") for s in _steps(proc)]
    tools = {"Read"}
    for app in apps:
        if app:
            tools |= _tools_for_app(app)
    return [t for t in _TOOL_ORDER if t in tools]


# ---- Small readers -------------------------------------------------------


def _steps(proc: dict) -> list[dict]:
    return [s for s in proc.get("steps") or [] if isinstance(s, dict)]


def _field(step: dict, key: str) -> str:
    """Read *key* from a step's top level, else from its ``parameters``.

    Procedures carry app/location/input/verify at the top level; SOP
    templates from the generator carry them under ``parameters``.  The
    shape decides, never the ``source`` string.
    """
    value = step.get(key)
    if not value:
        params = step.get("parameters")
        if isinstance(params, dict):
            value = params.get(key)
    return str(value).strip() if value else ""


def _first_sentence(text: str) -> str:
    text = " ".join(str(text).split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", text)
    sentence = match.group(1) if match else text
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _overview(proc: dict) -> dict:
    eo = proc.get("execution_overview")
    return eo if isinstance(eo, dict) else {}


def _when_to_use(proc: dict) -> str:
    return str(proc.get("when_to_use") or _overview(proc).get("when_to_use") or "").strip()


def _split_list(value) -> list[str]:
    """Normalise a list or a ``;``-joined string into clean items."""
    if not value:
        return []
    items = value if isinstance(value, list) else str(value).split(";")
    return [str(i).strip() for i in items if str(i).strip()]


def slugify(text: str) -> str:
    """Filesystem- and command-safe slug: lowercase, hyphens, no specials."""
    slug = unicodedata.normalize("NFKD", str(text))
    slug = re.sub(r"[^\w\s-]", "", slug).strip().lower()
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug[:80] if slug else "unknown"


def slug_of(proc: dict) -> str:
    return str(proc.get("id") or proc.get("slug") or "unknown")


def description(proc: dict) -> str:
    """One sentence: goal, description, task description, when to use, title.

    ``description`` is the procedure's own summary (``sop_to_procedure``
    fills it from the task description; the app edits it), so it wins
    over the raw ``task_description``.
    """
    for value in (
        proc.get("goal"), proc.get("description"), proc.get("task_description"),
        _overview(proc).get("goal"),
    ):
        if isinstance(value, str) and value.strip():
            return _first_sentence(value)
    when = _when_to_use(proc)
    if when:
        return _first_sentence(when)
    return _first_sentence(proc.get("title") or "Untitled Skill")


def _is_credential(inp: dict) -> bool:
    return bool(
        inp.get("credential")
        or inp.get("is_credential")
        or inp.get("sensitive")
        or str(inp.get("type", "")).lower() in ("password", "secret", "credential")
    )


def _inputs(proc: dict) -> list[dict]:
    raw = proc.get("inputs") or proc.get("variables") or []
    out: list[dict] = []
    for inp in raw:
        if isinstance(inp, str):
            out.append({"name": inp})
        elif isinstance(inp, dict) and inp.get("name"):
            out.append(inp)
    return out


def version_line() -> str:
    """Footer naming the worker version and the procedure schema."""
    from agenthandover_worker.procedure_schema import PROCEDURE_SCHEMA_VERSION

    return (
        f"Generated by AgentHandover v{__version__} "
        f"(procedure schema {PROCEDURE_SCHEMA_VERSION})"
    )


def body_hash(body: str) -> str:
    """SHA-256 of a packet body, used to detect manual edits on disk.

    Normalises Unicode to NFC and line endings so the hash is stable
    across platforms.
    """
    normalized = unicodedata.normalize(
        "NFC", body.strip("\n").replace("\r\n", "\n"),
    )
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---- The three pure functions -------------------------------------------


def frontmatter(proc: dict, target: Target) -> dict:
    """Per-target frontmatter.  Codex sections carry none."""
    if target not in TARGETS:
        raise ValueError(f"unknown render target: {target!r}")
    if target == "codex":
        return {}

    fm: dict = {}
    if target != "claude-command":
        fm["name"] = slugify(slug_of(proc))
    fm["description"] = description(proc)

    if target in ("claude-skill", "claude-command"):
        names = [i["name"] for i in _inputs(proc)]
        if names:
            fm["argument-hint"] = " ".join(f"[{n}]" for n in names)
        fm["allowed-tools"] = ", ".join(allowed_tools(proc))
    elif target in ("openclaw", "generic"):
        confidence = float(proc.get("confidence_avg") or 0.0)
        fm["title"] = proc.get("title") or "Untitled"
        fm["generated_by"] = f"agenthandover v{__version__}"
        fm["confidence_score_avg"] = round(confidence, 4)
        fm["apps_involved"] = list(proc.get("apps_involved") or [])
        fm["tags"] = list(proc.get("tags") or [])
    return fm


def serialize_frontmatter(fm: dict) -> str:
    """Render *fm* as a ``---`` fenced YAML block (empty dict → "")."""
    if not fm:
        return ""
    text = yaml.safe_dump(
        fm, sort_keys=False, allow_unicode=True, default_flow_style=False,
        width=10_000,
    )
    return f"---\n{text}---\n"


def _precondition(text: str) -> str:
    """Readable form of the miner's ``app_open:`` / ``url_open:`` preconditions."""
    for prefix, template in (("app_open:", "{} is open"), ("url_open:", "{} is open in the browser")):
        if text.startswith(prefix):
            return template.format(text[len(prefix):])
    return text


def _section(heading: str, lines: list[str]) -> str | None:
    if not lines:
        return None
    return "\n".join([f"## {heading}", "", *lines])


def _render_step(num: int, step: dict) -> list[str]:
    step_id = step.get("step_id") or f"step_{num}"
    intent = str(
        step.get("action") or step.get("step") or step.get("description") or "Perform the action"
    ).strip()
    lines = [f"{num}. [{step_id}] {intent}"]
    detail = str(step.get("description") or "").strip()
    if detail and detail != intent:
        lines.append(f"   Detail: {detail}")
    app = _field(step, "app")
    target = _field(step, "location") or _field(step, "target")
    for label, value in (
        ("App", app),
        ("Target", target),
        ("Input", _field(step, "input")),
        ("Verify", _field(step, "verify")),
        ("Anchor", f"`{step['selector']}`" if step.get("selector") else ""),
    ):
        if value:
            lines.append(f"   {label}: {value}")
    # Any other observed parameter (DOM-derived steps carry e.g. ``value``).
    params = step.get("parameters")
    if isinstance(params, dict):
        for key, value in params.items():
            if key not in _SKIPPED_PARAMS and value not in (None, "", [], {}):
                lines.append(f"   {key}: {value}")
    on_failure = step.get("on_failure")
    if isinstance(on_failure, dict) and on_failure.get("strategy") not in (None, "", "abort"):
        retries = on_failure.get("max_retries", 0)
        lines.append(f"   On failure: {on_failure['strategy']} (max retries {retries})")
    return lines


def _voice_lines(proc: dict) -> list[str]:
    """Voice and style guidance, when there is enough evidence for it."""
    vp = proc.get("voice_profile") or {}
    samples = proc.get("content_samples") or []
    lines: list[str] = []
    confident = not (
        vp.get("style_confidence", "low") == "low" and vp.get("sample_count", 0) < 3
    )
    if (vp or samples) and confident:
        lines.append("When producing text for this workflow, match the user's writing style:")
        tone = vp.get("tone") or vp.get("formality")
        for label, value in (
            ("Tone", tone),
            ("Sentences", vp.get("sentence_style")),
            ("Vocabulary", vp.get("vocabulary")),
            ("Personality", ", ".join(vp.get("personality_markers") or [])),
            ("Would say", f"\"{vp['would_say']}\"" if vp.get("would_say") else ""),
            ("Would never say", f"\"{vp['would_never_say']}\"" if vp.get("would_never_say") else ""),
        ):
            if value:
                lines.append(f"- {label}: {value}")
        texts = [s.get("text", "") for s in samples[:3] if isinstance(s, dict) and s.get("text")]
        if texts:
            lines.append("")
            lines.append("Example text from the user:")
            for text in texts:
                lines.append("> " + " ".join(str(text)[:200].split()))
    for ct in proc.get("content_templates") or []:
        if isinstance(ct, dict) and ct.get("template"):
            line = f"- Output template: {ct['template']}"
            if ct.get("variables"):
                line += f" (variables: {', '.join(ct['variables'])})"
            lines.append(line)
    return lines


def body_sections(proc: dict) -> list[str]:
    """The packet body as a list of markdown sections, identical for every target."""
    slug = slug_of(proc)
    title = str(proc.get("title") or "Untitled Skill")
    sections: list[str | None] = []

    intro = [f"# {title}"]
    summary = next(
        (
            v.strip()
            for v in (
                proc.get("goal"), proc.get("description"),
                proc.get("task_description"), _overview(proc).get("goal"),
            )
            if isinstance(v, str) and v.strip()
        ),
        "",
    )
    if summary:
        intro += ["", summary]
    outcome = str(proc.get("outcome") or "").strip()
    if outcome:
        intro += ["", f"Outcome: {outcome}"]
    sections.append("\n".join(intro))

    # Prerequisites
    prereqs: list[str] = []
    for item in (
        _split_list(proc.get("prerequisites"))
        + [_precondition(p) for p in _split_list(proc.get("preconditions"))]
        + _split_list(_overview(proc).get("prerequisites"))
    ):
        if item not in prereqs:
            prereqs.append(item)
    env = proc.get("environment") or {}
    for acct in env.get("accounts") or []:
        if isinstance(acct, dict):
            svc = acct.get("service", "unknown")
            ident = acct.get("identity", "")
            prereqs.append(f"Signed in to {svc}" + (f" as {ident}" if ident else ""))
    prereqs += [f"Setup: {a}" for a in env.get("setup_actions") or []]
    chain = proc.get("chain") or {}
    prereqs += [f"Run the `{d}` Skill first" for d in chain.get("depends_on") or []]
    sections.append(_section("Prerequisites", [f"- {p}" for p in prereqs]))

    # When to use / when not
    when = [f"- {w}" for w in _split_list(_when_to_use(proc))]
    strategy = str(proc.get("strategy") or "").strip()
    if strategy:
        when.append(f"- Strategy: {strategy}")
    elif proc.get("evidence_summary"):
        when.append(f"- Observed pattern: {proc['evidence_summary']}")
    when += [f"- Usually followed by the `{f}` Skill" for f in chain.get("followed_by") or []]
    sections.append(_section("When to use", when))
    when_not = _split_list(proc.get("when_not_to_use"))
    sections.append(_section("When not to use", [f"- {w}" for w in when_not]))

    # Inputs (credentials by reference only)
    input_lines: list[str] = []
    for inp in _inputs(proc):
        name = inp["name"]
        if _is_credential(inp):
            input_lines.append(
                f"- `{name}` (credential): take it from your credential store at "
                "run time; never write its value into files, messages or logs."
            )
            continue
        kind = str(inp.get("type") or "text")
        required = "required" if inp.get("required", True) else "optional"
        line = f"- `{name}` ({kind}, {required})"
        if inp.get("description"):
            line += f": {inp['description']}"
        extras = []
        if inp.get("example"):
            extras.append(f"example `{inp['example']}`")
        if inp.get("default") not in (None, ""):
            extras.append(f"default `{inp['default']}`")
        if inp.get("choices"):
            extras.append("one of " + ", ".join(f"`{c}`" for c in inp["choices"]))
        if extras:
            line += " (" + "; ".join(extras) + ")"
        input_lines.append(line)
    if input_lines:
        input_lines.append("")
        input_lines.append("Steps refer to inputs as `{{name}}`.")
    sections.append(_section("Inputs", input_lines))

    # Steps
    step_lines: list[str] = []
    for num, step in enumerate(_steps(proc), 1):
        step_lines.extend(_render_step(num, step))
    sections.append(_section("Steps", step_lines or ["No steps were recorded."]))

    # Success criteria and common errors
    success = _split_list(_overview(proc).get("success_criteria"))
    success += [
        str(o.get("description") or o.get("type") or "") if isinstance(o, dict) else str(o)
        for o in proc.get("expected_outcomes") or []
    ]
    sections.append(_section("Success criteria", [f"- {s}" for s in success if s]))
    errors = _split_list(_overview(proc).get("common_errors"))
    sections.append(_section("Common errors", [f"- {e}" for e in errors]))

    # Decisions
    decisions: list[str] = []
    for br in proc.get("branches") or []:
        if not isinstance(br, dict):
            continue
        if br.get("condition"):
            decisions.append(f"- If {br['condition']}: {br.get('action', '')}".rstrip(": "))
        elif br.get("alternatives"):
            alts = ", ".join(
                str(a[0] if isinstance(a, (list, tuple)) else a)
                for a in br["alternatives"]
            )
            pos = br.get("position")
            where = f"At step {pos + 1}" if isinstance(pos, int) else "At one step"
            decisions.append(
                f"- {where}, the usual action is \"{br.get('canonical_action', '')}\"; "
                f"alternatives seen: {alts}"
            )
    for sc in proc.get("selection_criteria") or []:
        if isinstance(sc, dict) and sc.get("criterion"):
            line = f"- Choose by: {sc['criterion']}"
            if sc.get("confidence") is not None:
                line += f" (confidence {float(sc['confidence']):.0%})"
            decisions.append(line)
            decisions += [f"  - Example: {ex}" for ex in (sc.get("examples") or [])[:2]]
    clar = proc.get("agent_clarifications") or []
    if isinstance(clar, dict):
        clar = [{"question": q, "answer": a} for q, a in clar.items()]
    for item in clar:
        if isinstance(item, dict) and item.get("question") and item.get("answer"):
            decisions.append(f"- The user answered \"{item['question']}\": {item['answer']}")
    sections.append(_section("Decisions", decisions))

    # Guardrails
    constraints = proc.get("constraints") or {}
    guards = [f"- {g}" for g in constraints.get("guardrails") or []]
    if constraints.get("trust_level"):
        guards.append(f"- Trust level: {constraints['trust_level']}")
    sections.append(_section("Guardrails", guards))

    # Voice
    sections.append(_section("Voice", _voice_lines(proc)))

    # Timing
    rhythm = proc.get("workflow_rhythm") or {}
    timing: list[str] = []
    if rhythm.get("avg_duration_minutes"):
        timing.append(f"- Typical duration: {float(rhythm['avg_duration_minutes']):.0f} minutes")
    for phase in rhythm.get("phases") or []:
        if isinstance(phase, dict) and phase.get("name"):
            line = f"- {phase['name']} (~{phase.get('typical_duration_minutes', 0)} min)"
            if phase.get("description"):
                line += f": {phase['description']}"
            timing.append(line)
    sections.append(_section("Timing", timing))

    # Evidence links
    evidence = proc.get("evidence") or {}
    ev_lines = [
        f"- Observed {proc.get('episode_count') or evidence.get('total_observations') or 0} "
        f"demonstration(s); confidence {float(proc.get('confidence_avg') or 0.0):.2f}"
    ]
    source = str(proc.get("source") or "")
    if source and source != "unknown":
        ev_lines.append(f"- Learned from: {source}")
    for obs in (evidence.get("observations") or [])[-10:]:
        if isinstance(obs, dict):
            parts = [str(obs.get(k)) for k in ("date", "type") if obs.get(k)]
            if obs.get("session_id"):
                parts.append(f"session `{obs['session_id']}`")
            ev_lines.append("- " + " ".join(parts))
    contradictions = evidence.get("contradictions") or []
    if contradictions:
        ev_lines.append(f"- Contradictions recorded: {len(contradictions)}")
    staleness = proc.get("staleness") or {}
    if staleness.get("last_confirmed"):
        ev_lines.append(f"- Last confirmed: {staleness['last_confirmed']}")
    sections.append(_section("Evidence", ev_lines))

    # Execution protocol
    sections.append(_section("Execution protocol", [
        "If you have access to the AgentHandover MCP server, report your progress:",
        f"1. Call `report_execution_start` with slug `{slug}` before starting.",
        "2. Call `report_step_result` after each step.",
        "3. Call `report_execution_complete` when finished.",
        "",
        "This feedback improves the Skill for future runs.",
    ]))

    return [s for s in sections if s]


def render_body(proc: dict) -> str:
    """Body text shared by every target, footer included."""
    return "\n\n".join(body_sections(proc)) + f"\n\n---\n*{version_line()}*\n"


def render_packet(proc: dict, *, target: Target) -> str:
    """Render *proc* for *target*.

    Every target shares :func:`render_body`.  Codex gets the body as a
    ``###`` section for ``AGENTS.md`` (headings demoted two levels, no
    frontmatter).  OpenClaw's frontmatter also records the body hash so
    a manual edit of the file on disk is detected before it is replaced.
    """
    fm = frontmatter(proc, target)
    body = render_body(proc)
    if target == "codex":
        return re.sub(r"(?m)^(#+) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", body)
    if target == "openclaw":
        fm["generated_body_hash"] = body_hash(body)
    return f"{serialize_frontmatter(fm)}\n{body}"
