"""DOM snapshots from the Chrome extension: one walk over what is stored,
the host a page belongs to, and the anchor an agent should use for a
clicked element.

A snapshot's ``metadata_json`` holds either

* ``anchors``: the current extension's anchor set, a flat list of
  ``{tag, role, name, ariaLabel, testId, selector, href, text}`` records for
  the element interacted with since the previous snapshot (flagged
  ``interacted``) and its ancestors up to the nearest landmark, next to a
  text ``digest`` of the viewport; or
* ``nodes``: the tree earlier extension versions stored (``children``
  nested, ``innerText`` per node), still present in rows inside the
  retention window.

:func:`flatten_dom` walks both, so the consumers never see the difference.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator
from urllib.parse import urlparse

# A capped snapshot ends with a synthetic ``#truncated`` marker node.
_MARKER_PREFIX = "#"

_NAME_MAX_CHARS = 100

# The ARIA role a tag has without an explicit ``role`` attribute
# (HTML-AAM, the subset an agent clicks or reads).
_IMPLICIT_ROLES = {
    "a": "link",
    "button": "button",
    "select": "combobox",
    "textarea": "textbox",
    "option": "option",
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "li": "listitem",
    "ul": "list",
    "ol": "list",
    "tr": "row",
    "td": "cell",
    "th": "columnheader",
    "table": "table",
    "img": "img",
    "nav": "navigation",
    "main": "main",
    "form": "form",
    "dialog": "dialog",
    "summary": "button",
}

_INPUT_ROLES = {
    "": "textbox", "text": "textbox", "email": "textbox", "tel": "textbox",
    "url": "textbox", "password": "textbox", "search": "searchbox",
    "number": "spinbutton", "checkbox": "checkbox", "radio": "radio",
    "range": "slider", "submit": "button", "button": "button",
    "reset": "button", "image": "button",
}

_INPUT_TAG = re.compile(r"^input\[type=([\w-]+)\]$")
_HOSTNAME = re.compile(r"^(localhost|[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}|\d{1,3}(\.\d{1,3}){3})$")


def flatten_dom(nodes: Iterable | None) -> Iterator[dict]:
    """Every element node of a stored snapshot (tree or anchor set), in
    document order, without the ``#truncated`` markers."""
    stack = list(reversed(list(nodes or [])))
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        children = node.get("children")
        if isinstance(children, list):
            stack.extend(reversed(children))
        if str(node.get("tag", "")).startswith(_MARKER_PREFIX):
            continue
        yield node


def node_tag(node: dict) -> tuple[str, str]:
    """``(tag, input type)``: the extension writes ``input[type=email]`` as
    the tag of a typed input."""
    tag = str(node.get("tag") or node.get("tagName") or "").strip().lower()
    match = _INPUT_TAG.match(tag)
    if match:
        return "input", match.group(1)
    return tag, str(node.get("type") or "").lower() if tag == "input" else ""


def implicit_role(node: dict) -> str:
    """The node's ARIA role: its ``role`` attribute, else its tag's."""
    role = str(node.get("role") or "").strip()
    if role:
        return role
    tag, input_type = node_tag(node)
    if tag == "input":
        return _INPUT_ROLES.get(input_type, "textbox")
    return _IMPLICIT_ROLES.get(tag, "")


def _one_line(text: object, limit: int = _NAME_MAX_CHARS) -> str:
    return " ".join(str(text or "").split())[:limit]


def _text(node: dict) -> str:
    return _one_line(node.get("text") or node.get("innerText"))


def accessible_name(node: dict) -> str:
    """The name an agent would read for the node.

    Anchor-set records carry the extension's computed ``name``.  In the
    older tree, ``name`` may be a class-name hint, so visible text wins
    over it there.
    """
    if node.get("ariaLabel"):
        return _one_line(node["ariaLabel"])
    if "selector" in node:  # anchor-set record
        return _one_line(node.get("name")) or _text(node)
    return _text(node) or _one_line(node.get("name"))


def url_host(text: str | None) -> str:
    """The host of a URL, with or without a scheme (``www.`` dropped,
    lowercased), or "" when *text* is not a web location."""
    raw = str(text or "").strip()
    if not raw or " " in raw.split("/", 1)[0]:
        return ""
    try:
        host = urlparse(raw if "://" in raw else "//" + raw).hostname or ""
    except ValueError:
        return ""
    host = host.lower().removeprefix("www.")
    return host if _HOSTNAME.match(host) else ""


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _norm(text: object) -> str:
    return " ".join(str(text or "").split()).lower()


def _match_score(node: dict, target: dict) -> int:
    """How surely *node* is the element *target* (a ClickIntent's
    ``target``) describes: 0 when nothing identifying matches."""
    score = 0
    test_id = target.get("testId")
    if test_id and node.get("testId") == test_id:
        score += 10
    aria = target.get("ariaLabel")
    if aria and node.get("ariaLabel") == aria:
        score += 8
    target_text = _norm(target.get("innerText"))
    node_text = _norm(node.get("text") or node.get("innerText"))
    if target_text and node_text:
        if node_text == target_text:
            score += 6
        elif len(node_text) >= 8 and (
            target_text.startswith(node_text) or node_text.startswith(target_text)
        ):
            score += 3
    if score and node_tag(node)[0] == node_tag(target)[0]:
        score += 1
    return score


def _path_selector(path: list | None, depth: int = 4) -> str:
    """A CSS selector from a click's composed path (target first): the
    target and its nearest ancestors, outermost first."""
    segments = [str(s) for s in (path or [])[:depth] if s]
    return " > ".join(reversed(segments))


def _anchor(node: dict, click_target: dict | None) -> dict | None:
    role = implicit_role(node)
    name = accessible_name(node)
    test_id = node.get("testId")
    aria = node.get("ariaLabel")
    if test_id:
        selector = f"[data-testid={_quote(str(test_id))}]"
    elif aria:
        selector = f"[aria-label={_quote(str(aria))}]"
    elif role and name:
        selector = f"role={role}[name={_quote(name)}]"
    else:
        selector = str(node.get("selector") or "") or _path_selector(
            (click_target or {}).get("composedPath")
        )
    if not (role or name or selector):
        return None
    return {"role": role, "name": name, "selector": selector}


def select_anchor(nodes: Iterable | None, click_target: dict | None) -> dict | None:
    """The clicked element's most stable anchor, ``{"role", "name",
    "selector"}``.

    The element is the snapshot node that matches *click_target* (a
    ClickIntent's ``target``: ``tagName``, ``role``, ``ariaLabel``,
    ``testId``, ``innerText``, ``composedPath``) best; without a click
    target it is the anchor set's ``interacted`` record.  When the snapshot
    does not hold the element, the click target itself is used.  The
    selector prefers ``data-testid``, then ``aria-label``, then role and
    name, then the composed-path selector.  None when there is no element.
    """
    flat = list(flatten_dom(nodes))
    if click_target:
        best, best_score = None, 0
        for node in flat:
            score = _match_score(node, click_target)
            if score > best_score:
                best, best_score = node, score
        return _anchor(best or click_target, click_target)
    for node in flat:
        if node.get("interacted"):
            return _anchor(node, None)
    return None
