"""Chosen versus visible: the entries of a list the user picks from, the
one they pick, and the selection criteria that separate the two.

Two halves:

* :func:`ground_visible_items` fills an annotation's ``visible_items``
  (``[{label, attrs, chosen}]``).  When the browser's DOM snapshot marks
  the element the user clicked and that element is one of the page's list
  rows (the snapshot's ``digest``), the rows come from the DOM, in page
  order, the clicked one ``chosen``.  Otherwise the annotation model's own
  entries stand.  Either way each entry's ``attrs`` are the numbers read
  from its text and from its OCR line (points, comments, price, age, and
  for DOM rows the position in the list).
* :func:`click_choice` is the same list for a click the extension reported
  (a ``ClickIntent``): a link that opens another page never reaches a
  snapshot's ``interacted`` anchor (the page it was on is gone before the
  next snapshot), so the pick is the click's target found among the rows
  of the snapshot taken before it.
* :func:`criteria_hypotheses` compares chosen with skipped entries across
  the recorded instances of a procedure and returns the criteria the
  observations support: a numeric threshold (``have at least 100 points``,
  ``are at most 2 hours old``) or a keyword the chosen entries share.  Each
  carries its support (choices it explains), its counter-examples and an
  ``instance`` evidence pointer per run it was seen in.  The behavioral
  synthesizer only phrases these; it never adds criteria of its own, and
  every criterion is a ``hypothesis`` until the user confirms it.
"""

from __future__ import annotations

import re

from agenthandover_worker.dom import accessible_name, flatten_dom, implicit_role, node_tag

#: At most this many entries per list: the chosen one and the rest in
#: page order.  The lists reach the SOP and frame-diff prompts.
MAX_VISIBLE_ITEMS = 8

#: A criterion needs this many choices (distinct picks) that it explains.
MIN_SUPPORT = 2

#: At most this many criteria per procedure, strongest first.
MAX_CRITERIA = 5

_ROW_ROLES = frozenset({"listitem", "row", "option", "treeitem", "article"})
_ROW_TAGS = frozenset({"li", "tr"})

_AGE_MINUTES = {
    "minute": 1, "min": 1, "hour": 60, "hr": 60, "day": 1440,
    "week": 10080, "month": 43200, "year": 525600,
}
_AGE = re.compile(
    r"\b(\d+|an?|one)\s*(minute|min|hour|hr|day|week|month|year)s?\s+ago\b", re.IGNORECASE,
)
_MONEY = re.compile(r"[$€£]\s?(\d[\d,]*(?:\.\d+)?)")
_COUNT = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:\s?([km])\b)?\s*(%|[a-z]{3,})", re.IGNORECASE,
)
_MULTIPLIER = {"k": 1_000, "m": 1_000_000}
# Words after a number that are not a countable property of the entry.
_NOT_UNITS = frozenset(set(_AGE_MINUTES) | {
    "minutes", "hours", "days", "weeks", "months", "years", "mins", "hrs", "ago",
    "seconds", "secs", "sec", "second", "and", "the", "for", "from", "with",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
})
_WORD = re.compile(r"[a-z][a-z'-]{2,}")
_STOPWORDS = frozenset("""
    the and for with from that this your you are was were has have had not but
    all any can its into our out over new how what why who when where which
    more most than then them they their there about after before just like
    will would should could via per one two use using get got
""".split())


def _norm(text: object) -> str:
    return " ".join(str(text or "").split()).lower()


# ---------------------------------------------------------------------------
# Attributes of one entry
# ---------------------------------------------------------------------------


def item_attrs(text: str) -> dict[str, float]:
    """The numbers *text* states about a list entry: ``age_minutes``
    (``3 hours ago``), ``price`` (``$12.99``) and counts by their unit
    (``182 points`` → ``point``, ``1.2k views`` → ``view``, ``40%`` →
    ``%``).  The first value per key wins."""
    attrs: dict[str, float] = {}
    rest = str(text or "")
    match = _AGE.search(rest)
    if match:
        amount = match.group(1).lower()
        count = 1 if amount in ("a", "an", "one") else int(amount)
        attrs["age_minutes"] = float(count * _AGE_MINUTES[match.group(2).lower()])
        rest = rest[:match.start()] + " " + rest[match.end():]
    match = _MONEY.search(rest)
    if match:
        attrs["price"] = float(match.group(1).replace(",", ""))
        rest = rest[:match.start()] + " " + rest[match.end():]
    for match in _COUNT.finditer(rest):
        unit = match.group(3).lower()
        if unit in _NOT_UNITS:
            continue
        if len(unit) > 3 and unit.endswith("s"):
            unit = unit[:-1]
        value = float(match.group(1).replace(",", ""))
        if match.group(2):
            value *= _MULTIPLIER[match.group(2).lower()]
        attrs.setdefault(unit, value)
    return attrs


def _line_attrs(label: str, lines: list[str], labels: list[str]) -> dict[str, float]:
    """*label*'s attributes, from the OCR line that holds it and the line
    after it (a list's subtext: points, age), unless that next line is
    another entry of the list."""
    key = _norm(label)
    others = {_norm(other) for other in labels if _norm(other) != key}
    for i, line in enumerate(lines):
        low = _norm(line)
        if key and (key in low or (len(low) >= 8 and low in key)):
            text = line
            if i + 1 < len(lines):
                nxt = _norm(lines[i + 1])
                if not any(o and (o in nxt or (len(nxt) >= 8 and nxt in o)) for o in others):
                    text += " " + lines[i + 1]
            return item_attrs(text)
    return item_attrs(label)


# ---------------------------------------------------------------------------
# visible_items of one frame
# ---------------------------------------------------------------------------


def _is_row(node: dict) -> bool:
    return implicit_role(node) in _ROW_ROLES or node_tag(node)[0] in _ROW_TAGS


def _tree_rows(nodes: object) -> list[str]:
    """The list rows of an earlier extension's snapshot tree (``children``
    nested, text per node), which has no ``digest``: the text of each row
    that holds no other row, in document order.  A row's text is its own
    ``text``/``innerText``, else its descendants' joined.  Empty for an
    anchor set (a flat list without ``children``)."""
    rows: list[str] = []

    def text_of(node: dict) -> str:
        own = " ".join(str(node.get("text") or node.get("innerText") or "").split())
        if own:
            return own
        return " ".join(
            text for child in node.get("children") or []
            if isinstance(child, dict) and (text := text_of(child))
        )

    def walk(node: dict) -> bool:
        """Collect the rows under *node*; True when it is or holds one."""
        if str(node.get("tag") or "").startswith("#"):
            return False
        holds_row = False
        for child in node.get("children") or []:
            if isinstance(child, dict) and walk(child):
                holds_row = True
        if not _is_row(node):
            return holds_row
        if not holds_row and (label := text_of(node)):
            rows.append(label)
        return True

    if not isinstance(nodes, list) or not any(
        isinstance(n, dict) and isinstance(n.get("children"), list) for n in nodes
    ):
        return []
    for node in nodes:
        if isinstance(node, dict):
            walk(node)
    return rows


def _row_of(lines: list[str], text: object) -> int | None:
    """The row *text* (the clicked element's text or name) identifies:
    the longest row equal to it or contained in it, else the one row that
    contains it (a link inside a row whose label adds a rank or a site)."""
    low = _norm(text)
    if not low:
        return None
    hits = [
        i for i, line in enumerate(lines)
        if _norm(line) == low or (len(_norm(line)) >= 8 and _norm(line) in low)
    ]
    if hits:
        return max(hits, key=lambda i: len(lines[i]))
    holding = [i for i, line in enumerate(lines) if len(low) >= 8 and low in _norm(line)]
    return holding[0] if len(holding) == 1 else None


def page_rows(page: dict | None, click: dict | None = None) -> list[dict]:
    """The list the user picked from on *page* (``{"nodes", "digest"}``, see
    :func:`scene_annotator.page_for_event`), from the DOM: its rows in page
    order (the ``digest``'s lines, or an earlier extension's tree rows),
    the clicked one ``chosen``, each with its 1-based ``position``; at most
    :data:`MAX_VISIBLE_ITEMS` (the chosen row and the first others).

    The clicked element is *click* (a ``ClickIntent``'s ``target``:
    ``innerText``, ``ariaLabel``) when given, else the element the snapshot
    marks ``interacted``.  Empty unless that element is, or sits in, one of
    the rows."""
    page = page or {}
    lines = [line.strip() for line in str(page.get("digest") or "").splitlines() if line.strip()]
    if not lines:
        lines = _tree_rows(page.get("nodes"))
    if len(lines) < 2:
        return []
    if click is not None:
        candidates = [click.get("innerText"), click.get("ariaLabel")]
    else:
        nodes = list(flatten_dom(page.get("nodes")))
        index = next((i for i, n in enumerate(nodes) if n.get("interacted")), None)
        if index is None:
            return []
        # The clicked element, then its enclosing rows (the anchor set
        # lists the element first, then its ancestors, innermost first).
        clicked = nodes[index]
        candidates = [accessible_name(clicked), _norm(clicked.get("text"))]
        candidates += [
            _norm(node.get("text")) or accessible_name(node)
            for node in nodes[index + 1:] if _is_row(node)
        ]
    chosen = next(
        (row for text in candidates if (row := _row_of(lines, text)) is not None), None,
    )
    if chosen is None:
        return []
    others = [i for i in range(len(lines)) if i != chosen][: MAX_VISIBLE_ITEMS - 1]
    return [
        {"label": lines[i], "attrs": {"position": float(i + 1)}, "chosen": i == chosen}
        for i in sorted(others + [chosen])
    ]


def _with_line_attrs(items: list[dict], ocr_text: str) -> list[dict]:
    """*items* with each entry's ``attrs`` read from its OCR line added
    (the entry's own attrs win)."""
    lines = [line for line in str(ocr_text or "").splitlines() if line.strip()]
    labels = [item["label"] for item in items]
    for item in items:
        item["attrs"] = {**_line_attrs(item["label"], lines, labels), **item["attrs"]}
    return items


def ground_visible_items(model_items: object, page: dict | None, ocr_text: str) -> list[dict]:
    """The frame's ``visible_items``: the DOM's rows when the snapshot shows
    which one was clicked (:func:`page_rows`), else the annotation model's
    entries; each ``{label, attrs, chosen}``, ``attrs`` read from the entry
    and its OCR line (:func:`item_attrs`)."""
    items = page_rows(page)
    if not items:
        items = [
            {"label": str(item["label"]), "attrs": {}, "chosen": bool(item.get("chosen"))}
            for item in (model_items if isinstance(model_items, list) else [])
            if isinstance(item, dict) and str(item.get("label") or "").strip()
        ]
    return _with_line_attrs(items, ocr_text)


def click_choice(page: dict | None, click: dict | None, ocr_text: str) -> list[dict]:
    """The list a reported click picked from: *page*'s rows (the snapshot
    of the page the click was made on) with the row *click* (a
    ``ClickIntent``'s ``target``) hit ``chosen`` (:func:`page_rows`), each
    entry's ``attrs`` read from its OCR line in *ocr_text* (the screen the
    click was made on).  Empty when the click was not on a list row."""
    if not isinstance(click, dict):
        return []
    return _with_line_attrs(page_rows(page, click), ocr_text)


# ---------------------------------------------------------------------------
# Criteria across instances
# ---------------------------------------------------------------------------


def _choices(instances: list[dict]) -> list[tuple[str, list[dict], list[dict]]]:
    """``(instance id, chosen, skipped)`` per distinct pick.  A list seen in
    several frames with the same pick counts once; an entry the user
    picked anywhere in the instance is never counted as skipped in it."""
    choices: list[tuple[str, list[dict], list[dict]]] = []
    for instance in instances:
        iid = str(instance.get("id") or "")
        lists = [
            [i for i in items if isinstance(i, dict) and _norm(i.get("label"))]
            for items in instance.get("lists") or [] if isinstance(items, list)
        ]
        picked = {_norm(i["label"]) for items in lists for i in items if i.get("chosen")}
        seen: set[frozenset] = set()
        for items in lists:
            chosen = [i for i in items if i.get("chosen")]
            skipped = [i for i in items if _norm(i["label"]) not in picked]
            key = frozenset(_norm(i["label"]) for i in chosen)
            if not iid or not chosen or not skipped or key in seen:
                continue
            seen.add(key)
            choices.append((iid, chosen, skipped))
    return choices


def _age(minutes: float) -> str:
    for unit, size in (("day", 1440), ("hour", 60)):
        if minutes >= size and minutes % size == 0:
            n = int(minutes // size)
            return f"{n} {unit}{'s' if n != 1 else ''}"
    return f"{minutes:g} minutes"


def _phrase(key: str, op: str, value: float) -> str:
    """The criterion as a predicate of an entry ("entries that ..."), e.g.
    ``have at least 120 points``."""
    bound = "at least" if op == ">=" else "at most"
    if key == "age_minutes":
        return f"are {bound} {_age(value)} old"
    if key == "price":
        return f"cost {bound} {value:g}"
    if key == "position":
        return (f"are among the first {value:g} entries of the list" if op == "<="
                else f"are at position {value:g} or later in the list")
    if key == "%":
        return f"show {bound} {value:g}%"
    return f"have {bound} {value:g} {key}s"


def _evidence(ids: list[str]) -> list[dict]:
    return [{"kind": "instance", "id": iid} for iid in dict.fromkeys(ids)]


def _examples(chosen_lists: list[list[dict]]) -> list[str]:
    labels = [str(i["label"]) for chosen in chosen_lists for i in chosen]
    return list(dict.fromkeys(labels))[:3]


def _numeric(choices: list[tuple[str, list[dict], list[dict]]]) -> list[tuple[int, dict]]:
    keys = sorted({
        k for _, chosen, skipped in choices for i in chosen + skipped
        for k in (i.get("attrs") or {})
    })
    found: list[tuple[int, dict]] = []
    for key in keys:
        comparable = [
            (iid, chosen, [i["attrs"][key] for i in chosen if key in (i.get("attrs") or {})],
             [i["attrs"][key] for i in skipped if key in (i.get("attrs") or {})])
            for iid, chosen, skipped in choices
        ]
        comparable = [c for c in comparable if c[2] and c[3]]
        if len(comparable) < MIN_SUPPORT:
            continue
        best: tuple[int, dict] | None = None
        for op in (">=", "<="):
            if op == ">=":
                value = min(min(c[2]) for c in comparable)
                excluded = [c for c in comparable if any(x < value for x in c[3])]
                rejected = sum(x < value for c in comparable for x in c[3])
                admitted = sum(x >= value for c in comparable for x in c[3])
            else:
                value = max(max(c[2]) for c in comparable)
                excluded = [c for c in comparable if any(x > value for x in c[3])]
                rejected = sum(x > value for c in comparable for x in c[3])
                admitted = sum(x <= value for c in comparable for x in c[3])
            if len(excluded) < MIN_SUPPORT or rejected <= admitted:
                continue
            hypothesis = {
                "criterion": _phrase(key, op, value),
                "attribute": key, "op": op, "value": value,
                "support": len(excluded), "counter": admitted,
                "evidence": _evidence([c[0] for c in excluded]),
                "examples": _examples([c[1] for c in excluded]),
            }
            score = rejected - admitted
            if best is None or score > best[0]:
                best = (score, hypothesis)
        if best is not None:
            found.append(best)
    return found


def _words(text: object) -> set[str]:
    return {w.strip("'-") for w in _WORD.findall(_norm(text))} - _STOPWORDS - {""}


def _keywords(choices: list[tuple[str, list[dict], list[dict]]]) -> list[tuple[int, dict]]:
    words = sorted({w for _, chosen, _ in choices for i in chosen for w in _words(i["label"])})
    found: list[tuple[int, dict]] = []
    for word in words:
        hits = [(iid, [i for i in chosen if word in _words(i["label"])])
                for iid, chosen, _ in choices]
        hits = [h for h in hits if h[1]]
        chosen_count = sum(len(h[1]) for h in hits)
        counter = sum(word in _words(i["label"]) for _, _, skipped in choices for i in skipped)
        if len(hits) < MIN_SUPPORT or chosen_count <= counter:
            continue
        found.append((chosen_count - counter, {
            "criterion": f'mention "{word}"',
            "attribute": "keyword", "op": "contains", "value": word,
            "support": len(hits), "counter": counter,
            "evidence": _evidence([h[0] for h in hits]),
            "examples": _examples([h[1] for h in hits]),
        }))
    return found


def criteria_hypotheses(instances: list[dict]) -> list[dict]:
    """Selection criteria that separate the chosen from the skipped entries.

    *instances* are the recorded runs of one procedure, each
    ``{"id": <session or observation id>, "lists": [visible_items, ...]}``
    with the ``visible_items`` of its frames that have a chosen entry.

    Returns at most :data:`MAX_CRITERIA` dicts, strongest first::

        {"criterion": "have at least 120 points",   # entries that ...
         "attribute": "point", "op": ">=", "value": 120.0,
         "support": 3,      # picks whose list it separates
         "counter": 1,      # skipped entries that also meet it
         "evidence": [{"kind": "instance", "id": ...}, ...],
         "examples": ["<chosen labels>"],
         "status": "hypothesis"}

    A numeric criterion's threshold is the weakest chosen value; it is kept
    when it excludes skipped entries in at least :data:`MIN_SUPPORT` picks
    and excludes more skipped entries than it admits.  A keyword is kept
    when chosen entries in at least :data:`MIN_SUPPORT` picks mention it
    and more chosen than skipped entries do.  An instance without an id
    contributes nothing (a criterion must cite where it was seen).
    """
    choices = _choices(instances)
    if len(choices) < MIN_SUPPORT:
        return []
    ranked = sorted(
        _numeric(choices) + _keywords(choices),
        key=lambda pair: (-pair[0], -pair[1]["support"], pair[1]["criterion"]),
    )
    return [{**hypothesis, "status": "hypothesis"} for _, hypothesis in ranked[:MAX_CRITERIA]]
