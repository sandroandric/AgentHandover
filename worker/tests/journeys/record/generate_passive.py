"""Generate the ``passive_3weeks`` fixture: three weeks of passive capture
with two planted routines, and its ``labels.json``.

The user works Monday to Friday (UTC, which the manifest names as the local
zone) and is off at weekends.  Planted routines:

* ``stripe-payout-reconciliation`` (daily, every workday around 09:10):
  open yesterday's payout in the Stripe dashboard, enter it in the
  reconciliation Google Sheet, post the total in Slack's #finance.  Variation:
  the order of Stripe and the Sheet swaps, an extra tab (Gmail, Hacker News)
  opens mid-routine, payout ids and amounts differ, and one instance runs in
  the evening (21:05) instead of the morning;
* ``weekly-expense-report`` (Mondays around 10:30): find last week's receipts
  in Gmail, create a report in Expensify, add each receipt as an expense with
  the PDF attached, submit.  The receipts, report ids and their number differ
  per week, and the week-2 instance is interrupted by a 40-minute Zoom call.

Around them: one-off work tasks, never repeated (a spec in Google Docs, a bug
fix in VS Code, ...), and non-work browsing (YouTube, Hacker News, X).

Rows: each annotated frame is a passive ``DwellSnapshot`` from the native
observer (``window_json`` from ``ObservationService``, ``screenshot_path``,
no ``focus_session_id``), and every Chrome step also has the extension's
``ClickIntent`` with the clicked element.  ``annotations.jsonl.gz`` holds what
the scene annotator wrote for each frame (``annotation_status`` and
``scene_annotation_json`` in the annotator's schema); frames carry no pixels,
so the fixture stands for the stream after annotation and the passive
pipeline from there on (segmenter, embeddings, clustering, SOP generation,
daily batch) runs for real.  ``matched_procedure`` is never written: which
procedure a task matches is the production matcher's answer.

``labels.json`` is the ground truth the replay scores against: each routine's
instances with their event ids (including the frames of extra tabs opened
mid-routine, flagged ``extra``), each one-off task, and the noise frames.

Usage (from the repository root)::

    .venv/bin/python worker/tests/journeys/record/generate_passive.py

The output is deterministic: running it twice gives identical files.
"""

from __future__ import annotations

import gzip
import json
import random
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

JOURNEYS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNEYS_DIR.parent))  # worker/tests, for daemon_db

from daemon_db import schema_version  # noqa: E402
from journeys.fixture_loader import (  # noqa: E402
    ANNOTATIONS_FILE,
    FIXTURES_DIR,
    LABELS_FILE,
    SCREENSHOTS_TOKEN,
    row_counts,
    write_manifest,
    write_rows,
)

NAME = "passive_3weeks"
SEED = 20260831
FIRST_DAY = datetime(2026, 8, 31, tzinfo=timezone.utc)  # a Monday
DAYS = 21

DISPLAY = {
    "display_id": "1",
    "bounds_global_px": [0, 0, 1512, 982],
    "scale_factor": 2.0,
    "orientation": 0,
}
APPS = {
    "Google Chrome": ("pid:812:Google Chrome", "4127", [0, 33, 1512, 949]),
    "Visual Studio Code": ("pid:2210:Code", "6021", [0, 33, 1512, 949]),
    "zoom.us": ("pid:3302:zoom.us", "7140", [120, 80, 1280, 800]),
    "Keynote": ("pid:2894:Keynote", "6533", [0, 33, 1512, 949]),
    "Terminal": ("pid:1990:Terminal", "5877", [200, 120, 1100, 720]),
    "Figma": ("pid:2750:Figma", "6310", [0, 33, 1512, 949]),
}

STRIPE = "stripe-payout-reconciliation"
EXPENSES = "weekly-expense-report"
EVENING_DAY = 10     # week 2, Thursday: the payout routine runs at 21:05
INTERRUPTED_DAY = 7  # week 2, Monday: a 40-minute Zoom call mid expense report
SHEET_URL = "https://docs.google.com/spreadsheets/d/1pQx7RkTz0Lm4aVn8sWc2YbE5dJh3uFg/edit#gid=0"
SHEET_TITLE = "Payout reconciliation 2026 - Google Sheets"
FINANCE_SLACK = "https://app.slack.com/client/T04ACME9/C05FINANCE"


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One annotated frame: what the screen showed and what the user did."""

    app: str
    title: str
    url: str
    what_doing: str
    likely_next: str
    is_workflow: bool
    headings: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    active_element: str = ""
    click: dict | None = None   # the element clicked (Chrome steps)
    names: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    typed: str = ""
    extra: bool = False          # an extra tab opened during a routine


def rfc3339(ts: datetime) -> str:
    """chrono ``DateTime<Utc>::to_rfc3339()``: fraction digits 0/3/6/9 (AutoSi)."""
    base = ts.strftime("%Y-%m-%dT%H:%M:%S")
    micros = ts.microsecond
    if micros == 0:
        frac = ""
    elif micros % 1000 == 0:
        frac = f".{micros // 1000:03d}"
    else:
        frac = f".{micros:06d}"
    return f"{base}{frac}+00:00"


def js_iso(ts: datetime) -> str:
    """JavaScript ``Date.prototype.toISOString()``."""
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"


def compact(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def metadata_text(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def annotation_for(step: Step) -> dict:
    """What the worker stores for a frame: the scene annotator's compact
    answer (``ANNOTATION_JSON_SCHEMA``) plus ``_screen_text``, the screen's
    own text, which the annotator keeps beside the answer instead of having
    the model copy it."""
    urls = [step.url] if step.url.startswith("http") else []
    clicked = (step.click or {}).get("innerText") or (step.click or {}).get("ariaLabel") or ""
    screen_text = [step.title, *step.headings, *step.labels, *step.values, *step.names]
    return {
        "app": step.app,
        "location": step.url,
        "task_context": {
            "is_workflow": step.is_workflow,
            "what_doing": step.what_doing,
            "next_likely": step.likely_next,
        },
        "ui_state": {
            "active_element": step.active_element,
            "modal": "",
        },
        "key_text": {
            "typed_text": step.typed,
            "urls": urls,
            "emails": step.emails,
        },
        "visible_items": [{"label": clicked, "chosen": True}] if clicked else [],
        "needs_vision": False,
        "_screen_text": "\n".join(t for t in screen_text if t),
    }

class Generator:
    def __init__(self) -> None:
        self.rng = random.Random(SEED)
        self.rows: list[dict] = []
        self.annotations: list[dict] = []

    def uuid(self) -> str:
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

    def frame(self, when: datetime, step: Step) -> str:
        """Add the frame (and its click, for a Chrome step); return the frame's id."""
        when = when.replace(microsecond=self.rng.randrange(1, 999_999))
        app_id, window_id, bounds = APPS[step.app]
        x, y, w, h = bounds
        frame_id = self.uuid()
        if step.click is not None:
            click_at = when - timedelta(seconds=self.rng.randrange(2, 9))
            label = step.click.get("ariaLabel") or step.click.get("innerText") or ""
            self.rows.append({
                "id": self.uuid(),
                "timestamp": rfc3339(click_at),
                "kind_json": compact({
                    "type": "ClickIntent",
                    "target_description": f"{label}<{step.click['tagName']}>",
                }),
                "window_json": compact({
                    "window_id": f"chrome-tab-{self.rng.randrange(1800, 1999)}",
                    "app_id": "com.google.Chrome",
                    "title": step.url,
                    "bounds_global_px": [0, 0, 0, 0],
                    "z_order": 0,
                    "is_fullscreen": False,
                }),
                "display_topology_json": "[]",
                "primary_display_id": "",
                "cursor_x": None,
                "cursor_y": None,
                "ui_scale": None,
                "artifact_ids_json": "[]",
                "metadata_json": metadata_text({
                    "tabId": self.rng.randrange(1800, 1999),
                    "url": step.url,
                    "x": self.rng.randrange(80, 900),
                    "y": self.rng.randrange(80, 700),
                    "target": step.click,
                    "timestamp": js_iso(click_at),
                }),
                "display_ids_spanned_json": None,
            })
        self.rows.append({
            "id": frame_id,
            "timestamp": rfc3339(when),
            "kind_json": compact({"type": "DwellSnapshot"}),
            "window_json": compact({
                "window_id": window_id,
                "app_id": app_id,
                "title": step.title,
                "bounds_global_px": bounds,
                "z_order": 0,
                "is_fullscreen": False,
            }),
            "display_topology_json": compact([DISPLAY]),
            "primary_display_id": DISPLAY["display_id"],
            "cursor_x": x + self.rng.randrange(40, w - 40),
            "cursor_y": y + self.rng.randrange(60, h - 40),
            "ui_scale": DISPLAY["scale_factor"],
            "artifact_ids_json": "[]",
            "metadata_json": metadata_text({
                "screenshot_path": f"{SCREENSHOTS_TOKEN}/{self.uuid()}.jpg",
            }),
            "display_ids_spanned_json": None,
        })
        self.annotations.append({
            "id": frame_id,
            "annotation_status": "completed",
            "scene_annotation_json": compact(annotation_for(step)),
        })
        return frame_id

    def run(self, start: datetime, steps: list[Step], gap: tuple[int, int]) -> tuple[list[str], list[str], datetime]:
        """Frames for *steps*, *gap* seconds apart; returns (ids, extra ids, end)."""
        ids: list[str] = []
        extra: list[str] = []
        when = start
        for step in steps:
            frame_id = self.frame(when, step)
            (extra if step.extra else ids).append(frame_id)
            end = when
            when += timedelta(seconds=self.rng.randrange(*gap))
        return ids, extra, end


# ---------------------------------------------------------------------------
# Routine 1: daily Stripe payout reconciliation
# ---------------------------------------------------------------------------


def payout_steps(rng: random.Random, day: datetime) -> tuple[list[Step], list[str]]:
    payout = "po_1" + "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789") for _ in range(23))
    amount = f"${rng.randrange(800, 9800):,}.{rng.randrange(0, 99):02d}"
    paid = (day - timedelta(days=3 if day.weekday() == 0 else 1)).strftime("%b %-d")
    row = 240 + (day - FIRST_DAY).days
    payouts = Step(
        "Google Chrome", "Payouts - Stripe", "https://dashboard.stripe.com/payouts",
        rng.choice([
            f"Reviewing the payouts list in the Stripe dashboard for the {paid} payout",
            f"Checking yesterday's payouts in the Stripe dashboard ({paid})",
            f"Looking up the {paid} payout in the Stripe dashboard payouts list",
        ]),
        "Open the most recent payout to see its amount and status", True,
        headings=["Payouts"], labels=["Amount", "Status", "Arrival date", "Paid"],
        values=[amount, paid, "Paid"],
        click={"tagName": "a", "innerText": "Payouts", "role": "link"},
    )
    detail = Step(
        "Google Chrome", f"{payout} - Stripe", f"https://dashboard.stripe.com/payouts/{payout}",
        rng.choice([
            f"Opening payout {payout} of {amount} in Stripe to confirm it was paid on {paid}",
            f"Viewing Stripe payout {payout} ({amount}, paid {paid}) and its balance transactions",
            f"Checking the status and amount ({amount}) of Stripe payout {payout}",
        ]),
        "Copy the payout amount into the reconciliation sheet", True,
        headings=[f"Payout {payout}"], labels=["Amount", "Arrival date", "Balance transactions"],
        values=[amount, paid], active_element=amount,
        click={"tagName": "a", "innerText": amount, "role": "link"},
    )
    sheet = Step(
        "Google Chrome", SHEET_TITLE, SHEET_URL,
        rng.choice([
            f"Entering the {amount} Stripe payout {payout} into row {row} of the payout reconciliation sheet",
            f"Typing payout {payout} and amount {amount} into the Payout reconciliation 2026 Google Sheet",
            f"Adding a row for the {paid} Stripe payout ({amount}) to the reconciliation spreadsheet",
        ]),
        "Mark the row as reconciled", True,
        headings=["Payout reconciliation 2026"], labels=["Date", "Payout ID", "Amount", "Reconciled"],
        values=[paid, payout, amount], active_element=f"Cell C{row}", typed=amount,
        click={"tagName": "div", "role": "gridcell", "ariaLabel": f"C{row}"},
    )
    reconcile = Step(
        "Google Chrome", SHEET_TITLE, SHEET_URL,
        rng.choice([
            f"Ticking the Reconciled checkbox for payout {payout} in the reconciliation sheet",
            f"Marking row {row} ({payout}, {amount}) as reconciled in Google Sheets",
        ]),
        "Post the reconciled total in the finance Slack channel", True,
        headings=["Payout reconciliation 2026"], labels=["Reconciled"],
        values=[payout, "TRUE"], active_element=f"Checkbox D{row}",
        click={"tagName": "div", "role": "checkbox", "ariaLabel": f"D{row} Reconciled"},
    )
    slack = Step(
        "Google Chrome", "finance (Channel) - Acme - Slack", FINANCE_SLACK,
        rng.choice([
            f"Posting 'Stripe payout {paid}: {amount} reconciled' in the #finance Slack channel",
            f"Writing a message in Slack #finance that the {amount} payout from {paid} is reconciled",
        ]),
        "Send the message", True,
        headings=["# finance"], labels=["Message #finance", "Send now"],
        values=[f"Stripe payout {paid}: {amount} reconciled"], typed=f"Stripe payout {paid}: {amount} reconciled",
        click={"tagName": "button", "ariaLabel": "Send now"},
    )
    variant: list[str] = []
    steps = [payouts, detail, sheet, reconcile, slack]
    if rng.random() < 0.3:
        # Sheet first: find the next empty row, then Stripe, then fill it in.
        variant.append("sheet_first")
        glance = Step(
            "Google Chrome", SHEET_TITLE, SHEET_URL,
            f"Scrolling to the next empty row ({row}) of the payout reconciliation sheet",
            "Look up the payout in Stripe", True,
            headings=["Payout reconciliation 2026"], labels=["Date", "Payout ID", "Amount"],
            values=[str(row)], click={"tagName": "div", "role": "gridcell", "ariaLabel": f"A{row}"},
        )
        steps = [glance, payouts, detail, sheet, reconcile, slack]
    if rng.random() < 0.35:
        variant.append("extra_tab")
        extra = rng.choice([
            Step("Google Chrome", "Inbox (3) - dana@acme.io - Gmail", "https://mail.google.com/mail/u/0/#inbox",
                 "Glancing at the Gmail inbox for new messages", "Go back to the previous tab", False,
                 headings=["Inbox"], extra=True),
            Step("Google Chrome", "Hacker News", "https://news.ycombinator.com/",
                 "Skimming the Hacker News front page", "Return to work", False,
                 headings=["Hacker News"], extra=True),
        ])
        steps.insert(rng.randrange(2, len(steps)), extra)
    return steps, variant


# ---------------------------------------------------------------------------
# Routine 2: weekly expense report (Mondays)
# ---------------------------------------------------------------------------

RECEIPTS = [
    ("Uber", "Travel", "uber.com"), ("Lyft", "Travel", "lyft.com"),
    ("United Airlines", "Airfare", "united.com"), ("Blue Bottle Coffee", "Meals", "bluebottlecoffee.com"),
    ("Marriott", "Lodging", "marriott.com"), ("WeWork", "Office", "wework.com"),
    ("Delta", "Airfare", "delta.com"), ("Sweetgreen", "Meals", "sweetgreen.com"),
]


def expense_steps(rng: random.Random, day: datetime, week: int) -> tuple[list[Step], list[Step]]:
    """(before the interruption point, after it)."""
    report = f"R00{rng.randrange(40000, 99999)}"
    since = (day - timedelta(days=7)).strftime("%Y/%m/%d")
    receipts = rng.sample(RECEIPTS, rng.randrange(2, 4))
    search = Step(
        "Google Chrome", "Search results - dana@acme.io - Gmail",
        f"https://mail.google.com/mail/u/0/#search/receipt+after%3A{since.replace('/', '%2F')}",
        rng.choice([
            f"Searching Gmail for receipts received since {since}",
            f"Looking through Gmail search results for last week's receipts (after {since})",
        ]),
        "Open the first receipt email", True,
        headings=["Search results"], labels=["receipt after:" + since],
        values=[r[0] for r in receipts], click={"tagName": "input", "ariaLabel": "Search mail"},
    )
    new_report = Step(
        "Google Chrome", "Reports - Expensify", "https://www.expensify.com/reports",
        rng.choice([
            f"Creating a new expense report 'Week {35 + week} expenses' in Expensify",
            f"Starting expense report {report} for week {35 + week} in Expensify",
        ]),
        "Add the first expense to the report", True,
        headings=["Reports"], labels=["New report", "Report name"], values=[f"Week {35 + week} expenses"],
        click={"tagName": "button", "innerText": "New report"},
    )
    before = [search]
    after: list[Step] = []
    for i, (merchant, category, domain) in enumerate(receipts):
        amount = f"${rng.randrange(9, 480)}.{rng.randrange(0, 99):02d}"
        date = (day - timedelta(days=rng.randrange(2, 7))).strftime("%b %-d")
        pdf = f"{merchant.replace(' ', '_')}_receipt_{date.replace(' ', '')}.pdf"
        email = Step(
            "Google Chrome", f"Your {merchant} receipt - dana@acme.io - Gmail",
            f"https://mail.google.com/mail/u/0/#search/receipt/{rng.randrange(10**15, 10**16):x}",
            rng.choice([
                f"Opening the {merchant} receipt email for {amount} from {date} and downloading {pdf}",
                f"Downloading the {pdf} attachment from the {merchant} receipt email ({amount})",
            ]),
            "Add this receipt as an expense in Expensify", True,
            headings=[f"Your {merchant} receipt"], labels=["Download attachment"],
            values=[amount, date, pdf], emails=[f"receipts@{domain}"],
            click={"tagName": "button", "ariaLabel": f"Download attachment {pdf}"},
        )
        expense = Step(
            "Google Chrome", f"Report {report} - Expensify", f"https://www.expensify.com/report?reportID={report[1:]}",
            rng.choice([
                f"Adding a {amount} {merchant} expense in category {category} to Expensify report {report}",
                f"Entering the {merchant} expense ({amount}, {category}) into expense report {report}",
            ]),
            f"Attach {pdf} to the expense", True,
            headings=[f"Report {report}"], labels=["Merchant", "Amount", "Category", "Date"],
            values=[merchant, amount, category, date], typed=amount,
            click={"tagName": "button", "innerText": "Add expense"},
        )
        attach = Step(
            "Google Chrome", f"Report {report} - Expensify", f"https://www.expensify.com/report?reportID={report[1:]}",
            rng.choice([
                f"Attaching the receipt {pdf} to the {merchant} expense in report {report}",
                f"Uploading {pdf} as the receipt for the {amount} {merchant} expense",
            ]),
            "Add the next receipt or submit the report", True,
            headings=[f"Report {report}"], labels=["Attach receipt"], values=[pdf],
            click={"tagName": "button", "ariaLabel": "Attach receipt"},
        )
        if i == 0:
            before += [email, new_report, expense]
            after.append(attach)
        else:
            after += [email, expense, attach]
    after.append(Step(
        "Google Chrome", f"Report {report} - Expensify", f"https://www.expensify.com/report?reportID={report[1:]}",
        rng.choice([
            f"Submitting expense report {report} with {len(receipts)} expenses to approver Mia Chen",
            f"Clicking Submit on Expensify report {report} to send it to Mia Chen for approval",
        ]),
        "Close the report", True,
        headings=[f"Report {report}"], labels=["Submit", "Approver"], values=["Mia Chen"],
        names=["Mia Chen"], click={"tagName": "button", "innerText": "Submit"},
    ))
    return before, after


def zoom_call(rng: random.Random) -> list[Step]:
    return [
        Step("zoom.us", "Zoom Meeting", "", what, "Keep listening to the call", False,
             headings=["Product sync"], names=["Mia Chen", "Sam Ortiz", "Priya Nair"])
        for what in [
            "Attending the weekly product sync call on Zoom with Mia, Sam and Priya",
            "Listening to Sam present the roadmap update in the Zoom product sync",
            "Discussing the launch date with Priya on the Zoom call",
            "Wrapping up the Zoom product sync call",
        ]
    ]


# ---------------------------------------------------------------------------
# One-off work and noise
# ---------------------------------------------------------------------------

# Each one-off task happens once in the three weeks: (app, title, url, what_doing lines).
ONE_OFF_TASKS = [
    ("Google Chrome", "Q4 hiring plan - Google Docs", "https://docs.google.com/document/d/1hQ4hiringPlan/edit",
     ["Writing the Q4 hiring plan section on the two backend roles in Google Docs",
      "Adding a budget table for the Q4 hires to the hiring plan document",
      "Revising the interview loop description in the Q4 hiring plan doc",
      "Leaving a comment for Mia on the Q4 hiring plan timeline"]),
    ("Visual Studio Code", "auth_service.py - acme-api", "~/code/acme-api/auth_service.py",
     ["Debugging the token refresh bug in auth_service.py",
      "Adding a retry around the OAuth token refresh call in auth_service.py",
      "Writing a regression test for expired refresh tokens in test_auth_service.py",
      "Running the auth service test suite in the VS Code terminal"]),
    ("Figma", "Onboarding v3 - Figma", "Onboarding v3 / Welcome screen",
     ["Reviewing the onboarding v3 welcome screen design in Figma",
      "Leaving feedback on the button copy of the onboarding welcome screen",
      "Comparing the two onboarding illustration options in Figma"]),
    ("Keynote", "Board update September.key", "~/Documents/Board update September.key",
     ["Building the revenue chart slide for the September board update in Keynote",
      "Writing speaker notes for the churn slide of the board deck",
      "Reordering the slides of the September board update deck"]),
    ("Google Chrome", "Vendor security questionnaire - Google Forms", "https://docs.google.com/forms/d/e/1FAIpQLvendorSec/viewform",
     ["Filling in the data retention answers of the Zendesk vendor security questionnaire",
      "Answering the encryption-at-rest questions of the vendor security form",
      "Submitting the vendor security questionnaire for Zendesk"]),
    ("Terminal", "zsh - acme-infra", "~/code/acme-infra",
     ["Upgrading the Postgres minor version on the staging database with terraform",
      "Reviewing the terraform plan for the staging Postgres upgrade",
      "Applying the Postgres upgrade on staging and watching the logs"]),
    ("Google Chrome", "Pricing page - Webflow", "https://webflow.com/design/acme-site/pricing",
     ["Editing the pricing page headline in Webflow",
      "Updating the Team plan price to $49 per seat on the pricing page",
      "Publishing the pricing page changes to staging in Webflow"]),
    ("Google Chrome", "Customer interview notes - Notion", "https://www.notion.so/acme/Interview-Brightline-4f2a",
     ["Writing up notes from the Brightline customer interview in Notion",
      "Tagging the Brightline interview insights by theme in Notion",
      "Linking the Brightline interview to the reporting feature request"]),
    ("Visual Studio Code", "invoice_export.ts - acme-web", "~/code/acme-web/src/invoice_export.ts",
     ["Implementing CSV export for invoices in invoice_export.ts",
      "Adding the currency column to the invoice CSV export",
      "Fixing a date formatting bug in the invoice export"]),
    ("Google Chrome", "Offsite planning - Google Sheets", "https://docs.google.com/spreadsheets/d/1offsitePlanning/edit",
     ["Comparing venue quotes for the October team offsite in a spreadsheet",
      "Adding flight cost estimates to the offsite planning sheet",
      "Sharing the offsite budget sheet with Sam"]),
    ("Google Chrome", "Incident 2026-09-02 postmortem - Google Docs", "https://docs.google.com/document/d/1postmortemSep2/edit",
     ["Writing the timeline of the September 2 API outage postmortem",
      "Listing the action items in the API outage postmortem",
      "Assigning owners to the postmortem action items"]),
    ("Figma", "Mobile nav redesign - Figma", "Mobile nav redesign / Tab bar",
     ["Redesigning the mobile tab bar icons in Figma",
      "Adjusting the spacing of the mobile navigation tab bar",
      "Exporting the new tab bar icons as SVG from Figma"]),
    ("Google Chrome", "Partnership agreement - DocuSign", "https://app.docusign.com/documents/details/7c1e",
     ["Reading the Northwind partnership agreement in DocuSign",
      "Signing the Northwind partnership agreement in DocuSign"]),
    ("Visual Studio Code", "README.md - acme-cli", "~/code/acme-cli/README.md",
     ["Rewriting the installation section of the acme-cli README",
      "Adding usage examples for the export command to the acme-cli README",
      "Fixing broken links in the acme-cli README"]),
    ("Google Chrome", "Job post: Senior Backend Engineer - Greenhouse", "https://app.greenhouse.io/jobs/4471/edit",
     ["Writing the Senior Backend Engineer job post in Greenhouse",
      "Adding the salary range and benefits to the backend job post",
      "Publishing the Senior Backend Engineer job post on the careers page"]),
    ("Keynote", "Sales kickoff.key", "~/Documents/Sales kickoff.key",
     ["Drafting the product roadmap slides for the sales kickoff in Keynote",
      "Adding customer logos to the sales kickoff deck"]),
    ("Google Chrome", "Data processing addendum - Google Docs", "https://docs.google.com/document/d/1dpaRedlines/edit",
     ["Reviewing the redlines on the Brightline data processing addendum",
      "Accepting the sub-processor list changes in the DPA",
      "Replying to legal's comment about the breach notification window"]),
    ("Terminal", "zsh - acme-api", "~/code/acme-api",
     ["Profiling the slow /reports endpoint with py-spy",
      "Adding an index on reports.created_at in a new migration",
      "Benchmarking the /reports endpoint after the index migration"]),
    ("Google Chrome", "Conference talk proposal - Sessionize", "https://sessionize.com/app/speaker/talk/551203",
     ["Writing the abstract of the talk proposal for PyCon about local LLMs",
      "Filling in the speaker bio on the Sessionize proposal form",
      "Submitting the PyCon talk proposal on Sessionize"]),
    ("Figma", "Invoice PDF template - Figma", "Invoice PDF template / A4",
     ["Designing the new invoice PDF template layout in Figma",
      "Choosing typography for the invoice PDF header"]),
    ("Google Chrome", "Annual plan discount analysis - Google Sheets", "https://docs.google.com/spreadsheets/d/1annualDiscount/edit",
     ["Modelling the revenue impact of a 20% annual plan discount in a spreadsheet",
      "Charting monthly versus annual plan conversion in the discount analysis sheet",
      "Writing the recommendation summary at the top of the discount analysis"]),
    ("Visual Studio Code", "docker-compose.yml - acme-api", "~/code/acme-api/docker-compose.yml",
     ["Adding a Redis service to docker-compose.yml for local development",
      "Configuring the API container to use the local Redis cache",
      "Documenting the new Redis setup in CONTRIBUTING.md"]),
    ("Google Chrome", "Customer churn survey - Typeform", "https://admin.typeform.com/form/churnSurvey/create",
     ["Building the customer churn survey questions in Typeform",
      "Adding a logic jump for customers who switched to a competitor",
      "Previewing the churn survey on mobile in Typeform"]),
    ("Google Chrome", "Security training - KnowBe4", "https://training.knowbe4.com/modules/phishing-2026",
     ["Watching the annual phishing awareness training module",
      "Answering the quiz at the end of the phishing training"]),
    ("Keynote", "All-hands October.key", "~/Documents/All-hands October.key",
     ["Preparing the October all-hands slides on hiring progress",
      "Adding the new team members' photos to the all-hands deck"]),
    ("Terminal", "zsh - acme-web", "~/code/acme-web",
     ["Upgrading the web app from Node 20 to Node 22",
      "Fixing TypeScript errors after the Node 22 upgrade",
      "Running the web app's end-to-end tests after the upgrade"]),
    ("Google Chrome", "Office lease renewal - Gmail", "https://mail.google.com/mail/u/0/#inbox/FMfcgzLeaseRenewal",
     ["Reading the office lease renewal terms from the landlord",
      "Replying to the landlord with questions about the lease renewal"]),
    ("Figma", "Marketing site hero - Figma", "Marketing site / Hero",
     ["Iterating on the marketing site hero illustration in Figma",
      "Testing three hero headline variants in the Figma mockup"]),
    ("Google Chrome", "SOC 2 evidence - Vanta", "https://app.vanta.com/tests/access-reviews",
     ["Uploading the quarterly access review evidence to Vanta",
      "Resolving the failing MFA test for two accounts in Vanta"]),
    ("Visual Studio Code", "rate_limiter.go - acme-gateway", "~/code/acme-gateway/rate_limiter.go",
     ["Implementing a token bucket rate limiter in rate_limiter.go",
      "Writing table-driven tests for the rate limiter",
      "Tuning the rate limiter burst size after load testing"]),
    ("Google Chrome", "Podcast guest prep - Google Docs", "https://docs.google.com/document/d/1podcastPrep/edit",
     ["Preparing talking points for the SaaS Founders podcast interview",
      "Writing answers about pricing strategy for the podcast prep doc"]),
]

NOISE = [
    ("Google Chrome", "YouTube", "https://www.youtube.com/watch?v={id}",
     ["Watching a YouTube video about {topic}", "Browsing YouTube recommendations after a video about {topic}"],
     ["espresso machines", "mechanical keyboards", "the history of Unix", "trail running in the Alps",
      "a Formula 1 race recap", "home workshop tours"]),
    ("Google Chrome", "Hacker News", "https://news.ycombinator.com/item?id={id}",
     ["Reading a Hacker News thread about {topic}", "Scrolling the comments of a Hacker News post on {topic}"],
     ["SQLite in production", "a new Rust release", "remote work salaries", "open-source funding",
      "a database outage postmortem", "keyboard layouts"]),
    ("Google Chrome", "Home / X", "https://x.com/home",
     ["Scrolling the X timeline", "Reading posts about {topic} on X"],
     ["the Apple event", "a startup acquisition", "AI chips", "the football transfer window"]),
]


def one_off_steps(rng: random.Random, task: tuple) -> list[Step]:
    app, title, url, lines = task
    return [
        Step(app, title, url, what, "Continue working on this task", True,
             headings=[title.split(" - ")[0]], click=(
                 {"tagName": "div", "role": "textbox", "ariaLabel": title.split(" - ")[0]}
                 if app == "Google Chrome" else None))
        for what in lines
    ]


def noise_steps(rng: random.Random) -> list[Step]:
    app, title, url, lines, topics = rng.choice(NOISE)
    topic = rng.choice(topics)
    page = url.format(id=rng.randrange(10**7, 10**8))
    return [
        Step(app, title, page, line.format(topic=topic), "Keep browsing", False, headings=[title])
        for line in lines
    ]


# ---------------------------------------------------------------------------
# Three weeks
# ---------------------------------------------------------------------------


def at(day: datetime, hour: int, minute: int, rng: random.Random, jitter: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute) + timedelta(minutes=rng.randrange(-jitter, jitter + 1))


def build() -> tuple[Generator, dict]:
    gen = Generator()
    rng = gen.rng
    tasks = list(ONE_OFF_TASKS)
    rng.shuffle(tasks)
    labels = {
        "fixture": NAME,
        "timezone": "UTC",
        "routines": [
            {"slug": STRIPE, "name": "Reconcile yesterday's Stripe payout", "cadence": "daily",
             "local_weekday": None, "instances": []},
            {"slug": EXPENSES, "name": "Weekly expense report", "cadence": "weekly",
             "local_weekday": 0, "instances": []},
        ],
        "one_off_tasks": [],
        "noise": [],
    }
    routines = {r["slug"]: r for r in labels["routines"]}

    def instance(slug: str, start: datetime, ids: list[str], extra: list[str], end: datetime,
                 variant: list[str]) -> None:
        routines[slug]["instances"].append({
            "instance_id": f"{slug}-{start.strftime('%Y%m%d')}",
            "start": rfc3339(start.replace(microsecond=0)),
            "end": rfc3339(end.replace(microsecond=0)),
            "variant": variant,
            "event_ids": ids,
            "extra_event_ids": extra,
        })

    for offset in range(DAYS):
        day = FIRST_DAY + timedelta(days=offset)
        if day.weekday() >= 5:
            continue  # weekends off

        if rng.random() < 0.6:
            start = at(day, 8, 40, rng, jitter=10)
            ids, _, _ = gen.run(start, noise_steps(rng), gap=(90, 300))
            labels["noise"].append({"start": rfc3339(start.replace(microsecond=0)), "event_ids": ids})

        steps, variant = payout_steps(rng, day)
        if offset == EVENING_DAY:
            variant.append("evening")
            start = at(day, 21, 5, rng)
        else:
            start = at(day, 9, 10, rng, jitter=15)
        ids, extra, end = gen.run(start, steps, gap=(50, 150))
        instance(STRIPE, start, ids, extra, end, variant)

        if day.weekday() == 0:
            week = offset // 7
            before, after = expense_steps(rng, day, week)
            start = at(day, 10, 30, rng, jitter=10)
            ids, _, end = gen.run(start, before, gap=(60, 150))
            variant = []
            if offset == INTERRUPTED_DAY:
                variant.append("interrupted_40min_call")
                call_start = end + timedelta(minutes=1)
                call_ids, _, _ = gen.run(call_start, zoom_call(rng), gap=(600, 601))
                labels["noise"].append({"start": rfc3339(call_start.replace(microsecond=0)),
                                        "event_ids": call_ids, "note": "40-minute Zoom call"})
                resume = call_start + timedelta(minutes=40)
            else:
                resume = end + timedelta(seconds=rng.randrange(60, 150))
            more, _, end = gen.run(resume, after, gap=(60, 150))
            instance(EXPENSES, start, ids + more, [], end, variant)

        for hour in rng.sample([11, 13, 14, 16], 2):
            task = tasks.pop()
            start = at(day, hour, 5, rng, jitter=5)
            ids, _, _ = gen.run(start, one_off_steps(rng, task), gap=(300, 600))
            labels["one_off_tasks"].append({
                "task": task[1], "start": rfc3339(start.replace(microsecond=0)), "event_ids": ids,
            })

        if rng.random() < 0.5:
            start = at(day, 15, 30, rng, jitter=20)
            ids, _, _ = gen.run(start, noise_steps(rng), gap=(90, 300))
            labels["noise"].append({"start": rfc3339(start.replace(microsecond=0)), "event_ids": ids})

    order = sorted(range(len(gen.rows)), key=lambda i: gen.rows[i]["timestamp"])
    gen.rows = [gen.rows[i] for i in order]
    return gen, labels


def manifest_for(gen: Generator, labels: dict) -> dict:
    return {
        "name": NAME,
        "description": (
            "Three weeks (Mon-Fri, weekends off) of annotated passive frames: a daily "
            "Stripe payout reconciliation, a Monday expense report, one-off work tasks "
            "and browsing noise."
        ),
        "source": {
            "kind": "generated",
            "by": "worker/tests/journeys/record/generate_passive.py",
            "note": (
                "Frames carry the scene annotator's output (annotations.jsonl.gz); "
                "labels.json is the ground truth."
            ),
        },
        "schema_version": schema_version(),
        "timezone": "UTC",
        "recorded_start": gen.rows[0]["timestamp"],
        "recorded_end": gen.rows[-1]["timestamp"],
        "planted_routines": [
            {"slug": r["slug"], "cadence": r["cadence"], "local_weekday": r["local_weekday"],
             "instances": len(r["instances"])}
            for r in labels["routines"]
        ],
        "annotated_rows": len(gen.annotations),
        "row_counts": row_counts(gen.rows),
    }


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(payload)


def main() -> None:
    gen, labels = build()
    out = FIXTURES_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out, gen.rows)
    write_jsonl_gz(out / ANNOTATIONS_FILE, gen.annotations)
    (out / LABELS_FILE).write_text(json.dumps(labels, indent=1) + "\n", encoding="utf-8")
    write_manifest(out, manifest_for(gen, labels))
    print(f"wrote {len(gen.rows)} rows ({len(gen.annotations)} annotated frames) to {out}")


if __name__ == "__main__":
    main()
