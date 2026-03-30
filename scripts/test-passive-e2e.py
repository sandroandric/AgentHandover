#!/usr/bin/env python3
"""End-to-end test of the passive observation pipeline.

Simulates 5 days of realistic user activity by injecting synthetic
annotated events into a temporary knowledge base, then runs the full
pipeline: daily summaries → profile → patterns → digest → embeddings.

Usage:
    python scripts/test-passive-e2e.py

No production data is modified — uses a temp directory for the KB and
a temp SQLite DB. Requires Ollama running with nomic-embed-text for
the embedding verification step (skipped gracefully if unavailable).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add worker source to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker" / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("e2e-test")

# -------------------------------------------------------------------------
# Synthetic event generator
# -------------------------------------------------------------------------

APPS_AND_TASKS = [
    # (app, location, intent, is_workflow)
    ("Google Chrome", "https://mail.google.com", "Reading and replying to emails", True),
    ("Google Chrome", "https://mail.google.com", "Composing a new email to team", True),
    ("Google Chrome", "https://github.com/pulls", "Reviewing pull requests on GitHub", True),
    ("Google Chrome", "https://github.com/issues", "Triaging GitHub issues", True),
    ("Google Chrome", "https://slack.com", "Reading Slack messages in #engineering", True),
    ("Google Chrome", "https://docs.google.com", "Writing product spec document", True),
    ("VS Code", "/src/main.py", "Editing Python code for API endpoint", True),
    ("VS Code", "/src/tests/test_api.py", "Writing unit tests for API", True),
    ("VS Code", "/src/models.py", "Refactoring data models", True),
    ("Terminal", "~/projects/app", "Running pytest in terminal", True),
    ("Terminal", "~/projects/app", "Running git status and committing code", True),
    ("Figma", "Design System v2", "Reviewing UI mockups from designer", True),
    ("Google Chrome", "https://producthunt.com", "Browsing Product Hunt for new tools", False),
    ("Google Chrome", "https://news.ycombinator.com", "Reading Hacker News", False),
    ("Finder", "~/Downloads", "Organizing downloaded files", False),
    ("Notes", "Meeting Notes", "Taking notes during standup meeting", True),
    ("Google Chrome", "https://calendar.google.com", "Checking calendar for next meeting", True),
    ("Google Chrome", "https://linear.app", "Updating ticket status in Linear", True),
    ("Terminal", "~/projects/app", "Deploying to staging environment", True),
    ("Google Chrome", "https://vercel.com", "Checking deployment logs on Vercel", True),
]

# Repeated workflows (should trigger pattern detection)
REPEATED_WORKFLOWS = [
    # Morning email check (every day)
    [
        ("Google Chrome", "https://mail.google.com", "Opening Gmail inbox", True),
        ("Google Chrome", "https://mail.google.com", "Reading new emails from overnight", True),
        ("Google Chrome", "https://mail.google.com", "Replying to urgent email from client", True),
    ],
    # Daily standup prep (every day)
    [
        ("Google Chrome", "https://linear.app", "Checking my assigned tickets in Linear", True),
        ("Notes", "Standup Notes", "Writing standup update", True),
        ("Google Chrome", "https://slack.com", "Posting standup update in Slack", True),
    ],
    # Code review (3 out of 5 days)
    [
        ("Google Chrome", "https://github.com/pulls", "Opening PR review queue", True),
        ("VS Code", "/src/feature.py", "Reading code changes in VS Code", True),
        ("Google Chrome", "https://github.com/pulls", "Leaving review comments on PR", True),
    ],
]


def _make_event(
    ts: datetime,
    app: str,
    location: str,
    intent: str,
    is_workflow: bool,
) -> dict:
    """Create a synthetic annotated event dict."""
    return {
        "id": str(uuid.uuid4()),
        "timestamp": ts.isoformat(),
        "event_type": "DwellSnapshot",
        "annotation_status": "completed",
        "scene_annotation_json": json.dumps({
            "app": app,
            "location": location,
            "visible_content": {
                "headings": [f"{app} - {location}"],
                "labels": [],
                "values": [],
            },
            "ui_state": {"focused_element": "main content"},
            "task_context": {
                "what_doing": intent,
                "is_workflow": is_workflow,
                "activity_type": "work" if is_workflow else "browsing",
            },
            "key_text": {
                "email_addresses": ["user@company.com"] if "mail" in location else [],
                "urls": [location] if location.startswith("http") else [],
            },
        }),
        "metadata": json.dumps({"app_name": app, "window_title": f"{app} - {location}"}),
    }


def generate_days(num_days: int = 5) -> dict[str, list[dict]]:
    """Generate synthetic events for N days of work."""
    base_date = datetime.now(timezone.utc).replace(
        hour=9, minute=0, second=0, microsecond=0,
    ) - timedelta(days=num_days)

    days: dict[str, list[dict]] = {}

    for day_offset in range(num_days):
        day_start = base_date + timedelta(days=day_offset)
        date_str = day_start.strftime("%Y-%m-%d")
        events: list[dict] = []
        ts = day_start

        # Morning routine (repeated workflow — email check)
        for app, loc, intent, wf in REPEATED_WORKFLOWS[0]:
            events.append(_make_event(ts, app, loc, intent, wf))
            ts += timedelta(minutes=3)

        ts += timedelta(minutes=10)

        # Standup prep (repeated workflow)
        for app, loc, intent, wf in REPEATED_WORKFLOWS[1]:
            events.append(_make_event(ts, app, loc, intent, wf))
            ts += timedelta(minutes=4)

        ts += timedelta(minutes=30)

        # Code review (3 out of 5 days)
        if day_offset % 2 == 0 or day_offset == 4:
            for app, loc, intent, wf in REPEATED_WORKFLOWS[2]:
                events.append(_make_event(ts, app, loc, intent, wf))
                ts += timedelta(minutes=5)
            ts += timedelta(minutes=15)

        # Random work tasks (varies per day)
        import random
        random.seed(day_offset)
        day_tasks = random.sample(APPS_AND_TASKS, min(8, len(APPS_AND_TASKS)))
        for app, loc, intent, wf in day_tasks:
            events.append(_make_event(ts, app, loc, intent, wf))
            ts += timedelta(minutes=random.randint(3, 15))

        days[date_str] = events
        logger.info("Day %s: %d events", date_str, len(events))

    return days


# -------------------------------------------------------------------------
# Pipeline runner
# -------------------------------------------------------------------------

def run_pipeline(days: dict[str, list[dict]]) -> dict:
    """Run the full passive pipeline on synthetic data."""
    results = {
        "daily_summaries": 0,
        "profile_built": False,
        "patterns_detected": 0,
        "digest_generated": False,
        "procedures_from_patterns": 0,
        "vectors_stored": 0,
        "errors": [],
    }

    # Create temp KB and DB
    with tempfile.TemporaryDirectory(prefix="ah-e2e-") as tmpdir:
        kb_root = Path(tmpdir) / "knowledge"
        db_path = Path(tmpdir) / "events.db"

        from agenthandover_worker.knowledge_base import KnowledgeBase
        kb = KnowledgeBase(root=kb_root)
        kb.ensure_structure()

        logger.info("KB root: %s", kb_root)
        logger.info("DB path: %s", db_path)

        # Initialize components
        from agenthandover_worker.daily_processor import DailyBatchProcessor
        from agenthandover_worker.profile_builder import ProfileBuilder
        from agenthandover_worker.pattern_detector import PatternDetector

        daily = DailyBatchProcessor(knowledge_base=kb)
        profile_builder = ProfileBuilder(kb)
        pattern_detector = PatternDetector(kb)

        # ---- Step 1: Process each day's events ----
        logger.info("\n=== Step 1: Daily summaries ===")
        for date_str in sorted(days.keys()):
            events = days[date_str]
            try:
                summary = daily.process_day(date_str, events)
                results["daily_summaries"] += 1
                logger.info(
                    "  %s: %d tasks, %.1fh active, apps: %s",
                    date_str,
                    summary.task_count,
                    summary.active_hours,
                    [a["app"] for a in summary.top_apps[:3]],
                )
            except Exception as e:
                results["errors"].append(f"daily_summary {date_str}: {e}")
                logger.error("  %s: FAILED — %s", date_str, e)

        # ---- Step 2: Profile ----
        logger.info("\n=== Step 2: Profile ===")
        try:
            profile = profile_builder.update_profile()
            results["profile_built"] = bool(profile and profile.get("tools"))
            if results["profile_built"]:
                tools = profile.get("tools", {})
                browser = tools.get("browser", "?")
                editor = tools.get("editor", "?")
                primary = [a["app"] for a in tools.get("primary_apps", [])[:3]]
                logger.info("  Browser: %s, Editor: %s", browser, editor)
                logger.info("  Primary apps: %s", primary)
                hours = profile.get("working_hours", {})
                logger.info("  Working hours: %s-%s", hours.get("start", "?"), hours.get("end", "?"))
                style = profile.get("communication_style", {})
                if style:
                    logger.info("  Communication: %s", style)
                writing = profile.get("writing_style", {})
                if writing:
                    logger.info("  Writing style: %s", writing)
            else:
                logger.warning("  Profile empty — not enough data")
        except Exception as e:
            results["errors"].append(f"profile: {e}")
            logger.error("  FAILED — %s", e)

        # ---- Step 3: Patterns ----
        logger.info("\n=== Step 3: Pattern detection ===")
        try:
            patterns = pattern_detector.detect_recurrence()
            results["patterns_detected"] = len(patterns)
            for p in patterns:
                logger.info(
                    "  Pattern: '%s' — %s (confidence %.0f%%)",
                    p.procedure_slug[:60],
                    p.pattern,
                    p.confidence * 100,
                )
            chains = pattern_detector.detect_chains()
            if chains:
                logger.info("  Chains detected: %d", len(chains))
        except Exception as e:
            results["errors"].append(f"patterns: {e}")
            logger.error("  FAILED — %s", e)

        # ---- Step 4: Digest ----
        logger.info("\n=== Step 4: Digest ===")
        try:
            from agenthandover_worker.daily_digest import DigestGenerator
            digest_gen = DigestGenerator(kb)
            # Generate digest for most recent day
            last_date = sorted(days.keys())[-1]
            digest = digest_gen.generate(last_date)
            if digest:
                results["digest_generated"] = True
                logger.info("  Summary: %s", digest.summary[:100])
                logger.info("  Highlights: %d", len(digest.highlights))
                for h in digest.highlights[:3]:
                    logger.info("    - %s: %s", h.type, h.title)
                for s in digest.sections:
                    logger.info("    Section '%s': %d items", s.title, len(s.items))
            else:
                logger.warning("  No digest generated")
        except Exception as e:
            results["errors"].append(f"digest: {e}")
            logger.error("  FAILED — %s", e)

        # ---- Step 4b: Voice / Writing Style (Qwen) ----
        logger.info("\n=== Step 4b: Voice & Writing Style (via Qwen) ===")
        try:
            from agenthandover_worker.llm_reasoning import LLMReasoner, ReasoningConfig
            from agenthandover_worker.style_analyzer import analyze_style, analyze_procedure_style

            reasoner = LLMReasoner(ReasoningConfig(
                model="qwen3.5:4b",
                ollama_host="http://localhost:11434",
            ))

            # Simulate text the user produced in different contexts
            email_samples = [
                "Hey Sarah, just wanted to follow up on the deployment timeline. "
                "We're targeting Friday but I want to make sure the staging tests "
                "pass first. Can you check the CI pipeline when you get a chance?",
                "Thanks for the quick turnaround on the mockups! The nav looks "
                "great. One small thing — can we make the CTA button more prominent? "
                "Maybe bump it up to 16px bold. Other than that, ship it!",
                "Hi team, quick update: API v2 is deployed to staging. All 47 tests "
                "pass. I'll monitor error rates over the weekend and we can cut the "
                "release Monday. Let me know if you spot anything weird.",
            ]

            reddit_samples = [
                "honestly the M4 is overkill for most people. unless you're doing "
                "ML inference or 8K video editing, the M3 Pro handles everything. "
                "saved $400 and zero regrets.",
                "lol at people saying RAG is dead. we literally just shipped a "
                "production RAG pipeline processing 50k docs/day. the trick is "
                "chunking strategy + reranking, not some magic model upgrade.",
                "this. people keep overcomplicating it. just use SQLite for anything "
                "under 10M rows and save yourself the DevOps headache. learned this "
                "the hard way after migrating off Postgres for a side project.",
            ]

            slack_samples = [
                "yo @channel heads up — deploying hotfix to prod in 10 min. "
                "should be zero downtime but keep an eye on #alerts",
                "lgtm, merging now. nice catch on the race condition btw",
                "haha yeah, classic off-by-one. fixed it and added a test so we "
                "don't get burned again",
            ]

            # Analyze each context
            contexts = {
                "email": email_samples,
                "reddit": reddit_samples,
                "slack": slack_samples,
            }

            voice_profiles = {}
            for ctx_name, samples in contexts.items():
                vp = analyze_style(samples, llm_reasoner=reasoner)
                if vp and vp.get("formality"):
                    voice_profiles[ctx_name] = vp
                    logger.info(
                        "  %s: %s, %s, %s",
                        ctx_name.upper(),
                        vp.get("formality", "?"),
                        vp.get("tone", "?"),
                        vp.get("sentence_style", "?"),
                    )
                    markers = vp.get("personality_markers", [])
                    if markers:
                        logger.info("    markers: %s", markers)
                    would_say = vp.get("would_say", "")
                    if would_say:
                        logger.info("    would say: \"%s\"", would_say[:80])
                    would_not = vp.get("would_never_say", "")
                    if would_not:
                        logger.info("    would NOT say: \"%s\"", would_not[:80])
                else:
                    logger.warning("  %s: no voice profile returned", ctx_name)

            results["voice_profiles"] = len(voice_profiles)
            if voice_profiles:
                # Save to KB as if it came from procedures
                for ctx, vp in voice_profiles.items():
                    proc = {
                        "id": f"test-{ctx}",
                        "title": f"Test {ctx} workflow",
                        "voice_profile": vp,
                    }
                    kb.save_procedure(proc)

                # Now test user-level aggregation
                from agenthandover_worker.style_analyzer import aggregate_user_style
                all_procs = [kb.get_procedure(p["id"]) for p in kb.list_procedures()]
                all_procs = [p for p in all_procs if p]
                user_style = aggregate_user_style(all_procs)
                if user_style:
                    per_wf = user_style.get("per_workflow", [])
                    logger.info("  User-level voice: %s / %s", user_style.get("formality"), user_style.get("tone"))
                    logger.info("  Per-workflow breakdown: %d contexts", len(per_wf))
                    for w in per_wf:
                        logger.info("    %s: %s", w.get("procedure"), w.get("formality"))
        except ConnectionError:
            logger.warning("  Qwen/Ollama not available — voice analysis skipped")
            results["voice_profiles"] = -1  # -1 = skipped
        except Exception as e:
            results["errors"].append(f"voice: {e}")
            logger.error("  FAILED — %s", e, exc_info=True)
            results["voice_profiles"] = 0

        # ---- Step 5: Embeddings ----
        logger.info("\n=== Step 5: Vector embeddings ===")
        try:
            from agenthandover_worker.vector_kb import VectorKB, VectorKBConfig
            vkb = VectorKB(db_path, VectorKBConfig())
            # Test embedding availability
            test = vkb.compute_embeddings(["health check"])
            if not test or not test[0]:
                logger.warning("  Embedding model not available — skipping")
            else:
                # Embed all procedures
                procs = kb.list_procedures()
                for proc in procs:
                    full = kb.get_procedure(proc.get("id", ""))
                    if full:
                        parts = [full.get("title", ""), full.get("description", "")]
                        for step in full.get("steps", []):
                            parts.append(step.get("action", ""))
                        text = "\n".join(p for p in parts if p)
                        vkb.upsert("procedure", full["id"], text[:4000])

                # Embed profile
                profile_data = kb.get_profile()
                if profile_data:
                    vkb.upsert("profile", "user", json.dumps(profile_data, default=str)[:4000])

                # Embed daily summaries
                for date_str in sorted(days.keys()):
                    summary_data = kb.get_daily_summary(date_str)
                    if summary_data:
                        vkb.upsert("daily_summary", date_str, json.dumps(summary_data, default=str)[:4000])

                results["vectors_stored"] = vkb.count()
                logger.info("  Vectors stored: %d", results["vectors_stored"])

                # Test semantic search
                search_results = vkb.search("email inbox morning", top_k=3)
                logger.info("  Search 'email inbox morning':")
                for r in search_results:
                    logger.info("    %s/%s — score %.3f", r.source_type, r.source_id, r.score)

                vkb.close()
        except Exception as e:
            results["errors"].append(f"embeddings: {e}")
            logger.error("  FAILED — %s", e)

        # ---- Step 6: KB inventory ----
        logger.info("\n=== Step 6: Knowledge Base inventory ===")
        for root, dirs, files in os.walk(kb_root):
            level = root.replace(str(kb_root), "").count(os.sep)
            indent = "  " * (level + 1)
            for f in sorted(files):
                fpath = os.path.join(root, f)
                size = os.path.getsize(fpath)
                logger.info("%s%s (%d bytes)", indent, f, size)

    return results


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("AgentHandover Passive Pipeline E2E Test")
    logger.info("=" * 60)

    days = generate_days(5)
    results = run_pipeline(days)

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    vp_count = results.get("voice_profiles", 0)
    vp_skipped = vp_count == -1
    checks = [
        ("Daily summaries", results["daily_summaries"] >= 5, f"{results['daily_summaries']}/5"),
        ("Profile built", results["profile_built"], "tools/hours/style detected"),
        ("Patterns detected", results["patterns_detected"] > 0, f"{results['patterns_detected']} patterns"),
        ("Digest generated", results["digest_generated"], "highlights + sections"),
        ("Voice profiles (Qwen)", vp_count > 0 or vp_skipped, f"{vp_count} contexts" if not vp_skipped else "skipped (no Ollama)"),
        ("Vectors stored", results["vectors_stored"] > 0, f"{results['vectors_stored']} vectors"),
    ]

    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        logger.info("  [%s] %s — %s", status, name, detail)

    if results["errors"]:
        logger.info("\n  Errors:")
        for err in results["errors"]:
            logger.info("    - %s", err)

    logger.info("")
    if all_pass:
        logger.info("ALL CHECKS PASSED")
    else:
        logger.info("SOME CHECKS FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
