# OpenMimic — Issue Tracker Lists

Compiled 2026-03-06 from external AI audit, verified against codebase.

---

## List 1 — Critical (breaks correctness, loses data, or blocks users)

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 1 | ISO vs `datetime()` timestamp comparison — passive windowing and episode retention queries over-include stale events because `timestamp` uses `T` separator while `datetime()` returns space separator, making text comparison unreliable | db.py:706,806 | DONE |
| 2 | Passive segment bookkeeping — transient cluster IDs overwrite `sop_generated` via INSERT OR REPLACE, causing duplicate SOP generation and wasted VLM budget | db.py:878, main.py | DONE |
| 3 | Passive generation uses in-memory segmentation instead of DB-backed pending state — no replay safety, bookkeeping is disconnected from generation decisions | main.py | DONE |
| 4 | Extension leaks native listeners on reconnect — `onNativeMessage` and `onNativeDisconnect` callbacks accumulate in Sets, never unsubscribed | background.ts:49, native-messaging.ts:217 | DONE |
| 5 | Extension reconnect replays buffered messages but daemon has no seq-based dedup — can create duplicate browser events | native-messaging.ts:195, native_messaging.rs | DONE |
| 6 | postinstall exits 1 on recoverable warnings — macOS installer treats this as package failure, may trigger rollback | postinstall:119 | DONE |
| 7 | Swift app pulls `llava:7b` but pipeline uses `qwen3.5:2b/4b` — user gets wrong model on first setup | OnboardingView.swift:383,715 | DONE |
| 8 | `just build-all` skips worker and app — documented source-build path is incomplete, setup.rs expects launchd assets that only exist post-install | justfile:7, README.md:38 | DONE |

---

## List 2 — Important (degrades UX, reliability, or scalability)

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 9 | Messages dropped during disconnect window never enter resend buffer — permanently lost | background.ts:186 | TODO |
| 10 | Onboarding blocks "Start Observing" on extension connection — native-only capture (screenshots, accessibility, clipboard) is unusable without Chrome | OnboardingView.swift:113 | TODO |
| 11 | Health checks use launchd job presence, not heartbeat/PID liveness — services can appear healthy while actually hung | ServiceController.swift:100, doctor.rs:74, setup.rs:232 | TODO |
| 12 | Focus recording first frame may diff against pre-session activity — polluted context from prior work | focus_processor.py | TODO |
| 13 | Passive SOP prompt dumps all raw frames from all demos — no sampling or summarization, scales poorly with long sessions | sop_generator.py:260 | TODO |
| 14 | Passive SOP JSON parse failure has no retry — single malformed VLM response wastes entire inference budget | sop_generator.py:861 | TODO |
| 15 | Homebrew/source installs don't ship menu bar app — best UX only exists in .pkg path | openmimic.rb:25, build-pkg.sh:69 | TODO |
| 16 | Extension README native-host install command path is wrong after `cd extension` | extension/README.md:21 | TODO |
| 17 | Passive discovery should produce draft SOPs for approve/edit/reject, not auto-trusted outputs | Product gap | TODO |
| 18 | User corrections to SOPs should feed back into future generation | Product gap | TODO |
| 19 | Failed focus/passive generations need retry/resume UX, not silent log failures | Product gap | TODO |
| 20 | First-run should be tiered: basic capture, browser-rich capture, vision-rich capture | Product gap | TODO |

---

## List 3 — Enhancements (product depth, agent quality, user trust)

| # | Issue | Location | Status |
|---|-------|----------|--------|
| 21 | App needs workflow inbox/gallery — show drafts, high-confidence, stale, recently improved SOPs | Product | TODO |
| 22 | Surface ROI: repeat count, estimated time saved, confidence trend, agent-ready score | Product | TODO |
| 23 | Install channels should be explicit: pkg=end-user, Homebrew=technical, source=developer | Product | TODO |
| 24 | SOPs should add Outcome block, Before You Start block, per-step verification | SOP format | TODO |
| 25 | Variables should be typed with names, examples, validation hints, sensitivity flags | SOP format | TODO |
| 26 | SOPs should model decision points/branches (if logged in, skip; if no results, check filters) | SOP format | TODO |
| 27 | SOPs should mark risky steps (submit, delete, send) for approval gates | SOP format | TODO |
| 28 | SOPs should include stop conditions, recovery paths, example runs | SOP format | TODO |
| 29 | SOPs should carry change history and freshness metadata (demo count, last confirmed, drift notes) | SOP format | TODO |
| 30 | Agent SOPs should be structured execution contracts (SOP.json) with SKILL.md as human-readable layer | Agent design | TODO |
| 31 | Agent SOPs need typed inputs, machine-checkable preconditions, step-level postconditions, retry/on_fail, risk flags | Agent design | TODO |
| 32 | Generator should lint/compile SOPs before export — validation gate for invalid workflows | Agent design | TODO |
| 33 | Better terminal/editor/IDE semantics for developer workflows | User segment | TODO |
| 34 | Stronger spreadsheet/table/form semantics for analyst/ops users | User segment | TODO |
| 35 | Privacy controls: per-app capture modes, pause/delete/audit history, clear data explanation | User trust | TODO |
| 36 | Preserve demonstration evidence behind each SOP for trust and debugging | SOP generation | TODO |
| 37 | Post-generation micro-questions: confirm name, what varies, what success looks like | SOP generation | TODO |
