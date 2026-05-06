# Changelog

All notable changes to AgentHandover are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project follows [Semantic Versioning](https://semver.org/) within the 0.x line
(minor bumps may include breaking changes; the README's Changelog section has the
prose version of every release).

## [0.3.0] — 2026-05-06

The biggest Skill-quality jump since v0.2.0. Fixes a chain of silent data losses between
the VLM, the saved procedure JSON, and the app UI.

### Fixed
- **Step descriptions saved AND rendered.** `procedure_schema.sop_template_to_procedure()`
  was dropping the per-step `description` field on the way to the saved JSON, and
  `SOPDetailView.swift` was rendering description as a `??`-fallback for action — meaning
  whenever action was populated (always), description was silently swallowed by the
  fallback chain. Both gaps closed.
- **Behavioral synthesis fails loud.** Empty-insights returns from the VLM now raise
  `EmptyInsightsError`, retry once, and only stamp `last_synthesized` when the
  extraction was substantive — instead of silently leaving procedures with empty
  behavioral fields.
- **Voice profile reads real user-authored text.** Style analyzer now reads from
  clipboard events, step inputs/descriptions, and content samples — not only the
  late-populated `extracted_evidence.content_produced`. Min combined text 50 → 30 chars.
- **Q&A no longer corrupts procedures.** Removed `_merge_credentials()` and
  `_merge_decision()` — free-text answers were being auto-merged into structured
  `accounts` / `branches` / `decision` fields, overwriting clean data with prose.
  `FocusQuestion` now carries a `step_indexes` field; clarifications rewrite the
  specific steps they cover, in place.
- **Variables wired into step text.** New `_wire_variables_into_steps()` post-pass
  substitutes example values with `{{varname}}` templates across step text and
  parameters.
- **Brace double-wrap fix:** `_unwrap_double_templated()` collapses
  `{{{{var}}suffix}}` (double-wrapped templated reference, occasionally emitted by
  Gemma) to `{{var}}suffix`.

### Changed
- VLM SOP prompts (`FOCUS_SOP_PROMPT`, `PASSIVE_SOP_PROMPT`, `ENRICHED_PASSIVE_PROMPT`)
  now require `description` and `verify` per step.
- Coherence-check rule refined to keep intermediate actions instead of dropping them
  as "unused later".
- Swift step card layout: action (bold) + description (dim) + `→ target` (monospaced)
  + `Input:` + `✓ verify`, all conditional.

### Tests
- 3026/3026 Python tests pass. New regression test
  `test_write_preserves_step_description` locks the procedure_schema fix.
- Validated end-to-end on the dailynews focus session: 19/19 steps with rich
  descriptions and verifies, 0 brace bugs, 0 Q&A corruption, behavioral synthesis
  confidence 0.81 with retries=0.

## [0.2.10] — 2026-04-18

Detach daemon from launching shell via `setsid()`. Fixes the daemon silently dying
when the launching shell exits — SIGHUP from shell exit was reaching the daemon
because it stayed in the shell's process group. `libc::setsid()` at startup makes
the daemon its own session leader; SIGHUP no longer reaches it. Defense-in-depth
SIGHUP handler also added.

## [0.2.9] — 2026-04-16

Removes the remaining three `tokio::time::timeout` + `spawn_blocking` crash sites
(clipboard monitoring, accessibility checks, AppleScript queries) that had the same
shape as the v0.2.8 OCR crash. Tokio dropping the future on timeout while the
blocking thread keeps running ObjC calls leads to uncaught exceptions when those
calls finally return — `abort()`. All four blocking calls now run to completion.

## [0.2.8] — 2026-04-15

Fixes a daemon abort caused by a Tokio-level OCR timeout orphaning an in-flight
Vision framework call. Drops the OCR timeout, installs a panic hook, and redirects
daemon stderr to `daemon.stderr.log` so future ObjC exceptions and panic
backtraces leave a diagnostic trail.

## [0.2.7] — 2026-04-14

Clean status reporting after "Observe Me" toggle. Daemon clears its own
`daemon-status.json` on shutdown; menu bar app removes the native messaging host
manifest on pause; CLI `stop daemon` always clears the status file.

## [0.2.6] — 2026-04-12

Worker state detection (`is_job_running()` checks the actual process via
`launchctl print` + pid parsing), native host manifest reliability on upgrade
(postinstall writes the manifest directly), and early status file visibility
(worker writes a "starting" status before heavy initialization).

## [0.2.5] — 2026-04-12

Postinstall extension manifest robustness — see README Changelog for details.

## [0.2.4] — 2026-04-11

Daemon spawn detach fix.

## [0.2.3] — 2026-04-11

Worker startup `UnboundLocalError` fix (`FocusProcessor` shadowing), doctor stale
checks cleaned up, SOP unused-variable cleanup pass, Q&A subprocess uses
configured model, latent shadow-imports proactively removed, SOP coherence
distraction filter rule.

## [0.2.2] — 2026-04-11

Preinstall stale-plist cleanup, CLI `start`/`stop` spawn daemon directly (no more
launchd plist that doesn't ship), rich observations ported to daily re-synthesis,
Homebrew formula → cask rewrite.

## [0.2.1] — 2026-04-10

Critical bugfixes for focus recordings + behavioral synthesis quality:
`LSUIElement=true` for menu bar registration, `v2_cfg` scoping bug, focus-session
race in passive annotation, clipboard attach reading wrong JSON field, OCR
injection into VLM annotation prompt, behavioral synthesizer goal field,
timeline evidence section in synthesizer prompt, rich pre-analysis observations,
compose-specific structured fields, post-SOP synthesizer rich observations.

## [0.2.0] — 2026-04-10

First public release.

[0.3.0]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.3.0
[0.2.10]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.10
[0.2.9]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.9
[0.2.8]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.8
[0.2.7]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.7
[0.2.6]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.6
[0.2.5]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.5
[0.2.4]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.4
[0.2.3]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.3
[0.2.2]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.2
[0.2.1]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.1
[0.2.0]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.2.0
