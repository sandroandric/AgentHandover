# Changelog

All notable changes to AgentHandover are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project follows [Semantic Versioning](https://semver.org/) within the 0.x line
(minor bumps may include breaking changes; the README's Changelog section has the
prose version of every release).

## [0.5.0]

The release where the product does what the README says. Every claim below
names the journey test that proves it (`worker/tests/journeys/`, replayed
from recorded model output) or the measurement behind it (single runs on a
16 GB test Mac with Ollama 0.34.2). Journeys still waiting on a recording
are marked *(not yet recorded)*.

### Changed
- **Models.** 16 GB: `qwen3.5:4b` annotates and writes Skills (was Gemma 4
  12B QAT). 8 GB: `qwen3.5:2b-q4_K_M` annotates, `qwen3.5:4b` writes Skills.
  24/48 GB keep Gemma 4 12B/31B QAT for Skills. Measured head to head on a
  recorded session's prompts: Skill in 153 s vs 230 s, 29.5 vs 12.1 tok/s,
  18 vs 10 steps; the 8 GB pair was measured on the 16 GB Mac, not on an
  8 GB one. Stop to questions on a live session: 50 min → ~2.5 min.
  Proof: J-focus (golden packet), `agenthandover bench`.
- **Memory-aware scheduling.** One model at a time on two-model tiers, a
  memory/swap/disk check before loading, early deferral for a model that
  does not load, one inference thread in priority order. Proof: J-phases
  *(not yet recorded)*, J-ollama-down, `test_inference_runner.py`.
- **Event-driven capture** of the focused window at one pixel per point,
  with change detection before OCR. Proof: J-compute, `event_loop_test.rs`.
- **Text-first annotation** with a cache and hourly/daily compute caps: on
  the recorded busy hour, 10 model calls and 422 model-seconds (bound 600).
  Proof: J-compute.
- **Grounded Skill steps**: clipboard evidence for copy/paste, screen
  evidence for scroll/save, exact typed lines, DOM anchors, frame citations
  and recorded order. Proof: J-focus, J-focus-clipboard, J-focus-typed.
- **Knowledge ledger, hand-off gate and budgeted questions** (up to 3 after
  a recording, 5 a week; step questions quote the recording). The four-check
  hand-off gate replaces the six readiness gates. Proof: J-approve, J-mcp,
  J-focus-typed (step questions and answer revision), J-ledger (its
  skip/diverge leg is *still skipped*).
- **Recognition of repeated workflows**: recurrence mining with at most 3
  new drafts a week; a session or recording of an existing Skill becomes a
  run and a branch. Proof: `test_workflow_observations.py`, J-ledger; the
  three-week journey J-passive is *not yet recorded*.
- **Agent runs feed Skills**: three runs as written confirm the steps; a
  deviation on a guessed decision reopens it as a question. Proof: J-mcp,
  `test_mcp_feedback.py`.
- **Doctor checks the machine**: serving Ollama, duplicate installs,
  memory/swap/disk headroom; an old Ollama fails only for Gemma 4 configs.
  Proof: CLI unit tests; J-install *(runs on the test Mac only)*.
- **Upgrades keep observation on**, reset stale permission grants after a
  signature change, and migrate app-written model configs from earlier
  defaults. Proof: `UpgradeLaunchTests`, `test_tier_migration.py`, J-install.
- **Privacy**: redaction before storage and before every prompt, URL query
  strings dropped, system password/permission prompts blocked, privacy zones
  before and after the shutter, clipboard text off by default, app-state
  reading opt-in. Proof: J-privacy, `privacy_zones_test.rs`,
  `storage_redaction_test.rs`.
- **The in-app FAQ** says where data goes: every model runs on the Mac
  through Ollama. Proof: `FAQContentTests`.
- **Docs are pinned to the code**: README's MCP, tier and CLI tables are
  generated, every `agenthandover` command the README and FAQ mention must
  exist, and `config.example.toml` lists only keys some code reads. Proof:
  `tests/integration/test_docs.py`, `crates/cli/src/docs_tests.rs`.

### Removed
- **Cloud mode** (OpenAI, Anthropic, Google): AgentHandover is fully local.
  Cloud mode never did the main work: screen annotation and Skill writing
  always ran on the local Ollama models, even in 0.4.0. Once the fallback
  queue was gone it only sent the older focus path's Skill steps (with
  `[vlm] annotation_enabled = false`) for a task description, and that path
  read the API key from an environment variable while the app kept it in
  the Keychain, so choosing "Cloud" in onboarding did effectively nothing.
  Gone: onboarding's Local/Cloud choice and API-key field, the cloud
  backends, `agenthandover-setup-vlm` (its `--remote` setup; `agenthandover
  setup --vlm` pulls the local models), `[vlm] mode = "remote"`, `provider`,
  `model` and `api_key_env`, and `vlm_mode`/`vlm_provider` in
  `worker-status.json`. The upgrade (installer and worker start) rewrites a
  cloud `[vlm]` block to `mode = "local"` with a backup and one log line,
  keeping its tier and models; `agenthandover doctor` warns about a block
  it could not rewrite; the app deletes the stored API keys from the
  Keychain once. A remote `[vlm] ollama_host` is still Ollama and still
  supported. The older focus path's SOP text enhancer runs the tier's SOP
  model (or `[llm] model`) on that Ollama. Proof: `test_cloud_mode_removed.py`,
  `CloudModeRemovedTests`, J-privacy (zones survive the upgrade's rewrite).
- The MLX and llama.cpp backends (they only served the fallback queue; the
  pipeline always used Ollama), the legacy PrefixSpan miner, the LLM-written
  focus questions, and config keys nothing read.
- The VLM fallback queue. Every translation the event pipeline rejected
  queued a job whose answer fed only a confidence boost that decided whether
  another job was queued; nothing else read it, in local or cloud mode. The
  queue, its jobs and their worker are gone, and so are `vlm_queue_pending`,
  `vlm_jobs_today` and `vlm_dropped_today` in `worker-status.json` (and the
  menu bar's "Processing" count and health warning built on them).

### Fixed
- **The daily pass runs to its end again.** It read the day's tasks
  (`TaskBoundary` records) as dicts and stopped on every day with work,
  logged only at DEBUG, so the app's daily digest was never written, and
  the profile, patterns, session links, trust suggestions, behavioural
  re-synthesis, default constraints and the day's questions never ran from
  it. A failed daily batch now logs a warning. The profile's embedding no
  longer includes its save time, which made it a new embedding call every
  day. Proof: `test_daily_processor.py`; the three-week replay and
  J-passive now run the worker's own daily pass *(not yet recorded)*.
- **Cluster labels and synthesis no longer stop for the day** after 50
  rejected translations. The fallback queue was also the daily budget of the
  LLM reasoner and the behavioral synthesizer, so its 50 jobs a day skipped
  every later reasoning call, logged only at DEBUG (312 cluster labels in the
  three-week replay). They now have their own budget, `[vlm]
  reasoning_max_calls_per_day` (300) and
  `reasoning_max_compute_minutes_per_day` (90); when it is spent,
  `worker-status.json` says `reasoning_paused_reason:
  reasoning_budget_exhausted`, `agenthandover status --compute` shows it,
  and the worker logs it at INFO once a day. Proof:
  `test_reasoning_budget.py`; the replay's new cluster labels are *not yet
  recorded*.
- A review card's free-text answer to a step question now revises the
  step, and the revision survives a worker restart.

### Corrections to 0.4.0
- The Homebrew `ollama` formula runs the QAT tags (0.34.2 on the test Mac),
  and the Gemma tiers need Ollama 0.34.0+, not 0.30.6.

## [0.4.0] — 2026-06-09

New default local models — Google's QAT (Quantization-Aware Training) Gemma 4
checkpoints — chosen by a head-to-head A/B test on a real focus recording.

### Changed
- **16 GB tier: Gemma 4 E4B → Gemma 4 12B QAT** (`gemma4:12b-it-qat`, ~7 GB,
  down from ~9.6 GB). A/B on the dailynews recording: E4B returned "Unclear …
  does not complete a final artifact" on both runs; 12B QAT correctly
  identified the digest-email task, extracted a typed variable, and produced
  goal-directed steps both runs. A 12B-dense model now fits where a
  4B-effective one used to, and reasons far better.
- **24 GB tier: Gemma 4 E4B Q8 → Gemma 4 12B QAT** (~7 GB, down from ~12 GB).
- **48 GB tier: Gemma 4 31B → Gemma 4 31B QAT** (`gemma4:31b-it-qat`, ~18 GB,
  down from ~20 GB).
- **8 GB tier: unchanged (Qwen 3.5).** We A/B-tested the smallest Gemma QAT
  (e2b) here too — it was a regression (got the task wrong both runs while
  Qwen 3.5:4b nailed it), so the most constrained tier stays on Qwen.
- New `_PROFILES` entries for `gemma4:12b-it-qat` and `gemma4:31b-it-qat`.

### Required
- **Ollama 0.30.6+** for the Gemma tiers (QAT tags return HTTP 412 on older
  Ollama). `MINIMUM_OLLAMA_VERSION_FOR_GEMMA4` bumped 0.20.0 → 0.30.6.
- Onboarding, README, and the Homebrew cask now direct users to the **official
  Ollama app** (`ollama.com/download` / `brew install --cask ollama-app`)
  instead of the Homebrew `ollama` formula, which does not bundle the GGUF
  runner and cannot run these models. The 8 GB Qwen tier still works on older
  Ollama.

### Tests
- 3026/3026 Python tests pass. Tier/profile changes validated; Swift + Rust
  builds clean.

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

[0.4.0]: https://github.com/sandroandric/AgentHandover/releases/tag/v0.4.0
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
