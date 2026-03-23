<p align="center">
  <img src="resources/favicon.png" width="140" alt="AgentHandover" />
</p>

<h1 align="center">AgentHandover</h1>

<p align="center">
  <strong>Work once. Hand over forever.</strong>
</p>

<p align="center">
  <a href="#what-you-get">What You Get</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#install">Install</a> &middot;
  <a href="#the-menu-bar-app">The App</a> &middot;
  <a href="#for-agent-developers">For Agent Developers</a> &middot;
  <a href="#privacy">Privacy</a>
</p>

---

You already know how to do your work. Now your agents can too.

AgentHandover watches you work on your Mac, understands what you're doing and why, and produces step-by-step procedures that agents like **OpenClaw**, **Claude Code**, **Codex**, and any AI agent can follow. Not screen recordings. Not macros. Actual procedures with strategy, decision criteria, guardrails, and timing - the kind of handoff a capable agent needs to do the work exactly how you would.

You keep working normally. AgentHandover writes the manual.

## What You Get

Here is a real example of what AgentHandover produces. This procedure was generated from a few focus recording sessions of Reddit community engagement:

```
Reddit Community Marketing
Daily engagement workflow - 6 steps - 4 sessions learned

STRATEGY
Browse target subreddits for posts about marketing tools or growth
hacking. Engage with high-signal posts (10+ comments, posted within
48h, not promotional). Write authentic replies that acknowledge the
problem, share personal experience, and softly mention the product.

STEPS
1. Open Reddit and navigate to r/startups
2. Scan posts - skip promotional, skip < 10 comments
3. Open high-signal post and read top comments
4. Write reply: acknowledge -> experience -> mention product
5. Submit and verify not auto-removed
6. Repeat for r/marketing, r/growthacking (max 5/day)

SELECTION CRITERIA              GUARDRAILS
- Posts with 10+ comments       - Max 5 replies per day
- Not promotional or competitor - Never identical phrasing
- Posted within 48 hours        - Never reply to own posts
- Relevant to [product category]- Empathy-first tone always

~15 min daily - 9-10am                     Confidence: 89%
```

This is what your agent receives. Not just steps, but the strategy, decisions, and guardrails behind them.

## How It Works

### Two ways to teach it

**Focus Recording** - Click Record in the menu bar, name the task, perform it, click Stop. A few minutes later, you get a complete procedure. Best for workflows you want to hand over right now.

**Passive Discovery** - Just work normally. When AgentHandover sees the same task appear in 2+ sessions, it generates a procedure automatically. No action required. Good for catching workflows you didn't think to record.

### Focus Q&A

After a focus recording, AgentHandover asks 1-3 targeted questions before producing the final deliverable. These fill in gaps that screen observation alone can't capture:

- "Does this workflow require logging into Reddit?"
- "What determines which posts you engage with vs. skip?"
- "How do you verify the reply wasn't auto-removed?"

Your answers are merged into the procedure so the agent has complete context. You can skip questions and it will use reasonable defaults.

### What happens under the hood

A local vision-language model (running on your machine, not in the cloud) watches each screenshot and produces a structured annotation: what app is open, what URL, what you're doing, what you'll likely do next. When the Chrome extension is loaded, this is enriched with real DOM context - CSS selectors, ARIA labels, form field IDs - giving agents precise targets for browser automation.

An activity classifier separates signal from noise. Your expense filing is "work." Your YouTube break is "entertainment." Only workflow-relevant activity feeds into learning.

A continuity tracker links related work across interruptions. Start a PR review, get pulled into Slack, come back 20 minutes later - AgentHandover knows it's the same task.

When the same workflow appears in multiple demonstrations, steps are aligned semantically (not by position), parameters are extracted, branches are detected, and the result is a single canonical procedure with confidence scores.

After 3+ observations accumulate, a behavioral synthesis pass extracts the higher-level patterns: overall strategy, selection criteria, content templates, guardrails, decision branches, and timing patterns.

### Screenshots are temporary

Screenshots are captured as half-resolution JPEGs, deduplicated via perceptual hashing (70% of frames are duplicates and get dropped immediately), and deleted after the vision model processes them - whether the annotation succeeds or fails. Only the structured annotation (~500 bytes) is kept. Your screen content never accumulates on disk.

## Install

### Download and run

Download the latest `.pkg` from [**Releases**](https://github.com/sandroandric/OpenMimic/releases) and double-click to install.

The app opens a guided setup that walks you through everything: permissions, vision model download, and Chrome extension install. Follow the prompts and you're done.

<details>
<summary><strong>Developer / advanced install</strong></summary>

### CLI setup (if you skip the onboarding)

```bash
agenthandover doctor     # Verify all prerequisites
agenthandover start all  # Start daemon + worker
```

Fix any `FAIL` items from `doctor`. Usually:
- **Accessibility** - System Settings > Privacy & Security > Accessibility > add `agenthandover-daemon`
- **Screen Recording** - same location

### Pull VLM models manually (~6 GB for default local models)

```bash
ollama pull qwen3.5:2b         # Scene annotation
ollama pull qwen3.5:4b         # SOP generation
ollama pull all-minilm:l6-v2   # Task embeddings
```

Or: `agenthandover setup --vlm` for guided setup (includes cloud API option).

### Load Chrome extension manually

Open `chrome://extensions` > Enable Developer Mode > Load unpacked > select the extension directory shown by `agenthandover doctor`.

### Homebrew

```bash
brew tap sandroandric/agenthandover
brew install --HEAD agenthandover
```

### Source build (requires Rust, Node.js 18+, Python 3.11+)

```bash
git clone https://github.com/sandroandric/OpenMimic.git && cd OpenMimic
just build-all           # Daemon, CLI, worker venv, extension, app
./scripts/setup.sh       # Native messaging host + VLM setup

# Install launchd services (rewrite paths for source build):
cp resources/launchd/com.agenthandover.*.plist ~/Library/LaunchAgents/
sed -i '' "s|/usr/local/bin/agenthandover-daemon|$(pwd)/target/release/agenthandover-daemon|" \
    ~/Library/LaunchAgents/com.agenthandover.daemon.plist
sed -i '' "s|/usr/local/lib/agenthandover/venv/bin/python|$(pwd)/worker/.venv/bin/python|" \
    ~/Library/LaunchAgents/com.agenthandover.worker.plist
sed -i '' "s|/usr/local/lib/agenthandover|$(pwd)/worker|" \
    ~/Library/LaunchAgents/com.agenthandover.worker.plist
```

</details>

### Choose your vision model

AgentHandover defaults to local Qwen models via Ollama because they're free, fast, and private. You can use any of 6 supported backends:

| Backend | How to use | Best for |
|---------|-----------|----------|
| **Ollama** (default) | `ollama pull qwen3.5:2b` | Local, free, private |
| **MLX** | Apple Silicon native | Fastest on Mac |
| **llama.cpp** | CPU/GPU flexible | Cross-platform local |
| **OpenAI** | `mode = "remote"` in config | Highest quality (GPT-4o) |
| **Anthropic** | `mode = "remote"` in config | Claude vision |
| **Google** | `mode = "remote"` in config | Gemini vision |

Switch models by editing `annotation_model` and `sop_model` in config, or run `agenthandover setup --vlm` for guided setup. Remote APIs require explicit opt-in and show a privacy warning.

## The Menu Bar App

AgentHandover lives in your menu bar. Click the icon and you'll see:

- **Status** - whether the daemon and worker are running, with a green/yellow/red indicator
- **Today's stats** - events captured, annotations completed, procedures generated
- **Attention items** - Focus Q&A questions waiting for answers, draft procedures ready for review
- **Record button** - one click to start a focus recording, name it, perform the workflow, stop
- **Workflows** - browse all your procedures, approve drafts, see confidence and observation counts
- **Digest** - daily summary of what was learned, what changed, what needs your attention

### Review and approve

Generated procedures appear as drafts in the Workflows view. Review the procedure - strategy, steps, guardrails, everything. Click "Approve for Agents" when it looks good. One click.

No procedure reaches agents without your sign-off. The system suggests promotions based on evidence (observation count, confidence, execution success rate) but never auto-promotes.

## For Agent Developers

### Query API

The worker runs a local HTTP API on port 9477:

```bash
# Discover agent-ready procedures (only returns executable ones)
curl http://localhost:9477/ready

# Discover ALL procedures with readiness info (includes drafts, blocked_by reasons)
curl http://localhost:9477/available

# Get a full handoff bundle for execution
curl http://localhost:9477/bundle/file-expense-report

# List all procedures (any lifecycle state)
curl http://localhost:9477/procedures

# Browse curation queue
curl http://localhost:9477/curation/queue

# Promote via API
curl -X POST http://localhost:9477/curation/promote \
  -H 'Content-Type: application/json' \
  -d '{"slug": "file-expense-report", "to_state": "agent_ready"}'
```

`/ready` returns only procedures that pass all readiness gates. If it's in the response, an agent can execute it. `/available` returns every procedure with its full readiness assessment, including `blocked_by` reasons - use it for browsing, dashboards, and agent discovery of draft work.

### Export formats

Approved procedures are compiled into target-specific formats from a single canonical source:

| Format | Location | Used by |
|--------|----------|---------|
| **SKILL.md** | `~/.openclaw/workspace/memory/apprentice/sops/` | OpenClaw agents |
| **Claude Code Skill** | `~/.claude/skills/<slug>/SKILL.md` | Claude Code (`/skill-name`) |
| **v3 Procedure JSON** | `~/.agenthandover/knowledge/procedures/` | Any agent via Query API |

### CLI reference

| Command | Description |
|---------|-------------|
| `agenthandover status` | Service health and stats |
| `agenthandover start all` | Start daemon + worker |
| `agenthandover stop all` | Stop services |
| `agenthandover focus start "title"` | Record a workflow |
| `agenthandover focus stop` | Stop recording, generate procedure |
| `agenthandover sops list` | List all procedures |
| `agenthandover sops drafts` | List procedures awaiting review |
| `agenthandover sops approve <slug>` | Approve a draft for export |
| `agenthandover sops promote <slug> <state>` | Promote lifecycle (e.g., `reviewed`, `agent_ready`) |
| `agenthandover doctor` | Pre-flight health check |
| `agenthandover watch` | Live dashboard |
| `agenthandover export --format claude-skill` | Re-export as Claude Code skills |
| `agenthandover logs worker -f` | Follow worker logs |

<details>
<summary><strong>How the lifecycle works</strong></summary>

Each procedure moves through a lifecycle with your approval at each stage:

```
Observed -> Draft -> Reviewed -> Verified -> Agent Ready
```

A procedure must pass multiple readiness gates before agents can use it:

| Gate | What it checks | Who decides |
|------|---------------|-------------|
| **Lifecycle** | Has the human reviewed and promoted it through each stage? | Human |
| **Trust level** | Is the agent authorized to execute (not just observe or draft)? | Human (via trust suggestions) |
| **Freshness** | Has the procedure been observed recently? Stale procedures auto-demote. | System |
| **Preflight** | Are required apps running? Any blocked domains? Steps present? | System |
| **Evidence** | How many observations? Any contradictions? What's the confidence? | System |
| **Execution history** | Has it succeeded when agents tried it before? 3+ failures auto-demote. | System |

All six must pass. A procedure with lifecycle=agent_ready but low freshness won't execute. A fresh procedure with trust=observe won't execute either. The system is truthful about readiness - it will never tell an agent a procedure is ready when it isn't.

The menu bar app surfaces:
- **Draft procedures** to approve or reject
- **Trust suggestions** - earned enough evidence for higher agent permissions
- **Lifecycle upgrades** - ready for promotion based on observation evidence
- **Merge candidates** - similar procedures that might be duplicates
- **Drift alerts** - procedures whose observed behavior has changed
- **Stale alerts** - procedures not seen recently

</details>

<details>
<summary><strong>Architecture</strong></summary>

```
┌──────────────────────────────────────────────────────┐
│              Menu Bar App (SwiftUI)                   │
│   Status · Focus recording · Workflows · Digest      │
└───────────────────────┬──────────────────────────────┘
                        │ trigger files (JSON)
                        ▼
Chrome Extension ──→ Daemon (Rust) ──SQLite WAL──→ Worker (Python)
  DOM snapshots       Screenshots                    ┌──────────────────┐
  Click intent        OS Accessibility               │ Pipeline v2 +VLM │
  Secure fields       Clipboard + dHash              │ Variant alignment │
                      Focus session tags             │ Behavioral synth  │
                                                     └────────┬─────────┘
                                                               │
                                    ┌──────────────────────────┼──────────────────────────┐
                                    ▼                          ▼                          ▼
                              SKILL.md SOPs          v3 Procedures (KB)            Query API
                              (OpenClaw,             (strategy, guardrails,        (port 9477)
                               Claude Code)           templates, evidence)
```

| Component | Language | Role |
|-----------|----------|------|
| **Daemon** | Rust | Always-on observer - screenshots, OS events, clipboard, dHash dedup |
| **Worker** | Python | Pipeline - VLM annotation, classification, segmentation, variant alignment, SOP generation, behavioral synthesis, evidence extraction, lifecycle, curation |
| **Extension** | TypeScript | Chrome MV3 - DOM snapshots, click intent, dwell/scroll tracking |
| **CLI** | Rust | Service management, focus recording, SOP approval, lifecycle promotion |
| **App** | SwiftUI | Menu bar - status, focus recording, workflows, daily digest |

</details>

<details>
<summary><strong>Processing budget</strong></summary>

AgentHandover uses ~40% of GPU time per work hour. 36 minutes of headroom remain for your own GPU tasks.

| Stage | Time per hour | Notes |
|-------|---------------|-------|
| Scene annotation | 15.6 min | ~75 frames after dedup and stale-skip |
| Frame diffs | 4.5 min | Consecutive frame comparison |
| Task segmentation | 0.8 min | CPU only (embeddings) |
| SOP generation | 2.4 min | Thinking mode, enriched with variant analysis |
| Behavioral synthesis | 0.5 min | Daily batch only, when 3+ observations accumulate |

</details>

<details>
<summary><strong>Why not just screen recording?</strong></summary>

Screen recording gives you pixels. AgentHandover gives you understanding.

| | Screen recording | AgentHandover |
|---|---|---|
| **What it captures** | Raw video frames | Structured intent + DOM context |
| **What it knows** | Nothing - just pixels | App context, task purpose, step sequence, CSS selectors, form field IDs, ARIA labels, verification criteria |
| **How it handles noise** | Records everything equally | Classifies activity into 8 types and filters noise automatically |
| **How it handles interruptions** | Breaks the recording | Tracks task continuity across interruptions and reconnects the workflow |
| **What happens with repetition** | Multiple identical recordings | Demonstrations are aligned and merged into one canonical procedure with typed variables and confidence scores |
| **What the output looks like** | A video file | A structured procedure with steps, strategy, selection criteria, guardrails, timing, DOM hints, and verification criteria |
| **Can an agent use it?** | No | Yes - with lifecycle gates, readiness checks, and execution monitoring |

</details>

## Privacy

AgentHandover is designed to never leave your machine:

- **Local-first.** All VLM inference runs locally via Ollama by default. Cloud APIs are opt-in with explicit consent and a privacy warning.
- **Screenshots are temporary.** Raw JPEGs are deleted immediately after the vision model annotates them. Only the structured annotation (~500 bytes) is kept - never the screenshot itself.
- **Auto-redaction.** API keys, tokens, passwords, and credit card numbers are detected and scrubbed before storage.
- **Secure field exclusion.** Password and credit card inputs are dropped entirely - never captured, never stored.
- **Encryption at rest.** Artifacts use zstd compression + XChaCha20-Poly1305.
- **Configurable retention.** Raw events pruned after 14 days, episodes after 90 days. Valuable evidence (content patterns, engagement signals, timing) is extracted and stored permanently on procedures before raw events expire.
- **No telemetry.** Pipeline metrics are local-only JSON files. Nothing phones home. Ever.

## Configuration

Config lives at `~/Library/Application Support/agenthandover/config.toml`.

<details>
<summary><strong>Full configuration reference</strong></summary>

### Observer

| Key | Default | Description |
|-----|---------|-------------|
| `t_dwell_seconds` | 3 | Inactivity before dwell snapshot |
| `screenshot_max_per_minute` | 20 | Screenshot rate limit |
| `screenshot_quality` | 70 | JPEG quality 1-100 |
| `screenshot_scale` | 0.5 | Resolution scale (0.5 = half) |

### VLM

| Key | Default | Description |
|-----|---------|-------------|
| `annotation_model` | qwen3.5:2b | Any Ollama model name, or cloud model ID |
| `sop_model` | qwen3.5:4b | SOP generation model |
| `mode` | local | `local` (Ollama) or `remote` (cloud API) |
| `provider` | - | For remote: `openai`, `anthropic`, or `google` |
| `max_jobs_per_day` | 50 | VLM inference budget |
| `max_compute_minutes_per_day` | 20 | GPU time budget |

### Features

| Key | Default | Description |
|-----|---------|-------------|
| `activity_classification` | true | 8-class activity taxonomy |
| `continuity_tracking` | true | Task continuity across interruptions |
| `lifecycle_management` | true | 7-state procedure lifecycle |
| `curation` | true | Merge/upgrade/drift detection |
| `runtime_validation` | true | Pre-execution app-running checks |

### Privacy

| Key | Default | Description |
|-----|---------|-------------|
| `enable_inline_secret_redaction` | true | Auto-redact API keys, tokens |
| `secure_field_drop` | true | Drop events from password fields |

### Storage

| Key | Default | Description |
|-----|---------|-------------|
| `retention_days_raw` | 14 | Days to keep raw events |
| `retention_days_episodes` | 90 | Days to keep episodes |

</details>

## Troubleshooting

<details>
<summary>Services not starting</summary>

```bash
agenthandover doctor        # Check all prerequisites
agenthandover logs daemon   # Daemon-specific errors
agenthandover logs worker   # Worker-specific errors
```

</details>

<details>
<summary>No events being captured</summary>

- Verify Accessibility permission: System Settings > Privacy & Security > Accessibility
- Check `agenthandover status` for daemon health
- Ensure Chrome extension is loaded and enabled

</details>

<details>
<summary>No procedures being generated</summary>

- **Passive mode** requires 2+ similar demonstrations of the same workflow within a 24-hour window
- Verify Ollama is running: `ollama list`
- Check worker logs: `agenthandover logs worker -f`
- Ensure `annotation_enabled = true` in config

</details>

<details>
<summary>Extension not connecting</summary>

- Run `agenthandover doctor` to verify native messaging host
- Reload the extension in `chrome://extensions`
- Check Chrome developer console for errors

</details>

## Uninstall

```bash
agenthandover uninstall              # Remove services, keep data
agenthandover uninstall --purge-data # Remove everything
```

## License

MIT
