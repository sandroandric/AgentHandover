<p align="center">
  <img src="https://raw.githubusercontent.com/sandroandric/OpenMimic/main/resources/icon.png" width="120" alt="OpenMimic icon" />
</p>

<h1 align="center">OpenMimic</h1>

<p align="center">
  <strong>Watch your screen. Learn your workflows. Hand them to agents.</strong>
</p>

<p align="center">
  <a href="#install">Install</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#usage">Usage</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#privacy">Privacy</a>
</p>

---

OpenMimic is a local, privacy-first apprentice that silently observes your macOS workflows, learns repeatable patterns, and produces semantic procedure files that AI agents can execute.

You keep working. OpenMimic watches, learns, and writes the manual.

## What You Get

```
You work normally on your laptop
        ↓
OpenMimic silently observes (screenshots + VLM annotation)
        ↓
Repeated workflows are detected automatically
        ↓
Semantic procedures generated (steps, variables, verification)
        ↓
Human reviews and approves in the menu bar app
        ↓
Agent-ready SKILL.md files exported to OpenClaw / Claude Code
```

**No macros. No DOM scripting.** OpenMimic captures *intent* — what you're doing and why — not pixel coordinates or CSS selectors. The output is a human-readable procedure that any AI agent can follow.

## Install

### Recommended: macOS Installer

Download the latest `.pkg` from [**Releases**](https://github.com/sandroandric/OpenMimic/releases) and double-click to install.

Then:

```bash
openmimic doctor     # Verify everything is set up
openmimic start all  # Start observing
```

That's it. The daemon, worker, CLI, and Chrome extension are installed to standard paths.

<details>
<summary><strong>Developer install</strong> (Homebrew or source)</summary>

**Homebrew:**
```bash
brew tap sandroandric/openmimic
brew install --HEAD openmimic
```

**Source build** (requires Rust, Node.js 18+, Python 3.11+):
```bash
git clone https://github.com/sandroandric/OpenMimic.git && cd OpenMimic
just build-all           # Daemon, CLI, worker venv, extension, app
./scripts/setup.sh       # Native messaging host + VLM setup

# Install launchd services (rewrite paths for source build):
cp resources/launchd/com.openmimic.*.plist ~/Library/LaunchAgents/
sed -i '' "s|/usr/local/bin/oc-apprentice-daemon|$(pwd)/target/release/oc-apprentice-daemon|" \
    ~/Library/LaunchAgents/com.openmimic.daemon.plist
sed -i '' "s|/usr/local/lib/openmimic/venv/bin/python|$(pwd)/worker/.venv/bin/python|" \
    ~/Library/LaunchAgents/com.openmimic.worker.plist
sed -i '' "s|/usr/local/lib/openmimic|$(pwd)/worker|" \
    ~/Library/LaunchAgents/com.openmimic.worker.plist
```

</details>

### First-time setup

After install, three things need to happen once:

**1. Grant permissions**

```bash
openmimic doctor
```

Fix any `FAIL` items. Usually:
- **Accessibility** — System Settings → Privacy & Security → Accessibility → add `oc-apprentice-daemon`
- **Screen Recording** — same location

**2. Pull VLM models** (~6 GB)

```bash
ollama pull qwen3.5:2b         # Scene annotation
ollama pull qwen3.5:4b         # SOP generation
ollama pull all-minilm:l6-v2   # Task embeddings
```

Or: `openmimic setup --vlm` for guided setup.

**3. Load Chrome extension** (optional, for richer SOPs)

Open `chrome://extensions` → Enable Developer Mode → Load unpacked → select the extension directory shown by `openmimic doctor`.

## How It Works

OpenMimic has two observation modes:

### Focus Recording — learn from one demonstration

Click **Record Workflow** in the menu bar, name it, perform the task, click **Stop**. A SKILL.md is generated in ~60 seconds.

```bash
openmimic focus start "File expense report"
# ... do the workflow ...
openmimic focus stop
```

### Passive Discovery — learn from repeated behavior

Just work normally. OpenMimic runs in the background:

| Stage | What happens | Speed |
|-------|-------------|-------|
| **Capture** | Screenshots deduplicated via perceptual hash (70% reduction) | Real-time |
| **Annotate** | Local VLM describes what's on screen and what you're doing | ~12s/frame |
| **Classify** | 8-class activity taxonomy separates work from noise | Instant |
| **Segment** | Embedding similarity clusters related work into tasks | Batch |
| **Generate** | SOP produced when same task seen 2+ times | ~72s |
| **Deduplicate** | Fingerprint matching prevents duplicate SOPs | Instant |

### Review and approve

Generated procedures appear as drafts in the menu bar app. You review, approve, and promote them through a lifecycle:

```
Observed → Draft → Reviewed → Verified → Agent Ready
```

Each promotion requires human approval. No procedure reaches agents without your sign-off.

The menu bar app shows:
- **Draft SOPs** to approve or reject
- **Trust suggestions** — system recommends when a procedure has earned enough evidence for higher trust
- **Stale alerts** — procedures that haven't been observed recently
- **Merge candidates** — similar procedures that might be duplicates
- **Drift alerts** — procedures whose observed behavior has changed
- **Lifecycle upgrades** — procedures ready for promotion based on evidence

### Agent-ready export

Approved procedures are exported as:

| Format | Location | Used by |
|--------|----------|---------|
| **SKILL.md** | `~/.openclaw/workspace/memory/apprentice/sops/` | OpenClaw agents |
| **Claude Code Skill** | `~/.claude/skills/<slug>/SKILL.md` | Claude Code (`/skill-name`) |
| **v3 Procedure JSON** | `~/.openmimic/knowledge/procedures/` | Any agent via Query API |

Agents query `GET /ready` on port 9477 to discover executable procedures, or `GET /bundle/<slug>` for a fully resolved handoff package with readiness assessment, preflight checks, and compiled outputs.

## Usage

### CLI quick reference

| Command | Description |
|---------|-------------|
| `openmimic status` | Service health and stats |
| `openmimic start all` | Start daemon + worker |
| `openmimic stop all` | Stop services |
| `openmimic focus start "title"` | Record a workflow |
| `openmimic focus stop` | Stop recording, generate SOP |
| `openmimic sops list` | List all SOPs |
| `openmimic sops drafts` | List SOPs awaiting review |
| `openmimic sops approve <slug>` | Approve a draft for export |
| `openmimic sops promote <slug> <state>` | Promote lifecycle (e.g., `reviewed`, `agent_ready`) |
| `openmimic doctor` | Pre-flight health check |
| `openmimic watch` | Live dashboard |
| `openmimic export --format claude-skill` | Re-export as Claude Code skills |
| `openmimic logs worker -f` | Follow worker logs |

### Query API (for agents)

The worker runs a local HTTP API on port 9477:

```bash
# List all procedures
curl http://localhost:9477/procedures

# Get agent-ready procedures
curl http://localhost:9477/ready

# Get a full handoff bundle
curl http://localhost:9477/bundle/file-expense-report

# Browse the curation queue
curl http://localhost:9477/curation/queue

# Promote a procedure via API
curl -X POST http://localhost:9477/curation/promote \
  -H 'Content-Type: application/json' \
  -d '{"slug": "file-expense-report", "to_state": "agent_ready"}'
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              Menu Bar App (SwiftUI)                   │
│   Status · Focus recording · Review queue · Digest   │
└───────────────────────┬──────────────────────────────┘
                        │ trigger files (JSON)
                        ▼
Chrome Extension ──→ Daemon (Rust) ──SQLite WAL──→ Worker (Python)
  DOM snapshots       Screenshots                    ┌──────────┐
  Click intent        OS Accessibility               │ Pipeline │
  Secure fields       Clipboard + dHash              │ v2 + VLM │
                      Focus session tags             └────┬─────┘
                                                          │
                                    ┌─────────────────────┼─────────────────────┐
                                    ▼                     ▼                     ▼
                              SKILL.md SOPs      v3 Procedures (KB)      Query API
                              (OpenClaw,         (lifecycle, trust,      (port 9477)
                               Claude Code)       evidence, curation)
```

| Component | Language | Role |
|-----------|----------|------|
| **Daemon** | Rust | Always-on observer — screenshots, OS events, clipboard, dHash dedup |
| **Worker** | Python | Pipeline — annotation, classification, segmentation, SOP generation, lifecycle, curation |
| **Extension** | TypeScript | Chrome MV3 — DOM snapshots, click intent, dwell/scroll tracking |
| **CLI** | Rust | Service management, focus recording, SOP approval, lifecycle promotion |
| **App** | SwiftUI | Menu bar — status, focus recording, review queue, daily digest |

### Processing budget

OpenMimic uses ~39% of GPU time per work hour. 37 minutes of headroom remain for your own GPU tasks.

| Stage | Time per hour | Notes |
|-------|---------------|-------|
| Scene annotation | 15.6 min | qwen3.5:2b, ~75 frames after stale-skip |
| Frame diffs | 4.5 min | Consecutive frame comparison |
| Task segmentation | 0.8 min | CPU only (embeddings) |
| SOP generation | 2.4 min | qwen3.5:4b thinking mode |

### Local models

| Model | Size | Purpose |
|-------|------|---------|
| `qwen3.5:2b` | 2.7 GB | Scene annotation + frame diff |
| `qwen3.5:4b` | 3.4 GB | SOP generation (thinking mode) |
| `all-minilm:l6-v2` | 45 MB | Task embedding for clustering |

## Privacy

OpenMimic is designed to never leave your machine:

- **Local-first.** All VLM inference runs locally via Ollama. Cloud APIs (OpenAI, Anthropic, Google) are opt-in with explicit consent.
- **Screenshots deleted after annotation.** Raw JPEGs (~270 KB) are deleted immediately after VLM annotation. Only structured JSON (~500 bytes) is kept.
- **Auto-redaction.** API keys, tokens, passwords, and credit card numbers are detected and scrubbed before storage.
- **Secure field exclusion.** Password inputs are dropped entirely — never captured, never stored.
- **Encryption at rest.** Artifacts use zstd compression + XChaCha20-Poly1305.
- **Configurable retention.** Raw events pruned after 14 days, episodes after 90 days.
- **No telemetry.** Pipeline metrics are local-only JSON files. Nothing phones home.

## Configuration

Config lives at `~/Library/Application Support/oc-apprentice/config.toml`.

<details>
<summary>Configuration reference</summary>

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
| `annotation_model` | qwen3.5:2b | Scene annotation model |
| `sop_model` | qwen3.5:4b | SOP generation model |
| `max_jobs_per_day` | 50 | VLM inference budget |
| `max_compute_minutes_per_day` | 20 | GPU time budget |

### Features

| Key | Default | Description |
|-----|---------|-------------|
| `activity_classification` | true | 8-class activity taxonomy |
| `continuity_tracking` | true | Task span continuity graph |
| `lifecycle_management` | true | 7-state procedure lifecycle |
| `curation` | true | Merge/upgrade/drift detection |
| `runtime_validation` | true | App-running checks via pgrep |

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
openmimic doctor        # Check all prerequisites
openmimic logs daemon   # Daemon-specific errors
openmimic logs worker   # Worker-specific errors
```

</details>

<details>
<summary>No events being captured</summary>

- Verify Accessibility permission: System Settings → Privacy & Security → Accessibility
- Check `openmimic status` for daemon health
- Ensure Chrome extension is loaded and enabled

</details>

<details>
<summary>No SOPs being generated</summary>

- **Passive mode** requires 2+ similar demonstrations of the same workflow
- Verify Ollama is running: `ollama list`
- Check worker logs: `openmimic logs worker -f`
- Ensure `annotation_enabled = true` in config

</details>

<details>
<summary>Extension not connecting</summary>

- Run `openmimic doctor` to verify native messaging host
- Reload the extension in `chrome://extensions`
- Check Chrome developer console for errors

</details>

## Uninstall

```bash
openmimic uninstall              # Remove services, keep data
openmimic uninstall --purge-data # Remove everything
```

## License

MIT
