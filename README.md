# OpenMimic

A local, privacy-first apprentice that silently observes your macOS workflows and generates semantic SOPs (Standard Operating Procedures) that AI agents can execute.

**Observation is always-on; learning is delayed.** OpenMimic captures UI intent (not raw macros), runs heavy processing only during idle windows, and never takes actions on your behalf.

## How It Works

OpenMimic operates in two modes:

### Focus Recording (User-Initiated)

Press **Record Workflow** in the menu bar, name the task, perform it, press **Stop**. A high-quality semantic SOP is generated immediately from a single demonstration.

```bash
openmimic focus start "Expense report filing"
# ... perform the workflow ...
openmimic focus stop
```

### Passive Discovery (Background)

OpenMimic continuously captures screenshots, annotates them with a local vision model, and detects repeated workflows automatically. When the same task appears in 2+ demonstrations, a SOP is generated without any user action.

## Install

### Recommended: macOS Installer (.pkg)

Download the latest `.pkg` from [Releases](https://github.com/sandroandric/OpenMimic/releases) and double-click to install.

This installs the daemon, worker, CLI (`openmimic`), and Chrome extension to standard system paths. No additional setup required beyond granting permissions.

After install:
```bash
openmimic doctor    # Verify everything is set up
openmimic start all # Start observing
```

### Developer Install

<details>
<summary>Homebrew (for developers)</summary>

```bash
brew tap sandroandric/openmimic
brew install --HEAD openmimic
```

</details>

<details>
<summary>Build from source (for contributors)</summary>

Requires: Rust toolchain, Node.js 18+, Python 3.11+

```bash
git clone https://github.com/sandroandric/OpenMimic.git
cd OpenMimic
just build-all          # Builds daemon, CLI, worker venv, extension, app
./scripts/setup.sh      # Install native messaging host + VLM setup

# Install launchd services (update paths for your source build):
cp resources/launchd/com.openmimic.daemon.plist ~/Library/LaunchAgents/
cp resources/launchd/com.openmimic.worker.plist ~/Library/LaunchAgents/
sed -i '' "s|/usr/local/bin/oc-apprentice-daemon|$(pwd)/target/release/oc-apprentice-daemon|" \
    ~/Library/LaunchAgents/com.openmimic.daemon.plist
sed -i '' "s|/usr/local/lib/openmimic/venv/bin/python|$(pwd)/worker/.venv/bin/python|" \
    ~/Library/LaunchAgents/com.openmimic.worker.plist
sed -i '' "s|/usr/local/lib/openmimic|$(pwd)/worker|" \
    ~/Library/LaunchAgents/com.openmimic.worker.plist
```

</details>

## First Run

### 1. Run the health check

```bash
openmimic doctor
```

Fix any `FAIL` checks before proceeding. Common first-time issues:
- **Accessibility permission** — System Settings > Privacy & Security > Accessibility, add `oc-apprentice-daemon`
- **Screen Recording permission** — Same location, add `oc-apprentice-daemon`

### 2. Set up VLM (Vision Language Model)

The v2 pipeline requires Ollama with local vision models for scene annotation and SOP generation.

```bash
# Install Ollama (if not already installed)
brew install ollama

# Pull required models (~6.2 GB total)
ollama pull qwen3.5:2b          # Scene annotation (fast, ~12s/frame)
ollama pull qwen3.5:4b          # SOP generation (thinking mode, ~72s)
ollama pull all-minilm:l6-v2    # Task embedding for clustering (45 MB)
```

Or run the guided setup:
```bash
openmimic setup --vlm
```

### 3. Load the Chrome Extension

The extension provides DOM context for richer SOPs (CSS selectors, form field IDs, ARIA labels).

1. Open `chrome://extensions` in Chrome
2. Enable **Developer Mode** (toggle in top-right)
3. Click **Load unpacked** and select:
   - **.pkg install:** `/usr/local/lib/openmimic/extension/`
   - **Homebrew:** Run `brew --prefix openmimic` to find path, then select `libexec/extension/`
   - **Source build:** `extension/dist/`
4. Verify the extension appears with ID `knldjmfmopnpolahpmmgbagdohdnhkik`

### 4. Start services

```bash
openmimic start all
```

Both daemon and worker start immediately and auto-restart on login.

### 5. Verify everything is working

```bash
openmimic status
```

Expected output:
```
  ● Daemon (running)
    PID:        12345
    Heartbeat:  2s ago
    Events:     0 captured today
    Perms:      OK
    Extension:  connected (5s ago)

  ● Worker (running)
    PID:        12346
    Heartbeat:  3s ago
    Events:     0 processed today
    SOPs:       0 generated
    VLM:        annotation pipeline active
```

## Usage

### Focus Recording

Record a specific workflow for immediate SOP generation:

**Menu Bar App:** Click the OpenMimic icon → **Record Workflow** → enter a title → perform the workflow → **Stop Recording**

**CLI:**
```bash
openmimic focus start "Deploy feature to staging"
# ... perform the workflow ...
openmimic focus stop
```

A SKILL.md file is generated within ~60 seconds after stopping, containing the exact steps observed with URLs, field names, and verification criteria.

### Passive Discovery

Just use your computer normally. OpenMimic continuously:
1. **Captures** screenshots (deduplicated via perceptual hashing, ~30% of raw frames)
2. **Annotates** each frame with a vision model (what app, what's on screen, what the user is doing)
3. **Diffs** consecutive frames (what changed, what was typed, what was clicked)
4. **Segments** annotations into task clusters using embedding similarity
5. **Generates** SOPs when a task cluster has 2+ demonstrations

### Viewing SOPs

```bash
openmimic sops list          # List generated SOPs
openmimic sops show <slug>   # View a specific SOP
openmimic sops dir           # Print SOPs directory path
```

SOPs are saved as `SKILL.<slug>.md` files in `~/.openclaw/workspace/memory/apprentice/sops/`.

### Exporting as Claude Code Skills

Export observed workflows directly as Claude Code personal skills:

```bash
openmimic export --format claude-skill
```

Skills are written to `~/.claude/skills/<slug>/SKILL.md` with YAML frontmatter (name, description, argument-hint, allowed-tools) and numbered natural language instructions. Use `/skill-name` in Claude Code to invoke them.

### Live Dashboard

```bash
openmimic watch              # Auto-refreshing status dashboard
```

## SOP Format (SKILL.md v2)

Generated SOPs are semantic workflow descriptions, not DOM automation scripts. They contain:

- **Description** — What the workflow accomplishes
- **When to Use** — Trigger conditions and prerequisites
- **Steps** — Each step with Action, App, Location, Input, and Verify fields
- **Variables** — Parameterized values detected across demonstrations (e.g., `{{amount}}`, `{{recipient}}`)
- **Success Criteria** — How to verify the workflow completed correctly
- **Common Errors** — Failure modes and recovery steps
- **DOM Hints** — CSS selectors for browser automation (collapsible appendix)
- **Confidence Score** — Multi-signal quality assessment (demo count, step consistency, annotation quality, variable detection)

### Deduplication

Recording the same workflow multiple times doesn't create duplicate SOPs. OpenMimic computes a structural fingerprint (apps + URL domains + action verbs) for each SOP and uses weighted Jaccard similarity to detect matches. When a match is found (≥70% similarity), the existing SOP is updated: episode count accumulates, steps are refined, new variables are discovered, and confidence increases. Different tasks (e.g., "Search Amazon" vs "Search eBay") are correctly kept separate due to domain divergence.

## CLI Reference

| Command | Description |
|---------|-------------|
| `openmimic status` | Show service health and stats |
| `openmimic start [daemon\|worker\|all]` | Start services via launchd |
| `openmimic stop [daemon\|worker\|all]` | Stop services |
| `openmimic restart [daemon\|worker\|all]` | Restart services |
| `openmimic focus start "<title>"` | Start recording a workflow |
| `openmimic focus stop` | Stop recording and generate SOP |
| `openmimic sops list\|show\|dir` | View generated SOPs |
| `openmimic logs <service> [-f] [-n N]` | View/follow log files |
| `openmimic config show\|edit\|path` | Manage configuration |
| `openmimic watch` | Live-updating status dashboard |
| `openmimic doctor` | Run pre-flight checks |
| `openmimic setup --vlm` | Configure VLM models |
| `openmimic export --format <fmt>` | Re-export SOPs (`skill-md`, `generic`, `openclaw`, `claude-skill`) |
| `openmimic uninstall [--purge-data]` | Remove OpenMimic |

## Configuration

The config file lives at `~/Library/Application Support/oc-apprentice/config.toml`.

### Observer

| Key | Default | Description |
|-----|---------|-------------|
| `t_dwell_seconds` | 3 | Inactivity before dwell snapshot |
| `screenshot_max_per_minute` | 20 | Screenshot rate limit |
| `screenshot_format` | jpeg | `jpeg` (half-res, recommended) or `png` (full-res) |
| `screenshot_quality` | 70 | JPEG quality 1-100 |
| `screenshot_scale` | 0.5 | Resolution scale (0.5 = half, saves 62% storage) |
| `dhash_threshold` | 10 | Perceptual hash dedup threshold (lower = stricter) |

### VLM (Vision Language Model)

| Key | Default | Description |
|-----|---------|-------------|
| `annotation_model` | qwen3.5:2b | Model for per-frame scene annotation |
| `sop_model` | qwen3.5:4b | Model for SOP generation (thinking mode) |
| `annotation_enabled` | true | Enable v2 continuous annotation pipeline |
| `stale_skip_count` | 3 | Skip after N consecutive non-workflow same-app frames |
| `sliding_window_max_age_sec` | 600 | Max age (seconds) for context window |
| `max_jobs_per_day` | 50 | VLM inference budget |

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

### Export

| Key | Default | Description |
|-----|---------|-------------|
| `adapter` | openclaw | Export adapter: `openclaw`, `generic`, `skill-md`, `claude-skill`, or `all` |
| `json_export` | false | Also write JSON alongside Markdown |

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Menu Bar App (SwiftUI)          │
                    │  Status indicator · Focus recording     │
                    │  Onboarding · Service controls          │
                    └──────────────┬──────────────────────────┘
                                   │ focus-session.json
                                   ▼
Chrome Extension ──native msg──> Daemon (Rust) ──SQLite WAL──> Worker (Python)
  DOM snapshots                    │                              │
  Click intent                OS Accessibility               ┌───┴───┐
  Page context                Screenshots (JPEG)             │  v2   │
                              Clipboard                      │Pipeline│
                              dHash dedup                    └───┬───┘
                              Focus session tagging              │
                                                                 ▼
                                                           SKILL.md SOPs
```

### Components

| Component | Language | Role |
|-----------|----------|------|
| **Daemon** | Rust | Always-on observer — screenshots, OS events, clipboard, dHash dedup, focus session tagging |
| **Worker** | Python | Processing pipeline — scene annotation, frame diff, task segmentation, SOP generation |
| **Extension** | TypeScript | Chrome MV3 — DOM snapshots, click intent, dwell/scroll-read tracking |
| **CLI** | Rust | Service management — start/stop, status, focus recording, logs, doctor |
| **Menu Bar App** | Swift | Visual controls — status indicator, focus recording UI, onboarding wizard |

### v2 Processing Pipeline

```
Screenshot ─→ dHash Dedup (70% reduction)
                  │
                  ▼
          Scene Annotator (qwen3.5:2b, ~12.5s/frame)
          Extracts: app, location, visible content, UI state, task context
          Uses 3-frame sliding window for cross-app continuity
                  │
                  ▼
           Frame Differ (qwen3.5:2b, ~3.6s/pair)
           Produces: action descriptions, typed inputs, navigation changes
           Edge markers: app_switch, session_gap, no_change (free, no LLM)
                  │
                  ▼
          Task Segmenter (all-minilm embeddings + clustering)
           Groups annotations by semantic similarity (cosine > 0.75)
           Filters noise (browsing, chatting, reading)
           Stitches interrupted workflows
                  │
                  ▼
           SOP Generator (qwen3.5:4b thinking, ~72s)
           Single-demo (focus) or multi-demo (passive)
           Variable detection across demonstrations
                  │
                  ▼
           SOP Deduplication (structural fingerprint)
           Matches by app + domain + action verb similarity
           Merges repeated recordings into one SOP
                  │
                  ▼
             Export: SKILL.md + OpenClaw + Claude Code Skills
```

### Processing Budget (Per Work Hour)

| Stage | Time | GPU % |
|-------|------|-------|
| Annotation (~75 frames after stale-skip) | 15.6 min | Primary |
| Frame diffs (~75 pairs) | 4.5 min | Secondary |
| Segmentation (batch) | 0.8 min | CPU only |
| SOP generation (~2 workflows) | 2.4 min | Burst |
| **Total** | **23.3 min** | **39%** |

37 minutes of headroom per hour for user GPU tasks.

### Models on Disk

| Model | Size | Role |
|-------|------|------|
| `qwen3.5:2b` | 2.7 GB | Scene annotation + frame diff (think=False) |
| `qwen3.5:4b` | 3.4 GB | SOP generation (think=True, num_predict=8000) |
| `all-minilm:l6-v2` | 45 MB | Task label embeddings for segmentation |

## Troubleshooting

**Services not starting:**
```bash
openmimic doctor           # Check all prerequisites
openmimic logs daemon      # Check daemon logs for errors
openmimic logs worker      # Check worker logs
```

**No events being captured:**
- Verify Accessibility permission is granted
- Check `openmimic status` for daemon health
- Ensure Chrome extension is loaded and enabled

**No SOPs being generated (passive mode):**
- SOPs require 2+ similar demonstrations of the same workflow
- Check `openmimic logs worker` for pipeline activity
- Verify Ollama is running: `ollama list`
- Ensure annotation models are pulled: `ollama pull qwen3.5:2b`

**Focus recording not producing SOPs:**
- Check `openmimic logs worker` for focus processing messages
- Verify the workflow had observable screen changes (identical screenshots are deduplicated)
- Ensure `annotation_enabled = true` in config

**VLM annotation not running:**
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check model availability: `ollama list` should show `qwen3.5:2b`
- Check worker logs: `openmimic logs worker -f`

**Extension not connecting:**
- Verify native messaging host: `openmimic doctor`
- Check Chrome developer console for the extension
- Reload the extension in `chrome://extensions`

## Uninstall

```bash
openmimic uninstall              # Remove services, keep data
openmimic uninstall --purge-data # Remove everything including database
```

Or run the standalone uninstaller:
```bash
bash /usr/local/lib/openmimic/scripts/uninstall.sh
```

## Privacy

- **Local by default, opt-in remote.** VLM inference runs locally via Ollama. Remote cloud APIs (OpenAI, Anthropic, Google) can be enabled via `mode = "remote"` in config — requires explicit consent and shows a privacy warning. API keys stored in macOS Keychain or env vars, never in config files.
- **Screenshots deleted after annotation.** Raw JPEG screenshots (~270 KB each) are deleted immediately after successful VLM annotation. Only the structured JSON annotation (~500 bytes) is retained. Screenshots are kept on annotation failure for retry.
- **Auto-redaction.** API keys, tokens, passwords, and credit card numbers are detected and redacted before storage.
- **Secure field exclusion.** Password and credit-card input fields are dropped entirely.
- **Encryption at rest.** Artifacts use zstd compression + XChaCha20-Poly1305 encryption.
- **Prompt injection defense.** DOM text is sanitized against 15 regex patterns across 7 threat categories.
- **Configurable retention.** Raw events pruned after 14 days, episodes after 90 days.

## License

MIT
