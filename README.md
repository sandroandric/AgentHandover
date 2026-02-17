# OpenMimic

A local, always-on "apprentice" subsystem that silently observes your day-to-day laptop work, learns your workflows, and produces semantic SOPs (Standard Operating Procedures) that agents like [OpenClaw](https://github.com/sandroandric) can later execute safely.

**Observation is always-on; learning is delayed.** OpenMimic captures UI intent (not raw macros), runs heavy processing only during idle windows, and never takes actions on your behalf.

## Architecture

OpenMimic is a three-process system:

| Component | Language | Role |
|-----------|----------|------|
| **oc-apprentice-daemon** | Rust | Always-on observer — captures OS events, screenshots, accessibility trees, clipboard, and manages the local SQLite event store |
| **oc-apprentice-worker** | Python | Idle-time processor — builds episodes from raw events, runs VLM inference, mines SOPs via PrefixSpan, exports to OpenClaw |
| **openmimic-extension** | TypeScript | Chrome MV3 extension — captures DOM snapshots, click intent, dwell tracking, and secure field detection |

The daemon and worker communicate through a **SQLite WAL-mode database** acting as a local event broker. The Chrome extension communicates with the daemon via **Chrome Native Messaging** (stdio, 4-byte LE length-prefix framing).

```
Chrome Extension ──native messaging──► Daemon ──SQLite WAL──► Worker ──► SOP YAML files
                                         │
                                    OS Accessibility
                                    Screenshots
                                    Clipboard
```

## Prerequisites

- **macOS** (primary target; Linux support planned)
- **Rust** stable toolchain (the repo pins `stable` via `rust-toolchain.toml`)
- **Python 3.11+**
- **Node.js 18+** and npm
- **Google Chrome** (for the extension)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/sandroandric/OpenMimic.git
cd OpenMimic
```

### 2. Build the Rust daemon

```bash
cargo build --release
```

This builds four crates:
- `oc-apprentice-daemon` — the main binary
- `oc-apprentice-common` — shared types and privacy filters
- `oc-apprentice-storage` — SQLite WAL storage layer with encryption
- `oc-apprentice-test-harness` — test utilities (dev only)

### 3. Install the Python worker

```bash
cd worker
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cd ..
```

This installs:
- `prefixspan` — sequential pattern mining for SOP induction
- `pyyaml` — YAML serialization for SOP export
- `pytest`, `pytest-cov` — testing (dev dependency)

**Optional VLM backends** (for screenshot understanding during idle processing):
```bash
# Apple Silicon (recommended):
pip install mlx-vlm

# OR CPU/cross-platform fallback:
pip install llama-cpp-python
```

### 4. Build the Chrome extension

```bash
cd extension
npm install
npm run build
cd ..
```

### 5. Install the Chrome extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked** and select the `extension/dist/` directory
4. Note the **Extension ID** shown on the card (e.g., `knldjmfmopn...`)

### 6. Register the Native Messaging host

This tells Chrome how to launch the daemon when the extension connects:

```bash
./scripts/install-native-host.sh --extension-id YOUR_EXTENSION_ID
```

The script places a JSON manifest in:
- **macOS:** `~/Library/Application Support/Google/Chrome/NativeMessagingHosts/`
- **Linux:** `~/.config/google-chrome/NativeMessagingHosts/`

### 7. Configure OpenMimic

Copy the example configuration and edit as needed:

```bash
# macOS
mkdir -p ~/Library/Application\ Support/oc-apprentice
cp config.example.toml ~/Library/Application\ Support/oc-apprentice/config.toml

# Linux
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/openclaw-apprentice"
cp config.example.toml "${XDG_CONFIG_HOME:-$HOME/.config}/openclaw-apprentice/config.toml"
```

Key settings in `config.toml`:

| Section | Setting | Default | Description |
|---------|---------|---------|-------------|
| `[observer]` | `t_dwell_seconds` | 3 | Seconds of inactivity before capturing a dwell snapshot |
| `[observer]` | `screenshot_max_per_minute` | 20 | Rate limit for screenshot captures |
| `[privacy]` | `enable_inline_secret_redaction` | true | Automatically redact API keys, tokens, and credentials |
| `[privacy]` | `secure_field_drop` | true | Drop events from password/credit-card fields |
| `[storage]` | `retention_days_raw` | 14 | Days to keep raw events before pruning |
| `[storage]` | `retention_days_episodes` | 90 | Days to keep processed episodes |
| `[vlm]` | `max_jobs_per_day` | 50 | Budget cap for VLM inference jobs |
| `[vlm]` | `max_compute_minutes_per_day` | 20 | Compute time budget for VLM processing |
| `[idle_jobs]` | `run_window_local_time` | 01:00-05:00 | Local time window for background processing |
| `[idle_jobs]` | `require_ac_power` | true | Only run idle jobs when plugged in |

### 8. Grant macOS permissions

OpenMimic requires two macOS permissions to observe your workflow:

1. **Accessibility** — System Settings > Privacy & Security > Accessibility > Add `oc-apprentice-daemon`
2. **Screen Recording** — System Settings > Privacy & Security > Screen Recording > Add `oc-apprentice-daemon`

These permissions allow the daemon to read window titles, UI element trees, and capture screenshots. No data leaves your machine.

## Running

```bash
# Start the daemon (foreground for initial testing)
./target/release/oc-apprentice-daemon

# In a separate terminal, start the worker
cd worker
source .venv/bin/activate
python -m oc_apprentice_worker.main
```

The daemon starts observing immediately. The worker processes events during the configured idle window (default: 1:00-5:00 AM) when the machine is plugged in and idle.

## Running tests

```bash
# Rust tests (175 tests)
cargo test

# Python worker tests (383+ tests)
cd worker
source .venv/bin/activate
pytest tests/ -v

# Chrome extension tests (133 tests)
cd extension
npm test

# Integration + load tests (72+ tests)
cd ..  # back to repo root
pytest tests/ -v
```

## Project structure

```
OpenMimic/
├── Cargo.toml                  # Rust workspace root
├── config.example.toml         # Example configuration file
├── crates/
│   ├── common/                 # Shared types, event models, privacy redaction
│   ├── daemon/                 # Main observer daemon
│   │   └── src/
│   │       ├── capture/        # Screenshot, clipboard, CSS filter
│   │       ├── ipc/            # Native messaging, CDP bridge
│   │       ├── observer/       # Event loop, dwell detection, health monitor
│   │       └── platform/       # macOS accessibility, power, windows, Electron detection
│   ├── storage/                # SQLite WAL storage, encryption, artifact pipeline
│   └── test-harness/           # Deterministic test utilities
├── extension/                  # Chrome MV3 extension
│   └── src/
│       ├── background.ts       # Service worker, native messaging bridge
│       ├── content.ts          # Content script entry point
│       ├── dom-capture.ts      # DOM snapshot with viewport filtering
│       ├── click-capture.ts    # Click intent capture
│       ├── dwell-tracker.ts    # Reading detection (dwell + scroll)
│       ├── native-messaging.ts # 4-byte LE framing protocol
│       └── secure-field.ts     # Password/credit-card field detection
├── worker/                     # Python idle-time worker
│   └── src/oc_apprentice_worker/
│       ├── episode_builder.py  # Event-to-episode segmentation
│       ├── translator.py       # Raw events to semantic steps
│       ├── confidence.py       # Confidence scoring with VLM fallback
│       ├── sop_inducer.py      # PrefixSpan-based SOP mining
│       ├── sop_format.py       # YAML frontmatter formatting
│       ├── sop_versioner.py    # SHA-256 drift detection
│       ├── exporter.py         # Atomic SOP export
│       ├── openclaw_writer.py  # OpenClaw workspace writer
│       ├── vlm_worker.py       # VLM inference (mlx-vlm / llama-cpp)
│       ├── vlm_queue.py        # Budget-aware VLM job queue
│       ├── injection_defense.py# Prompt injection detection + sanitization
│       ├── negative_demo.py    # Undo/cancel detection
│       ├── clipboard_linker.py # Copy-paste chain linking
│       ├── scheduler.py        # Power/thermal-gated job scheduler
│       └── db.py               # Worker-side SQLite access
├── tests/
│   ├── integration/            # Cross-component integration tests
│   └── load/                   # Power-user-day load simulation
└── scripts/
    └── install-native-host.sh  # Chrome Native Messaging host installer
```

## Privacy and security

- All data stays local. No network egress from the daemon or worker.
- API keys, tokens, passwords, and credit card numbers are automatically redacted before storage.
- Password and credit-card input fields are detected and their events are dropped entirely.
- Screenshots and artifacts are compressed (zstd level 3) and encrypted (XChaCha20-Poly1305) at rest.
- DOM text from web pages is treated as untrusted input with prompt injection defense (15 regex patterns across 7 threat categories).
- Raw events are pruned after 14 days; episodes after 90 days (configurable).

## License

MIT
