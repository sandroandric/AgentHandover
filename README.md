<p align="center">
  <img src="resources/favicon.png" width="140" alt="AgentHandover" />
</p>

<h1 align="center">AgentHandover</h1>

<p align="center">
  <strong>Work once. Hand over forever.</strong>
</p>

<p align="center">
  <a href="#what-agents-actually-get">What Agents Get</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#connect-your-agent">Connect Your Agent</a> &middot;
  <a href="#install">Install</a> &middot;
  <a href="#privacy">Privacy</a>
</p>

---

<p align="center">
  <img src="AgentHandover.png" alt="The Struggle - The Discovery - The Transformation" />
</p>

You already know how to do your work. Your agents don't.

AgentHandover watches you work on your Mac, understands what you're doing and *why*, and produces complete handoff documents that agents like **Claude Code**, **OpenClaw**, **Codex**, or any MCP-compatible agent can follow. Not macros. Not screen recordings. Actual procedures with strategy, decision logic, guardrails, and your voice.

The kind of handoff you'd give a capable new hire -- except the new hire is an AI agent, and the handoff writes itself.

## What Agents Actually Get

This is a real procedure generated from a few focus recording sessions:

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

VOICE & STYLE
Tone: casual | Sentences: short and punchy | Uses emoji
> Hey great point about the engagement metrics! We should
> def try that approach with the subreddit

~15 min daily - 9-10am                     Confidence: 89%
```

Not just steps. Strategy, decisions, guardrails, and the user's actual voice -- so the agent doesn't just do the work, it does it the way you would.

### What makes this different

| What | How |
|------|-----|
| **Strategy, not just steps** | Behavioral synthesis extracts the reasoning behind your actions -- not "click here" but "target high-signal posts because..." |
| **Your voice, not generic text** | Style analysis captures formality, sentence patterns, vocabulary, emoji -- per workflow. Casual on Reddit, formal in email. |
| **Semantic understanding** | Vector KB (nomic-embed-text, 768d) powers similarity search, deduplication, and cross-session linking. "Deploy to staging" matches "push to stage environment." |
| **Visual intelligence** | Optional SigLIP image embeddings (1152d) let agents find visually similar screens even when text differs. |
| **Universal agent support** | MCP server, Claude Code /skills, Codex AGENTS.md, OpenClaw SOPs, REST API. One observation, every agent format. |
| **Voice strengthens over time** | Style confidence builds across sessions. One reply is a guess. Twenty replies is a voice profile the agent can match. |

## How It Works

### Two ways to teach

**Focus Recording** -- Click Record, name the task, do the work, click Stop. A few minutes later you have a complete procedure. Best for workflows you want to hand off now.

**Passive Discovery** -- Just work normally. When the same task appears in 2+ sessions, AgentHandover generates a procedure automatically. You don't have to do anything.

### What happens under the hood

```
You work normally
    |
    v
Daemon captures screenshots + OS events + clipboard + DOM context
    |
    v
Local VLM (Qwen 3.5) annotates each frame:
  what app, what URL, what you're doing, what you'll do next
    |
    v
Activity classifier separates signal from noise
  (your expense filing is "work", YouTube is "entertainment")
    |
    v
Vector KB embeds each annotation (nomic-embed-text, 768d)
  + optional image embeddings (SigLIP, 1152d)
    |
    v
Task segmenter clusters similar activity by semantic similarity
  (not keywords -- "deploy staging" = "push to stage")
    |
    v
Session linker connects the same workflow across days
  (vector cosine similarity, not brittle token matching)
    |
    v
After 3+ observations: behavioral synthesis extracts
  strategy, selection criteria, content templates, guardrails,
  decision branches, timing patterns
    |
    v
Style analyzer captures your voice from typed text:
  formality, sentence length, vocabulary, emoji, tone
  (strengthens over sessions -- not a one-shot guess)
    |
    v
Procedure compiled to every agent format simultaneously:
  MCP tools, Claude Code /skills, Codex AGENTS.md, OpenClaw SOPs
```

### Focus Q&A

After a focus recording, AgentHandover asks 1-3 targeted questions from the agent's perspective:

- "Does this workflow require logging into Reddit?"
- "What determines which posts you engage with vs. skip?"
- "How do you verify the reply wasn't auto-removed?"

Answers merge into the procedure. Skip any question and it uses reasonable defaults.

### Screenshots are temporary

Screenshots are captured at half resolution, deduplicated via perceptual hashing (70% are duplicates and get dropped), then deleted after VLM processing. Only the structured annotation (~500 bytes) survives. If image embeddings are enabled, the visual embedding is computed before deletion -- the screenshot itself never accumulates on disk.

## Connect Your Agent

### MCP Server (recommended -- works with any agent)

Add one line to your agent's settings:

```json
{
  "mcpServers": {
    "agenthandover": {
      "command": "agenthandover-mcp"
    }
  }
}
```

Works with Claude Code, Cursor, Windsurf, and any MCP-compatible tool. Exposes 5 tools:

| Tool | What it does |
|------|-------------|
| `list_ready_procedures` | Procedures ready for execution (all gates passed) |
| `get_procedure(slug)` | Full procedure with steps, strategy, voice, guardrails |
| `search_procedures(query)` | Semantic search -- find workflows by meaning |
| `list_all_procedures` | All procedures including drafts |
| `get_user_profile` | User's tools, working hours, writing style |

### Claude Code

```bash
agenthandover connect claude-code
```

Procedures appear as `/slash-commands`. Type `/reddit-community-marketing` and Claude Code gets the full procedure.

### Codex

```bash
agenthandover connect codex
```

Generates `AGENTS.md` in your project with all agent-ready procedures, strategy, guardrails, and voice guidance.

### OpenClaw

```bash
agenthandover connect openclaw
```

Procedures auto-sync to `~/.openclaw/workspace/memory/apprentice/sops/`. Nothing to configure.

### REST API

Already running on localhost:9477:

```bash
curl http://localhost:9477/ready              # Executable procedures
curl http://localhost:9477/bundle/my-workflow  # Full handoff bundle
curl -X POST http://localhost:9477/search/semantic \
  -d '{"query": "deploy to production"}'      # Semantic search
```

## Install

### Download and run

Download the latest `.pkg` from [**Releases**](https://github.com/sandroandric/OpenMimic/releases) and double-click.

The app opens a guided setup: permissions, model download, Chrome extension, and optional image embeddings. Follow the prompts.

<details>
<summary><strong>Developer / advanced install</strong></summary>

### CLI setup

```bash
agenthandover doctor     # Verify prerequisites
agenthandover start all  # Start daemon + worker
```

### Pull models manually

```bash
ollama pull qwen3.5:2b         # Scene annotation (~2.7 GB)
ollama pull qwen3.5:4b         # SOP generation (~3.4 GB)
ollama pull nomic-embed-text   # Semantic search (~274 MB)
```

### Chrome extension

Open `chrome://extensions` > Enable Developer Mode > Load unpacked > select the extension directory shown by `agenthandover doctor`.

### Homebrew

```bash
brew tap sandroandric/agenthandover
brew install --HEAD agenthandover
```

### Source build (Rust, Node.js 18+, Python 3.11+)

```bash
git clone https://github.com/sandroandric/OpenMimic.git && cd OpenMimic
just build-all
./scripts/setup.sh
```

</details>

### Choose your vision model

| Backend | Best for |
|---------|----------|
| **Ollama** (default) | Local, free, private |
| **MLX** | Fastest on Apple Silicon |
| **llama.cpp** | Cross-platform local |
| **OpenAI / Anthropic / Google** | Highest quality (remote, opt-in) |

Switch via `config.toml` or `agenthandover setup --vlm`.

## The Menu Bar App

AgentHandover lives in your menu bar:

- **Status** -- daemon and worker health with green/yellow/red indicator
- **Today's stats** -- events captured, annotations completed, procedures generated
- **Attention items** -- Focus Q&A questions waiting, drafts ready for review
- **Record button** -- one click to start a focus recording
- **Workflows** -- browse procedures, approve drafts, see confidence scores
- **Digest** -- daily summary of what was learned and what needs attention

### Review and approve

Procedures appear as drafts. Review the strategy, steps, and guardrails. Click "Approve for Agents" when it looks right. One click.

No procedure reaches agents without your sign-off. The system suggests promotions based on evidence but never auto-promotes.

## Privacy

Everything runs on your machine:

- **Local-first.** VLM inference via Ollama. Cloud APIs are opt-in with explicit consent.
- **Screenshots are temporary.** Deleted after VLM annotation. Only structured text survives.
- **Auto-redaction.** API keys, tokens, passwords, credit card numbers scrubbed before storage.
- **Secure field exclusion.** Password and credit card inputs are never captured.
- **Encryption at rest.** Artifacts use zstd + XChaCha20-Poly1305.
- **Configurable retention.** Raw events pruned at 14 days. Valuable evidence extracted and preserved permanently before expiry.
- **No telemetry.** Nothing phones home. Ever.

<details>
<summary><strong>CLI reference</strong></summary>

| Command | Description |
|---------|-------------|
| `agenthandover status` | Service health and stats |
| `agenthandover start all` | Start daemon + worker |
| `agenthandover stop all` | Stop services |
| `agenthandover focus start "title"` | Record a workflow |
| `agenthandover focus stop` | Stop recording |
| `agenthandover sops list` | List all procedures |
| `agenthandover sops approve <slug>` | Approve for agents |
| `agenthandover sops promote <slug> <state>` | Promote lifecycle |
| `agenthandover connect <agent>` | Set up agent integration |
| `agenthandover doctor` | Pre-flight health check |
| `agenthandover watch` | Live dashboard |
| `agenthandover logs worker -f` | Follow worker logs |

</details>

<details>
<summary><strong>Architecture</strong></summary>

```
Chrome Extension ----> Daemon (Rust) --SQLite WAL--> Worker (Python)
  DOM snapshots          Screenshots                   Pipeline v2 + VLM
  Click intent           OS events                     Vector KB (nomic-embed-text)
  Secure fields          Clipboard                     Style analyzer
                         dHash dedup                   Behavioral synthesis
                                                       |
                    Menu Bar App (SwiftUI)              |
                    Status - Record - Workflows         |
                                                       v
                                            +---------------------+
                                            |   Knowledge Base    |
                                            |   Procedures (v3)   |
                                            |   Vector store      |
                                            |   Voice profiles    |
                                            +-----+-----+--------+
                                                  |     |
                                    +-------------+     +-------------+
                                    v             v                   v
                              MCP Server    Claude Code        OpenClaw SOPs
                              (any agent)   /slash-commands    (auto-sync)
```

| Component | Language | Role |
|-----------|----------|------|
| **Daemon** | Rust | Always-on observer -- screenshots, OS events, clipboard, dedup |
| **Worker** | Python | Intelligence -- VLM, classification, segmentation, vector KB, behavioral synthesis, style analysis, lifecycle, export |
| **Extension** | TypeScript | Chrome MV3 -- DOM snapshots, click intent, form field context |
| **CLI** | Rust | Service management, focus recording, agent connection |
| **App** | SwiftUI | Menu bar -- status, recording, workflows, digest |
| **MCP Server** | Python | Universal agent interface -- tools + resources via MCP protocol |

</details>

<details>
<summary><strong>Lifecycle gates</strong></summary>

```
Observed -> Draft -> Reviewed -> Verified -> Agent Ready
```

Six gates must pass before an agent can execute:

| Gate | Checks | Decides |
|------|--------|---------|
| **Lifecycle** | Human reviewed and promoted? | Human |
| **Trust** | Agent authorized to execute? | Human |
| **Freshness** | Observed recently? Stale = auto-demote | System |
| **Preflight** | Required apps running? Blocked domains? | System |
| **Evidence** | Observation count, confidence, contradictions | System |
| **Execution** | Prior success rate? 3+ failures = auto-demote | System |

</details>

## Uninstall

```bash
agenthandover uninstall              # Remove services, keep data
agenthandover uninstall --purge-data # Remove everything
```

## License

[BSL 1.1](LICENSE) — source available, non-commercial. Converts to Apache 2.0 on 2030-03-25.
