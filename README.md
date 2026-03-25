<p align="center">
  <img src="resources/favicon.png" width="140" alt="AgentHandover" />
</p>

<h1 align="center">AgentHandover</h1>

<p align="center">
  <strong>Work once. Hand over forever.</strong>
</p>

<p align="center">
  <a href="#what-a-skill-looks-like">What a Skill Looks Like</a> &middot;
  <a href="#how-it-works">How It Works</a> &middot;
  <a href="#the-knowledge-base">Knowledge Base</a> &middot;
  <a href="#connect-your-agent">Connect Your Agent</a> &middot;
  <a href="#install">Install</a> &middot;
  <a href="#privacy">Privacy</a>
</p>

---

<p align="center">
  <img src="AgentHandover.png" alt="The Struggle - The Discovery - The Transformation" />
</p>

You already know how to do your work. Your agents don't.

AgentHandover watches you work on your Mac, understands what you're doing and *why*, and produces **Skills** -- structured playbooks that tell AI agents exactly what to do and how to do it. Each Skill contains the steps, the strategy behind them, decision logic, guardrails, and your writing voice. **Claude Code**, **OpenClaw**, **Codex**, or any MCP-compatible agent can pick up a Skill and execute the workflow the way you would.

Not macros. Not screen recordings. Not a list of clicks. A complete understanding of your work -- the kind of handoff you'd give a sharp new hire, except it writes itself.

## What a Skill Looks Like

Here's an illustrative example of what a Skill looks like:

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

Skills follow the same format as Claude Code's native skills -- same frontmatter, same markdown structure -- but go further. Hand-written skills say "do X then Y." AgentHandover Skills include the strategy behind the steps, selection criteria, guardrails, your voice, and evidence-backed confidence from real observations. No hand-written skill has that.

## How It Works

### Two ways to teach

**Focus Recording** -- Click Record in the menu bar, name the task, perform it, click Stop. AgentHandover asks 1-3 targeted questions from the agent's perspective ("What determines which posts you engage with vs. skip?"), then generates a complete Skill. Best for workflows you want to hand off right now.

**Passive Discovery** -- Just work normally. AgentHandover recognizes recurring workflows across sessions using semantic similarity, accumulates observations, and when it has enough evidence, runs behavioral analysis to extract the strategy, decisions, and patterns behind your actions -- then generates a Skill automatically. You don't have to do anything.

### An 11-stage intelligence pipeline

This is not a screen recorder with ChatGPT on top. AgentHandover runs an 11-stage pipeline that turns raw screen activity into agent-ready Skills:

| Stage | What it does |
|-------|-------------|
| **1. Screen capture** | Half-resolution screenshots, deduplicated by perceptual hashing (70% of frames are duplicates and get dropped) |
| **2. VLM annotation** | Local Qwen 3.5 model reads each frame -- what app, what URL, what you're doing, what you'll do next |
| **3. Activity classification** | 8-class taxonomy separates work from noise. Your expense filing is "work." Your YouTube break is "entertainment." |
| **4. Text embedding** | Every annotation embedded into a vector knowledge base (nomic-embed-text, 768d) for semantic matching |
| **5. Image embedding** | Optional SigLIP embeddings (1152d) capture what your screen looked like -- find visually similar screens even when text differs |
| **6. Semantic clustering** | Groups related activity by meaning, not keywords. "Deploy to staging" matches "push to stage environment." |
| **7. Cross-session linking** | Connects the same workflow across days and interruptions using vector cosine similarity |
| **8. Behavioral synthesis** | After 3+ observations: extracts strategy, selection criteria, content templates, guardrails, decision branches, and timing patterns |
| **9. Voice analysis** | Captures your writing style from typed text -- formality, sentence length, vocabulary, emoji. Per workflow. Strengthens over sessions. |
| **10. Skill generation** | Produces a canonical Skill with semantic dedup (won't create duplicates even if you describe the same workflow differently) |
| **11. Human review** | You approve before any agent can execute. Six readiness gates must pass. Nothing auto-promotes. |

Every stage runs locally on your Mac. No cloud APIs required.

### You stay in control

Every Skill starts as a draft in your menu bar app. Six gates must pass before an agent can execute:

| Gate | What it checks |
|------|---------------|
| **Lifecycle** | You reviewed and promoted it through each stage (Observed > Draft > Reviewed > Verified > Agent Ready) |
| **Trust** | You authorized the agent to execute, not just observe |
| **Freshness** | The Skill was observed recently -- stale Skills auto-demote |
| **Preflight** | Required apps are running, no blocked domains |
| **Evidence** | Enough observations, high confidence, no contradictions |
| **Execution history** | Past success rate -- 3+ failures auto-demote |

The system suggests promotions based on evidence. You decide.

## The Knowledge Base

Everything AgentHandover learns lives in a local knowledge base on your machine. It's not a flat list of files -- it's an active intelligence layer that gets smarter the more you work.

**Vector store** -- Every observation is embedded (nomic-embed-text, 768d) so the system finds similar workflows by meaning, deduplicates Skills that describe the same task differently, and links activity across sessions. Optional image embeddings (SigLIP, 1152d) capture what your screen looked like.

**Voice profiles** -- Your writing style accumulates per workflow and strengthens over sessions. One reply is a guess. Twenty replies is a fingerprint the agent can match. Casual on Reddit, formal in client emails -- the system knows the difference.

**User profile** -- Aggregated across all workflows: your tools, working hours, communication patterns, and overall writing style. Agents read this to adapt to you.

**Semantic search** -- Agents can search the knowledge base by meaning via the MCP server or REST API. "Find something about deploying" returns your staging deployment Skill even if it's titled "Push to Prod."

## Execution Feedback Loop

Most tools stop at "here's a procedure, good luck." AgentHandover closes the loop. When an agent executes a Skill, it reports back what happened -- and the Skill gets better.

**How it works**: Every Skill includes an execution protocol. The agent calls `report_execution_start` before beginning, `report_step_result` after each step, and `report_execution_complete` when done. AgentHandover processes the results:

- **Success** -- Confidence goes up. Freshness confirmed. Timing updated via exponential moving average.
- **Deviation** -- The system tracks what the agent actually did vs. what was expected. After 2+ deviations on the same step, it suggests a decision branch.
- **Failure** -- Confidence drops. After 3 failures in 7 days, the Skill auto-demotes from agent-ready.

Skills don't just describe your workflows -- they learn from every execution and improve over time.

### One-click agent pairing

The menu bar app detects installed agents (Claude Code, Cursor, Windsurf) and connects them with one click -- writes the MCP config automatically. No terminal, no config files.

## Connect Your Agent

### MCP Server (recommended)

One config line, any agent. Works with Claude Code, Cursor, Windsurf, and any MCP-compatible tool.

```json
{
  "mcpServers": {
    "agenthandover": {
      "command": "agenthandover-mcp"
    }
  }
}
```

Exposes 8 tools:

| Tool | What it does |
|------|-------------|
| `list_ready_skills` | Skills ready for execution (all gates passed) |
| `get_skill(slug)` | Full Skill with steps, strategy, voice, guardrails + execution protocol |
| `search_skills(query)` | Semantic search -- find Skills by meaning |
| `list_all_skills` | All Skills including drafts |
| `get_user_profile` | User's tools, working hours, writing style |
| `report_execution_start(slug)` | Tell AgentHandover you're starting to execute a Skill |
| `report_step_result(id, step)` | Report each step's outcome (completed or deviated) |
| `report_execution_complete(id)` | Report final status -- triggers Skill improvement |

### Claude Code

```bash
agenthandover connect claude-code
```

Skills appear as `/slash-commands`. Type `/reddit-community-marketing` and Claude Code gets the full Skill.

### Codex

```bash
agenthandover connect codex
```

Generates `AGENTS.md` with all agent-ready Skills, strategy, guardrails, and voice guidance.

### OpenClaw

```bash
agenthandover connect openclaw
```

Skills auto-sync to the OpenClaw workspace. Nothing to configure.

### REST API

Already running on localhost:9477:

```bash
curl http://localhost:9477/ready              # Agent-ready Skills
curl http://localhost:9477/bundle/my-workflow  # Full handoff bundle
curl -X POST http://localhost:9477/search/semantic \
  -d '{"query": "deploy to production"}'      # Semantic search
```

## Install

### Download and run

Download the latest `.pkg` from [**Releases**](https://github.com/sandroandric/OpenMimic/releases) and double-click.

The onboarding app walks you through: permissions, AI model downloads (Qwen for screen understanding and Skill generation, nomic-embed-text for semantic search, optional SigLIP for image embeddings), Chrome extension, and your first recording.

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
ollama pull qwen3.5:4b         # Skill generation (~3.4 GB)
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

### Choose your AI model

AgentHandover defaults to local Qwen models via Ollama -- free, fast, private. Six backends supported:

| Backend | Best for |
|---------|----------|
| **Ollama** (default) | Local, free, private |
| **MLX** | Fastest on Apple Silicon |
| **llama.cpp** | Cross-platform local |
| **OpenAI / Anthropic / Google** | Highest quality (remote, opt-in) |

Switch via `config.toml` or `agenthandover setup --vlm`.

## The Menu Bar App

AgentHandover lives in your menu bar:

- **Status** -- daemon and worker health
- **Today's stats** -- events captured, annotations completed, Skills generated
- **Attention items** -- Focus Q&A questions waiting, drafts ready for review
- **Record** -- one click to start a focus recording
- **Workflows** -- browse all Skills, approve drafts, see confidence and evidence
- **Digest** -- daily summary of what was learned and what needs attention

Review the strategy, steps, and guardrails. Click "Approve for Agents" when it looks right. One click. No Skill reaches agents without your sign-off.

## Privacy

Everything runs on your machine:

- **Local-first.** VLM inference via Ollama. Cloud APIs are opt-in with explicit consent.
- **Screenshots are temporary.** Deleted after VLM annotation. Only structured text survives. Image embeddings computed before deletion.
- **Auto-redaction.** API keys, tokens, passwords, credit card numbers scrubbed before storage.
- **Secure field exclusion.** Password and credit card inputs are never captured.
- **Knowledge base is local.** Vector store, voice profiles, and all Skills live on your machine. Never uploaded.
- **Encryption at rest.** Artifacts use zstd + XChaCha20-Poly1305.
- **Configurable retention.** Raw events pruned at 14 days. Valuable evidence extracted and preserved permanently before expiry.
- **No telemetry.** Nothing phones home. Ever.

<details>
<summary><strong>Architecture</strong></summary>

```
                              You work normally
                                    |
                                    v
Chrome Extension -----> Daemon (Rust) ---SQLite WAL---> Worker (Python)
  DOM snapshots           Screenshots                     |
  Click targets           OS events                       v
  Form field IDs          Clipboard                  11-stage pipeline:
                          Perceptual dedup           VLM annotation
                                                     Activity classification
                    Menu Bar App (SwiftUI)            Text + image embedding
                    Status - Record - Workflows      Semantic clustering
                    Digest - Focus Q&A               Behavioral synthesis
                                                     Voice analysis
                                                     Skill generation
                                                          |
                                                          v
                                              +------------------------+
                                              |    Knowledge Base      |
                                              |                        |
                                              |  Skills (v3 schema)    |
                                              |  Vector store (768d)   |
                                              |  Image vectors (1152d) |
                                              |  Voice profiles        |
                                              |  User profile          |
                                              |  Evidence + history    |
                                              +-----+------+-----------+
                                                    |      |
                                      +-------------+      +------------+
                                      v              v                  v
                                MCP Server     Claude Code       OpenClaw SOPs
                                (any agent)    /slash-commands    (auto-sync)
                                Codex          REST API
                                AGENTS.md      localhost:9477
```

| Component | Language | Role |
|-----------|----------|------|
| **Daemon** | Rust | Always-on observer -- screenshots, OS events, clipboard, dedup |
| **Worker** | Python | Intelligence -- 11-stage pipeline, vector KB, behavioral synthesis, voice analysis, lifecycle, export |
| **Extension** | TypeScript | Chrome MV3 -- DOM snapshots, click targets, form field context, ARIA labels |
| **CLI** | Rust | Service management, focus recording, agent connection |
| **App** | SwiftUI | Menu bar -- status, recording, workflows, digest, Focus Q&A |
| **MCP Server** | Python | Universal agent interface -- 5 tools + 3 resources via MCP protocol |
| **Knowledge Base** | SQLite + JSON | Vector store, Skills, voice profiles, user profile, evidence |

</details>

<details>
<summary><strong>CLI reference</strong></summary>

| Command | Description |
|---------|-------------|
| `agenthandover status` | Service health and stats |
| `agenthandover start all` | Start daemon + worker |
| `agenthandover stop all` | Stop services |
| `agenthandover focus start "title"` | Record a workflow |
| `agenthandover focus stop` | Stop recording |
| `agenthandover skills list` | List all Skills |
| `agenthandover skills approve <slug>` | Approve for agents |
| `agenthandover skills promote <slug> <state>` | Promote lifecycle |
| `agenthandover connect <agent>` | Set up agent integration |
| `agenthandover doctor` | Pre-flight health check |
| `agenthandover watch` | Live dashboard |
| `agenthandover logs worker -f` | Follow worker logs |

</details>

## Uninstall

```bash
agenthandover uninstall              # Remove services, keep data
agenthandover uninstall --purge-data # Remove everything
```

## License

[BSL 1.1](LICENSE) -- source available, non-commercial. Converts to Apache 2.0 on 2030-03-25.
