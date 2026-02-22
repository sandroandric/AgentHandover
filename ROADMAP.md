# OpenMimic Roadmap

Last updated: 2026-02-22

---

## 1. Remote VLM API Integration

**Status:** Not started
**Priority:** High
**Estimated effort:** 1-2 weeks

### Problem

OpenMimic currently enforces `deny_network_egress` — VLM backends must run locally (Ollama, MLX, llama.cpp) or through a localhost-only OpenAI-compatible proxy. This means users need decent hardware (Apple Silicon with 16GB+ RAM) or must manually set up a local proxy to use cloud models. Most users just want to paste an API key and go.

### Goal

Let users choose between local VLM (privacy-first, current behavior) and remote API (convenience, SOTA quality) with a single config change or onboarding toggle. Support the major multimodal API providers directly.

### Providers to support

| Provider | Model(s) | API style | Auth |
|----------|----------|-----------|------|
| OpenAI | gpt-4o, gpt-4o-mini | OpenAI Chat Completions | `OPENAI_API_KEY` |
| Anthropic | claude-sonnet, claude-haiku | Anthropic Messages API (vision via base64 image blocks) | `ANTHROPIC_API_KEY` |
| Google | gemini-2.0-flash, gemini-2.5-pro | Google Generative AI / OpenAI-compat endpoint | `GOOGLE_API_KEY` |

### Design

**Config (`config.toml`):**

```toml
[vlm]
# "local" (current default) or "remote"
mode = "remote"

# Only used when mode = "remote"
provider = "openai"          # openai | anthropic | google
model = "gpt-4o-mini"        # provider-specific model name
api_key_env = "OPENAI_API_KEY"  # env var holding the key (never store key in config)

# Budget controls apply regardless of mode
max_jobs_per_day = 50
```

**Privacy model change:**

- `mode = "local"` — current behavior, `deny_network_egress` enforced, nothing leaves the machine
- `mode = "remote"` — screenshots and DOM context are sent to the configured provider's API
- Users must explicitly opt in to `remote` mode; `local` remains the default
- On first switch to `remote`, log a clear warning: "Screenshots and DOM context will be sent to [provider]. Sensitive data is redacted before transmission but visual content (screenshots) cannot be fully sanitized."

**Implementation:**

1. **New backends** in `worker/src/oc_apprentice_worker/backends/`:
   - `anthropic.py` — Anthropic Messages API with base64 image content blocks
   - `google.py` — Google Generative AI (or their OpenAI-compat endpoint)
   - Existing `openai_compat.py` already works for OpenAI — just lift the localhost restriction when `mode = "remote"`

2. **Config parsing** — extend `[vlm]` section in `crates/common/src/config.rs` and `worker/main.py` to read `mode`, `provider`, `model`, `api_key_env`

3. **Backend selection** — when `mode = "remote"`, skip local detection entirely and instantiate the specified remote backend. When `mode = "local"`, current priority chain (mlx > ollama > llama_cpp > openai_compat@localhost)

4. **`setup_vlm.py` / `openmimic setup --vlm`** — add remote option: "Do you want to use a cloud API instead? Paste your API key." Validate key with a lightweight test call (small image, simple prompt)

5. **Onboarding (SwiftUI)** — VLM step (step 4) gets a segmented control: Local / Cloud API. Cloud path: provider picker, API key text field (stored in Keychain via `SecItemAdd`, env var set at worker launch), test connection button

6. **Pre-send redaction** — before sending screenshots to remote APIs, apply existing inline secret redaction patterns to any text overlays detected in the image (OCR → redact → re-render is too heavy; document the limitation clearly)

### Privacy considerations

- API keys stored in macOS Keychain, never in config files
- Config only references the env var name (`api_key_env = "OPENAI_API_KEY"`)
- Worker reads key from env at startup, never logs it
- Clear user consent at first remote mode enable
- Redaction pipeline runs on DOM/text context before API call
- Screenshots sent as-is (visual redaction is not feasible in real-time) — document this tradeoff

---

## 2. LLM-Enhanced SOP Descriptions

**Status:** Not started
**Priority:** High
**Estimated effort:** 1-2 weeks
**Depends on:** Item 1 (shares the same API backend infrastructure)

### Problem

Current SOPs are structured step sequences — great for machines, but an AI agent reading one has no high-level understanding of *what the workflow accomplishes* or *why* these steps matter. The agent sees "click → type → click → submit" but doesn't know it's "filing a bug report in Jira." This context gap means agents need more inference cycles to plan execution, and errors in early steps cascade because there's no goal to sanity-check against.

### Goal

After SOP induction, run an LLM pass that adds two new sections to each SOP:

1. **Task Description** — a natural language paragraph explaining what this workflow does, when you'd use it, and what the expected outcome is. This becomes the first thing an agent reads.

2. **Execution Overview** — a structured summary of the task sequence: prerequisites, key decision points, expected state transitions, and success criteria. Helps agents plan before executing.

### Example output

```markdown
# Create Jira Bug Ticket
slug: create-jira-bug-ticket

## Task Description

This workflow creates a new bug report in Jira. It navigates to the team's
Jira board, opens the issue creation dialog, selects "Bug" as the issue type,
fills in the summary and description fields with the bug details, and submits
the form. The workflow assumes you are already logged into Jira and have
permission to create issues on the target board.

## Execution Overview

- **Goal:** Create a bug ticket in Jira with a title and description
- **Prerequisites:** Authenticated Jira session, project board access
- **Key inputs:** Bug summary (title), reproduction steps (description)
- **Decision points:** Issue type selection (always "Bug" in observed pattern)
- **Success criteria:** Issue creation confirmation page appears with new ticket ID
- **Typical duration:** ~30 seconds

## Steps
1. navigate → Jira board ...
2. click → "Create" button ...
...
```

### Design

**LLM backend reuse:**

This feature shares the same backend infrastructure as VLM (item 1). The difference is:
- VLM: sends screenshots + DOM context, asks "what UI element is this?"
- LLM: sends the raw SOP step sequence as text, asks "describe this workflow"

Both use the same `mode = "local" | "remote"` config, same provider selection, same API key management. The LLM call is text-only (no images), so it works with any model — doesn't require multimodal.

**Config (`config.toml`):**

```toml
[llm]
# Inherits mode/provider/api_key_env from [vlm] by default, or override:
# mode = "remote"
# provider = "anthropic"
# model = "claude-haiku"

# Set to false to skip LLM enhancement entirely
enhance_sops = true
```

**Implementation:**

1. **New module** `worker/src/oc_apprentice_worker/sop_enhancer.py`:
   - `enhance_sop(sop_template: dict) -> dict` — takes a raw SOP template from the inducer, calls LLM with a structured prompt, parses response, adds `task_description` and `execution_overview` fields
   - Prompt template includes: the step sequence, app names, URLs observed, number of repetitions, confidence scores
   - Structured output: request JSON response with `task_description` (string) and `execution_overview` (object with goal, prerequisites, key_inputs, decision_points, success_criteria, typical_duration)
   - Retry with simpler prompt on parse failure, fall back to no enhancement (never block SOP export)

2. **Integration point** in `worker/main.py` — after `sop_inducer.induce()` returns templates and before `openclaw_writer.write_all_sops()`:
   ```python
   if sop_enhancer and sop_templates:
       for template in sop_templates:
           enhanced = sop_enhancer.enhance_sop(template)
           template.update(enhanced)
   ```

3. **Export format update**:
   - `openclaw_writer.py` — write `task_description` and `execution_overview` as frontmatter sections before the step list
   - `generic_writer.py` — include in both markdown and JSON output
   - `sop_schema.py` — bump schema version to 1.1.0, add optional `task_description` and `execution_overview` fields

4. **Budget management** — LLM calls are cheap (text-only, ~500 tokens in, ~300 tokens out) but should still be budgeted:
   - Only enhance newly induced SOPs (not on every version bump)
   - Share the daily job budget with VLM or have a separate `[llm] max_enhancements_per_day = 20`
   - Cache: if SOP steps haven't changed, don't re-enhance

5. **Local LLM option** — when `mode = "local"`, use the same Ollama/MLX backend but with a text-only model (e.g., `llama3.2:3b` instead of `llava:7b`). Lighter weight, faster, doesn't need vision capabilities.

### OpenClaw integration

The `task_description` field is specifically designed for OpenClaw's agent planning phase. When OpenClaw loads a SOP to execute:
1. Reads `task_description` first → understands the goal
2. Reads `execution_overview` → builds execution plan with checkpoints
3. Reads `steps` → executes sequentially with goal-awareness

This means if step 3 fails, the agent knows *why* step 3 matters (from the overview) and can attempt recovery strategies aligned with the overall goal rather than blindly retrying.

---

## 3. Safari Web Extension

**Status:** Not started
**Priority:** Medium
**Estimated effort:** 2-3 weeks
**Depends on:** None (can be done in parallel with items 1-2)

### Problem

OpenMimic's browser extension is Chrome-only (MV3 + Native Messaging). Safari users get no browser observation — only system-level capture (screenshots, window titles, app switches, clipboard). This means significantly less detailed SOPs for web workflows: no CSS selectors, no DOM snapshots, no click targets, no page URLs.

### Key architectural difference

**Chrome:** Extension JS → `chrome.runtime.connectNative()` → spawns native binary → stdin/stdout pipe → Rust daemon. Direct, persistent, bidirectional.

**Safari:** Extension JS → `browser.runtime.sendNativeMessage()` → `SafariWebExtensionHandler` (Swift class inside app bundle) → must manually bridge to Rust daemon. Request-response only, no persistent connection, no daemon-initiated pushes.

### Code sharing analysis

| Module | Lines | Portable | Notes |
|--------|-------|----------|-------|
| `dom-capture.ts` | 662 | 100% | Pure DOM APIs |
| `click-capture.ts` | 185 | 100% | Standard event listeners |
| `secure-field.ts` | 180 | 100% | DOM attribute checks |
| `dwell-tracker.ts` | 174 | 100% | Timers + scroll events |
| `types.ts` | 114 | 100% | TypeScript interfaces |
| `content.ts` | 306 | ~90% | `chrome.runtime` → `browser.runtime` |
| `background.ts` | 226 | ~60% | Needs platform messaging adapter |
| `native-messaging.ts` | 242 | ~20% | Chrome-specific, needs Safari rewrite |

**~62% direct code reuse.** All observation logic is portable. Only the messaging plumbing changes.

### Design

**Platform abstraction layer:**

Refactor the extension to separate observation (portable) from messaging (platform-specific):

```
extension/
  src/
    shared/              # 100% shared between Chrome and Safari
      dom-capture.ts
      click-capture.ts
      secure-field.ts
      dwell-tracker.ts
      types.ts
    chrome/              # Chrome-specific
      native-messaging.ts
      background.ts
      content.ts         # thin wrapper importing shared + chrome messaging
    safari/              # Safari-specific
      native-messaging.ts  # uses browser.runtime.sendNativeMessage
      background.ts        # polling-based command check
      content.ts           # thin wrapper importing shared + safari messaging
    manifest.json        # shared (Safari supports Chrome manifest format)
```

**Safari native messaging bridge (`SafariWebExtensionHandler.swift`):**

```swift
class SafariWebExtensionHandler: NSObject, NSExtensionRequestHandling {
    func beginRequest(with context: NSExtensionContext) {
        // 1. Extract message from context
        // 2. Connect to Rust daemon via Unix domain socket
        // 3. Forward message, read response
        // 4. Return response via context.completeRequest()
    }
}
```

The Rust daemon already listens on a Unix socket (or can be extended to). The Swift handler connects, sends JSON, reads response. ~150 lines.

**Containing app:**

Safari extensions must live inside a macOS `.app` bundle. Two options:

- **Option A:** Embed in existing OpenMimic SwiftUI menu bar app. Pro: single app, shared UI. Con: requires migrating from SPM to Xcode project, adds complexity to existing app.
- **Option B:** Separate lightweight app (`OpenMimic Safari.app`). Pro: no risk to existing app, clean separation. Con: two apps to install.

**Recommendation:** Option B for initial implementation. Can merge later once stable.

**Daemon-to-extension messaging (the hard part):**

Chrome provides persistent bidirectional ports. Safari doesn't. For daemon-initiated commands (like `request_snapshot`):

- **Approach:** Extension background script polls the daemon every 2-5 seconds for pending commands. Daemon maintains a small command queue per extension connection. Extension processes commands and responds.
- **Tradeoff:** 2-5s latency on daemon-initiated requests vs. Chrome's near-instant. Acceptable for OpenMimic's use case (observation, not real-time interaction).

### Known risks

1. **Safari service worker instability** — documented bug (Safari 17-18) where background service workers crash after ~30s and sometimes fail to restart. Mitigation: aggressive reconnection logic in content script, fall back to content-script-only mode if background dies.

2. **Distribution** — Safari extensions need code signing + notarization (Developer ID) or Mac App Store. Adds a build/release step. Can distribute as a DMG alongside the Homebrew install initially.

3. **Testing surface** — Safari's Web Extension debugging tools are less mature than Chrome DevTools. Budget extra time for debugging.

### Implementation plan

1. **Refactor shared code** — move `dom-capture`, `click-capture`, `secure-field`, `dwell-tracker`, `types` to `shared/` directory. Create platform adapter interface. Update webpack config for two build targets.

2. **Safari extension skeleton** — use `xcrun safari-web-extension-converter` to bootstrap Xcode project from existing extension. Set up app wrapper with minimal UI ("OpenMimic Safari Extension is active").

3. **SafariWebExtensionHandler** — implement Unix socket bridge to Rust daemon. Handle message serialization, connection lifecycle, error recovery.

4. **Safari-specific messaging** — rewrite `native-messaging.ts` for Safari using `browser.runtime.sendNativeMessage()`. Implement polling for daemon-initiated commands.

5. **Build pipeline** — webpack config produces two bundles (Chrome dist, Safari dist). Xcode project references Safari dist. Add to `justfile`.

6. **Daemon socket endpoint** — if daemon doesn't already expose a Unix socket for extension communication, add one. Lightweight JSON-over-socket protocol matching the existing NM message format.

7. **Testing** — manual testing in Safari, verify DOM capture parity with Chrome, stress-test service worker recovery, verify message delivery under load.

8. **Distribution** — code signing with Developer ID, notarization via `xcrun notarytool`, DMG packaging, Homebrew cask formula.

---

## Execution Order

```
[Item 1: Remote VLM APIs]  ──────────────>  [Item 2: LLM SOP Enhancement]
   1-2 weeks                                    1-2 weeks
   (backend infra shared)                       (reuses API backends from item 1)

[Item 3: Safari Extension]  ──────────────────────────────>
   2-3 weeks (can run in parallel with items 1-2)
```

Items 1 and 2 are sequential (item 2 reuses item 1's API backend infrastructure).
Item 3 is independent and can be developed in parallel.

**Total estimated timeline: 4-6 weeks** if items 1-2 are done sequentially and item 3 in parallel.
