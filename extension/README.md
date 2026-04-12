# AgentHandover Chrome Extension

Browser observer for the AgentHandover system. Captures DOM snapshots, click intent, scroll-read patterns, and dwell time — sent to the local daemon via Chrome Native Messaging.

## Setup

The extension uses `nativeMessaging` and cannot be published to the Chrome Web Store. It must be loaded as an unpacked extension.

### 1. Build

```bash
cd extension
npm install
npx webpack --mode production
```

### 2. Install Native Messaging Host

First, run `agenthandover doctor` to check whether the native messaging host is
already installed. If it reports missing, install it with the script below.

Alternatively, run the install script **from the repo root**:

```bash
# From the repository root directory
bash scripts/install-native-host.sh --extension-id jpemkdcihaijkolbkankcldmiimmmnfo
```

> **Important:** You must provide `--extension-id` with your actual extension ID.
> After loading the extension in Chrome, find the ID on `chrome://extensions`.
> The default ID `jpemkdcihaijkolbkankcldmiimmmnfo` comes from the `key` field
> in `manifest.json` and is stable for unpacked loads from the same key.

### 3. Load in Chrome

1. Open `chrome://extensions`
2. Enable **Developer Mode** (toggle in top-right)
3. Click **Load unpacked**
4. Select the `extension/dist/` directory (or `extension/` if using source)

### 4. Verify

The extension connects to the daemon automatically. Check `agenthandover status` to see if browser events are flowing.

## Privacy

- All data stays local — no network requests from the extension
- `deny_network_egress: true` in config enforces this
- Secure fields (password inputs, credit card fields) are automatically excluded
- No content is captured from incognito windows unless explicitly allowed

## Architecture

- `background.ts` — Service worker, manages native messaging connection
- `content.ts` — Content script injected into all pages
- `dom-capture.ts` — DOM snapshot extraction (structure, not content)
- `click-capture.ts` — Click intent with UI element metadata
- `dwell-tracker.ts` — Reading/scrolling pattern detection
- `secure-field.ts` — Password/sensitive field exclusion
- `native-messaging.ts` — Chrome Native Messaging protocol handler
