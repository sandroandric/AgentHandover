# OpenMimic Chrome Extension

Browser observer for the OpenMimic apprentice system. Captures DOM snapshots, click intent, scroll-read patterns, and dwell time — sent to the local daemon via Chrome Native Messaging.

## Setup

The extension uses `nativeMessaging` and cannot be published to the Chrome Web Store. It must be loaded as an unpacked extension.

### 1. Build

```bash
cd extension
npm install
npx webpack --mode production
```

### 2. Install Native Messaging Host

```bash
# Use the stable extension ID (from manifest.json key field)
bash scripts/install-native-host.sh --extension-id knldjmfmopnpolahpmmgbagdohdnhkik
```

> **Important:** You must provide `--extension-id` with your actual extension ID.
> After loading the extension in Chrome, find the ID on `chrome://extensions`.
> The default ID `knldjmfmopnpolahpmmgbagdohdnhkik` comes from the `key` field
> in `manifest.json` and is stable for unpacked loads from the same key.

Or use the installer: `openmimic doctor` will verify this is configured.

### 3. Load in Chrome

1. Open `chrome://extensions`
2. Enable **Developer Mode** (toggle in top-right)
3. Click **Load unpacked**
4. Select the `extension/dist/` directory (or `extension/` if using source)

### 4. Verify

The extension connects to the daemon automatically. Check `openmimic status` to see if browser events are flowing.

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
