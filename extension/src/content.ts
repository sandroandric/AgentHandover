/**
 * OpenMimic Observer — Content Script
 *
 * Injected at `document_idle` on every page (`<all_urls>`).
 *
 * Responsibilities:
 *   1. Log injection (URL, timestamp) for debugging.
 *   2. Notify the background service worker that the script is ready.
 *   3. Listen for commands from the background (e.g. "take a snapshot").
 *   4. Expose hook points that future modules plug into:
 *        - DOM snapshot capture  (dom-capture)
 *        - Click intent tracking (click-capture)
 *        - Dwell / scroll-read   (dwell)
 *        - Secure field detection (secure-field)
 */

import { captureViewportDOM } from './dom-capture';
import { initClickCapture } from './click-capture';
import { initSecureFieldDetection } from './secure-field';
import { initDwellTracker, type DwellConfig } from './dwell-tracker';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A module hook that the content script can register. */
interface ContentModule {
  /** Human-readable name for logging. */
  name: string;
  /** Called once when the content script initialises. */
  init: () => void;
  /** Called when the content script is about to be torn down (navigation). */
  destroy: () => void;
}

/** Messages the background may send to this content script. */
interface BackgroundCommand {
  type: string;
  payload?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Module registry
// ---------------------------------------------------------------------------

const registeredModules: ContentModule[] = [];

/**
 * Register a content-side module.  Modules are initialised in registration
 * order.
 *
 * Future files (dom-capture.ts, click-capture.ts, etc.) will import and
 * call this function during their top-level execution.
 */
export function registerModule(mod: ContentModule): void {
  console.log('[OpenMimic:content] Registering module:', mod.name);
  registeredModules.push(mod);
  mod.init();
}

// ---------------------------------------------------------------------------
// Communication helpers
// ---------------------------------------------------------------------------

/**
 * Send a typed message to the background service worker.
 *
 * Wraps `chrome.runtime.sendMessage` with consistent error handling.
 */
export function sendToBackground(
  type: string,
  payload: Record<string, unknown> = {},
  retries: number = 2,
): void {
  chrome.runtime.sendMessage({ type, payload }, (response) => {
    if (chrome.runtime.lastError) {
      // This can happen legitimately if the service worker is restarting.
      // Retry with a 1-second delay to give the service worker time to wake up.
      if (retries > 0) {
        console.warn(
          '[OpenMimic:content] sendMessage error, retrying in 1s:',
          chrome.runtime.lastError.message,
        );
        setTimeout(() => sendToBackground(type, payload, retries - 1), 1000);
      } else {
        console.warn(
          '[OpenMimic:content] sendMessage error (retries exhausted):',
          chrome.runtime.lastError.message,
        );
      }
      return;
    }
    if (response && !response.ok) {
      console.warn(
        '[OpenMimic:content] Background rejected message:',
        response.error,
      );
    }
  });
}

// ---------------------------------------------------------------------------
// Secure field state — shared across modules
// ---------------------------------------------------------------------------

/**
 * Global flag: true when a password / secure input field is focused.
 * While true, other capture modules must suppress data collection.
 */
let secureFieldActive = false;

export function isSecureFieldActive(): boolean {
  return secureFieldActive;
}

/**
 * Guarded send function: wraps sendToBackground but drops messages
 * while a secure field is active. Used by click-capture and dwell modules.
 */
function guardedSend(type: string, payload: Record<string, unknown>): void {
  if (secureFieldActive) {
    console.log('[OpenMimic:content] Suppressed', type, '— secure field active');
    return;
  }
  sendToBackground(type, payload);
}

// ---------------------------------------------------------------------------
// Background command listener
// ---------------------------------------------------------------------------

chrome.runtime.onMessage.addListener(
  (
    message: BackgroundCommand,
    _sender: chrome.runtime.MessageSender,
    sendResponse: (response: unknown) => void,
  ) => {
    console.log('[OpenMimic:content] Received command:', message.type);

    switch (message.type) {
      case 'request_snapshot': {
        console.log('[OpenMimic:content] Snapshot requested — capturing viewport DOM');
        const nodes = captureViewportDOM();
        console.log('[OpenMimic:content] Captured', nodes.length, 'top-level nodes');
        sendToBackground('dom_snapshot', {
          nodes,
          url: window.location.href,
          captureReason: 'request',
        });
        sendResponse({ ok: true, nodeCount: nodes.length });
        break;
      }

      default:
        console.log('[OpenMimic:content] Unknown command:', message.type);
        sendResponse({ ok: false, error: 'unknown_command' });
    }

    return true;
  },
);

// ---------------------------------------------------------------------------
// Teardown
// ---------------------------------------------------------------------------

/**
 * Clean up all registered modules.  Called automatically before the page
 * unloads so modules can remove listeners / observers.
 */
function destroyAllModules(): void {
  for (const mod of registeredModules) {
    try {
      mod.destroy();
      console.log('[OpenMimic:content] Destroyed module:', mod.name);
    } catch (err) {
      console.error('[OpenMimic:content] Error destroying module', mod.name, err);
    }
  }
  registeredModules.length = 0;
}

window.addEventListener('beforeunload', destroyAllModules);

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

console.log(
  '[OpenMimic:content] Content script injected at',
  window.location.href,
  'timestamp:',
  new Date().toISOString(),
);

sendToBackground('content_ready', {
  url: window.location.href,
  title: document.title,
  timestamp: new Date().toISOString(),
});

// ---------------------------------------------------------------------------
// Register built-in modules
// ---------------------------------------------------------------------------

// IMPORTANT: Module registration order matters. Secure-field MUST be first
// to ensure the guard is active before capture modules start. Do not change
// this order or make init() async without adding explicit synchronization.

// --- Secure Field Detection (must be registered first so the guard is active
//     before other modules start capturing) ---
{
  let cleanupSecure: (() => void) | null = null;

  registerModule({
    name: 'secure-field',
    init() {
      cleanupSecure = initSecureFieldDetection(
        sendToBackground, // secure field status always goes through unguarded
        (isSecure: boolean) => {
          secureFieldActive = isSecure;
          console.log(
            '[OpenMimic:content] Secure field state:',
            isSecure ? 'ACTIVE' : 'inactive',
          );
        },
      );
    },
    destroy() {
      cleanupSecure?.();
      cleanupSecure = null;
    },
  });
}

// --- Click Intent Capture ---
{
  let cleanupClick: (() => void) | null = null;

  registerModule({
    name: 'click-capture',
    init() {
      cleanupClick = initClickCapture(guardedSend);
    },
    destroy() {
      cleanupClick?.();
      cleanupClick = null;
    },
  });
}

// --- Dwell + Scroll Snapshot Triggers ---
{
  let cleanupDwell: (() => void) | null = null;

  const dwellConfig: DwellConfig = {
    dwellThresholdMs: 3000,
    scrollReadThresholdMs: 8000,
  };

  registerModule({
    name: 'dwell-tracker',
    init() {
      cleanupDwell = initDwellTracker(
        dwellConfig,
        () => {
          if (secureFieldActive) {
            console.log('[OpenMimic:content] Suppressed dwell_snapshot — secure field active');
            return;
          }
          console.log('[OpenMimic:content] Dwell snapshot triggered');
          sendToBackground('dwell_snapshot', {
            url: window.location.href,
            title: document.title,
            timestamp: new Date().toISOString(),
          });
        },
        () => {
          if (secureFieldActive) {
            console.log('[OpenMimic:content] Suppressed scroll_snapshot — secure field active');
            return;
          }
          console.log('[OpenMimic:content] Scroll-read snapshot triggered');
          sendToBackground('scroll_snapshot', {
            url: window.location.href,
            title: document.title,
            timestamp: new Date().toISOString(),
          });
        },
      );
    },
    destroy() {
      cleanupDwell?.();
      cleanupDwell = null;
    },
  });
}

// Runtime assertion: verify secure-field is the first registered module
if (registeredModules.length > 0 && registeredModules[0].name !== 'secure-field') {
  console.error(
    '[OpenMimic:content] CRITICAL: secure-field must be the first registered module, but found:',
    registeredModules[0].name,
  );
}
