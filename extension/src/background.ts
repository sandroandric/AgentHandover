/**
 * OpenMimic Observer — Background Service Worker (MV3)
 *
 * Responsibilities:
 *   1. Log service worker lifecycle events (install, activate).
 *   2. Listen for messages from content scripts and relay them onward.
 *   3. Manage the native messaging connection to the local daemon.
 *
 * This service worker is the single coordination point between the
 * per-tab content scripts and the local daemon process.
 */

import {
  connectNativeHost,
  disconnectNativeHost,
  sendToNative,
  onNativeMessage,
  onNativeDisconnect,
  isConnected,
  resendBufferedMessages,
  type NativeInboundMessage,
} from './native-messaging';

// ---------------------------------------------------------------------------
// Service worker lifecycle
// ---------------------------------------------------------------------------

self.addEventListener('install', () => {
  console.log('[OpenMimic:bg] Service worker installed');
});

self.addEventListener('activate', () => {
  console.log('[OpenMimic:bg] Service worker activated');
  initNativeConnection();
});

// ---------------------------------------------------------------------------
// Native messaging bootstrap
// ---------------------------------------------------------------------------

let reconnectAttempts = 0;
const MAX_RECONNECT_DELAY = 300_000; // 5 minutes
const MAX_RECONNECT_ATTEMPTS = 100;

// Unsubscribe handles for native messaging listeners — prevents listener
// accumulation across reconnect cycles (each initNativeConnection call
// registers new callbacks; old ones must be cleaned up first).
let unsubMessage: (() => void) | null = null;
let unsubDisconnect: (() => void) | null = null;

/**
 * Establish a connection to the daemon.  If the connection drops (e.g. daemon
 * restarts) we schedule a reconnect with exponential backoff and jitter.
 */
function initNativeConnection(): void {
  if (isConnected()) {
    console.log('[OpenMimic:bg] Native connection already active');
    return;
  }

  // Clean up previous listeners to prevent accumulation across reconnects
  unsubMessage?.();
  unsubDisconnect?.();
  unsubMessage = null;
  unsubDisconnect = null;

  try {
    connectNativeHost();

    // Reset reconnect counter on successful connection
    reconnectAttempts = 0;

    unsubMessage = onNativeMessage((message: NativeInboundMessage) => {
      console.log('[OpenMimic:bg] Daemon message:', message.type);
      handleDaemonMessage(message);
    });

    unsubDisconnect = onNativeDisconnect(() => {
      reconnectAttempts++;
      if (reconnectAttempts > MAX_RECONNECT_ATTEMPTS) {
        console.error('[OpenMimic:bg] Max reconnect attempts reached. Giving up.');
        return;
      }
      const delay = Math.min(5_000 * Math.pow(1.5, reconnectAttempts - 1), MAX_RECONNECT_DELAY);
      const jitter = delay * 0.1 * Math.random();
      console.warn(
        `[OpenMimic:bg] Lost daemon connection — reconnect attempt ${reconnectAttempts} in ${Math.round(delay + jitter)}ms`,
      );
      setTimeout(() => {
        initNativeConnection();
        // Resend buffered messages after reconnecting
        if (isConnected()) {
          resendBufferedMessages();
        }
      }, delay + jitter);
    });
  } catch (err) {
    console.error('[OpenMimic:bg] Failed to connect to daemon:', err);
    reconnectAttempts++;
    if (reconnectAttempts > MAX_RECONNECT_ATTEMPTS) {
      console.error('[OpenMimic:bg] Max reconnect attempts reached. Giving up.');
      return;
    }
    const delay = Math.min(5_000 * Math.pow(1.5, reconnectAttempts - 1), MAX_RECONNECT_DELAY);
    const jitter = delay * 0.1 * Math.random();
    console.log(`[OpenMimic:bg] Retrying in ${Math.round(delay + jitter)}ms`);
    setTimeout(initNativeConnection, delay + jitter);
  }
}

// ---------------------------------------------------------------------------
// Message handling: content script -> background
// ---------------------------------------------------------------------------

/**
 * Content scripts send messages via `chrome.runtime.sendMessage`.
 * We forward relevant ones to the daemon over the native port.
 */
chrome.runtime.onMessage.addListener(
  (
    message: ContentScriptMessage,
    sender: chrome.runtime.MessageSender,
    sendResponse: (response: BackgroundResponse) => void,
  ) => {
    const tabId = sender.tab?.id ?? -1;
    const url = sender.tab?.url ?? sender.url ?? 'unknown';

    console.log(
      '[OpenMimic:bg] Content script message from tab', tabId,
      'type:', message.type,
      'url:', url,
    );

    switch (message.type) {
      case 'content_ready':
        handleContentReady(tabId, url);
        sendResponse({ ok: true });
        break;

      case 'dom_snapshot':
      case 'click_intent':
      case 'dwell_snapshot':
      case 'scroll_snapshot':
      case 'secure_field_status':
        forwardToDaemon(message, tabId, url);
        sendResponse({ ok: true });
        break;

      default:
        console.warn('[OpenMimic:bg] Unknown message type:', message.type);
        sendResponse({ ok: false, error: 'unknown_message_type' });
    }

    // Return true to keep the message channel open for async sendResponse.
    return true;
  },
);

// ---------------------------------------------------------------------------
// Content script message types
// ---------------------------------------------------------------------------

/** Union of all messages a content script may send to the background. */
interface ContentScriptMessage {
  type:
    | 'content_ready'
    | 'dom_snapshot'
    | 'click_intent'
    | 'dwell_snapshot'
    | 'scroll_snapshot'
    | 'secure_field_status';
  payload?: Record<string, unknown>;
}

/** Standard response back to content scripts. */
interface BackgroundResponse {
  ok: boolean;
  error?: string;
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

function handleContentReady(tabId: number, url: string): void {
  console.log('[OpenMimic:bg] Content script ready in tab', tabId, '—', url);

  if (isConnected()) {
    sendToNative('content_ready', { tabId, url });
  }
}

function forwardToDaemon(
  message: ContentScriptMessage,
  tabId: number,
  url: string,
): void {
  if (!isConnected()) {
    console.warn('[OpenMimic:bg] Daemon not connected — dropping', message.type);
    return;
  }

  sendToNative(message.type, {
    tabId,
    url,
    ...(message.payload ?? {}),
  });
}

/**
 * Handle messages arriving from the daemon.
 *
 * Currently a pass-through logger.  Future modules will route commands
 * (e.g. "request DOM snapshot for tab X") to the appropriate content script.
 */
function handleDaemonMessage(message: NativeInboundMessage): void {
  switch (message.type) {
    case 'ping':
      console.log('[OpenMimic:bg] Daemon ping received');
      sendToNative('pong', {});
      break;

    case 'request_snapshot': {
      const targetTab = message.payload.tabId as number | undefined;
      if (targetTab !== undefined) {
        chrome.tabs.sendMessage(targetTab, {
          type: 'request_snapshot',
          payload: message.payload,
        });
      } else {
        // Fallback: query active tab when no specific tabId provided
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          const activeTab = tabs[0]?.id;
          if (activeTab) {
            console.log('[OpenMimic:bg] request_snapshot fallback to active tab', activeTab);
            chrome.tabs.sendMessage(activeTab, {
              type: 'request_snapshot',
              payload: message.payload,
            });
          }
        });
      }
      break;
    }

    default:
      console.log('[OpenMimic:bg] Unhandled daemon message type:', message.type);
  }
}
