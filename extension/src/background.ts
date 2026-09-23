/**
 * AgentHandover Observer — Background Service Worker (MV3)
 *
 * Responsibilities:
 *   1. Keep the native messaging connection to the local daemon alive
 *      across service-worker restarts.
 *   2. Listen for messages from content scripts and relay them onward.
 *   3. Report navigations of the active tab (`tabs.onUpdated`, no DOM).
 *
 * This service worker is the single coordination point between the
 * per-tab content scripts and the local daemon process.
 *
 * Lifecycle: Chrome stops an idle MV3 worker after ~30 s and starts it
 * again, re-evaluating this module, for the next event (a message, a tab
 * update, an alarm). `install`/`activate` fire only once per version, so
 * the connection is opened at module top level (every wake), on
 * `runtime.onStartup`/`onInstalled`, whenever an event is forwarded while
 * disconnected, and by a one-minute alarm. A `setTimeout` retry does not
 * survive the worker being stopped; an alarm does.
 */

import {
  connectNativeHost,
  sendToNative,
  onNativeMessage,
  onNativeDisconnect,
  isConnected,
  clearReconnectBuffer,
  type NativeInboundMessage,
} from './native-messaging';

// ---------------------------------------------------------------------------
// Disconnect queue — buffers messages while the daemon is unreachable
// ---------------------------------------------------------------------------

interface QueuedMessage {
  message: ContentScriptMessage;
  tabId: number;
  url: string;
}

const MAX_DISCONNECT_QUEUE = 100;
const disconnectQueue: QueuedMessage[] = [];

/** Name of the alarm that reconnects a dropped native connection. */
export const RECONNECT_ALARM = 'ah-reconnect';

// Unsubscribe handles for native messaging listeners — prevents listener
// accumulation across reconnect cycles (each initNativeConnection call
// registers new callbacks; old ones must be cleaned up first).
let unsubMessage: (() => void) | null = null;
let unsubDisconnect: (() => void) | null = null;

/**
 * Connect to the daemon if not connected, then flush what was queued while
 * disconnected.  Safe to call on every wake and every event.
 */
function initNativeConnection(): void {
  if (isConnected()) {
    return;
  }

  // Clean up previous listeners to prevent accumulation across reconnects
  unsubMessage?.();
  unsubDisconnect?.();
  unsubMessage = null;
  unsubDisconnect = null;

  try {
    connectNativeHost();
  } catch (err) {
    console.error('[AgentHandover:bg] Failed to connect to daemon (the reconnect alarm retries):', err);
    return;
  }

  unsubMessage = onNativeMessage((message: NativeInboundMessage) => {
    console.log('[AgentHandover:bg] Daemon message:', message.type);
    handleDaemonMessage(message);
  });
  unsubDisconnect = onNativeDisconnect(() => {
    console.warn('[AgentHandover:bg] Lost daemon connection — the next event or the reconnect alarm reconnects');
  });

  // Clear the resend buffer (do NOT replay — the daemon cannot deduplicate
  // across reconnects, so replaying causes duplicate browser events), then
  // forward what was queued while disconnected.
  clearReconnectBuffer();
  drainDisconnectQueue();
}

/** Create the reconnect alarm unless it already exists (alarms outlive the worker). */
function ensureReconnectAlarm(): void {
  chrome.alarms.get(RECONNECT_ALARM, (existing) => {
    if (!existing) {
      chrome.alarms.create(RECONNECT_ALARM, { periodInMinutes: 1 });
    }
  });
}

chrome.runtime.onStartup.addListener(() => {
  initNativeConnection();
});

chrome.runtime.onInstalled.addListener(() => {
  ensureReconnectAlarm();
  initNativeConnection();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === RECONNECT_ALARM && !isConnected()) {
    initNativeConnection();
  }
});

// ---------------------------------------------------------------------------
// Navigation of the active tab
// ---------------------------------------------------------------------------

/** Last URL reported per tab, so each navigation is reported once. */
const lastNavigatedUrl = new Map<number, string>();

/**
 * Report a navigation when the active tab finishes loading a new URL (a
 * full load reports `status: 'complete'`; a history push reports the new
 * `url`).  Title-only updates are not navigations.  URL and title come from
 * the tab (host permission), never from the page's DOM.
 */
chrome.tabs.onUpdated.addListener(
  (tabId: number, changeInfo: chrome.tabs.TabChangeInfo, tab: chrome.tabs.Tab) => {
    if (changeInfo.url === undefined && changeInfo.status !== 'complete') return;
    if (!tab.active || tab.status !== 'complete' || !tab.url) return;
    if (lastNavigatedUrl.get(tabId) === tab.url) return;
    lastNavigatedUrl.set(tabId, tab.url);
    forwardToDaemon(
      { type: 'navigation_change', payload: { title: tab.title ?? '' } },
      tabId,
      tab.url,
    );
  },
);

chrome.tabs.onRemoved.addListener((tabId: number) => {
  lastNavigatedUrl.delete(tabId);
});

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
      '[AgentHandover:bg] Content script message from tab', tabId,
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
      case 'form_submit':
        forwardToDaemon(message, tabId, url);
        sendResponse({ ok: true });
        break;

      default:
        console.warn('[AgentHandover:bg] Unknown message type:', message.type);
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
    | 'secure_field_status'
    | 'form_submit'
    | 'navigation_change';
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
  console.log('[AgentHandover:bg] Content script ready in tab', tabId, '—', url);

  initNativeConnection();
  if (isConnected()) {
    sendToNative('content_ready', { tabId, url });
  }
}

/**
 * Drain the disconnect queue by re-forwarding each queued message.
 * Called after a successful (re)connect.
 */
function drainDisconnectQueue(): void {
  if (disconnectQueue.length === 0) return;

  console.log('[AgentHandover:bg] Draining disconnect queue:', disconnectQueue.length, 'messages');
  // Splice the queue so forwardToDaemon sees an empty queue (avoids re-queuing
  // if the connection drops again mid-drain — those would be freshly queued).
  const pending = disconnectQueue.splice(0);
  for (const item of pending) {
    forwardToDaemon(item.message, item.tabId, item.url);
  }
}

function forwardToDaemon(
  message: ContentScriptMessage,
  tabId: number,
  url: string,
): void {
  // The worker may have been restarted, or the port dropped, since the last
  // event: reconnect before deciding to queue.
  if (!isConnected()) {
    initNativeConnection();
  }
  if (!isConnected()) {
    if (disconnectQueue.length >= MAX_DISCONNECT_QUEUE) {
      console.warn('[AgentHandover:bg] Disconnect queue full — dropping oldest to enqueue', message.type);
      disconnectQueue.shift();
    }
    console.log('[AgentHandover:bg] Daemon not connected — queuing', message.type, `(${disconnectQueue.length + 1}/${MAX_DISCONNECT_QUEUE})`);
    disconnectQueue.push({ message, tabId, url });
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
      console.log('[AgentHandover:bg] Daemon ping received');
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
            console.log('[AgentHandover:bg] request_snapshot fallback to active tab', activeTab);
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
      console.log('[AgentHandover:bg] Unhandled daemon message type:', message.type);
  }
}

// ---------------------------------------------------------------------------
// Bootstrap — runs on every service-worker start
// ---------------------------------------------------------------------------

ensureReconnectAlarm();
initNativeConnection();
