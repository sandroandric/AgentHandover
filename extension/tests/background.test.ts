/**
 * Tests for the background service worker: native-connection lifecycle
 * across MV3 worker restarts, and navigation events from tabs.onUpdated.
 *
 * Each "worker start" is a fresh evaluation of background.ts (and of the
 * native-messaging module it imports) against a chrome mock, the way Chrome
 * re-evaluates the service worker after stopping it.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

type Listener = (...args: any[]) => unknown;

interface MockPort {
  postMessage: ReturnType<typeof vi.fn>;
  disconnect: () => void;
  onMessage: { addListener: (fn: Listener) => void };
  onDisconnect: { addListener: (fn: Listener) => void };
  /** Simulate the host going away (Chrome fires onDisconnect). */
  drop: () => void;
}

interface ChromeMock {
  connectNative: ReturnType<typeof vi.fn>;
  ports: MockPort[];
  listeners: Record<string, Listener[]>;
  alarms: Map<string, { periodInMinutes?: number }>;
  fire: (name: string, ...args: unknown[]) => void;
}

function makePort(): MockPort {
  const disconnectListeners: Listener[] = [];
  const port: MockPort = {
    postMessage: vi.fn(),
    disconnect: () => port.drop(),
    onMessage: { addListener: () => {} },
    onDisconnect: { addListener: (fn) => disconnectListeners.push(fn) },
    drop: () => {
      for (const fn of disconnectListeners) fn();
    },
  };
  return port;
}

/**
 * Install a chrome mock. Alarms persist across worker restarts in Chrome,
 * so a restart can pass the previous run's alarm map.
 */
function installChrome(alarms = new Map<string, { periodInMinutes?: number }>()): ChromeMock {
  const listeners: Record<string, Listener[]> = {};
  const event = (name: string) => ({
    addListener: (fn: Listener) => {
      (listeners[name] ??= []).push(fn);
    },
  });
  const mock: ChromeMock = {
    connectNative: vi.fn(),
    ports: [],
    listeners,
    alarms,
    fire: (name, ...args) => {
      for (const fn of listeners[name] ?? []) fn(...args);
    },
  };
  mock.connectNative.mockImplementation(() => {
    const port = makePort();
    mock.ports.push(port);
    return port;
  });

  (globalThis as Record<string, unknown>).chrome = {
    runtime: {
      connectNative: mock.connectNative,
      lastError: undefined,
      onMessage: event('runtime.onMessage'),
      onStartup: event('runtime.onStartup'),
      onInstalled: event('runtime.onInstalled'),
    },
    tabs: {
      onUpdated: event('tabs.onUpdated'),
      onRemoved: event('tabs.onRemoved'),
      sendMessage: vi.fn(),
      query: vi.fn(),
    },
    alarms: {
      get: (name: string, cb: (alarm?: unknown) => void) => {
        const alarm = alarms.get(name);
        cb(alarm ? { name, ...alarm } : undefined);
      },
      create: vi.fn((name: string, info: { periodInMinutes?: number }) => {
        alarms.set(name, info);
      }),
      onAlarm: event('alarms.onAlarm'),
    },
  };
  return mock;
}

/** Start (or restart) the service worker: a fresh module evaluation. */
async function startWorker(): Promise<void> {
  vi.resetModules();
  await import('../src/background');
}

const activeTab = (url: string, title = 'Page') => ({
  id: 5,
  active: true,
  status: 'complete',
  url,
  title,
});

function sentMessages(port: MockPort): Array<{ type: string; payload: Record<string, unknown> }> {
  return port.postMessage.mock.calls.map((call) => call[0]);
}

beforeEach(() => {
  vi.spyOn(console, 'log').mockImplementation(() => {});
  vi.spyOn(console, 'warn').mockImplementation(() => {});
  vi.spyOn(console, 'error').mockImplementation(() => {});
});

afterEach(() => {
  delete (globalThis as Record<string, unknown>).chrome;
  vi.restoreAllMocks();
});

describe('native connection lifecycle', () => {
  it('connects when the worker starts, outside any lifecycle event', async () => {
    const chrome = installChrome();
    await startWorker();
    expect(chrome.connectNative).toHaveBeenCalledTimes(1);
    expect(chrome.connectNative).toHaveBeenCalledWith('com.agenthandover.host');
  });

  it('registers onStartup/onInstalled and a one-minute reconnect alarm', async () => {
    const chrome = installChrome();
    await startWorker();
    expect(chrome.listeners['runtime.onStartup']).toHaveLength(1);
    expect(chrome.listeners['runtime.onInstalled']).toHaveLength(1);
    expect(chrome.alarms.get('ah-reconnect')).toEqual({ periodInMinutes: 1 });
  });

  it('does not recreate an alarm that survived a worker restart', async () => {
    const first = installChrome();
    await startWorker();
    const second = installChrome(first.alarms);
    await startWorker();
    const create = (globalThis as any).chrome.alarms.create;
    expect(create).not.toHaveBeenCalled();
    expect(second.alarms.get('ah-reconnect')).toEqual({ periodInMinutes: 1 });
  });

  it('after the worker is terminated between events, the next tabs.onUpdated reconnects and forwards', async () => {
    const before = installChrome();
    await startWorker();
    before.fire('tabs.onUpdated', 5, { status: 'complete' }, activeTab('https://a.example/one'));
    expect(sentMessages(before.ports[0]).map((m) => m.type)).toEqual(['navigation_change']);

    // Chrome stops the idle worker: its port closes with it. The next tab
    // update starts a new worker (fresh module state, same alarms).
    before.ports[0].drop();
    const after = installChrome(before.alarms);
    await startWorker();
    after.fire('tabs.onUpdated', 5, { status: 'complete' }, activeTab('https://a.example/two', 'Two'));

    expect(after.connectNative).toHaveBeenCalledTimes(1);
    const sent = sentMessages(after.ports[0]);
    expect(sent).toHaveLength(1);
    expect(sent[0].type).toBe('navigation_change');
    expect(sent[0].payload).toEqual({ tabId: 5, url: 'https://a.example/two', title: 'Two' });
  });

  it('reconnects on the next event when the port dropped while the worker kept running', async () => {
    const chrome = installChrome();
    await startWorker();
    chrome.ports[0].drop();

    chrome.fire('tabs.onUpdated', 5, { url: 'https://a.example/spa' }, activeTab('https://a.example/spa'));

    expect(chrome.connectNative).toHaveBeenCalledTimes(2);
    expect(sentMessages(chrome.ports[0])).toHaveLength(0);
    expect(sentMessages(chrome.ports[1]).map((m) => m.type)).toEqual(['navigation_change']);
  });

  it('queues while the host is unreachable and the alarm reconnects and drains', async () => {
    const chrome = installChrome();
    await startWorker();
    chrome.ports[0].drop();
    chrome.connectNative.mockImplementationOnce(() => {
      throw new Error('host not found');
    });

    // Forwarded while the host is down: queued, not lost.
    chrome.fire(
      'runtime.onMessage',
      { type: 'form_submit', payload: { formAnchor: { tag: 'form', name: 'login' } } },
      { tab: { id: 9, url: 'https://b.example/login' } },
      () => {},
    );
    expect(chrome.ports).toHaveLength(1);

    chrome.fire('alarms.onAlarm', { name: 'ah-reconnect' });

    expect(chrome.ports).toHaveLength(2);
    const sent = sentMessages(chrome.ports[1]);
    expect(sent.map((m) => m.type)).toEqual(['form_submit']);
    expect(sent[0].payload).toMatchObject({
      tabId: 9,
      url: 'https://b.example/login',
      formAnchor: { tag: 'form', name: 'login' },
    });
  });

  it('the alarm leaves a live connection alone', async () => {
    const chrome = installChrome();
    await startWorker();
    chrome.fire('alarms.onAlarm', { name: 'ah-reconnect' });
    chrome.fire('alarms.onAlarm', { name: 'something-else' });
    expect(chrome.connectNative).toHaveBeenCalledTimes(1);
  });
});

describe('navigation events', () => {
  it('reports each committed navigation of the active tab once, from the tab, not the DOM', async () => {
    const chrome = installChrome();
    await startWorker();
    const url = 'https://mail.example/inbox?q=1';

    // A full load: loading -> title -> complete.
    chrome.fire('tabs.onUpdated', 5, { status: 'loading', url }, { ...activeTab(url), status: 'loading' });
    chrome.fire('tabs.onUpdated', 5, { title: 'Inbox' }, { ...activeTab(url, 'Inbox'), status: 'loading' });
    chrome.fire('tabs.onUpdated', 5, { status: 'complete' }, activeTab(url, 'Inbox'));
    // Later title-only change (unread counter) is not a navigation.
    chrome.fire('tabs.onUpdated', 5, { title: '(1) Inbox' }, activeTab(url, '(1) Inbox'));
    // A duplicate complete for the same URL is not reported twice.
    chrome.fire('tabs.onUpdated', 5, { status: 'complete' }, activeTab(url, 'Inbox'));

    const sent = sentMessages(chrome.ports[0]);
    expect(sent).toHaveLength(1);
    expect(sent[0]).toMatchObject({
      type: 'navigation_change',
      payload: { tabId: 5, url, title: 'Inbox' },
    });
  });

  it('ignores background tabs and pages without a readable URL', async () => {
    const chrome = installChrome();
    await startWorker();
    chrome.fire('tabs.onUpdated', 6, { status: 'complete' }, { ...activeTab('https://c.example/'), active: false });
    chrome.fire('tabs.onUpdated', 7, { status: 'complete' }, { ...activeTab(''), url: undefined });
    expect(sentMessages(chrome.ports[0])).toHaveLength(0);
  });
});
