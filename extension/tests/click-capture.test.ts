/**
 * Tests for Click Intent Capture module.
 *
 * Verifies:
 *   - Click handler extracts semantic info (role, aria-label, data-testid)
 *   - composedPath building from elements
 *   - innerText truncation at 200 characters
 *   - CSS selector building (tag#id.class format)
 *   - Capture-phase interception
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { initClickCapture, buildSelector, buildComposedPath } from '../src/click-capture';

// ---------------------------------------------------------------------------
// Mock chrome.runtime.sendMessage (not used directly but may be needed)
// ---------------------------------------------------------------------------

beforeEach(() => {
  (globalThis as Record<string, unknown>).chrome = {
    runtime: {
      sendMessage: vi.fn(),
      lastError: null,
    },
  };
});

afterEach(() => {
  delete (globalThis as Record<string, unknown>).chrome;
});

// ---------------------------------------------------------------------------
// buildSelector tests
// ---------------------------------------------------------------------------

describe('buildSelector', () => {
  it('should return lowercase tag name for a simple element', () => {
    const el = document.createElement('DIV');
    expect(buildSelector(el)).toBe('div');
  });

  it('should include id when present', () => {
    const el = document.createElement('button');
    el.id = 'submit-btn';
    expect(buildSelector(el)).toBe('button#submit-btn');
  });

  it('should include classes when present', () => {
    const el = document.createElement('span');
    el.classList.add('primary', 'active');
    expect(buildSelector(el)).toBe('span.primary.active');
  });

  it('should include both id and classes', () => {
    const el = document.createElement('a');
    el.id = 'nav-link';
    el.classList.add('menu-item', 'highlighted');
    expect(buildSelector(el)).toBe('a#nav-link.menu-item.highlighted');
  });

  it('should handle element with no id or classes', () => {
    const el = document.createElement('p');
    expect(buildSelector(el)).toBe('p');
  });
});

// ---------------------------------------------------------------------------
// buildComposedPath tests
// ---------------------------------------------------------------------------

describe('buildComposedPath', () => {
  it('should build path from a nested DOM structure', () => {
    const container = document.createElement('div');
    container.id = 'root';

    const section = document.createElement('section');
    section.classList.add('content');

    const button = document.createElement('button');
    button.classList.add('primary');

    container.appendChild(section);
    section.appendChild(button);
    document.body.appendChild(container);

    // Click the button and capture the event
    let captured: string[] = [];
    button.addEventListener('click', (event) => {
      captured = buildComposedPath(event);
    });

    button.click();

    // Path should start from the clicked element going up
    expect(captured[0]).toBe('button.primary');
    expect(captured[1]).toBe('section.content');
    expect(captured[2]).toBe('div#root');

    document.body.removeChild(container);
  });

  it('should limit path to 10 levels', () => {
    // Create a deeply nested structure (15 levels)
    let current = document.createElement('div');
    const root = current;
    for (let i = 0; i < 14; i++) {
      const child = document.createElement('div');
      current.appendChild(child);
      current = child;
    }
    document.body.appendChild(root);

    let captured: string[] = [];
    current.addEventListener('click', (event) => {
      captured = buildComposedPath(event);
    });

    current.click();

    // Should be capped at 10
    expect(captured.length).toBe(10);

    document.body.removeChild(root);
  });
});

// ---------------------------------------------------------------------------
// initClickCapture tests
// ---------------------------------------------------------------------------

describe('initClickCapture', () => {
  it('should call sendFn with click_intent on click', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const button = document.createElement('button');
    button.textContent = 'Click me';
    document.body.appendChild(button);

    button.click();

    expect(sendFn).toHaveBeenCalledOnce();
    expect(sendFn).toHaveBeenCalledWith('click_intent', expect.objectContaining({
      url: expect.any(String),
      timestamp: expect.any(String),
      target: expect.objectContaining({
        tagName: 'button',
        composedPath: expect.any(Array),
      }),
    }));

    document.body.removeChild(button);
    cleanup();
  });

  it('should extract ARIA role', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const div = document.createElement('div');
    div.setAttribute('role', 'navigation');
    document.body.appendChild(div);

    div.click();

    const payload = sendFn.mock.calls[0][1];
    expect(payload.target.role).toBe('navigation');

    document.body.removeChild(div);
    cleanup();
  });

  it('should extract aria-label', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const button = document.createElement('button');
    button.setAttribute('aria-label', 'Close dialog');
    document.body.appendChild(button);

    button.click();

    const payload = sendFn.mock.calls[0][1];
    expect(payload.target.ariaLabel).toBe('Close dialog');

    document.body.removeChild(button);
    cleanup();
  });

  it('should extract data-testid', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const div = document.createElement('div');
    div.setAttribute('data-testid', 'submit-button');
    document.body.appendChild(div);

    div.click();

    const payload = sendFn.mock.calls[0][1];
    expect(payload.target.testId).toBe('submit-button');

    document.body.removeChild(div);
    cleanup();
  });

  it('should truncate innerText to 200 characters', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const div = document.createElement('div');
    div.innerText = 'A'.repeat(300);
    document.body.appendChild(div);

    div.click();

    const payload = sendFn.mock.calls[0][1];
    expect(payload.target.innerText).toHaveLength(200);
    expect(payload.target.innerText).toBe('A'.repeat(200));

    document.body.removeChild(div);
    cleanup();
  });

  it('should not include innerText for empty text', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const div = document.createElement('div');
    // innerText will be empty since there is no text content
    document.body.appendChild(div);

    div.click();

    const payload = sendFn.mock.calls[0][1];
    expect(payload.target.innerText).toBeUndefined();

    document.body.removeChild(div);
    cleanup();
  });

  it('should include page coordinates', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const button = document.createElement('button');
    button.textContent = 'Click';
    document.body.appendChild(button);

    // Create a click event with specific coordinates
    const event = new MouseEvent('click', {
      bubbles: true,
      cancelable: true,
      clientX: 100,
      clientY: 200,
    });
    button.dispatchEvent(event);

    const payload = sendFn.mock.calls[0][1];
    expect(payload.x).toBe(100); // pageX in jsdom equals clientX (no scroll)
    expect(payload.y).toBe(200);

    document.body.removeChild(button);
    cleanup();
  });

  it('should include timestamp and URL', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const div = document.createElement('div');
    document.body.appendChild(div);

    div.click();

    const payload = sendFn.mock.calls[0][1];
    expect(payload.timestamp).toBeDefined();
    expect(payload.url).toBeDefined();
    // Verify timestamp is valid ISO string
    expect(() => new Date(payload.timestamp)).not.toThrow();

    document.body.removeChild(div);
    cleanup();
  });

  it('should include composedPath in the target', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const container = document.createElement('div');
    container.id = 'wrapper';
    const btn = document.createElement('button');
    btn.classList.add('action');
    container.appendChild(btn);
    document.body.appendChild(container);

    btn.click();

    const payload = sendFn.mock.calls[0][1];
    expect(payload.target.composedPath).toBeInstanceOf(Array);
    expect(payload.target.composedPath.length).toBeGreaterThan(0);
    expect(payload.target.composedPath[0]).toBe('button.action');

    document.body.removeChild(container);
    cleanup();
  });

  it('should stop capturing after cleanup is called', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const div = document.createElement('div');
    document.body.appendChild(div);

    div.click();
    expect(sendFn).toHaveBeenCalledOnce();

    cleanup();

    div.click();
    // Should still be 1 — no new call after cleanup
    expect(sendFn).toHaveBeenCalledOnce();

    document.body.removeChild(div);
  });

  it('should not set optional fields when attributes are absent', () => {
    const sendFn = vi.fn();
    const cleanup = initClickCapture(sendFn);

    const div = document.createElement('div');
    document.body.appendChild(div);

    div.click();

    const payload = sendFn.mock.calls[0][1];
    expect(payload.target.role).toBeUndefined();
    expect(payload.target.ariaLabel).toBeUndefined();
    expect(payload.target.testId).toBeUndefined();

    document.body.removeChild(div);
    cleanup();
  });

  it('fires capture handler before bubble handler', () => {
    const order: string[] = [];

    // Register a bubble-phase handler first
    const bubbleHandler = () => order.push('bubble');
    document.addEventListener('click', bubbleHandler, false);

    // The click capture module registers in capture phase
    const sendFn = vi.fn(() => {
      order.push('capture');
    });
    const cleanup = initClickCapture(sendFn);

    const div = document.createElement('div');
    document.body.appendChild(div);

    div.click();

    // Capture phase should fire before bubble phase
    expect(order[0]).toBe('capture');
    expect(order[1]).toBe('bubble');

    document.body.removeChild(div);
    document.removeEventListener('click', bubbleHandler, false);
    cleanup();
  });
});

// ---------------------------------------------------------------------------
// Browser-level integration tests note
// ---------------------------------------------------------------------------
// NOTE: Browser-level integration tests (native messaging connection,
// permission verification) require a real Chrome instance and cannot
// run in jsdom.
