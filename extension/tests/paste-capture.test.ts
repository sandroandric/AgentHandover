/**
 * Tests for Paste Capture: each paste is reported with the element pasted
 * into, never with the pasted content.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { initPasteCapture } from '../src/paste-capture';

function firePaste(target: Element, text: string): void {
  const event = new Event('paste', { bubbles: true, cancelable: true });
  // jsdom has no ClipboardEvent/DataTransfer: attach the data the way a
  // browser exposes it, so the test proves it is not read.
  Object.defineProperty(event, 'clipboardData', {
    value: { getData: vi.fn(() => text), types: ['text/plain'] },
  });
  target.dispatchEvent(event);
}

describe('initPasteCapture', () => {
  let cleanup: (() => void) | null = null;

  afterEach(() => {
    cleanup?.();
    cleanup = null;
    document.body.innerHTML = '';
  });

  it('reports a paste with its target and without its content', () => {
    document.body.innerHTML =
      '<form><textarea name="comment" aria-label="Comment" class="c"></textarea></form>';
    const textarea = document.querySelector('textarea')!;
    const send = vi.fn();
    cleanup = initPasteCapture(send);

    firePaste(textarea, 'https://example.com/secret-link');

    expect(send).toHaveBeenCalledTimes(1);
    const [type, payload] = send.mock.calls[0];
    expect(type).toBe('paste');
    expect(payload.target).toEqual({
      tagName: 'textarea',
      role: undefined,
      ariaLabel: 'Comment',
      fieldName: 'comment',
      selector: 'textarea.c',
    });
    expect(payload.url).toBe(window.location.href);
    expect(typeof payload.timestamp).toBe('string');
    expect(JSON.stringify(payload)).not.toContain('secret-link');
  });

  it('sees a paste even when the page stops its propagation', () => {
    document.body.innerHTML = '<input placeholder="Search" />';
    const input = document.querySelector('input')!;
    input.addEventListener('paste', (e) => e.stopPropagation());
    const send = vi.fn();
    cleanup = initPasteCapture(send);

    firePaste(input, 'x');

    expect(send).toHaveBeenCalledTimes(1);
    expect(send.mock.calls[0][1].target.fieldName).toBe('Search');
  });

  it('stops reporting after cleanup', () => {
    document.body.innerHTML = '<input />';
    const send = vi.fn();
    initPasteCapture(send)();

    firePaste(document.querySelector('input')!, 'x');

    expect(send).not.toHaveBeenCalled();
  });
});
