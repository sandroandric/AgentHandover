/**
 * Paste Capture
 *
 * Reports each `paste` event (Cmd-V, Edit > Paste, the context menu) with
 * the element pasted into.  The pasted content itself is never read: the
 * daemon hashes the pasteboard when the message arrives and links the
 * paste to the copy it came from (`capture::paste` in the daemon).
 */

import { buildSelector } from './click-capture';

export interface PasteTarget {
  tagName: string;
  role?: string;
  ariaLabel?: string;
  /** The `name` or `placeholder` of a form field, when it has one. */
  fieldName?: string;
  selector: string;
}

export interface PasteIntent {
  target: PasteTarget;
  timestamp: string;
  url: string;
}

function describeTarget(el: Element): PasteTarget {
  return {
    tagName: el.tagName.toLowerCase(),
    role: el.getAttribute('role') || undefined,
    ariaLabel: el.getAttribute('aria-label') || undefined,
    fieldName: el.getAttribute('name') || el.getAttribute('placeholder') || undefined,
    selector: buildSelector(el),
  };
}

/**
 * Listen for pastes on the current document (capture phase, so a page
 * handler that stops propagation does not hide the paste).
 *
 * @param sendFn  Called with ('paste', payload) for each paste.  The content
 *                script passes its secure-field guard, so a paste into a
 *                password field is never reported.
 * @returns       A cleanup function that removes the listener.
 */
export function initPasteCapture(
  sendFn: (type: string, payload: Record<string, unknown>) => void,
): () => void {
  function handlePaste(event: Event): void {
    const target = event.composedPath().find(
      (node) => (node as Node).nodeType === Node.ELEMENT_NODE,
    ) as Element | undefined;
    if (!target) return;

    const paste: PasteIntent = {
      target: describeTarget(target),
      timestamp: new Date().toISOString(),
      url: window.location.href,
    };
    sendFn('paste', paste as unknown as Record<string, unknown>);
  }

  document.addEventListener('paste', handlePaste, true);
  return () => document.removeEventListener('paste', handlePaste, true);
}
