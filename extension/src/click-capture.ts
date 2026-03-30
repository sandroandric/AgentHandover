/**
 * AgentHandover Observer — Click Intent Capture
 *
 * Captures click events with semantic information about the target element.
 * Extracts composedPath (CSS selector path through the composed tree), ARIA role,
 * accessible name, data-testid, visible innerText, click coordinates, tab ID, and URL.
 *
 * Uses capture-phase listener for reliable interception before any handler
 * can stopPropagation.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ClickIntent {
  x: number;
  y: number;
  target: {
    tagName: string;
    role?: string;
    ariaLabel?: string;
    testId?: string;
    innerText?: string;
    composedPath: string[];
  };
  timestamp: string;
  url: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a CSS selector string for a single element: tag#id.class1.class2
 * Returns lowercase tag name with optional id and class selectors.
 */
export function buildSelector(el: Element): string {
  const tag = el.tagName.toLowerCase();
  let selector = tag;

  if (el.id) {
    selector += `#${el.id}`;
  }

  if (el.classList && el.classList.length > 0) {
    for (let i = 0; i < el.classList.length; i++) {
      selector += `.${el.classList[i]}`;
    }
  }

  return selector;
}

/**
 * Build an array of CSS selectors from the event's composedPath.
 * Walks the composed path (which pierces Shadow DOM boundaries),
 * collecting selectors for Element nodes. Stops at 10 levels or
 * at the document/window boundary.
 */
export function buildComposedPath(event: Event): string[] {
  const path = event.composedPath();
  const selectors: string[] = [];

  for (const node of path) {
    // Stop at document or window nodes
    if (node === document || node === window) {
      break;
    }

    // Only process Element nodes (nodeType === 1)
    if ((node as Node).nodeType === Node.ELEMENT_NODE) {
      selectors.push(buildSelector(node as Element));
    }

    // Limit to 10 levels to keep the path manageable
    if (selectors.length >= 10) {
      break;
    }
  }

  return selectors;
}

/**
 * Extract semantic information from the clicked element.
 */
function extractTargetInfo(el: Element, event: Event): ClickIntent['target'] {
  const tagName = el.tagName.toLowerCase();

  const role = el.getAttribute('role') || undefined;
  const ariaLabel = el.getAttribute('aria-label') || undefined;
  const testId = el.getAttribute('data-testid') || undefined;

  // Get visible innerText, truncated to 200 chars.
  // Only available on HTMLElement (not SVGElement etc.), so check.
  let innerText: string | undefined;
  if ('innerText' in el) {
    const raw = (el as HTMLElement).innerText;
    if (raw && raw.trim().length > 0) {
      innerText = raw.trim().slice(0, 200);
    }
  }

  const composedPath = buildComposedPath(event);

  return {
    tagName,
    role,
    ariaLabel,
    testId,
    innerText,
    composedPath,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Initialise click intent capture on the current document.
 *
 * @param sendFn  Callback to emit captured data. Called with
 *                ('click_intent', payload) on each click.
 * @returns       A cleanup function that removes the listener.
 */
export function initClickCapture(
  sendFn: (type: string, payload: Record<string, unknown>) => void,
): () => void {
  function handleClick(event: MouseEvent): void {
    // Find the actual element target. Use composedPath for Shadow DOM support.
    const path = event.composedPath();
    let targetEl: Element | null = null;

    for (const node of path) {
      if ((node as Node).nodeType === Node.ELEMENT_NODE) {
        targetEl = node as Element;
        break;
      }
    }

    if (!targetEl) {
      return;
    }

    const clickIntent: ClickIntent = {
      x: event.pageX,
      y: event.pageY,
      target: extractTargetInfo(targetEl, event),
      timestamp: new Date().toISOString(),
      url: window.location.href,
    };

    sendFn('click_intent', clickIntent as unknown as Record<string, unknown>);
  }

  // Use capture phase (third argument = true) to intercept clicks before
  // any handler can call stopPropagation.
  //
  // LIMITATION: Other capture-phase listeners registered earlier on the same
  // target can call stopImmediatePropagation() and prevent this handler from
  // firing. To mitigate this, we also register on `window` as a fallback —
  // this gives two chances to catch the click event. The dedup flag prevents
  // the sendFn from being called twice for the same click.
  let lastProcessedTimestamp = -1;

  function deduplicatedHandleClick(event: MouseEvent): void {
    // Skip if we already processed this exact event
    if (event.timeStamp === lastProcessedTimestamp) return;
    lastProcessedTimestamp = event.timeStamp;
    handleClick(event);
  }

  document.addEventListener('click', deduplicatedHandleClick, true);
  window.addEventListener('click', deduplicatedHandleClick, true);

  // Return cleanup function
  return () => {
    document.removeEventListener('click', deduplicatedHandleClick, true);
    window.removeEventListener('click', deduplicatedHandleClick, true);
  };
}
