/**
 * AgentHandover Observer — Form Submit Capture
 *
 * Listens for `submit` (capture phase, so a page handler that stops
 * propagation cannot hide it) and reports which form was submitted and by
 * which control. The anchor identifies the form the way an agent would
 * find it again (role, accessible label, test id, action path, submit
 * button text); it never contains the values typed into the form.
 */

/** The submit control that sent the form. */
export interface SubmitterAnchor {
  tagName: string;
  role?: string;
  ariaLabel?: string;
  testId?: string;
  /** Visible label of the control (button text or submit input's label), ≤ 80 chars. */
  text?: string;
}

/** Identifies a submitted form without any of its field values. */
export interface FormAnchor {
  tag: 'form';
  id?: string;
  name?: string;
  role?: string;
  ariaLabel?: string;
  testId?: string;
  /** Origin + path of the form's action (query and fragment dropped). */
  action?: string;
  method: string;
  submitter?: SubmitterAnchor;
}

const MAX_TEXT = 80;

/** Origin + path of `raw` resolved against the page URL, or undefined. */
function originAndPath(raw: string): string | undefined {
  try {
    const url = new URL(raw, window.location.href);
    return `${url.origin}${url.pathname}`;
  } catch {
    return undefined;
  }
}

/** An attribute's value, or undefined when absent or empty. */
function attr(el: Element, name: string): string | undefined {
  return el.getAttribute(name) || undefined;
}

function submitterAnchor(el: HTMLElement): SubmitterAnchor {
  // A submit <input>'s value is its visible label, not user data.
  const label =
    el instanceof HTMLInputElement ? el.value : (el.innerText || el.textContent || '');
  const text = label.trim().slice(0, MAX_TEXT) || undefined;
  return {
    tagName: el.tagName.toLowerCase(),
    role: attr(el, 'role'),
    ariaLabel: attr(el, 'aria-label'),
    testId: attr(el, 'data-testid'),
    text,
  };
}

/**
 * Build the anchor for `form`, submitted by `submitter` (null when the form
 * was submitted by pressing Enter or from script).
 *
 * Attributes are read with getAttribute: `form.id`, `form.name` and
 * `form.action` are shadowed by fields named "id", "name" or "action".
 */
export function buildFormAnchor(form: HTMLFormElement, submitter: HTMLElement | null): FormAnchor {
  const action = submitter?.getAttribute('formaction') ?? form.getAttribute('action') ?? '';
  const method = (submitter?.getAttribute('formmethod') ?? form.getAttribute('method') ?? 'get')
    .toLowerCase();
  return {
    tag: 'form',
    id: attr(form, 'id'),
    name: attr(form, 'name'),
    role: attr(form, 'role'),
    ariaLabel: attr(form, 'aria-label'),
    testId: attr(form, 'data-testid'),
    action: originAndPath(action),
    method,
    submitter: submitter ? submitterAnchor(submitter) : undefined,
  };
}

/**
 * Start reporting form submissions on the current document.
 *
 * @param sendFn  Called with ('form_submit', payload) on each submission.
 * @returns       A cleanup function that removes the listener.
 */
export function initFormSubmitCapture(
  sendFn: (type: string, payload: Record<string, unknown>) => void,
): () => void {
  function handleSubmit(event: Event): void {
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return;
    const submitter = (event as SubmitEvent).submitter ?? null;
    sendFn('form_submit', {
      url: window.location.href,
      title: document.title,
      formAnchor: buildFormAnchor(form, submitter),
      timestamp: new Date().toISOString(),
    });
  }

  document.addEventListener('submit', handleSubmit, true);
  return () => document.removeEventListener('submit', handleSubmit, true);
}
