/**
 * AgentHandover Observer — Secure Field Detection
 *
 * Detects when a user focuses a password or other sensitive input field,
 * and signals other capture modules to suppress data collection while
 * the secure field is active.
 *
 * Detection covers:
 *   - <input type="password">, and an input that was one before a
 *     "show password" toggle turned it into type="text"
 *   - inputs and textareas whose name or id mentions a password
 *     (reveal toggles, hidden password-manager inputs)
 *   - autocomplete tokens current-password, new-password, one-time-code
 *     and the cc-* credit card tokens
 *   - name/id patterns for credit cards (card-number, cvv, cvc, expiry,
 *     ...) and one-time codes (otp, mfa, 2fa, totp, verification-code,
 *     backup-code, recovery-code)
 *   - fields inside open shadow roots: focus is resolved through
 *     `composedPath()` / `shadowRoot.activeElement`, not the shadow host
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SecureFieldState {
  isSecure: boolean;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Autocomplete tokens that mark a password, one-time code or card field. */
const SECURE_AUTOCOMPLETE_VALUES: ReadonlySet<string> = new Set([
  'current-password',
  'new-password',
  'one-time-code',
  'cc-number',
  'cc-exp',
  'cc-exp-month',
  'cc-exp-year',
  'cc-csc',
  'cc-name',
  'cc-type',
  'cc-given-name',
  'cc-family-name',
  'cc-additional-name',
]);

/** Name/ID substrings indicating password, card or one-time-code fields. */
const SECURE_NAME_PATTERNS: readonly string[] = [
  'password',
  'passwd',
  'card-number',
  'cardnumber',
  'cc-num',
  'cvv',
  'cvc',
  'security-code',
  'expiry',
  'card-expiry',
  'card-holder',
  'otp',
  'one-time',
  'mfa',
  '2fa',
  'totp',
  'verification-code',
  'backup-code',
  'recovery-code',
];

/** Set on an input once it has been seen as type="password", so a
 *  "show password" toggle to type="text" does not make it capturable. */
export const WAS_SECURE_ATTR = 'ahWasSecure';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Determine whether a given element is a secure field (password, one-time
 * code or credit card).  Inputs and textareas only.
 */
export function isSecureField(el: Element | null): boolean {
  if (!el) return false;
  const isInput = el instanceof HTMLInputElement;
  if (!isInput && !(el instanceof HTMLTextAreaElement)) return false;

  if (isInput && el.type === 'password') {
    el.dataset[WAS_SECURE_ATTR] = '1';
    return true;
  }
  if (el.dataset[WAS_SECURE_ATTR] === '1') {
    return true;
  }

  // Autocomplete may be a token list ("section-pay billing cc-number").
  const autocomplete = (el.getAttribute('autocomplete') || '').toLowerCase();
  for (const token of autocomplete.split(/\s+/)) {
    if (SECURE_AUTOCOMPLETE_VALUES.has(token)) {
      return true;
    }
  }

  const elName = (el.getAttribute('name') || '').toLowerCase();
  const elId = (el.id || '').toLowerCase();
  for (const pattern of SECURE_NAME_PATTERNS) {
    if (elName.includes(pattern) || elId.includes(pattern)) {
      return true;
    }
  }

  return false;
}

/**
 * The element that really has focus: `document.activeElement` is the
 * shadow host when focus is inside an open shadow root, so descend.
 */
export function deepActiveElement(root: Document | ShadowRoot = document): Element | null {
  let active = root.activeElement;
  while (active && active.shadowRoot && active.shadowRoot.activeElement) {
    active = active.shadowRoot.activeElement;
  }
  return active;
}

/** The event's real target, inside open shadow roots (not the retargeted host). */
function eventTarget(event: Event): Element | null {
  const first = event.composedPath()[0];
  return first instanceof Element ? first : (event.target as Element | null);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Initialise secure field detection on the current document.
 *
 * @param sendFn        Callback to emit secure field status changes.
 *                      Called with ('secure_field_status', { isSecure }).
 * @param onStateChange Called synchronously whenever the secure state changes.
 *                      Other capture modules should check this to suppress.
 * @returns             A cleanup function that removes all listeners.
 */
export function initSecureFieldDetection(
  sendFn: (type: string, payload: Record<string, unknown>) => void,
  onStateChange: (isSecure: boolean) => void,
): () => void {
  let currentlySecure = false;
  /** Monotonically increasing counter to track focus transitions and avoid stale callbacks. */
  let focusSeq = 0;

  function setSecure(value: boolean): void {
    if (value === currentlySecure) return;
    currentlySecure = value;
    onStateChange(value);
    sendFn('secure_field_status', { isSecure: value });
  }

  function handleFocusIn(event: FocusEvent): void {
    focusSeq++;
    const target = eventTarget(event);
    if (isSecureField(target)) {
      setSecure(true);
    } else if (currentlySecure) {
      // Focus moved to a non-secure field: release the lock
      setSecure(false);
    }
  }

  function handleFocusOut(event: FocusEvent): void {
    // When focus leaves a secure field and nothing else is focused yet
    // (relatedTarget is null, e.g. clicking outside), clear secure state.
    if (!currentlySecure) return;

    const relatedTarget = event.relatedTarget as Element | null;
    // If the new target is also a secure field, stay secure
    if (isSecureField(relatedTarget)) return;

    // Use requestAnimationFrame to check the actual new activeElement,
    // since focusout fires before focusin on the new element.
    // requestAnimationFrame fires after the browser has processed the
    // focus change, avoiding the timing race that setTimeout(0) creates.
    // We track the focus sequence so that if focus has changed again by
    // the time the callback fires, we skip the stale check.
    const seq = ++focusSeq;
    requestAnimationFrame(() => {
      if (seq !== focusSeq) return; // Focus changed again, skip
      const active = deepActiveElement();
      if (!isSecureField(active)) {
        setSecure(false);
      }
    });
  }

  // Listen on capture phase to detect focus before bubbling
  document.addEventListener('focusin', handleFocusIn, true);
  document.addEventListener('focusout', handleFocusOut, true);

  // Return cleanup function
  return () => {
    document.removeEventListener('focusin', handleFocusIn, true);
    document.removeEventListener('focusout', handleFocusOut, true);
  };
}
