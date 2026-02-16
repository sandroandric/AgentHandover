/**
 * OpenMimic Observer — Secure Field Detection
 *
 * Detects when a user focuses a password or other sensitive input field,
 * and signals other capture modules to suppress data collection while
 * the secure field is active.
 *
 * Detection covers:
 *   - <input type="password">
 *   - <input autocomplete="current-password">
 *   - <input autocomplete="new-password">
 *   - <input type="hidden" name="...password...">
 *   - Credit card fields (autocomplete: cc-number, cc-exp, cc-exp-month,
 *     cc-exp-year, cc-csc, cc-name, cc-type)
 *   - Credit card fields (name/id patterns: card-number, cardnumber,
 *     cc-num, cvv, cvc, security-code, expiry, card-expiry, card-holder)
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

/** Autocomplete attribute values indicating credit card fields. */
const CC_AUTOCOMPLETE_VALUES: ReadonlySet<string> = new Set([
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

/** Name/ID substrings indicating credit card fields. */
const CC_NAME_PATTERNS: readonly string[] = [
  'card-number',
  'cardnumber',
  'cc-num',
  'cvv',
  'cvc',
  'security-code',
  'expiry',
  'card-expiry',
  'card-holder',
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Determine whether a given element is a secure (password or credit card) field.
 */
export function isSecureField(el: Element | null): boolean {
  if (!el) return false;
  if (!(el instanceof HTMLInputElement)) return false;

  // Explicit password type
  if (el.type === 'password') {
    return true;
  }

  // Autocomplete attributes indicating password or credit card fields
  const autocomplete = el.getAttribute('autocomplete');
  if (autocomplete) {
    const normalised = autocomplete.toLowerCase().trim();
    if (normalised === 'current-password' || normalised === 'new-password') {
      return true;
    }
    if (CC_AUTOCOMPLETE_VALUES.has(normalised)) {
      return true;
    }
  }

  // Hidden inputs with "password" in their name (e.g. password managers)
  if (el.type === 'hidden') {
    const name = (el.name || '').toLowerCase();
    if (name.includes('password')) {
      return true;
    }
  }

  // Credit card fields detected by name or id patterns
  const elName = (el.name || '').toLowerCase();
  const elId = (el.id || '').toLowerCase();
  for (const pattern of CC_NAME_PATTERNS) {
    if (elName.includes(pattern) || elId.includes(pattern)) {
      return true;
    }
  }

  return false;
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
    const target = event.target as Element | null;
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
      const active = document.activeElement;
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
