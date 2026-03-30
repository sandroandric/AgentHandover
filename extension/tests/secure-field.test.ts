/**
 * Tests for Secure Field Detection module.
 *
 * Verifies:
 *   - Password field detection (<input type="password">)
 *   - Autocomplete attribute detection (current-password, new-password)
 *   - Hidden input with "password" in name
 *   - State transitions (focus in -> secure, focus out -> not secure)
 *   - onStateChange callback fires correctly
 *   - sendFn called with correct messages
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { initSecureFieldDetection, isSecureField } from '../src/secure-field';

// ---------------------------------------------------------------------------
// Mock chrome.runtime.sendMessage
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
// isSecureField tests
// ---------------------------------------------------------------------------

describe('isSecureField', () => {
  it('should detect <input type="password">', () => {
    const input = document.createElement('input');
    input.type = 'password';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect <input autocomplete="current-password">', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'current-password');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect <input autocomplete="new-password">', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'new-password');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect <input type="hidden" name="user_password">', () => {
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = 'user_password';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect hidden input with "password" anywhere in name', () => {
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = 'encrypted_password_field';
    expect(isSecureField(input)).toBe(true);
  });

  it('should not flag a regular text input', () => {
    const input = document.createElement('input');
    input.type = 'text';
    expect(isSecureField(input)).toBe(false);
  });

  it('should not flag a non-input element', () => {
    const div = document.createElement('div');
    expect(isSecureField(div)).toBe(false);
  });

  it('should not flag null', () => {
    expect(isSecureField(null)).toBe(false);
  });

  it('should not flag a regular hidden input without password in name', () => {
    const input = document.createElement('input');
    input.type = 'hidden';
    input.name = 'csrf_token';
    expect(isSecureField(input)).toBe(false);
  });

  it('should handle case-insensitive autocomplete', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'Current-Password');
    expect(isSecureField(input)).toBe(true);
  });

  // ----- Credit card autocomplete values -----

  it('should detect autocomplete="cc-number"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'cc-number');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect autocomplete="cc-exp"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'cc-exp');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect autocomplete="cc-exp-month"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'cc-exp-month');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect autocomplete="cc-exp-year"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'cc-exp-year');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect autocomplete="cc-csc"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'cc-csc');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect autocomplete="cc-name"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'cc-name');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect autocomplete="cc-type"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'cc-type');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect case-insensitive cc autocomplete (CC-NUMBER)', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'CC-NUMBER');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect autocomplete="cc-given-name"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'cc-given-name');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect autocomplete="cc-family-name"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'cc-family-name');
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect autocomplete="cc-additional-name"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'cc-additional-name');
    expect(isSecureField(input)).toBe(true);
  });

  // ----- Credit card name/id patterns -----

  it('should detect name="card-number"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'card-number';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect name="cardnumber"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'cardnumber';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect name="cc-num"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'cc-num';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect name="cvv"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'cvv';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect name="cvc"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'cvc';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect name="security-code"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'security-code';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect name="expiry"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'expiry';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect name="card-expiry"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'card-expiry';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect name="card-holder"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'card-holder';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect id="card-number" (via id, not name)', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.id = 'card-number';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect id containing cc pattern (e.g. "payment-cvv-field")', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.id = 'payment-cvv-field';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect name with cc pattern substring (e.g. "billing_cardnumber_input")', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'billing_cardnumber_input';
    expect(isSecureField(input)).toBe(true);
  });

  it('should detect case-insensitive name pattern (e.g. "CardNumber")', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'CardNumber';
    expect(isSecureField(input)).toBe(true);
  });

  // ----- Non-credit-card fields should NOT be flagged -----

  it('should NOT flag a regular text input with unrelated name', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.name = 'username';
    expect(isSecureField(input)).toBe(false);
  });

  it('should NOT flag a tel input', () => {
    const input = document.createElement('input');
    input.type = 'tel';
    input.name = 'phone';
    expect(isSecureField(input)).toBe(false);
  });

  it('should NOT flag an email input', () => {
    const input = document.createElement('input');
    input.type = 'email';
    input.name = 'email';
    expect(isSecureField(input)).toBe(false);
  });

  it('should NOT flag an input with autocomplete="name"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'name');
    expect(isSecureField(input)).toBe(false);
  });

  it('should NOT flag an input with autocomplete="email"', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'email');
    expect(isSecureField(input)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// initSecureFieldDetection tests
// ---------------------------------------------------------------------------

describe('initSecureFieldDetection', () => {
  it('should call onStateChange(true) when password field gains focus', () => {
    const sendFn = vi.fn();
    const onStateChange = vi.fn();
    const cleanup = initSecureFieldDetection(sendFn, onStateChange);

    const passwordInput = document.createElement('input');
    passwordInput.type = 'password';
    document.body.appendChild(passwordInput);

    // Dispatch focusin event
    const focusEvent = new FocusEvent('focusin', {
      bubbles: true,
      cancelable: true,
    });
    Object.defineProperty(focusEvent, 'target', { value: passwordInput });
    passwordInput.dispatchEvent(focusEvent);

    expect(onStateChange).toHaveBeenCalledWith(true);
    expect(sendFn).toHaveBeenCalledWith('secure_field_status', { isSecure: true });

    document.body.removeChild(passwordInput);
    cleanup();
  });

  it('should call onStateChange(false) when focus moves to non-secure field', () => {
    const sendFn = vi.fn();
    const onStateChange = vi.fn();
    const cleanup = initSecureFieldDetection(sendFn, onStateChange);

    const passwordInput = document.createElement('input');
    passwordInput.type = 'password';
    const textInput = document.createElement('input');
    textInput.type = 'text';
    document.body.appendChild(passwordInput);
    document.body.appendChild(textInput);

    // Focus password field first
    passwordInput.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    expect(onStateChange).toHaveBeenCalledWith(true);

    // Now focus a text field
    textInput.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    expect(onStateChange).toHaveBeenCalledWith(false);

    document.body.removeChild(passwordInput);
    document.body.removeChild(textInput);
    cleanup();
  });

  it('should not fire duplicate state changes', () => {
    const sendFn = vi.fn();
    const onStateChange = vi.fn();
    const cleanup = initSecureFieldDetection(sendFn, onStateChange);

    const textInput = document.createElement('input');
    textInput.type = 'text';
    document.body.appendChild(textInput);

    // Focus a non-secure field: should not trigger since already not secure
    textInput.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    expect(onStateChange).not.toHaveBeenCalled();

    document.body.removeChild(textInput);
    cleanup();
  });

  it('should send secure_field_status messages', () => {
    const sendFn = vi.fn();
    const onStateChange = vi.fn();
    const cleanup = initSecureFieldDetection(sendFn, onStateChange);

    const passwordInput = document.createElement('input');
    passwordInput.type = 'password';
    document.body.appendChild(passwordInput);

    passwordInput.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));

    expect(sendFn).toHaveBeenCalledWith('secure_field_status', { isSecure: true });

    document.body.removeChild(passwordInput);
    cleanup();
  });

  it('should clean up listeners after destroy', () => {
    const sendFn = vi.fn();
    const onStateChange = vi.fn();
    const cleanup = initSecureFieldDetection(sendFn, onStateChange);

    cleanup();

    const passwordInput = document.createElement('input');
    passwordInput.type = 'password';
    document.body.appendChild(passwordInput);

    passwordInput.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));

    // Should not have been called after cleanup
    expect(onStateChange).not.toHaveBeenCalled();

    document.body.removeChild(passwordInput);
  });

  it('should detect autocomplete="current-password" field on focus', () => {
    const sendFn = vi.fn();
    const onStateChange = vi.fn();
    const cleanup = initSecureFieldDetection(sendFn, onStateChange);

    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('autocomplete', 'current-password');
    document.body.appendChild(input);

    input.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));

    expect(onStateChange).toHaveBeenCalledWith(true);

    document.body.removeChild(input);
    cleanup();
  });

  it('should handle rapid focus transitions between secure and non-secure', () => {
    const sendFn = vi.fn();
    const onStateChange = vi.fn();
    const cleanup = initSecureFieldDetection(sendFn, onStateChange);

    const passwordInput = document.createElement('input');
    passwordInput.type = 'password';
    const textInput = document.createElement('input');
    textInput.type = 'text';
    const anotherPassword = document.createElement('input');
    anotherPassword.type = 'password';

    document.body.appendChild(passwordInput);
    document.body.appendChild(textInput);
    document.body.appendChild(anotherPassword);

    // password -> text -> password
    passwordInput.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    textInput.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));
    anotherPassword.dispatchEvent(new FocusEvent('focusin', { bubbles: true }));

    // State transitions: false->true, true->false, false->true
    expect(onStateChange).toHaveBeenCalledTimes(3);
    expect(onStateChange.mock.calls[0][0]).toBe(true);
    expect(onStateChange.mock.calls[1][0]).toBe(false);
    expect(onStateChange.mock.calls[2][0]).toBe(true);

    document.body.removeChild(passwordInput);
    document.body.removeChild(textInput);
    document.body.removeChild(anotherPassword);
    cleanup();
  });
});
