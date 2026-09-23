/**
 * Tests for form submit capture: which form and control were used, never
 * what was typed.
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { buildFormAnchor, initFormSubmitCapture } from '../src/form-submit';

let cleanup: (() => void) | null = null;

afterEach(() => {
  cleanup?.();
  cleanup = null;
  document.body.innerHTML = '';
});

function mountLoginForm(): { form: HTMLFormElement; button: HTMLButtonElement } {
  document.body.innerHTML = `
    <form id="login" name="signin" aria-label="Sign in" data-testid="login-form"
          action="/session?next=/inbox#top" method="POST">
      <input name="email" value="jane@example.com">
      <input type="password" name="password" value="hunter2">
      <input name="id" value="shadowing-field">
      <button type="submit" data-testid="submit-btn">  Sign in  </button>
    </form>`;
  const form = document.querySelector('form') as HTMLFormElement;
  // Keep jsdom from attempting the (unimplemented) navigation.
  form.addEventListener('submit', (e) => e.preventDefault());
  return { form, button: form.querySelector('button') as HTMLButtonElement };
}

describe('buildFormAnchor', () => {
  it('identifies the form and submitter by attributes, with the action reduced to origin + path', () => {
    const { form, button } = mountLoginForm();
    const anchor = buildFormAnchor(form, button);
    expect(anchor).toEqual({
      tag: 'form',
      id: 'login',
      name: 'signin',
      role: undefined,
      ariaLabel: 'Sign in',
      testId: 'login-form',
      action: `${window.location.origin}/session`,
      method: 'post',
      submitter: {
        tagName: 'button',
        role: undefined,
        ariaLabel: undefined,
        testId: 'submit-btn',
        text: 'Sign in',
      },
    });
  });

  it('never contains field values', () => {
    const { form, button } = mountLoginForm();
    const json = JSON.stringify(buildFormAnchor(form, button));
    expect(json).not.toContain('jane@example.com');
    expect(json).not.toContain('hunter2');
    expect(json).not.toContain('shadowing-field');
  });

  it('uses the submitter formaction/formmethod and a submit input label', () => {
    document.body.innerHTML = `
      <form action="/a"><input type="submit" value="Save draft" formaction="/drafts?x=1" formmethod="PUT"></form>`;
    const form = document.querySelector('form') as HTMLFormElement;
    const input = form.querySelector('input') as HTMLInputElement;
    const anchor = buildFormAnchor(form, input);
    expect(anchor.action).toBe(`${window.location.origin}/drafts`);
    expect(anchor.method).toBe('put');
    expect(anchor.submitter).toMatchObject({ tagName: 'input', text: 'Save draft' });
  });

  it('works without a submitter (Enter key or script)', () => {
    document.body.innerHTML = '<form><input name="q"></form>';
    const form = document.querySelector('form') as HTMLFormElement;
    const anchor = buildFormAnchor(form, null);
    expect(anchor.submitter).toBeUndefined();
    expect(anchor.method).toBe('get');
    expect(anchor.action).toBe(`${window.location.origin}${window.location.pathname}`);
  });
});

describe('initFormSubmitCapture', () => {
  it('reports a submission with its submitter, even when the page stops propagation', () => {
    const { form, button } = mountLoginForm();
    form.addEventListener('submit', (e) => e.stopImmediatePropagation());
    const send = vi.fn();
    cleanup = initFormSubmitCapture(send);

    form.requestSubmit(button);

    expect(send).toHaveBeenCalledTimes(1);
    const [type, payload] = send.mock.calls[0];
    expect(type).toBe('form_submit');
    expect(payload.url).toBe(window.location.href);
    expect(payload.formAnchor).toMatchObject({ id: 'login', submitter: { text: 'Sign in' } });
  });

  it('stops reporting after cleanup', () => {
    const { form, button } = mountLoginForm();
    const send = vi.fn();
    const stop = initFormSubmitCapture(send);
    stop();
    form.requestSubmit(button);
    expect(send).not.toHaveBeenCalled();
  });
});
