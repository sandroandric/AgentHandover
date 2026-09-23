/**
 * Shared TypeScript type definitions for the AgentHandover IPC protocol.
 *
 * These types define the contract between the Chrome extension and the
 * local daemon process (com.agenthandover.host) communicating over
 * Chrome Native Messaging.
 *
 * Message direction:
 *   Extension -> Daemon:  ExtensionMessage (discriminated by ExtensionMessageType)
 *   Daemon -> Extension:  DaemonMessage   (discriminated by DaemonMessageType)
 */

import type { FormAnchor } from './form-submit';

// ---------------------------------------------------------------------------
// Extension -> Daemon message types
// ---------------------------------------------------------------------------

/** Discriminator values for messages sent from the extension to the daemon. */
export type ExtensionMessageType =
  | 'content_ready'
  | 'dom_snapshot'
  | 'click_intent'
  | 'dwell_snapshot'
  | 'scroll_snapshot'
  | 'secure_field_status'
  | 'paste'
  | 'navigation_change'
  | 'form_submit';

// ---------------------------------------------------------------------------
// Daemon -> Extension message types
// ---------------------------------------------------------------------------

/** Discriminator values for messages sent from the daemon to the extension. */
export type DaemonMessageType =
  | 'ping'
  | 'request_snapshot'
  | 'config_update'
  | 'ack';

// ---------------------------------------------------------------------------
// Extension -> Daemon payloads
// ---------------------------------------------------------------------------

/** Sent when a content script is injected and ready in a tab. */
export interface ContentReadyPayload {
  url: string;
  title: string;
  tabId: number;
}

/** Sent when the user clicks an element in the page. */
export interface ClickIntentPayload {
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
  tabId: number;
  url: string;
}

/** A single node in a captured DOM snapshot. */
export interface DomNode {
  tag: string;
  role?: string;
  name?: string;
  ariaLabel?: string;
  testId?: string;
  innerText?: string;
  bbox?: { x: number; y: number; width: number; height: number };
  state?: { enabled?: boolean; checked?: boolean; selected?: boolean };
  children?: DomNode[];
  /** True when this node represents an element rendered inside a Shadow DOM. */
  isShadowRoot?: boolean;
}

/** One element of the anchor set. */
export interface DomAnchor {
  tag: string;
  /** Input type (`email`, `checkbox`, ...) for `<input>`. */
  type?: string;
  id?: string;
  role?: string;
  /** Accessible name: aria-label, aria-labelledby, `<label>`, placeholder, title or text. */
  name?: string;
  ariaLabel?: string;
  testId?: string;
  href?: string;
  /** Visible text, one line. */
  text?: string;
  /** Typed value of the interacted field, only with `captureInputValues`
   *  (never a password). */
  value?: string;
  /** Composed-path CSS selector from the chain's outermost element. */
  selector: string;
  /** True on the element the user interacted with. */
  interacted?: boolean;
}

/** Sent when a DOM snapshot is captured (initial, dwell, scroll, or on-demand).
 *  Carries the anchor set and text digest of `captureSnapshot`
 *  (dom-capture.ts), not the viewport tree. */
export interface DomSnapshotPayload {
  anchors: DomAnchor[];
  digest: string;
  truncated: boolean;
  url: string;
  tabId: number;
  captureReason: 'dwell' | 'scroll' | 'request' | 'initial';
}

/** Sent when a password/secure input field gains or loses focus. */
export interface SecureFieldPayload {
  isSecure: boolean;
  tabId: number;
  url: string;
}

/**
 * Sent by the background when the active tab commits a navigation
 * (`tabs.onUpdated`, no DOM access).
 */
export interface NavigationChangePayload {
  url: string;
  title: string;
  tabId: number;
}

/** Sent when a form is submitted (content-script `submit` listener). */
export interface FormSubmitPayload {
  url: string;
  title: string;
  tabId: number;
  formAnchor: FormAnchor;
}

// ---------------------------------------------------------------------------
// Daemon -> Extension payloads
// ---------------------------------------------------------------------------

/** Payload for a snapshot request from the daemon. */
export interface RequestSnapshotPayload {
  tabId: number;
}

/** Payload for a configuration update pushed from the daemon. */
export interface ConfigUpdatePayload {
  /** Key-value pairs of configuration changes. */
  changes: Record<string, unknown>;
}

/** Payload for an acknowledgement message. */
export interface AckPayload {
  /** The sequence number of the message being acknowledged. */
  ackSeq: number;
}
