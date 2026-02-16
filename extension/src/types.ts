/**
 * Shared TypeScript type definitions for the OpenMimic IPC protocol.
 *
 * These types define the contract between the Chrome extension and the
 * local daemon process (com.openclaw.apprentice) communicating over
 * Chrome Native Messaging.
 *
 * Message direction:
 *   Extension -> Daemon:  ExtensionMessage (discriminated by ExtensionMessageType)
 *   Daemon -> Extension:  DaemonMessage   (discriminated by DaemonMessageType)
 */

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
  | 'secure_field_status';

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

/** Sent when a DOM snapshot is captured (initial, dwell, scroll, or on-demand). */
export interface DomSnapshotPayload {
  nodes: DomNode[];
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
