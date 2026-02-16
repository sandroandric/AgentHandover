//! Optional CDP (Chrome DevTools Protocol) bridge for Electron app inspection.
//!
//! Provides a controlled interface for connecting to Electron apps that expose
//! a CDP debugging port on localhost. Enforces:
//! - Opt-in configuration (disabled by default)
//! - Bundle-ID allowlisting
//! - Localhost-only connections (never remote)
//! - Data/instruction separation for DOM queries (safe eval scripts)

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Configuration for CDP bridge connections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdpConfig {
    /// Whether CDP bridge is enabled at all (default: false).
    pub enabled: bool,
    /// Allowlisted bundle IDs that can use CDP.
    pub allowlist: HashSet<String>,
    /// Connection timeout in milliseconds.
    pub connect_timeout_ms: u64,
    /// Maximum retries for connection attempts.
    pub max_retries: u32,
}

impl Default for CdpConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            allowlist: HashSet::new(),
            connect_timeout_ms: 5000,
            max_retries: 3,
        }
    }
}

/// Result of a CDP connection attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum CdpConnectionResult {
    /// Successfully connected to the CDP endpoint.
    Connected {
        port: u16,
        targets: Vec<CdpTarget>,
    },
    /// CDP bridge is disabled in configuration.
    Disabled,
    /// The bundle ID is not in the allowlist.
    NotAllowlisted { bundle_id: String },
    /// Connection to the CDP endpoint failed.
    ConnectionFailed { reason: String },
    /// The host is not localhost (remote connections are forbidden).
    NotLocalhost,
}

/// A CDP target (tab/page in the Electron app).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CdpTarget {
    /// Unique target identifier.
    pub id: String,
    /// Page title.
    pub title: String,
    /// Page URL.
    pub url: String,
    /// Target type (e.g. "page", "background_page", "service_worker").
    pub target_type: String,
}

/// CDP bridge for Electron app inspection.
///
/// Connects to local DevTools ports only, never remote. Requires explicit
/// opt-in via `CdpConfig::enabled` and bundle-ID allowlisting.
pub struct CdpBridge {
    config: CdpConfig,
}

impl CdpBridge {
    /// Create a new CDP bridge with the given configuration.
    pub fn new(config: CdpConfig) -> Self {
        Self { config }
    }

    /// Check if a connection to the given bundle is allowed.
    ///
    /// Returns `true` only when CDP is enabled AND the bundle ID is in the
    /// allowlist.
    pub fn is_allowed(&self, bundle_id: &str) -> bool {
        self.config.enabled && self.config.allowlist.contains(bundle_id)
    }

    /// Validate that a connection target is localhost only.
    ///
    /// Accepts "127.0.0.1", "localhost", "::1", and "[::1]".
    /// Rejects everything else (LAN addresses, hostnames, etc.).
    pub fn validate_localhost(host: &str) -> bool {
        matches!(host, "127.0.0.1" | "localhost" | "::1" | "[::1]")
    }

    /// Attempt to connect to a CDP endpoint.
    ///
    /// Returns `CdpConnectionResult` indicating the outcome:
    /// - `Disabled` if CDP is not enabled
    /// - `NotAllowlisted` if the bundle is not in the allowlist
    /// - `NotLocalhost` if the host is not a localhost address
    /// - `Connected` or `ConnectionFailed` for actual connection attempts
    pub fn connect(&self, bundle_id: &str, host: &str, port: u16) -> CdpConnectionResult {
        if !self.config.enabled {
            return CdpConnectionResult::Disabled;
        }

        if !self.config.allowlist.contains(bundle_id) {
            return CdpConnectionResult::NotAllowlisted {
                bundle_id: bundle_id.to_string(),
            };
        }

        if !Self::validate_localhost(host) {
            return CdpConnectionResult::NotLocalhost;
        }

        // Attempt TCP connection to localhost:port
        match self.try_connect(port) {
            Ok(targets) => CdpConnectionResult::Connected { port, targets },
            Err(e) => CdpConnectionResult::ConnectionFailed { reason: e },
        }
    }

    /// Try a TCP connection to the given localhost port.
    ///
    /// In production, this would send `GET /json` to enumerate CDP targets.
    /// Currently returns an empty target list on successful TCP connect.
    fn try_connect(&self, port: u16) -> Result<Vec<CdpTarget>, String> {
        use std::net::TcpStream;
        use std::time::Duration;

        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
        let timeout = Duration::from_millis(self.config.connect_timeout_ms);

        match TcpStream::connect_timeout(&addr, timeout) {
            Ok(_stream) => {
                // In production: send HTTP GET /json to enumerate targets
                // For now, return empty targets (port was reachable)
                Ok(vec![])
            }
            Err(e) => Err(format!("Failed to connect to localhost:{}: {}", port, e)),
        }
    }
}

/// Build a safe CDP eval script for DOM/AX element extraction.
///
/// Constructs a JavaScript IIFE that queries a single DOM element by CSS
/// selector and returns its semantic properties (tagName, role, aria-label,
/// textContent, bounding rect). The selector is JSON-encoded to prevent
/// injection, enforcing data/instruction separation per spec section 7.2.
///
/// This function does NOT allow arbitrary JS execution -- only safe DOM queries.
///
/// Returns `Err` if the selector cannot be serialized to JSON.
pub fn build_cdp_eval_script(selector: &str) -> Result<String, String> {
    let encoded_selector = serde_json::to_string(selector)
        .map_err(|e| format!("Failed to JSON-encode selector {:?}: {}", selector, e))?;
    Ok(format!(
        r#"(function() {{
            const el = document.querySelector({selector});
            if (!el) return null;
            return {{
                tagName: el.tagName,
                role: el.getAttribute('role'),
                ariaLabel: el.getAttribute('aria-label'),
                textContent: (el.textContent || '').substring(0, 500),
                rect: el.getBoundingClientRect().toJSON()
            }};
        }})()"#,
        selector = encoded_selector
    ))
}
