//! Electron/CEF app detection for macOS.
//!
//! Detects Electron and Chromium Embedded Framework (CEF) apps by inspecting
//! `.app` bundle structures on macOS. Looks for framework directories inside
//! `Contents/Frameworks/` that indicate the app is built on Electron or CEF
//! rather than being a native Cocoa/SwiftUI application.
//!
//! Also provides scanning of `/Applications` and arbitrary directories for
//! `.app` bundles, reading bundle identifiers from `Info.plist`, and checking
//! common CDP debugging ports.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Known Electron apps for quick name-based detection.
const KNOWN_ELECTRON_APPS: &[&str] = &[
    "Slack",
    "Visual Studio Code",
    "Code",
    "Discord",
    "Notion",
    "Figma",
    "Spotify",
    "Microsoft Teams",
    "Obsidian",
    "Postman",
    "Atom",
    "Signal",
    "WhatsApp",
    "Skype",
    "Bitwarden",
    "1Password",
    "GitHub Desktop",
    "Hyper",
    "Insomnia",
];

/// The detected runtime environment of a macOS application.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AppRuntime {
    /// Native macOS application (Cocoa, SwiftUI, etc.)
    Native,
    /// Electron-based application with optional version string.
    Electron { version: Option<String> },
    /// Chromium Embedded Framework application.
    CEF,
    /// Could not determine the runtime.
    Unknown,
}

/// Information about a detected macOS application bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppInfo {
    /// The CFBundleIdentifier from the app's Info.plist.
    pub bundle_id: String,
    /// The display name of the application (derived from the .app filename).
    pub name: String,
    /// Absolute path to the .app bundle.
    pub path: PathBuf,
    /// Detected runtime environment.
    pub runtime: AppRuntime,
    /// Chrome DevTools Protocol port if detected.
    pub cdp_port: Option<u16>,
}

/// Check if an app bundle at the given path is an Electron or CEF app.
///
/// Inspects `Contents/Frameworks/` inside the bundle for:
/// - `Electron Framework.framework` → `AppRuntime::Electron`
/// - `Chromium Embedded Framework.framework` → `AppRuntime::CEF`
/// - Any framework containing "Electron" in its name → `AppRuntime::Electron`
///
/// Returns `AppRuntime::Native` if no web-runtime frameworks are found.
pub fn detect_runtime(app_path: &Path) -> AppRuntime {
    let frameworks_dir = app_path.join("Contents/Frameworks");

    if !frameworks_dir.exists() {
        return AppRuntime::Native;
    }

    // Check for Electron Framework.framework (standard location)
    let electron_framework = frameworks_dir.join("Electron Framework.framework");
    if electron_framework.exists() {
        let version = read_electron_version(&frameworks_dir);
        return AppRuntime::Electron { version };
    }

    // Check for CEF (Chromium Embedded Framework)
    let cef_framework = frameworks_dir.join("Chromium Embedded Framework.framework");
    if cef_framework.exists() {
        return AppRuntime::CEF;
    }

    // Only scan the full Frameworks directory if the standard checks above
    // didn't find anything — some apps bundle Electron under a different name.
    if let Ok(entries) = std::fs::read_dir(&frameworks_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.contains("Electron") && name.ends_with(".framework") {
                let version = read_electron_version(&frameworks_dir);
                return AppRuntime::Electron { version };
            }
        }
    }

    AppRuntime::Native
}

/// Try to read the Electron version from the framework bundle's Info.plist.
///
/// Looks for `CFBundleShortVersionString` inside
/// `Electron Framework.framework/Versions/Current/Resources/Info.plist`.
fn read_electron_version(frameworks_dir: &Path) -> Option<String> {
    let version_file = frameworks_dir
        .join("Electron Framework.framework")
        .join("Versions")
        .join("Current")
        .join("Resources")
        .join("Info.plist");

    if version_file.exists() {
        if let Ok(content) = std::fs::read_to_string(&version_file) {
            // Look for CFBundleShortVersionString pattern in XML plist
            if let Some(pos) = content.find("CFBundleShortVersionString") {
                let after = &content[pos..];
                if let Some(start) = after.find("<string>") {
                    let version_start = start + 8;
                    if let Some(end) = after[version_start..].find("</string>") {
                        return Some(after[version_start..version_start + end].to_string());
                    }
                }
            }
        }
    }
    None
}

/// Check if an app name is a known Electron app.
///
/// Performs case-insensitive matching: the app name can either exactly match
/// (ignoring case) or contain a known Electron app name as a substring.
pub fn is_known_electron_app(app_name: &str) -> bool {
    KNOWN_ELECTRON_APPS.iter().any(|&known| {
        app_name.eq_ignore_ascii_case(known)
            || app_name
                .to_lowercase()
                .contains(&known.to_lowercase())
    })
}

/// Scan `/Applications` for macOS app bundles and detect their runtimes.
pub fn scan_applications() -> Vec<AppInfo> {
    let apps_dir = Path::new("/Applications");
    scan_directory(apps_dir)
}

/// Scan a directory for `.app` bundles and detect their runtimes.
///
/// Returns an `AppInfo` for each `.app` found directly inside `dir`.
/// Does not recurse into subdirectories.
pub fn scan_directory(dir: &Path) -> Vec<AppInfo> {
    let mut results = Vec::new();

    if !dir.exists() {
        return results;
    }

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "app") {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_default();

                let runtime = detect_runtime(&path);

                let bundle_id = read_bundle_id(&path).unwrap_or_else(|| {
                    format!("unknown.{}", name.to_lowercase().replace(' ', "-"))
                });

                results.push(AppInfo {
                    bundle_id,
                    name,
                    path,
                    runtime,
                    cdp_port: None,
                });
            }
        }
    }

    results
}

/// Read `CFBundleIdentifier` from an app bundle's `Contents/Info.plist`.
fn read_bundle_id(app_path: &Path) -> Option<String> {
    let plist_path = app_path.join("Contents/Info.plist");
    if let Ok(content) = std::fs::read_to_string(&plist_path) {
        if let Some(pos) = content.find("CFBundleIdentifier") {
            let after = &content[pos..];
            if let Some(start) = after.find("<string>") {
                let id_start = start + 8;
                if let Some(end) = after[id_start..].find("</string>") {
                    return Some(after[id_start..id_start + end].to_string());
                }
            }
        }
    }
    None
}

/// Detect if a running app has exposed a CDP debugging port.
///
/// Checks common Electron debug ports (9222, 9229, 5858) by attempting
/// a TCP connection to localhost.
pub fn detect_cdp_port(_bundle_id: &str) -> Option<u16> {
    let common_ports: &[u16] = &[9222, 9229, 5858];

    for &port in common_ports {
        if is_port_open(port) {
            return Some(port);
        }
    }
    None
}

/// Check if a localhost TCP port is accepting connections (non-blocking).
fn is_port_open(port: u16) -> bool {
    use std::net::TcpStream;
    use std::time::Duration;

    TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(100),
    )
    .is_ok()
}
