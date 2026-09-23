//! Whether the AgentHandover browser extension is loaded in a browser.
//!
//! One check for `agenthandover doctor` and the app's setup screen (which
//! runs `agenthandover doctor --extension-json`): the extension's id in a
//! Chromium-family browser profile's preferences. Chrome records every
//! installed extension, unpacked ones included, under
//! `extensions.settings.<id>` in `Preferences` or `Secure Preferences`.
//! Finding the built extension on disk (see `paths::find_any_extension_path`)
//! only says it can be loaded, not that it is.

use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The extension's id: fixed by the public key in extension/manifest.json
/// (the same id the native messaging manifests allow).
pub const EXTENSION_ID: &str = "jpemkdcihaijkolbkankcldmiimmmnfo";

/// Chromium-family browsers: the profile root under
/// `~/Library/Application Support` and the name people know. The same list
/// postinstall writes native messaging manifests for.
const BROWSERS: &[(&str, &str)] = &[
    ("Google/Chrome", "Google Chrome"),
    ("Chromium", "Chromium"),
    ("BraveSoftware/Brave-Browser", "Brave"),
    ("Microsoft Edge", "Microsoft Edge"),
    ("Arc/User Data", "Arc"),
    ("Comet", "Comet"),
];

/// One browser profile that has the extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Install {
    pub browser: String,
    pub profile: String,
    /// False when the user turned it off in the browser's extensions page.
    pub enabled: bool,
}

/// What `doctor --extension-json` prints.
#[derive(Debug, Serialize)]
pub struct Report {
    pub extension_id: &'static str,
    /// The folder to load unpacked (installed or a source build); null when
    /// none is on disk.
    pub build_path: Option<PathBuf>,
    pub installed_in: Vec<Install>,
}

impl Report {
    pub fn gather() -> Report {
        Report {
            extension_id: EXTENSION_ID,
            build_path: crate::paths::find_any_extension_path(),
            installed_in: installs(&home_dir()),
        }
    }
}

fn home_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string()))
}

/// Every browser profile under `home` that has the extension installed.
pub fn installs(home: &Path) -> Vec<Install> {
    let support = home.join("Library/Application Support");
    let mut found = Vec::new();
    for (root, browser) in BROWSERS {
        let Ok(entries) = std::fs::read_dir(support.join(root)) else {
            continue;
        };
        let mut profiles: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_dir())
            .collect();
        profiles.sort();
        for profile in profiles {
            if let Some(enabled) = profile_state(&profile) {
                found.push(Install {
                    browser: (*browser).to_string(),
                    profile: profile
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    enabled,
                });
            }
        }
    }
    found
}

/// Some(enabled) when the profile has the extension, None when it does not.
fn profile_state(profile: &Path) -> Option<bool> {
    let mut state = None;
    for file in ["Secure Preferences", "Preferences"] {
        let Ok(text) = std::fs::read_to_string(profile.join(file)) else {
            continue;
        };
        let Ok(prefs) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        if let Some(entry) = prefs
            .pointer("/extensions/settings")
            .and_then(|s| s.get(EXTENSION_ID))
        {
            // Either file may carry the disabled state; disabled in one wins.
            let enabled = entry_enabled(entry);
            state = Some(state.unwrap_or(true) && enabled);
        }
    }
    state
}

/// Chrome marks a turned-off extension with a non-empty `disable_reasons`
/// (a bitmask in older versions, a list in newer ones) or, in older
/// versions, `state: 0`.
fn entry_enabled(entry: &Value) -> bool {
    let reasons_set = match entry.get("disable_reasons") {
        Some(Value::Number(n)) => n.as_u64() != Some(0),
        Some(Value::Array(a)) => !a.is_empty(),
        _ => false,
    };
    let state_off = entry.get("state").and_then(Value::as_u64) == Some(0);
    !reasons_set && !state_off
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_prefs(home: &Path, root: &str, profile: &str, file: &str, json: &str) {
        let dir = home.join("Library/Application Support").join(root).join(profile);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(file), json).unwrap();
    }

    fn settings(entry: &str) -> String {
        format!(r#"{{"extensions": {{"settings": {{"{EXTENSION_ID}": {entry}}}}}}}"#)
    }

    #[test]
    fn no_browsers_means_not_installed() {
        let home = tempfile::tempdir().unwrap();
        assert!(installs(home.path()).is_empty());
    }

    #[test]
    fn an_unpacked_extension_in_secure_preferences_is_found() {
        let home = tempfile::tempdir().unwrap();
        write_prefs(
            home.path(),
            "Google/Chrome",
            "Default",
            "Secure Preferences",
            &settings(r#"{"location": 4, "path": "/usr/local/lib/agenthandover/extension"}"#),
        );
        assert_eq!(
            installs(home.path()),
            vec![Install {
                browser: "Google Chrome".into(),
                profile: "Default".into(),
                enabled: true
            }]
        );
    }

    #[test]
    fn other_extensions_and_other_profiles_do_not_count() {
        let home = tempfile::tempdir().unwrap();
        write_prefs(
            home.path(),
            "Google/Chrome",
            "Default",
            "Preferences",
            r#"{"extensions": {"settings": {"aaaabbbbccccddddeeeeffffgggghhhh": {}}}}"#,
        );
        write_prefs(home.path(), "Google/Chrome", "Profile 1", "Preferences", "not json");
        write_prefs(home.path(), "BraveSoftware/Brave-Browser", "Profile 2", "Preferences", &settings("{}"));
        assert_eq!(
            installs(home.path()),
            vec![Install {
                browser: "Brave".into(),
                profile: "Profile 2".into(),
                enabled: true
            }]
        );
    }

    #[test]
    fn a_turned_off_extension_is_reported_disabled() {
        let home = tempfile::tempdir().unwrap();
        write_prefs(home.path(), "Google/Chrome", "Default", "Preferences", &settings(r#"{"disable_reasons": 1}"#));
        write_prefs(home.path(), "Microsoft Edge", "Default", "Preferences", &settings(r#"{"state": 0}"#));
        write_prefs(home.path(), "Chromium", "Default", "Preferences", &settings(r#"{"disable_reasons": [1]}"#));
        write_prefs(home.path(), "Arc/User Data", "Default", "Preferences", &settings(r#"{"disable_reasons": 0, "state": 1}"#));
        let found = installs(home.path());
        let enabled: Vec<(&str, bool)> =
            found.iter().map(|i| (i.browser.as_str(), i.enabled)).collect();
        assert_eq!(
            enabled,
            vec![
                ("Google Chrome", false),
                ("Chromium", false),
                ("Microsoft Edge", false),
                ("Arc", true)
            ]
        );
    }

    #[test]
    fn disabled_in_either_preferences_file_wins() {
        let home = tempfile::tempdir().unwrap();
        write_prefs(home.path(), "Google/Chrome", "Default", "Secure Preferences", &settings("{}"));
        write_prefs(home.path(), "Google/Chrome", "Default", "Preferences", &settings(r#"{"disable_reasons": 1}"#));
        assert!(!installs(home.path())[0].enabled);
    }

    #[test]
    fn the_report_serializes_as_the_app_reads_it() {
        let report = Report {
            extension_id: EXTENSION_ID,
            build_path: Some(PathBuf::from("/usr/local/lib/agenthandover/extension")),
            installed_in: vec![Install {
                browser: "Google Chrome".into(),
                profile: "Default".into(),
                enabled: true,
            }],
        };
        let json: Value = serde_json::to_value(&report).unwrap();
        assert_eq!(json["extension_id"], EXTENSION_ID);
        assert_eq!(json["build_path"], "/usr/local/lib/agenthandover/extension");
        assert_eq!(json["installed_in"][0]["browser"], "Google Chrome");
        assert_eq!(json["installed_in"][0]["enabled"], true);
    }
}
