//! AppleScript bridge for querying native app state.
//!
//! Queries known macOS apps via `osascript -e` to extract context like
//! current folder, document name, email subject, etc. Follows the
//! app-specific adapter pattern used in `electron_detect.rs`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use std::time::Duration;
use tracing::{debug, warn};

/// State extracted from a native macOS app via AppleScript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppState {
    pub app_id: String,
    pub properties: HashMap<String, String>,
}

/// Known macOS apps that support AppleScript queries.
const KNOWN_APPS: &[(&str, &str)] = &[
    ("Finder", "com.apple.finder"),
    ("Mail", "com.apple.mail"),
    ("Notes", "com.apple.Notes"),
    ("Calendar", "com.apple.iCal"),
    ("Messages", "com.apple.MobileSMS"),
    ("Safari", "com.apple.Safari"),
    ("Preview", "com.apple.Preview"),
    ("TextEdit", "com.apple.TextEdit"),
    ("Reminders", "com.apple.reminders"),
    ("Contacts", "com.apple.AddressBook"),
];

/// Check if an app name (case-insensitive) is in the known apps list.
pub fn is_supported_app(name: &str) -> bool {
    let lower = name.to_lowercase();
    KNOWN_APPS.iter().any(|(n, _)| n.to_lowercase() == lower)
}

/// Check if a bundle ID is in the known apps list.
pub fn is_supported_bundle_id(bundle_id: &str) -> bool {
    KNOWN_APPS.iter().any(|(_, bid)| *bid == bundle_id)
}

/// Get the bundle ID for a known app name (case-insensitive).
pub fn bundle_id_for_app(name: &str) -> Option<&'static str> {
    let lower = name.to_lowercase();
    KNOWN_APPS
        .iter()
        .find(|(n, _)| n.to_lowercase() == lower)
        .map(|(_, bid)| *bid)
}

/// Get the app display name for a bundle ID.
pub fn app_name_for_bundle_id(bundle_id: &str) -> Option<&'static str> {
    KNOWN_APPS
        .iter()
        .find(|(_, bid)| *bid == bundle_id)
        .map(|(name, _)| *name)
}

/// Build an AppleScript snippet for the given app name.
fn build_script(app_name: &str) -> Option<String> {
    let lower = app_name.to_lowercase();
    let script = match lower.as_str() {
        "finder" => {
            r#"tell application "Finder"
    set currentFolder to (target of front Finder window) as text
    set selectedItems to selection as text
    return "current_folder:" & currentFolder & linefeed & "selected_items:" & selectedItems
end tell"#
        }
        "mail" => {
            r#"tell application "Mail"
    set msgSubject to ""
    set msgSender to ""
    set msgMailbox to ""
    try
        set selectedMsgs to selection
        if (count of selectedMsgs) > 0 then
            set firstMsg to item 1 of selectedMsgs
            set msgSubject to subject of firstMsg
            set msgSender to sender of firstMsg
            set msgMailbox to name of mailbox of firstMsg
        end if
    end try
    return "subject:" & msgSubject & linefeed & "sender:" & msgSender & linefeed & "mailbox:" & msgMailbox
end tell"#
        }
        "notes" => {
            r#"tell application "Notes"
    set noteTitle to ""
    set noteFolder to ""
    try
        set noteTitle to name of first note
        set noteFolder to name of container of first note
    end try
    return "note_title:" & noteTitle & linefeed & "folder:" & noteFolder
end tell"#
        }
        "calendar" => {
            r#"tell application "Calendar"
    set currentDate to current date as text
    return "current_date:" & currentDate
end tell"#
        }
        "messages" => {
            r#"tell application "Messages"
    set chatCount to count of chats
    return "chat_count:" & chatCount
end tell"#
        }
        "safari" => {
            r#"tell application "Safari"
    set pageURL to ""
    set pageTitle to ""
    try
        set pageURL to URL of front document
        set pageTitle to name of front document
    end try
    return "url:" & pageURL & linefeed & "title:" & pageTitle
end tell"#
        }
        "textedit" => {
            r#"tell application "TextEdit"
    set docName to ""
    try
        set docName to name of front document
    end try
    return "document_name:" & docName
end tell"#
        }
        "preview" => {
            r#"tell application "Preview"
    set docName to ""
    try
        set docName to name of front document
    end try
    return "document_name:" & docName
end tell"#
        }
        _ => return None,
    };
    Some(script.to_string())
}

/// Run an AppleScript via `osascript -e` with a 500ms timeout.
/// Returns the stdout output or None on failure/timeout.
fn run_osascript(script: &str) -> Option<String> {
    let mut child = Command::new("osascript")
        .arg("-e")
        .arg(script)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;

    // Wait with timeout via polling loop (runs inside spawn_blocking)
    let start = std::time::Instant::now();
    let timeout = Duration::from_millis(500);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process exited — read stdout directly from the child's pipe.
                // Do NOT call wait_with_output() here (it would re-wait).
                if status.success() {
                    let mut stdout_buf = Vec::new();
                    if let Some(ref mut stdout) = child.stdout {
                        use std::io::Read;
                        let _ = stdout.read_to_end(&mut stdout_buf);
                    }
                    return String::from_utf8(stdout_buf).ok();
                }
                return None;
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    warn!("osascript timed out after 500ms, killing process");
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

/// Parse `key:value\n` output into a HashMap.
fn parse_output(output: &str) -> HashMap<String, String> {
    let mut result = HashMap::new();
    for line in output.lines() {
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim().to_string();
            let value = value.trim().to_string();
            if !key.is_empty() {
                result.insert(key, value);
            }
        }
    }
    result
}

/// Query the state of a known app by its display name.
/// Returns None if the app is unsupported, not running, or query fails.
pub fn query_app_state(app_name: &str) -> Option<AppState> {
    let script = build_script(app_name)?;
    let bundle_id = bundle_id_for_app(app_name)?;

    let output = run_osascript(&script)?;
    let properties = parse_output(&output);

    if properties.is_empty() {
        return None;
    }

    debug!(
        app = app_name,
        properties = properties.len(),
        "AppleScript state captured"
    );

    Some(AppState {
        app_id: bundle_id.to_string(),
        properties,
    })
}

/// Query app state by bundle ID instead of display name.
pub fn query_app_state_by_bundle_id(bundle_id: &str) -> Option<AppState> {
    let app_name = app_name_for_bundle_id(bundle_id)?;
    query_app_state(app_name)
}

/// Async wrapper that runs the AppleScript query in a blocking thread with a 1s timeout.
pub async fn query_app_state_async(app_name: String) -> Option<AppState> {
    let task = tokio::task::spawn_blocking(move || query_app_state(&app_name));

    match tokio::time::timeout(Duration::from_secs(1), task).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            warn!(error = %e, "AppleScript query task panicked");
            None
        }
        Err(_) => {
            warn!("AppleScript query timed out after 1s");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_supported_finder() {
        assert!(is_supported_app("Finder"));
    }

    #[test]
    fn test_is_supported_mail() {
        assert!(is_supported_app("Mail"));
    }

    #[test]
    fn test_is_supported_notes() {
        assert!(is_supported_app("Notes"));
    }

    #[test]
    fn test_is_supported_case_insensitive() {
        assert!(is_supported_app("finder"));
        assert!(is_supported_app("FINDER"));
        assert!(is_supported_app("FiNdEr"));
    }

    #[test]
    fn test_unsupported_app_returns_false() {
        assert!(!is_supported_app("Photoshop"));
        assert!(!is_supported_app("VSCode"));
        assert!(!is_supported_app(""));
    }

    #[test]
    fn test_is_supported_bundle_id() {
        assert!(is_supported_bundle_id("com.apple.finder"));
        assert!(is_supported_bundle_id("com.apple.mail"));
        assert!(!is_supported_bundle_id("com.microsoft.VSCode"));
    }

    #[test]
    fn test_bundle_id_lookup() {
        assert_eq!(bundle_id_for_app("Finder"), Some("com.apple.finder"));
        assert_eq!(bundle_id_for_app("Mail"), Some("com.apple.mail"));
        assert_eq!(bundle_id_for_app("Safari"), Some("com.apple.Safari"));
    }

    #[test]
    fn test_bundle_id_unknown_returns_none() {
        assert_eq!(bundle_id_for_app("UnknownApp"), None);
    }

    #[test]
    fn test_app_state_serde_roundtrip() {
        let state = AppState {
            app_id: "com.apple.finder".to_string(),
            properties: HashMap::from([
                ("current_folder".to_string(), "Documents".to_string()),
                ("selected_items".to_string(), "file.txt".to_string()),
            ]),
        };
        let json = serde_json::to_string(&state).unwrap();
        let deserialized: AppState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.app_id, "com.apple.finder");
        assert_eq!(deserialized.properties.len(), 2);
    }

    #[test]
    fn test_parse_output_key_value() {
        let output = "current_folder:Documents\nselected_items:file.txt\n";
        let parsed = parse_output(output);
        assert_eq!(parsed.get("current_folder"), Some(&"Documents".to_string()));
        assert_eq!(
            parsed.get("selected_items"),
            Some(&"file.txt".to_string())
        );
    }

    #[test]
    fn test_parse_output_empty() {
        let parsed = parse_output("");
        assert!(parsed.is_empty());
    }

    #[test]
    fn test_parse_output_colon_in_value() {
        let output = "url:https://example.com\n";
        let parsed = parse_output(output);
        assert_eq!(
            parsed.get("url"),
            Some(&"https://example.com".to_string())
        );
    }

    #[test]
    fn test_build_script_finder() {
        let script = build_script("Finder");
        assert!(script.is_some());
        assert!(script.unwrap().contains("Finder"));
    }

    #[test]
    fn test_build_script_unknown() {
        let script = build_script("UnknownApp");
        assert!(script.is_none());
    }
}
