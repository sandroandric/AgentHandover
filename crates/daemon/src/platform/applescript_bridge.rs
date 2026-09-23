//! AppleScript bridge for querying native app state.
//!
//! Queries known macOS apps via `osascript -e` to extract context like
//! current folder, document name, email subject, etc.
//!
//! Only with the user's consent: each query first asks macOS, without a
//! prompt, whether AgentHandover may send Apple Events to the app
//! (`AEDeterminePermissionToAutomateTarget`, `askUserIfNeeded = false`).
//! When it may not (denied, or never asked), the query is skipped
//! silently; the answer per app is kept for the daemon status, where
//! `agenthandover doctor` reports the apps that are not granted.  macOS
//! asks the user only from the app's onboarding, where the option is
//! explained.

use agenthandover_common::status::AutomationConsent;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::process::Command;
use std::sync::Mutex;
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

/// The known app a window's `app_id` belongs to, as its AppleScript name.
///
/// The app reports native windows as `pid:<pid>:<owner name>`; tests and
/// older rows use a bundle id or a bare name. All three resolve here.
pub fn app_name_for_app_id(app_id: &str) -> Option<&'static str> {
    if let Some(name) = app_name_for_bundle_id(app_id) {
        return Some(name);
    }
    let owner = crate::observer::privacy_zones::strip_pid_prefix(app_id);
    let lower = owner.to_lowercase();
    KNOWN_APPS
        .iter()
        .find(|(n, _)| n.to_lowercase() == lower)
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

/// The last consent seen per app name.
static CONSENT: Mutex<BTreeMap<String, AutomationConsent>> = Mutex::new(BTreeMap::new());

/// The Automation consent last seen for each app a read was attempted for.
pub fn consent_snapshot() -> BTreeMap<String, AutomationConsent> {
    CONSENT.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

fn record_consent(app_name: &str, consent: AutomationConsent) {
    CONSENT
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(app_name.to_string(), consent);
}

/// Apple Event Manager, in CoreServices.
mod ae {
    use std::ffi::c_void;

    #[repr(C)]
    pub struct AEDesc {
        pub descriptor_type: u32,
        pub data_handle: *mut c_void,
    }

    /// 'bund': an address by bundle identifier.
    pub const TYPE_APPLICATION_BUNDLE_ID: u32 = u32::from_be_bytes(*b"bund");
    /// '****': any event class and id.
    pub const TYPE_WILD_CARD: u32 = u32::from_be_bytes(*b"****");

    #[link(name = "CoreServices", kind = "framework")]
    extern "C" {
        pub fn AECreateDesc(
            type_code: u32,
            data_ptr: *const c_void,
            data_size: isize,
            result: *mut AEDesc,
        ) -> i16;
        pub fn AEDisposeDesc(desc: *mut AEDesc) -> i16;
        pub fn AEDeterminePermissionToAutomateTarget(
            target: *const AEDesc,
            event_class: u32,
            event_id: u32,
            ask_user_if_needed: u8,
        ) -> i32;
    }
}

/// Whether macOS lets this process (AgentHandover.app, which spawned the
/// daemon and owns its privacy grants) send Apple Events to `bundle_id`.
/// Never shows a prompt.
pub fn automation_consent(bundle_id: &str) -> AutomationConsent {
    let mut target = ae::AEDesc {
        descriptor_type: 0,
        data_handle: std::ptr::null_mut(),
    };
    // SAFETY: AECreateDesc copies `bundle_id`'s bytes into `target`, which
    // is disposed of below; the permission call only reads it.
    let status = unsafe {
        if ae::AECreateDesc(
            ae::TYPE_APPLICATION_BUNDLE_ID,
            bundle_id.as_ptr().cast(),
            bundle_id.len() as isize,
            &mut target,
        ) != 0
        {
            return AutomationConsent::Unknown;
        }
        let status = ae::AEDeterminePermissionToAutomateTarget(
            &target,
            ae::TYPE_WILD_CARD,
            ae::TYPE_WILD_CARD,
            0,
        );
        ae::AEDisposeDesc(&mut target);
        status
    };
    AutomationConsent::from_status(status)
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
/// Returns None if the app is unsupported, not running, not granted to
/// AgentHandover (see the module docs), or the query fails.
pub fn query_app_state(app_name: &str) -> Option<AppState> {
    query_app_state_with(app_name, automation_consent, run_osascript)
}

/// [`query_app_state`] with the consent check and the script runner given.
fn query_app_state_with(
    app_name: &str,
    consent: impl Fn(&str) -> AutomationConsent,
    run: impl Fn(&str) -> Option<String>,
) -> Option<AppState> {
    let script = build_script(app_name)?;
    let bundle_id = bundle_id_for_app(app_name)?;
    let app_name = app_name_for_bundle_id(bundle_id).unwrap_or(app_name);

    let granted = consent(bundle_id);
    record_consent(app_name, granted);
    if granted != AutomationConsent::Granted {
        debug!(app = app_name, consent = ?granted, "App state not read: not granted");
        return None;
    }

    let output = run(&script)?;
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

/// Async wrapper that runs the AppleScript query in a blocking thread.
///
/// No outer Tokio timeout — the inner `run_osascript()` already has its
/// own 500ms polling timeout that kills the `osascript` child process.
/// A Tokio-level timeout on top would orphan the blocking thread
/// (same class of bug as the v0.2.8 OCR/clipboard crash).
pub async fn query_app_state_async(app_name: String) -> Option<AppState> {
    match tokio::task::spawn_blocking(move || query_app_state(&app_name)).await {
        Ok(result) => result,
        Err(e) => {
            warn!(error = %e, "AppleScript query task panicked");
            None
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn app_id_forms_resolve_to_the_known_app() {
        assert_eq!(app_name_for_app_id("pid:412:Mail"), Some("Mail"));
        assert_eq!(app_name_for_app_id("pid:9:notes"), Some("Notes"));
        assert_eq!(app_name_for_app_id("com.apple.finder"), Some("Finder"));
        assert_eq!(app_name_for_app_id("TextEdit"), Some("TextEdit"));
        assert_eq!(app_name_for_app_id("pid:412:Microsoft Excel"), None);
        assert_eq!(app_name_for_app_id("com.example.Mail"), None);
    }
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
    fn an_app_that_is_not_granted_is_never_queried() {
        use std::cell::Cell;
        for consent in [
            AutomationConsent::Denied,
            AutomationConsent::NotAsked,
            AutomationConsent::NotRunning,
            AutomationConsent::Unknown,
        ] {
            let ran = Cell::new(false);
            let state = query_app_state_with(
                "notes",
                |bid| {
                    assert_eq!(bid, "com.apple.Notes");
                    consent
                },
                |_| {
                    ran.set(true);
                    Some("document_name:x".into())
                },
            );
            assert!(state.is_none());
            assert!(!ran.get(), "osascript ran for {consent:?}");
            assert_eq!(consent_snapshot().get("Notes"), Some(&consent));
        }
        let state = query_app_state_with(
            "TextEdit",
            |_| AutomationConsent::Granted,
            |_| Some("document_name:Draft\n".into()),
        )
        .unwrap();
        assert_eq!(state.app_id, "com.apple.TextEdit");
        assert_eq!(state.properties["document_name"], "Draft");
        assert_eq!(consent_snapshot().get("TextEdit"), Some(&AutomationConsent::Granted));
    }

    /// The real call links and answers at once, without a prompt: the test
    /// runner has no consent for an app that does not exist.
    #[test]
    fn the_consent_call_returns_without_prompting() {
        assert_ne!(
            automation_consent("com.agenthandover.no-such-app"),
            AutomationConsent::Granted
        );
    }

    #[test]
    fn consent_codes() {
        assert_eq!(AutomationConsent::from_status(0), AutomationConsent::Granted);
        assert_eq!(AutomationConsent::from_status(-1743), AutomationConsent::Denied);
        assert_eq!(AutomationConsent::from_status(-1744), AutomationConsent::NotAsked);
        assert_eq!(AutomationConsent::from_status(-600), AutomationConsent::NotRunning);
        assert_eq!(AutomationConsent::from_status(-50), AutomationConsent::Unknown);
    }

    #[test]
    fn test_build_script_unknown() {
        let script = build_script("UnknownApp");
        assert!(script.is_none());
    }
}
