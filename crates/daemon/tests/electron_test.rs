//! Integration tests for Electron/CEF app detection.
//!
//! Tests cover:
//!   - detect_runtime with mock .app bundles (Electron, CEF, Native)
//!   - is_known_electron_app for known and unknown apps
//!   - scan_directory with temporary mock .app bundles
//!   - AppRuntime serialization roundtrip
//!   - Case-insensitive matching for known Electron apps

use agenthandover_daemon::platform::electron_detect::{
    detect_runtime, is_known_electron_app, scan_directory, AppInfo, AppRuntime,
};
use std::fs;
use std::path::PathBuf;

/// Helper: create a mock .app bundle directory structure.
fn create_mock_app(base_dir: &std::path::Path, app_name: &str) -> PathBuf {
    let app_path = base_dir.join(format!("{}.app", app_name));
    fs::create_dir_all(app_path.join("Contents")).unwrap();
    app_path
}

/// Helper: add Electron Framework.framework to a mock .app bundle.
fn add_electron_framework(app_path: &std::path::Path) {
    let framework_dir = app_path.join("Contents/Frameworks/Electron Framework.framework");
    fs::create_dir_all(&framework_dir).unwrap();
}

/// Helper: add Electron Framework.framework with a version Info.plist.
fn add_electron_framework_with_version(app_path: &std::path::Path, version: &str) {
    let framework_dir = app_path.join("Contents/Frameworks/Electron Framework.framework");
    let resources_dir = framework_dir
        .join("Versions")
        .join("Current")
        .join("Resources");
    fs::create_dir_all(&resources_dir).unwrap();

    let plist_content = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>CFBundleShortVersionString</key>
    <string>{}</string>
</dict>
</plist>"#,
        version
    );
    fs::write(resources_dir.join("Info.plist"), plist_content).unwrap();
}

/// Helper: add CEF framework to a mock .app bundle.
fn add_cef_framework(app_path: &std::path::Path) {
    let framework_dir =
        app_path.join("Contents/Frameworks/Chromium Embedded Framework.framework");
    fs::create_dir_all(&framework_dir).unwrap();
}

/// Helper: add a custom-named Electron framework variant.
fn add_custom_electron_framework(app_path: &std::path::Path, name: &str) {
    let framework_dir = app_path.join(format!("Contents/Frameworks/{}", name));
    fs::create_dir_all(&framework_dir).unwrap();
}

/// Helper: add an Info.plist with a CFBundleIdentifier.
fn add_bundle_id(app_path: &std::path::Path, bundle_id: &str) {
    let plist_content = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>{}</string>
</dict>
</plist>"#,
        bundle_id
    );
    fs::write(app_path.join("Contents/Info.plist"), plist_content).unwrap();
}

// ---------------------------------------------------------------------------
// detect_runtime tests
// ---------------------------------------------------------------------------

#[test]
fn electron_detect_runtime_with_electron_framework() {
    let tmp = tempfile::tempdir().unwrap();
    let app_path = create_mock_app(tmp.path(), "TestElectron");
    add_electron_framework(&app_path);

    let runtime = detect_runtime(&app_path);
    match runtime {
        AppRuntime::Electron { version } => {
            // No version file added, so version should be None
            assert!(version.is_none());
        }
        other => panic!("Expected Electron, got {:?}", other),
    }
}

#[test]
fn electron_detect_runtime_with_electron_version() {
    let tmp = tempfile::tempdir().unwrap();
    let app_path = create_mock_app(tmp.path(), "TestElectronVersioned");
    add_electron_framework_with_version(&app_path, "28.1.0");

    let runtime = detect_runtime(&app_path);
    match runtime {
        AppRuntime::Electron { version } => {
            assert_eq!(version, Some("28.1.0".to_string()));
        }
        other => panic!("Expected Electron with version, got {:?}", other),
    }
}

#[test]
fn electron_detect_runtime_native_without_frameworks() {
    let tmp = tempfile::tempdir().unwrap();
    let app_path = create_mock_app(tmp.path(), "NativeApp");
    // No frameworks directory at all

    let runtime = detect_runtime(&app_path);
    assert_eq!(runtime, AppRuntime::Native);
}

#[test]
fn electron_detect_runtime_native_with_empty_frameworks() {
    let tmp = tempfile::tempdir().unwrap();
    let app_path = create_mock_app(tmp.path(), "NativeAppFrameworks");
    fs::create_dir_all(app_path.join("Contents/Frameworks")).unwrap();

    let runtime = detect_runtime(&app_path);
    assert_eq!(runtime, AppRuntime::Native);
}

#[test]
fn electron_detect_runtime_cef_framework() {
    let tmp = tempfile::tempdir().unwrap();
    let app_path = create_mock_app(tmp.path(), "CefApp");
    add_cef_framework(&app_path);

    let runtime = detect_runtime(&app_path);
    assert_eq!(runtime, AppRuntime::CEF);
}

#[test]
fn electron_detect_runtime_custom_electron_naming() {
    let tmp = tempfile::tempdir().unwrap();
    let app_path = create_mock_app(tmp.path(), "CustomElectron");
    add_custom_electron_framework(&app_path, "MyElectronCustom.framework");

    let runtime = detect_runtime(&app_path);
    match runtime {
        AppRuntime::Electron { .. } => {} // pass
        other => panic!("Expected Electron for custom naming, got {:?}", other),
    }
}

#[test]
fn electron_detect_runtime_nonexistent_path() {
    let path = PathBuf::from("/nonexistent/app.app");
    let runtime = detect_runtime(&path);
    assert_eq!(runtime, AppRuntime::Native);
}

// ---------------------------------------------------------------------------
// is_known_electron_app tests
// ---------------------------------------------------------------------------

#[test]
fn electron_known_app_slack() {
    assert!(is_known_electron_app("Slack"));
}

#[test]
fn electron_known_app_discord() {
    assert!(is_known_electron_app("Discord"));
}

#[test]
fn electron_known_app_vscode() {
    assert!(is_known_electron_app("Visual Studio Code"));
}

#[test]
fn electron_known_app_code_short() {
    assert!(is_known_electron_app("Code"));
}

#[test]
fn electron_known_app_obsidian() {
    assert!(is_known_electron_app("Obsidian"));
}

#[test]
fn electron_unknown_app_safari() {
    assert!(!is_known_electron_app("Safari"));
}

#[test]
fn electron_unknown_app_finder() {
    assert!(!is_known_electron_app("Finder"));
}

#[test]
fn electron_unknown_app_empty() {
    assert!(!is_known_electron_app(""));
}

#[test]
fn electron_known_app_case_insensitive_lower() {
    assert!(is_known_electron_app("slack"));
}

#[test]
fn electron_known_app_case_insensitive_upper() {
    assert!(is_known_electron_app("DISCORD"));
}

#[test]
fn electron_known_app_case_insensitive_mixed() {
    assert!(is_known_electron_app("oBsIdIaN"));
}

#[test]
fn electron_known_app_substring_match() {
    // "Visual Studio Code - Insiders" contains "Visual Studio Code"
    assert!(is_known_electron_app("Visual Studio Code - Insiders"));
}

// ---------------------------------------------------------------------------
// scan_directory tests
// ---------------------------------------------------------------------------

#[test]
fn electron_scan_directory_finds_electron_app() {
    let tmp = tempfile::tempdir().unwrap();

    let electron_app = create_mock_app(tmp.path(), "MyElectron");
    add_electron_framework(&electron_app);
    add_bundle_id(&electron_app, "com.test.myelectron");

    let results = scan_directory(tmp.path());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "MyElectron");
    assert_eq!(results[0].bundle_id, "com.test.myelectron");
    match &results[0].runtime {
        AppRuntime::Electron { .. } => {}
        other => panic!("Expected Electron, got {:?}", other),
    }
}

#[test]
fn electron_scan_directory_finds_native_and_electron() {
    let tmp = tempfile::tempdir().unwrap();

    let native_app = create_mock_app(tmp.path(), "NativeApp");
    add_bundle_id(&native_app, "com.test.native");

    let electron_app = create_mock_app(tmp.path(), "ElectronApp");
    add_electron_framework(&electron_app);
    add_bundle_id(&electron_app, "com.test.electron");

    let results = scan_directory(tmp.path());
    assert_eq!(results.len(), 2);

    let electron_count = results
        .iter()
        .filter(|app| matches!(app.runtime, AppRuntime::Electron { .. }))
        .count();
    let native_count = results
        .iter()
        .filter(|app| matches!(app.runtime, AppRuntime::Native))
        .count();

    assert_eq!(electron_count, 1);
    assert_eq!(native_count, 1);
}

#[test]
fn electron_scan_directory_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let results = scan_directory(tmp.path());
    assert!(results.is_empty());
}

#[test]
fn electron_scan_directory_nonexistent() {
    let results = scan_directory(&PathBuf::from("/nonexistent/directory"));
    assert!(results.is_empty());
}

#[test]
fn electron_scan_directory_ignores_non_app_items() {
    let tmp = tempfile::tempdir().unwrap();

    // Create a regular directory (not .app)
    fs::create_dir_all(tmp.path().join("NotAnApp")).unwrap();
    // Create a regular file
    fs::write(tmp.path().join("readme.txt"), "hello").unwrap();
    // Create one actual .app
    let app_path = create_mock_app(tmp.path(), "RealApp");
    add_bundle_id(&app_path, "com.test.real");

    let results = scan_directory(tmp.path());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].name, "RealApp");
}

#[test]
fn electron_scan_directory_generates_fallback_bundle_id() {
    let tmp = tempfile::tempdir().unwrap();

    // App without Info.plist
    create_mock_app(tmp.path(), "NoInfoPlist");

    let results = scan_directory(tmp.path());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].bundle_id, "unknown.noinfoplist");
}

#[test]
fn electron_scan_directory_cef_app() {
    let tmp = tempfile::tempdir().unwrap();

    let cef_app = create_mock_app(tmp.path(), "CefApp");
    add_cef_framework(&cef_app);
    add_bundle_id(&cef_app, "com.test.cef");

    let results = scan_directory(tmp.path());
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].runtime, AppRuntime::CEF);
}

// ---------------------------------------------------------------------------
// Serialization roundtrip tests
// ---------------------------------------------------------------------------

#[test]
fn electron_app_runtime_serde_roundtrip_native() {
    let runtime = AppRuntime::Native;
    let json = serde_json::to_string(&runtime).unwrap();
    let deserialized: AppRuntime = serde_json::from_str(&json).unwrap();
    assert_eq!(runtime, deserialized);
}

#[test]
fn electron_app_runtime_serde_roundtrip_electron_with_version() {
    let runtime = AppRuntime::Electron {
        version: Some("28.1.0".to_string()),
    };
    let json = serde_json::to_string(&runtime).unwrap();
    let deserialized: AppRuntime = serde_json::from_str(&json).unwrap();
    assert_eq!(runtime, deserialized);
}

#[test]
fn electron_app_runtime_serde_roundtrip_electron_no_version() {
    let runtime = AppRuntime::Electron { version: None };
    let json = serde_json::to_string(&runtime).unwrap();
    let deserialized: AppRuntime = serde_json::from_str(&json).unwrap();
    assert_eq!(runtime, deserialized);
}

#[test]
fn electron_app_runtime_serde_roundtrip_cef() {
    let runtime = AppRuntime::CEF;
    let json = serde_json::to_string(&runtime).unwrap();
    let deserialized: AppRuntime = serde_json::from_str(&json).unwrap();
    assert_eq!(runtime, deserialized);
}

#[test]
fn electron_app_runtime_serde_roundtrip_unknown() {
    let runtime = AppRuntime::Unknown;
    let json = serde_json::to_string(&runtime).unwrap();
    let deserialized: AppRuntime = serde_json::from_str(&json).unwrap();
    assert_eq!(runtime, deserialized);
}

#[test]
fn electron_app_info_serde_roundtrip() {
    let info = AppInfo {
        bundle_id: "com.test.app".to_string(),
        name: "TestApp".to_string(),
        path: PathBuf::from("/Applications/TestApp.app"),
        runtime: AppRuntime::Electron {
            version: Some("25.0.0".to_string()),
        },
        cdp_port: Some(9222),
    };

    let json = serde_json::to_string(&info).unwrap();
    let deserialized: AppInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.bundle_id, info.bundle_id);
    assert_eq!(deserialized.name, info.name);
    assert_eq!(deserialized.path, info.path);
    assert_eq!(deserialized.runtime, info.runtime);
    assert_eq!(deserialized.cdp_port, info.cdp_port);
}
