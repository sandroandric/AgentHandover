//! The built CLI, run with HOME pointed at a temporary directory (so it
//! never reads or writes the real data dir): `focus start` refuses when no
//! daemon runs, and `doctor --extension-json` reports a browser profile
//! that has the extension.

use std::process::Command;

fn cli(home: &std::path::Path, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_agenthandover"))
        .args(args)
        .env("HOME", home)
        .env("AH_LIB_DIR", home.join("no-lib"))
        .output()
        .expect("run agenthandover")
}

fn data_dir(home: &std::path::Path) -> std::path::PathBuf {
    if cfg!(target_os = "macos") {
        home.join("Library/Application Support/agenthandover")
    } else {
        home.join(".local/share/agenthandover")
    }
}

#[test]
fn focus_start_without_a_daemon_fails_and_records_nothing() {
    let home = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(data_dir(home.path())).unwrap();
    let out = cli(home.path(), &["focus", "start", "Send weekly report"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("not observing"), "stderr: {stderr}");
    assert!(stderr.contains("Observe Me"), "stderr: {stderr}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.contains("Focus recording started"), "stdout: {stdout}");
    assert!(!data_dir(home.path()).join("focus-session.json").exists());
}

#[cfg(target_os = "macos")]
#[test]
fn doctor_extension_json_lists_a_profile_with_the_extension() {
    let home = tempfile::tempdir().unwrap();
    let profile = home.path().join("Library/Application Support/Google/Chrome/Default");
    std::fs::create_dir_all(&profile).unwrap();
    std::fs::write(
        profile.join("Secure Preferences"),
        r#"{"extensions": {"settings": {"jpemkdcihaijkolbkankcldmiimmmnfo": {"location": 4}}}}"#,
    )
    .unwrap();
    let out = cli(home.path(), &["doctor", "--extension-json"]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["extension_id"], "jpemkdcihaijkolbkankcldmiimmmnfo");
    assert_eq!(
        report["installed_in"],
        serde_json::json!([{"browser": "Google Chrome", "profile": "Default", "enabled": true}])
    );
}
