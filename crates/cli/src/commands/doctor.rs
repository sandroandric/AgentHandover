use anyhow::Result;
use chrono::Utc;
use colored::Colorize;

use agenthandover_common::observation::{
    capture_socket_path, observation_socket_path, snapshot_at,
};

use agenthandover_common::status::AutomationConsent;
use agenthandover_common::tiers;
use crate::{browser_extension, machine, ollama, paths};

/// Detect which install channel produced this AgentHandover installation.
fn detect_install_channel() -> &'static str {
    // Check for pkg install (the daemon is installed as <lib dir>/ah-observer)
    if paths::lib_dir().join("ah-observer").exists() {
        return "pkg";
    }
    // Check for Homebrew
    if let Ok(output) = std::process::Command::new("brew")
        .args(["--prefix", "agenthandover"])
        .output()
    {
        if output.status.success() {
            return "homebrew";
        }
    }
    // Check for source build (look for Cargo.toml in ancestor dirs)
    let mut dir = std::env::current_dir().unwrap_or_default();
    for _ in 0..5 {
        if dir.join("Cargo.toml").exists() && dir.join("worker").exists() {
            return "source";
        }
        if !dir.pop() {
            break;
        }
    }
    "unknown"
}

/// Print a channel-aware install/repair hint.
fn print_install_hint(channel: &str) {
    match channel {
        "pkg" => eprintln!("  Fix: Re-run the .pkg installer to repair."),
        "homebrew" => eprintln!("  Fix: brew reinstall agenthandover"),
        "source" => eprintln!("  Fix: just build-all"),
        _ => eprintln!("  Fix: Re-install AgentHandover (.pkg recommended)."),
    }
}

/// Counters for the check result states.  A warning does not fail doctor
/// but is counted, so a summary never reads "all passed" over one.
struct CheckCounts {
    passes: u32,
    skips: u32,
    warnings: u32,
    failures: u32,
}

impl CheckCounts {
    fn new() -> Self {
        Self {
            passes: 0,
            skips: 0,
            warnings: 0,
            failures: 0,
        }
    }

    fn record_pass(&mut self) {
        self.passes += 1;
    }

    fn record_skip(&mut self) {
        self.skips += 1;
    }

    fn record_fail(&mut self) {
        self.failures += 1;
    }
}

pub fn run() -> Result<()> {
    println!("{}", "AgentHandover Doctor".bold());
    println!("{}", "=".repeat(50));
    println!();

    let channel = detect_install_channel();
    let mut counts = CheckCounts::new();

    // Check 1: Daemon binary exists
    //
    // v0.2.1 renamed the installed daemon from `agenthandover-daemon` to
    // `ah-observer` and moved it from `/usr/local/bin/` to
    // `/usr/local/lib/agenthandover/`. The Swift menu bar app hardcodes
    // that new path as the place to spawn the daemon from. We use
    // paths::find_daemon_binary() so pkg/cask installs (ah-observer),
    // legacy installs, homebrew libexec, and cargo source builds all
    // resolve through a single helper.
    let daemon_path = paths::find_daemon_binary();
    if let Some(ref p) = daemon_path {
        check_pass(&mut counts, &format!("Daemon binary ({})", p.display()));
    } else {
        check_fail(&mut counts, "Daemon binary");
        print_install_hint(channel);
    }

    // Check 2: CLI binary (we're running it, so it exists)
    check_pass(&mut counts, "CLI binary");

    // Check 3: Data directory exists
    let data_dir = agenthandover_common::status::status_dir();
    if data_dir.exists() {
        check_pass(&mut counts, "Data directory");
    } else {
        check_fail(&mut counts, "Data directory");
    }

    // Check 4: Config file exists (optional — defaults are fine)
    let config_path = data_dir.join("config.toml");
    if config_path.exists() {
        check_pass(&mut counts, "Config file");
    } else {
        check_skip(&mut counts, "Config file", "Using defaults");
    }

    // Check 5-8: the app's capture servers and its TCC grants.
    //
    // Accessibility and Screen Recording are granted to AgentHandover.app,
    // not to this CLI (the app spawns the daemon, so the daemon runs under
    // the app's grants). Asking TCC about this process would always say
    // "no", so we ask the app: its observation server reports both grants.
    // If it does not answer, the app is not running or its server failed
    // to bind, and the daemon captures nothing.
    check_capture_servers(&mut counts);

    // Check 7: Database exists and is writable
    let db_path = data_dir.join("events.db");
    let daemon_pid_exists = data_dir.join("daemon.pid").exists();
    let db_ok = db_path.exists()
        && std::fs::OpenOptions::new()
            .write(true)
            .open(&db_path)
            .is_ok();
    if daemon_pid_exists {
        // Daemon has started at least once — DB should exist
        if db_ok {
            check_pass(&mut counts, "Database");
        } else {
            check_fail(&mut counts, "Database");
        }
    } else {
        // Fresh install — daemon never ran, DB expected to not exist
        if db_ok {
            check_pass(&mut counts, "Database");
        } else {
            check_skip(
                &mut counts,
                "Database",
                "Run 'agenthandover start' to create database",
            );
        }
    }

    // Check 8: Native messaging host manifest (content validation)
    if check_native_messaging_manifest() {
        check_pass(&mut counts, "Native messaging host");
    } else {
        check_fail(&mut counts, "Native messaging host");
    }

    // Check 9: Worker launchd plist
    //
    // Only the worker uses launchd in v0.2.1+. The daemon is spawned
    // directly by the menu bar app via Process() (see ServiceController
    // .startDaemon()) so it has no launchd plist anymore. The old
    // com.agenthandover.daemon.plist was deleted in v0.2.1 — do NOT
    // add a check for it here, it will always fail and mislead users.
    let launch_agents = launch_agents_dir();
    if launch_agents.join("com.agenthandover.worker.plist").exists() {
        check_pass(&mut counts, "Worker launchd plist");
    } else {
        check_fail(&mut counts, "Worker launchd plist");
        print_install_hint(channel);
    }

    // Check 10: Heartbeat freshness (advisory — does not affect counts)
    check_heartbeat_freshness("Daemon", "daemon-status.json");
    check_heartbeat_freshness("Worker", "worker-status.json");
    check_embedding_upgrade(&mut counts);

    // Check 11: Disk space
    if free_disk_gb() > 1 {
        check_pass(&mut counts, "Disk space (>1GB free)");
    } else {
        check_fail(&mut counts, "Disk space (>1GB free)");
    }

    // Check 12: Python virtual environment. Absent is fine for a source
    // build; when the lib dir exists (a pkg install) a missing or broken
    // venv means the worker cannot run at all, so it is a failure.
    check_venv(&mut counts, channel);

    // Check 13: Browser extension (optional — not all users need it):
    // loaded in a browser profile, not merely built on disk.
    let extension = browser_extension::Report::gather();
    let enabled: Vec<String> = extension
        .installed_in
        .iter()
        .filter(|i| i.enabled)
        .map(|i| format!("{} ({})", i.browser, i.profile))
        .collect();
    if !enabled.is_empty() {
        check_pass(&mut counts, &format!("Browser extension (in {})", enabled.join(", ")));
        print_extension_path(&extension);
    } else if let Some(off) = extension.installed_in.first() {
        let hint = format!("turned off in {}: switch it on in its extensions page", off.browser);
        check_skip(&mut counts, "Browser extension", &hint);
        print_extension_path(&extension);
    } else if let Some(path) = &extension.build_path {
        let hint = format!(
            "not loaded in a browser: open chrome://extensions, turn on Developer mode, \
             click Load unpacked and choose {}",
            path.display()
        );
        check_skip(&mut counts, "Browser extension", &hint);
    } else {
        let hint = match channel {
            "pkg" => "Re-run .pkg installer to repair",
            "homebrew" => "brew reinstall agenthandover",
            "source" => "cd extension && npm run build",
            _ => "Re-install AgentHandover (.pkg recommended)",
        };
        check_skip(&mut counts, "Browser extension", hint);
    }

    // Check 14: Worker process alive (optional — may not be started yet)
    if agenthandover_common::pid::check_pid_file("worker").is_some() {
        check_pass(&mut counts, "Worker process");
    } else {
        check_skip(
            &mut counts,
            "Worker process",
            "Start with: agenthandover start",
        );
    }

    // Check 15: the Ollama service (the configured host), its version
    // against the minimum (a failure only when a configured model is
    // Gemma 4, a warning otherwise), the installs on this Mac and which one serves
    // the port, with Memory on its own minimum (a warning), and memory
    // headroom for the configured tier.
    let config_str = std::fs::read_to_string(&config_path).unwrap_or_default();
    let host_url = read_config_string_field(&config_str, "vlm", "ollama_host").unwrap_or_default();
    let report = machine::gather(&host_url, &configured_models(&config_str));
    print_findings(&mut counts, &report.findings);
    let ollama_ok = report.version.is_some();

    // Check 15b: `[privacy] read_app_state` (opt-in): the apps macOS has
    // not let AgentHandover read, from the daemon's status.
    let read_app_state = read_config_string_field(&config_str, "privacy", "read_app_state")
        .is_some_and(|v| v == "true");
    let consent = agenthandover_common::status::read_status_file::<
        agenthandover_common::status::DaemonStatus,
    >("daemon-status.json")
    .ok()
    .and_then(|s| s.app_state_consent);
    print_findings(&mut counts, &app_state_findings(read_app_state, consent.as_ref()));

    // Check 15c: a config still pinning its tier's earlier default models
    // (tiers.json `previous_defaults`): the tier migration the installer
    // and the worker run at startup did not happen or failed.
    print_findings(
        &mut counts,
        &tier_migration_findings(&config_str, tiers::machine_ram_gb()),
    );

    // Check 15d: a config still selecting cloud mode, which 0.5.0 removed:
    // the upgrade's migration (the installer and the worker at startup)
    // could not rewrite it.
    print_findings(&mut counts, &cloud_mode_findings(&config_str));

    // Check 16: Required models pulled
    //
    // The models depend on the RAM tier (tiers.json), so a hard-coded
    // list reports false failures on every other tier.
    //
    // Correct approach: read the user's config.toml to find their
    // ACTUALLY configured annotation_model + sop_model + embedding
    // model, then check Ollama for those specific names. Gracefully
    // fall back if config.toml is missing or fields aren't set.
    if ollama_ok {
        // Missing model keys come from the config's tier, else the
        // RAM tier, exactly as the worker resolves them.
        let machine::ConfiguredModels {
            annotation: annotation_model,
            sop: sop_model,
            ..
        } = configured_models(&config_str);
        let embedding_model = read_config_string_field(&config_str, "embedding", "model")
            .unwrap_or_else(|| tiers::table().embedding_model.clone());

        // Over HTTP, as `setup` asks (see crate::ollama for why).
        let installed = ollama::installed_models(&report.host.url).unwrap_or_default();

        // The recommended tier runs one model for both jobs; deduplicate
        // so we don't report the same model twice.
        let mut models_to_check: Vec<(String, &str)> = Vec::new();
        if !annotation_model.is_empty() {
            models_to_check.push((annotation_model.clone(), "scene annotation"));
        }
        if !sop_model.is_empty() && sop_model != annotation_model {
            models_to_check.push((sop_model.clone(), "SOP generation"));
        }
        if !embedding_model.is_empty() {
            models_to_check.push((embedding_model, "semantic search"));
        }

        for (model, purpose) in &models_to_check {
            if ollama::model_installed(&installed, model) {
                check_pass(&mut counts, &format!("Model {} ({})", model, purpose));
            } else {
                check_fail(&mut counts, &format!("Model {} ({})", model, purpose));
                eprintln!("  Fix: ollama pull {}", model);
            }
        }
    }

    // Summary
    println!();
    println!(
        "Results: {} passed, {} skipped, {} warning(s), {} failed",
        counts.passes, counts.skips, counts.warnings, counts.failures
    );
    if counts.failures > 0 {
        println!(
            "{}",
            "Some checks failed. See above for details."
                .yellow()
                .bold()
        );
    } else if counts.warnings > 0 {
        println!(
            "{}",
            format!(
                "Required checks passed, with {} warning(s): see WARN above",
                counts.warnings
            )
            .yellow()
            .bold()
        );
    } else if counts.skips > 0 {
        println!(
            "{}",
            format!(
                "All required checks passed ({} optional check(s) skipped)",
                counts.skips
            )
            .green()
            .bold()
        );
    } else {
        println!("{}", "All checks passed!".green().bold());
    }

    if counts.failures > 0 {
        anyhow::bail!("{} doctor check(s) failed", counts.failures);
    }
    Ok(())
}

/// `agenthandover doctor --extension-json`: whether the browser extension
/// is built and which browser profiles have it (what the app's setup shows).
pub fn print_extension_json() -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&browser_extension::Report::gather())?);
    Ok(())
}

/// `agenthandover doctor --ollama-json`: the Ollama and memory report the
/// app's onboarding shows (the same findings `doctor` prints).
pub fn print_ollama_json() -> Result<()> {
    let config_path = agenthandover_common::status::status_dir().join("config.toml");
    let config_str = std::fs::read_to_string(config_path).unwrap_or_default();
    let host_url = read_config_string_field(&config_str, "vlm", "ollama_host").unwrap_or_default();
    let report = machine::gather(&host_url, &configured_models(&config_str));
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// The config's tier and models, as the worker resolves them.
fn configured_models(config_str: &str) -> machine::ConfiguredModels {
    let (tier, annotation, sop) = tiers::resolve_models(
        read_config_string_field(config_str, "vlm", "tier").as_deref(),
        read_config_string_field(config_str, "vlm", "annotation_model").as_deref(),
        read_config_string_field(config_str, "vlm", "sop_model").as_deref(),
    );
    machine::ConfiguredModels {
        tier_label: tier.label.clone(),
        annotation,
        sop,
        memory_enabled: read_config_string_field(config_str, "memory", "enabled")
            .is_some_and(|v| v == "true"),
    }
}

/// What doctor says about `[privacy] read_app_state`: nothing when it is
/// off; otherwise which apps macOS lets AgentHandover read (the daemon
/// records the answer it gets, asked without a prompt, per app).
fn app_state_findings(
    enabled: bool,
    consent: Option<&std::collections::BTreeMap<String, AutomationConsent>>,
) -> Vec<machine::Finding> {
    use machine::{Finding, Level};
    if !enabled {
        return vec![];
    }
    let Some(consent) = consent.filter(|c| !c.is_empty()) else {
        return vec![Finding {
            level: Level::Pass,
            title: "Reading open documents (on; no supported app opened yet)".into(),
            lines: vec![],
        }];
    };
    let granted: Vec<&str> = consent
        .iter()
        .filter(|(_, c)| **c == AutomationConsent::Granted)
        .map(|(a, _)| a.as_str())
        .collect();
    let mut lines = Vec::new();
    for (app, c) in consent {
        match c {
            AutomationConsent::Denied => lines.push(format!(
                "{app}: turned off. Fix: System Settings > Privacy & Security > Automation > \
                 AgentHandover: turn on {app}."
            )),
            AutomationConsent::NotAsked => lines.push(format!(
                "{app}: macOS has not asked yet; AgentHandover asks the next time you switch \
                 to {app} while it runs."
            )),
            AutomationConsent::Unknown => {
                lines.push(format!("{app}: macOS gave no answer; it is not read."))
            }
            AutomationConsent::Granted | AutomationConsent::NotRunning => {}
        }
    }
    if lines.is_empty() {
        let title = if granted.is_empty() {
            "Reading open documents (on)".to_string()
        } else {
            format!("Reading open documents (on: {})", granted.join(", "))
        };
        return vec![Finding { level: Level::Pass, title, lines }];
    }
    vec![Finding {
        level: Level::Warn,
        title: format!("Reading open documents: {} app(s) not granted", lines.len()),
        lines,
    }]
}

/// A warning when `[vlm]` still says `mode = "remote"`: cloud mode was
/// removed in 0.5.0 and the upgrade's migration
/// (`model_profiles.migrate_vlm_block`) did not rewrite the block.  The
/// worker runs the local models either way.  Nothing otherwise.
fn cloud_mode_findings(config_str: &str) -> Vec<machine::Finding> {
    let Some(mode) = read_config_string_field(config_str, "vlm", "mode") else {
        return vec![];
    };
    if mode == "local" {
        return vec![];
    }
    vec![machine::Finding {
        level: machine::Level::Warn,
        title: format!("Config still selects cloud mode ([vlm] mode = \"{mode}\")"),
        lines: vec![
            "Cloud mode was removed in 0.5.0: every model runs locally through Ollama, \
             whatever this key says. The upgrade could not rewrite config.toml (edited by hand?)."
                .into(),
            "Fix: under [vlm] in config.toml set mode = \"local\" and delete provider, model \
             and api_key_env, or run: agenthandover setup --vlm"
                .into(),
        ],
    }]
}

/// A warning when `[vlm]` (local mode) pins earlier default models: one
/// of its tier's earlier pairs, or (no `tier` key) a pair v0.4.0 wrote,
/// whose tier comes from `ram_gb` as the worker picks it.  The upgrade's
/// switch (`model_profiles.migrate_vlm_block`) waits until Ollama serves
/// the new models; the app offers the download, `setup --tier` makes it.
/// Nothing otherwise.
fn tier_migration_findings(config_str: &str, ram_gb: u64) -> Vec<machine::Finding> {
    let field = |key: &str| read_config_string_field(config_str, "vlm", key);
    if field("mode").is_some_and(|m| m != "local") {
        return vec![];
    }
    let (Some(annotation), Some(sop)) = (field("annotation_model"), field("sop_model")) else {
        return vec![];
    };
    let table = tiers::table();
    let (tier, title) = match field("tier") {
        Some(id) => match table.get(&id) {
            Some(tier) if tier.is_previous_default(&annotation, &sop) => (
                tier,
                format!("Config pins the {} tier's earlier models ({annotation} + {sop})", tier.label),
            ),
            _ => return vec![],
        },
        None => match table.untiered_default_tier(&annotation, &sop, ram_gb) {
            Some(tier) => (tier, format!("Config pins 0.4.0's models ({annotation} + {sop})")),
            None => return vec![],
        },
    };
    vec![machine::Finding {
        level: machine::Level::Warn,
        title,
        lines: vec![
            format!(
                "The {} tier now runs {} + {}; the switch waits until they are downloaded \
                 (AgentHandover offers the download).",
                tier.label, tier.annotation_model, tier.sop_model
            ),
            format!("Fix: agenthandover setup --tier {}", tier.id),
        ],
    }]
}

fn print_findings(counts: &mut CheckCounts, findings: &[machine::Finding]) {
    for f in findings {
        match f.level {
            machine::Level::Pass => check_pass(counts, &f.title),
            machine::Level::Warn => check_warn(counts, &f.title),
            machine::Level::Fail => check_fail(counts, &f.title),
        }
        for line in &f.lines {
            eprintln!("  {line}");
        }
    }
}

fn check_capture_servers(counts: &mut CheckCounts) {
    let obs_sock = observation_socket_path();
    match snapshot_at(&obs_sock) {
        Ok(snap) => {
            check_pass(
                counts,
                &format!("Observation server ({})", obs_sock.display()),
            );
            if snap.accessibility_granted {
                check_pass(counts, "Accessibility permission (AgentHandover.app)");
            } else {
                check_fail(counts, "Accessibility permission (AgentHandover.app)");
                eprintln!(
                    "  Fix: System Settings > Privacy & Security > Accessibility: enable AgentHandover"
                );
            }
            if snap.screen_recording_granted {
                check_pass(counts, "Screen Recording permission (AgentHandover.app)");
            } else {
                check_fail(counts, "Screen Recording permission (AgentHandover.app)");
                eprintln!(
                    "  Fix: System Settings > Privacy & Security > Screen Recording: enable AgentHandover"
                );
            }
        }
        Err(e) => {
            check_fail(
                counts,
                &format!("Observation server ({})", obs_sock.display()),
            );
            eprintln!(
                "  AgentHandover.app is not running or its capture server failed to bind \
                 — the daemon captures nothing without it ({})",
                e
            );
            eprintln!("  Fix: open AgentHandover.app (it lives in /Applications)");
        }
    }

    let cap_sock = capture_socket_path();
    if is_socket(&cap_sock) {
        check_pass(counts, &format!("Capture socket ({})", cap_sock.display()));
    } else {
        check_fail(counts, &format!("Capture socket ({})", cap_sock.display()));
        eprintln!("  AgentHandover.app serves screenshots on this socket; open the app.");
    }
}

#[cfg(unix)]
fn is_socket(path: &std::path::Path) -> bool {
    use std::os::unix::fs::FileTypeExt;
    std::fs::metadata(path)
        .map(|m| m.file_type().is_socket())
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_socket(_path: &std::path::Path) -> bool {
    false
}

/// Python run in the venv: the worker and its MCP dependency must import,
/// and it prints the installed worker version.
const VENV_PROBE: &str = "import importlib.metadata as m, agenthandover_worker, mcp; \
                          print(m.version('agenthandover-worker'))";

fn check_venv(counts: &mut CheckCounts, channel: &str) {
    const NAME: &str = "Python virtual environment";
    let lib_dir = paths::lib_dir();
    let Some(python) = paths::find_venv_python() else {
        if lib_dir.exists() {
            check_fail(counts, NAME);
            eprintln!("  {} has no working venv/bin/python", lib_dir.display());
            print_install_hint(channel);
        } else {
            let hint = match channel {
                "homebrew" => "brew reinstall agenthandover",
                "source" => "venv not found — expected for source builds",
                _ => "Re-install AgentHandover (.pkg recommended)",
            };
            check_skip(counts, NAME, hint);
        }
        return;
    };

    let probe = std::process::Command::new(&python)
        .args(["-c", VENV_PROBE])
        .output();
    let outcome = match probe {
        Ok(out) if out.status.success() => {
            venv_version_verdict(&String::from_utf8_lossy(&out.stdout), env!("CARGO_PKG_VERSION"))
        }
        Ok(out) => Err(format!(
            "worker does not import: {}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .last()
                .unwrap_or("unknown error")
        )),
        Err(e) => Err(format!("cannot run {}: {}", python.display(), e)),
    };
    match outcome {
        Ok(version) => check_pass(counts, &format!("{} (worker {})", NAME, version)),
        Err(reason) => {
            check_fail(counts, &format!("{} ({})", NAME, reason));
            print_install_hint(channel);
        }
    }
}

/// Compare the venv's reported worker version with the CLI's own.
fn venv_version_verdict(probe_stdout: &str, cli_version: &str) -> Result<String, String> {
    let worker_version = probe_stdout.trim();
    if worker_version == cli_version {
        Ok(worker_version.to_string())
    } else {
        Err(format!(
            "worker {} does not match CLI {}",
            if worker_version.is_empty() { "?" } else { worker_version },
            cli_version
        ))
    }
}

fn check_pass(counts: &mut CheckCounts, name: &str) {
    counts.record_pass();
    println!("  {} {}", "pass".green(), name);
}

fn check_fail(counts: &mut CheckCounts, name: &str) {
    counts.record_fail();
    println!("  {} {}", "FAIL".red(), name);
}

fn check_warn(counts: &mut CheckCounts, name: &str) {
    counts.warnings += 1;
    println!("  {} {}", "WARN".yellow(), name);
}

fn check_skip(counts: &mut CheckCounts, name: &str, reason: &str) {
    counts.record_skip();
    println!("  {} {} ({})", "SKIP".yellow(), name, reason);
}

/// "Upgrade everything" (the worker's re-embed with EmbeddingGemma 2), in
/// one line when it was asked for: a warning when it failed (text search
/// stays on its current model until it is fixed), else information.
fn check_embedding_upgrade(counts: &mut CheckCounts) {
    let Ok(worker) = agenthandover_common::status::read_status_file::<serde_json::Value>(
        "worker-status.json",
    ) else {
        return;
    };
    let Some(line) = super::status::embedding_upgrade_line(&worker) else {
        return;
    };
    let name = format!("Embedding upgrade: {line}");
    if line.starts_with("failed") {
        check_warn(counts, &name);
    } else {
        println!("  {} {}", "info".dimmed(), name);
    }
}

/// Advisory heartbeat freshness check for a service status file
/// (`~/Library/Application Support/agenthandover/<filename>`).  Never
/// affects the overall doctor result.
fn check_heartbeat_freshness(service_name: &str, status_filename: &str) {
    let label = format!("{} heartbeat", service_name);
    let path = agenthandover_common::status::status_dir().join(status_filename);
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => Some(c),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(_) => {
            println!("  {} {} (Cannot read status file)", "info".dimmed(), label);
            return;
        }
    };
    let (level, detail) = heartbeat_verdict(
        content.as_deref(),
        Utc::now(),
        agenthandover_common::pid::is_process_running,
    );
    let tag = match level {
        HeartbeatLevel::Pass => "pass".green(),
        HeartbeatLevel::Info => "info".dimmed(),
        HeartbeatLevel::Warn => "WARN".yellow(),
    };
    println!("  {} {} ({})", tag, label, detail);
}

#[derive(Debug, PartialEq)]
enum HeartbeatLevel {
    Pass,
    Info,
    Warn,
}

/// Classifies a status file's heartbeat.  `content` is None when the file
/// does not exist.  A status file whose `pid` is not a live process is left
/// over from an earlier run (an upgrade, a crash, a stop): the service is
/// not running, and the age of that file's heartbeat means nothing.
fn heartbeat_verdict(
    content: Option<&str>,
    now: chrono::DateTime<Utc>,
    pid_alive: impl Fn(u32) -> bool,
) -> (HeartbeatLevel, String) {
    let info = |m: &str| (HeartbeatLevel::Info, m.to_string());
    let Some(content) = content else {
        return info("No status file yet (service may not have started)");
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(content) else {
        return info("Invalid JSON in status file");
    };
    let pid = parsed
        .get("pid")
        .and_then(|v| v.as_u64())
        .and_then(|p| u32::try_from(p).ok());
    if !pid.map(&pid_alive).unwrap_or(false) {
        return info("not running");
    }
    let Some(heartbeat_str) = parsed.get("heartbeat").and_then(|v| v.as_str()) else {
        return info("No heartbeat field in status file");
    };
    let Ok(heartbeat) = chrono::DateTime::parse_from_rfc3339(heartbeat_str) else {
        return info("Cannot parse heartbeat timestamp");
    };
    let age_secs = now
        .signed_duration_since(heartbeat.with_timezone(&Utc))
        .num_seconds();
    if age_secs <= 60 {
        (HeartbeatLevel::Pass, format!("{}s ago", age_secs))
    } else {
        (
            HeartbeatLevel::Warn,
            format!("last heartbeat {}s ago — service may be hung", age_secs),
        )
    }
}

// Path resolution functions (find_homebrew_libexec, find_venv_python,
// find_extension_dir, find_local_extension_dist, find_daemon_binary)
// live in crate::paths. Don't duplicate them here.
//
// The dead helpers `which`, `source_build_daemon_exists`,
// `accessibility_sys_check`, and `screen_recording_check` were removed
// in v0.2.3. The first two are subsumed by paths::find_daemon_binary;
// the last two checked TCC permissions for the CLI process instead of
// the app bundle (wrong semantics) and were replaced with advisory
// info output — see Check 5/6 above.

/// Extract a string field from a TOML config file without pulling in
/// a TOML parser dependency. Handles:
///
/// - Section headers (`[section]`)
/// - `key = "value"` (quoted)
/// - `key = value` (unquoted, e.g. numbers or booleans)
/// - `#` comments (whole line or trailing)
/// - Empty lines
///
/// Returns `None` if the section or key isn't found. Does NOT handle
/// nested tables, arrays, or multi-line strings — only flat scalar
/// fields, which is all doctor actually needs to read. If you need
/// anything more, add the `toml` crate as a dependency instead.
fn read_config_string_field(config: &str, section: &str, key: &str) -> Option<String> {
    let target_header = format!("[{}]", section);
    let mut in_section = false;
    for line in config.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Section header
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_section = trimmed == target_header;
            continue;
        }
        if !in_section {
            continue;
        }
        // key = value
        if let Some((k, v)) = trimmed.split_once('=') {
            if k.trim() == key {
                let v = v.trim();
                // Strip trailing comment (after the value)
                let v = v.split('#').next().unwrap_or(v).trim();
                // Strip surrounding quotes
                let v = v.trim_matches(|c| c == '"' || c == '\'');
                return Some(v.to_string());
            }
        }
    }
    None
}

fn native_messaging_manifest_path() -> std::path::PathBuf {
    // Check all supported Chromium browsers, return the first found manifest
    let manifest_name = "com.agenthandover.host.json";
    for dir in crate::paths::native_messaging_hosts_dirs() {
        let path = dir.join(manifest_name);
        if path.exists() {
            return path;
        }
    }
    // Fallback to Chrome default for error reporting
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    std::path::PathBuf::from(home).join(
        "Library/Application Support/Google/Chrome/NativeMessagingHosts/com.agenthandover.host.json",
    )
}

/// `AH_LAUNCH_AGENTS` (a temporary-prefix install, as in postinstall) or
/// `~/Library/LaunchAgents`.
fn launch_agents_dir() -> std::path::PathBuf {
    if let Some(dir) = std::env::var_os("AH_LAUNCH_AGENTS").filter(|d| !d.is_empty()) {
        return std::path::PathBuf::from(dir);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    std::path::PathBuf::from(home).join("Library/LaunchAgents")
}

fn free_disk_gb() -> u64 {
    machine::free_disk_now().unwrap_or(0) / (1024 * 1024 * 1024)
}

fn check_native_messaging_manifest() -> bool {
    let manifest_path = native_messaging_manifest_path();
    if !manifest_path.exists() {
        println!(
            "    {} Manifest not found at: {}",
            "→".dimmed(),
            manifest_path.display()
        );
        println!(
            "    Run {} to install it.",
            "agenthandover setup --extension".cyan()
        );
        return false;
    }

    // Read and parse
    let content = match std::fs::read_to_string(&manifest_path) {
        Ok(c) => c,
        Err(e) => {
            println!("    {} Cannot read manifest: {}", "→".dimmed(), e);
            return false;
        }
    };

    let parsed: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            println!("    {} Invalid JSON in manifest: {}", "→".dimmed(), e);
            return false;
        }
    };

    // Verify name
    let name = parsed.get("name").and_then(|v| v.as_str()).unwrap_or("");
    if name != "com.agenthandover.host" {
        println!(
            "    {} Wrong host name: expected 'com.agenthandover.host', got '{}'",
            "→".dimmed(),
            name
        );
        return false;
    }

    // Verify path points to existing executable
    let daemon_path_str = parsed.get("path").and_then(|v| v.as_str()).unwrap_or("");
    if daemon_path_str.is_empty() {
        println!("    {} Manifest 'path' is empty", "→".dimmed());
        return false;
    }
    let daemon_path = std::path::Path::new(daemon_path_str);
    if !daemon_path.exists() {
        println!(
            "    {} Daemon binary not found at manifest path: {}",
            "→".dimmed(),
            daemon_path_str
        );
        println!(
            "    Re-run {} to update.",
            "agenthandover setup --extension".cyan()
        );
        return false;
    }

    // Verify allowed_origins contains expected extension ID
    let origins = parsed.get("allowed_origins").and_then(|v| v.as_array());
    let expected_origin = format!("chrome-extension://{}/", browser_extension::EXTENSION_ID);
    let has_origin = origins
        .map(|arr| arr.iter().any(|v| v.as_str() == Some(expected_origin.as_str())))
        .unwrap_or(false);
    if !has_origin {
        println!(
            "    {} allowed_origins missing expected extension ID",
            "→".dimmed()
        );
        return false;
    }

    true
}

/// The extension directory doctor found (what Load unpacked takes, and
/// what `agenthandover setup --extension` copies), under its check line.
fn print_extension_path(extension: &browser_extension::Report) {
    if let Some(path) = &extension.build_path {
        println!("    → {}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn venv_version_matches_cli() {
        assert_eq!(venv_version_verdict("0.4.0\n", "0.4.0"), Ok("0.4.0".to_string()));
    }

    #[test]
    fn venv_version_mismatch_fails() {
        let err = venv_version_verdict("0.3.2\n", "0.4.0").unwrap_err();
        assert!(err.contains("0.3.2") && err.contains("0.4.0"), "{err}");
    }

    #[test]
    fn venv_version_empty_output_fails() {
        assert!(venv_version_verdict("", "0.4.0").is_err());
    }

    fn now() -> chrono::DateTime<Utc> {
        chrono::DateTime::parse_from_rfc3339("2026-09-23T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn stale_status_file_of_a_dead_process_reads_not_running() {
        // A 177-day-old status file left by the previous install.
        let old = r#"{"pid": 4242, "heartbeat": "2026-03-30T12:00:00Z"}"#;
        assert_eq!(
            heartbeat_verdict(Some(old), now(), |_| false),
            (HeartbeatLevel::Info, "not running".to_string())
        );
    }

    #[test]
    fn live_process_heartbeat_age_is_reported() {
        let fresh = r#"{"pid": 4242, "heartbeat": "2026-09-23T11:59:50Z"}"#;
        assert_eq!(
            heartbeat_verdict(Some(fresh), now(), |p| p == 4242),
            (HeartbeatLevel::Pass, "10s ago".to_string())
        );
        let hung = r#"{"pid": 4242, "heartbeat": "2026-09-23T11:50:00Z"}"#;
        let (level, detail) = heartbeat_verdict(Some(hung), now(), |_| true);
        assert_eq!(level, HeartbeatLevel::Warn);
        assert!(detail.contains("600s ago"), "{detail}");
    }

    #[test]
    fn app_state_reads_report_the_apps_not_granted() {
        use std::collections::BTreeMap;
        use AutomationConsent::*;
        assert!(app_state_findings(false, None).is_empty());
        let none = app_state_findings(true, None);
        assert_eq!(none[0].level, machine::Level::Pass);

        let mut seen = BTreeMap::new();
        seen.insert("Notes".to_string(), Granted);
        seen.insert("Safari".to_string(), NotRunning);
        let ok = app_state_findings(true, Some(&seen));
        assert_eq!(ok[0].level, machine::Level::Pass);
        assert_eq!(ok[0].title, "Reading open documents (on: Notes)");

        seen.insert("Mail".to_string(), Denied);
        seen.insert("TextEdit".to_string(), NotAsked);
        let warn = app_state_findings(true, Some(&seen));
        assert_eq!(warn[0].level, machine::Level::Warn);
        assert_eq!(warn[0].title, "Reading open documents: 2 app(s) not granted");
        assert_eq!(
            warn[0].lines,
            [
                "Mail: turned off. Fix: System Settings > Privacy & Security > Automation > \
                 AgentHandover: turn on Mail.",
                "TextEdit: macOS has not asked yet; AgentHandover asks the next time you switch \
                 to TextEdit while it runs.",
            ]
        );
    }

    /// `[memory] enabled = true` asks for the shipped table's Memory
    /// minimum: doctor warns "Memory needs Ollama 0.40.0 or later" on an
    /// older service, and says nothing when Memory is off or absent.
    #[test]
    fn memory_on_warns_about_an_ollama_too_old_for_it() {
        let warning = |config: &str, version: &str| {
            machine::diagnose_memory(
                Some(version),
                &tiers::table().minimum_ollama_version_for_memory,
                configured_models(config).memory_enabled,
            )
        };
        let on = "[vlm]\ntier = \"recommended\"\n\n[memory]\nenabled = true  # setup\n";
        let f = warning(on, "0.34.2").expect("a warning");
        assert_eq!(f.level, machine::Level::Warn);
        assert_eq!(f.title, "Memory needs Ollama 0.40.0 or later (you have 0.34.2)");
        assert_eq!(warning(on, "0.40.3"), None);
        assert_eq!(warning("[memory]\nenabled = false\n", "0.34.2"), None);
        assert_eq!(warning("[vlm]\ntier = \"recommended\"\n", "0.34.2"), None);
        // `enabled` under another section is not Memory's.
        assert_eq!(warning("[privacy]\nenabled = true\n", "0.34.2"), None);
    }

    /// An old Ollama fails only when the models the worker would use (the
    /// config's explicit models, else its tier's, else the RAM tier's)
    /// include a Gemma 4 tag; the shipped tier table decides the tiers.
    #[test]
    fn an_old_ollama_fails_only_for_configured_gemma4_models() {
        let level = |config: &str| {
            let m = configured_models(config);
            let host = machine::parse_host("http://localhost:11434");
            let findings = machine::diagnose_service(&machine::ServiceFacts {
                host: &host,
                version: Some("0.30.6"),
                minimum: &tiers::table().minimum_ollama_version_for_gemma4,
                models: &[m.annotation.as_str(), m.sop.as_str()],
                installs: &[],
                launch_items: &[],
                serving: None,
            });
            assert_eq!(findings[0].title, "Ollama 0.30.6 is older than 0.34.0");
            findings[0].level
        };
        use machine::Level::{Fail, Warn};
        assert_eq!(level("[vlm]\ntier = \"standard\"\n"), Warn);
        assert_eq!(level("[vlm]\ntier = \"recommended\"\n"), Warn);
        assert_eq!(level("[vlm]\ntier = \"performance\"\n"), Fail);
        assert_eq!(level("[vlm]\ntier = \"max_quality\"\n"), Fail);
        // Explicit models win over the tier, either way.
        assert_eq!(level("[vlm]\ntier = \"recommended\"\nsop_model = \"gemma4:12b-it-qat\"\n"), Fail);
        assert_eq!(
            level("[vlm]\ntier = \"performance\"\nannotation_model = \"qwen3.5:4b\"\nsop_model = \"qwen3.5:4b\"\n"),
            Warn
        );
        // No [vlm] section: this Mac's RAM tier, as the worker resolves it.
        let ram_tier = tiers::recommended_tier();
        let expected = if ollama::is_gemma4(&ram_tier.annotation_model) || ollama::is_gemma4(&ram_tier.sop_model) {
            Fail
        } else {
            Warn
        };
        assert_eq!(level(""), expected);
    }

    #[test]
    fn a_config_on_a_tiers_earlier_models_warns_with_the_fix() {
        let old = "[vlm]\nmode = \"local\"\ntier = \"recommended\"\n\
                   annotation_model = \"qwen3.5:4b\"\nsop_model = \"gemma4:12b-it-qat\"\n";
        let findings = tier_migration_findings(old, 16);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].level, machine::Level::Warn);
        assert_eq!(
            findings[0].title,
            "Config pins the Recommended tier's earlier models (qwen3.5:4b + gemma4:12b-it-qat)"
        );
        assert_eq!(
            findings[0].lines,
            [
                "The Recommended tier now runs qwen3.5:4b + qwen3.5:4b; the switch waits until \
                 they are downloaded (AgentHandover offers the download).",
                "Fix: agenthandover setup --tier recommended",
            ]
        );
        // No mode key means local.
        assert_eq!(tier_migration_findings(&old.replace("mode = \"local\"\n", ""), 16).len(), 1);

        // Migrated, customised, remote, another or unknown tier, no pins: silent.
        for config in [
            old.replace("gemma4:12b-it-qat", "qwen3.5:4b"),
            old.replace("gemma4:12b-it-qat", "gemma4:31b-it-qat"),
            old.replace("\"local\"", "\"remote\""),
            old.replace("\"recommended\"", "\"performance\""),
            old.replace("\"recommended\"", "\"huge\""),
            old.replace("sop_model = \"gemma4:12b-it-qat\"\n", ""),
            String::new(),
        ] {
            assert!(tier_migration_findings(&config, 16).is_empty(), "{config}");
        }
    }

    /// WP-F16: the file v0.4.0 left (no `tier` key) warns, naming the tier
    /// this Mac's RAM picks for it and the fix; another pair does not.
    #[test]
    fn a_v040_config_warns_with_the_tier_its_ram_picks() {
        let v040 = std::fs::read_to_string(format!(
            "{}/../../tests/fixtures/vlm_block/v0.4.0-onboarding.toml",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        for (ram, label, runs, id) in [
            (16, "Recommended", "qwen3.5:4b + qwen3.5:4b", "recommended"),
            (24, "Performance", "qwen3.5:4b + gemma4:12b-it-qat", "performance"),
        ] {
            let findings = tier_migration_findings(&v040, ram);
            assert_eq!(findings.len(), 1, "{ram} GB");
            assert_eq!(findings[0].level, machine::Level::Warn);
            assert_eq!(
                findings[0].title,
                "Config pins 0.4.0's models (gemma4:12b-it-qat + gemma4:12b-it-qat)"
            );
            assert_eq!(
                findings[0].lines,
                [
                    format!(
                        "The {label} tier now runs {runs}; the switch waits until they are \
                         downloaded (AgentHandover offers the download)."
                    ),
                    format!("Fix: agenthandover setup --tier {id}"),
                ]
            );
        }
        let custom = v040.replace(
            "sop_model = \"gemma4:12b-it-qat\"",
            "sop_model = \"gemma4:31b-it-qat\"",
        );
        assert!(tier_migration_findings(&custom, 24).is_empty());
    }

    /// WP-F13: a `[vlm]` block the upgrade could not move off cloud mode
    /// warns with the fix; a local block or none says nothing.
    #[test]
    fn a_config_still_in_cloud_mode_warns_with_the_fix() {
        let remote = "[vlm]\nmode = \"remote\"\nprovider = \"openai\"\nmodel = \"gpt-4.1-mini\"\n\
                      api_key_env = \"OPENAI_API_KEY\"\n";
        let findings = cloud_mode_findings(remote);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].level, machine::Level::Warn);
        assert_eq!(findings[0].title, "Config still selects cloud mode ([vlm] mode = \"remote\")");
        assert!(findings[0].lines[1].starts_with("Fix: under [vlm] in config.toml set mode = \"local\""));
        for config in [
            "[vlm]\nmode = \"local\"\ntier = \"standard\"\n",
            "[vlm]\ntier = \"standard\"\n",
            "[embedding]\nmode = \"remote\"\n",
            "",
        ] {
            assert!(cloud_mode_findings(config).is_empty(), "{config}");
        }
    }

    /// WP-F8: an 8 GB config still annotating with qwen3.5:4b warns; the
    /// migrated block (qwen3.5:2b-q4_K_M) and a customised one do not.
    #[test]
    fn a_config_on_the_standard_tiers_earlier_models_warns() {
        let old = "[vlm]\nmode = \"local\"\ntier = \"standard\"\n\
                   annotation_model = \"qwen3.5:4b\"\nsop_model = \"qwen3.5:4b\"\n";
        let findings = tier_migration_findings(old, 16);
        assert_eq!(findings.len(), 1);
        assert_eq!(
            findings[0].title,
            "Config pins the Standard tier's earlier models (qwen3.5:4b + qwen3.5:4b)"
        );
        assert_eq!(
            findings[0].lines,
            [
                "The Standard tier now runs qwen3.5:2b-q4_K_M + qwen3.5:4b; the switch waits \
                 until they are downloaded (AgentHandover offers the download).",
                "Fix: agenthandover setup --tier standard",
            ]
        );
        for config in [
            old.replace("annotation_model = \"qwen3.5:4b\"", "annotation_model = \"qwen3.5:2b-q4_K_M\""),
            old.replace("sop_model = \"qwen3.5:4b\"", "sop_model = \"gemma4:12b-it-qat\""),
            old.replace("\"standard\"", "\"recommended\""),
        ] {
            assert!(tier_migration_findings(&config, 16).is_empty(), "{config}");
        }
    }

    /// Doctor's memory need for the shipped 8 GB tier: the larger of its two
    /// models (qwen3.5:4b, 3.4 GB; the 2B is 1.9 GB), the embedding model
    /// and the margin, since the worker loads them one at a time.
    #[test]
    fn the_standard_tiers_memory_need_is_the_skill_model() {
        let t = tiers::table();
        let standard = t.get("standard").unwrap();
        assert_eq!(
            machine::need_bytes(t, &[], &standard.annotation_model, &standard.sop_model),
            4_670_000_000
        );
    }

    #[test]
    fn missing_file_and_missing_pid() {
        assert_eq!(heartbeat_verdict(None, now(), |_| true).0, HeartbeatLevel::Info);
        assert_eq!(
            heartbeat_verdict(Some(r#"{"heartbeat": "2026-09-23T11:59:50Z"}"#), now(), |_| true),
            (HeartbeatLevel::Info, "not running".to_string())
        );
    }
}
