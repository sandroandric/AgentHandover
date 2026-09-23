//! This Mac as the local models see it: which Ollama installs exist, which
//! one serves the port, and whether the chosen tier fits in memory now.
//!
//! People run Ollama.app, Homebrew's formula, both, a custom build or a
//! remote host, next to whatever else they use.  Nothing here assumes one
//! setup: it detects what is there and explains it.  `doctor` prints the
//! findings; the app's onboarding reads the same report as JSON
//! (`agenthandover doctor --ollama-json`), so both show the same words.
//!
//! Every probe of the filesystem goes through [`Roots`] and every probe of
//! a running process or of memory is a separate function, so the logic
//! (`diagnose_*`) is tested against fake roots and recorded command output.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use agenthandover_common::tiers::TierTable;

use crate::ollama;

/// Decimal, as the tier table and `ollama list` count.
pub const GB: u64 = 1_000_000_000;
/// Over the largest model's size: its context (the tier table lists
/// download sizes).  The worker's `model_scheduler.MEMORY_MARGIN_BYTES`.
pub const MEMORY_MARGIN_BYTES: u64 = GB;
/// Swap in use above this: the Mac is short of memory, and the worker does
/// not load the Skills model (`model_scheduler.SWAP_LIMIT_BYTES`).
pub const SWAP_LIMIT_BYTES: u64 = 8 * GB;
/// Free disk below this: swap cannot grow (`model_scheduler.DISK_FLOOR_BYTES`).
pub const DISK_FLOOR_BYTES: u64 = 10 * GB;

// ---------------------------------------------------------------------------
// Where Ollama can be installed
// ---------------------------------------------------------------------------

/// Where to look for installs and launch items.
pub struct Roots {
    /// Candidate `Ollama.app` bundles.
    pub apps: Vec<PathBuf>,
    /// Homebrew prefixes (`/opt/homebrew`, `/usr/local`, `brew --prefix`).
    pub brew_prefixes: Vec<PathBuf>,
    /// Directories of launchd plists.
    pub launch_dirs: Vec<PathBuf>,
}

impl Roots {
    pub fn this_mac() -> Self {
        let home = PathBuf::from(std::env::var("HOME").unwrap_or_default());
        let mut brew_prefixes: Vec<PathBuf> = Vec::new();
        // `brew` is often not on the PATH of a GUI-launched process; the two
        // standard prefixes cover that case.
        if let Ok(out) = Command::new("brew").arg("--prefix").output() {
            let prefix = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if out.status.success() && !prefix.is_empty() {
                brew_prefixes.push(PathBuf::from(prefix));
            }
        }
        for p in ["/opt/homebrew", "/usr/local"] {
            let p = PathBuf::from(p);
            if !brew_prefixes.contains(&p) {
                brew_prefixes.push(p);
            }
        }
        Roots {
            apps: vec![
                PathBuf::from("/Applications/Ollama.app"),
                home.join("Applications/Ollama.app"),
            ],
            brew_prefixes,
            launch_dirs: vec![
                home.join("Library/LaunchAgents"),
                PathBuf::from("/Library/LaunchAgents"),
                PathBuf::from("/Library/LaunchDaemons"),
            ],
        }
    }
}

/// One Ollama installation.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Install {
    /// `Ollama.app` or `Homebrew`.
    pub kind: &'static str,
    /// The app bundle, or the Homebrew keg (`<prefix>/Cellar/ollama/<version>`).
    pub path: PathBuf,
    pub version: Option<String>,
}

pub const APP: &str = "Ollama.app";
pub const HOMEBREW: &str = "Homebrew";

impl Install {
    /// "Ollama.app 0.30.6" / "Homebrew 0.34.2".
    fn name(&self) -> String {
        match &self.version {
            Some(v) => format!("{} {}", self.kind, v),
            None => self.kind.to_string(),
        }
    }

    /// Whether `executable` (canonical) belongs to this install.
    fn owns(&self, executable: &Path) -> bool {
        let root = canonical(&self.path);
        match self.kind {
            // Any keg of the formula: a service can still run an older keg.
            HOMEBREW => root.parent().is_some_and(|cellar| executable.starts_with(cellar)),
            _ => executable.starts_with(&root),
        }
    }
}

/// A launchd plist that starts an Ollama server.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LaunchItem {
    pub label: String,
    pub plist: PathBuf,
    /// What it runs (`Program`, else the first of `ProgramArguments`).
    pub program: Option<PathBuf>,
}

/// Every install under `roots`: each Ollama.app bundle, and the Homebrew
/// formula's keg in each prefix (the one `opt/ollama` links to, else the
/// highest version).
pub fn find_installs(roots: &Roots) -> Vec<Install> {
    let mut out = Vec::new();
    for app in &roots.apps {
        let info = app.join("Contents/Info.plist");
        if info.exists() {
            let version =
                read_plist(&info).and_then(|x| plist_string(&x, "CFBundleShortVersionString"));
            out.push(Install { kind: APP, path: app.clone(), version });
        }
    }
    for prefix in &roots.brew_prefixes {
        let cellar = prefix.join("Cellar/ollama");
        let Ok(entries) = std::fs::read_dir(&cellar) else {
            continue;
        };
        let mut kegs: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        kegs.sort_by_key(|k| version_key(k));
        let linked = std::fs::canonicalize(prefix.join("opt/ollama"))
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .filter(|k| kegs.contains(k));
        let Some(keg) = linked.or_else(|| kegs.last().cloned()) else {
            continue;
        };
        // A keg is "<version>" or "<version>_<revision>".
        let version = keg.split('_').next().unwrap_or(&keg).to_string();
        out.push(Install { kind: HOMEBREW, path: cellar.join(&keg), version: Some(version) });
    }
    out
}

/// Every launchd plist under `roots` whose file name mentions Ollama
/// (`com.ollama.ollama`, `homebrew.mxcl.ollama`, a hand-written one ...).
pub fn find_launch_items(roots: &Roots) -> Vec<LaunchItem> {
    let mut out = Vec::new();
    for dir in &roots.launch_dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        let mut plists: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase());
                name.is_some_and(|n| n.contains("ollama") && n.ends_with(".plist"))
            })
            .collect();
        plists.sort();
        for plist in plists {
            let xml = read_plist(&plist).unwrap_or_default();
            let label = plist_string(&xml, "Label").unwrap_or_else(|| {
                plist.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
            });
            let program = plist_string(&xml, "Program")
                .or_else(|| plist_string(&xml, "ProgramArguments"))
                .map(PathBuf::from);
            out.push(LaunchItem { label, plist, program });
        }
    }
    out
}

/// Sort key of a version or keg name: its numbers.
fn version_key(v: &str) -> Vec<u64> {
    v.split(['.', '_', '-']).map(|p| p.parse().unwrap_or(0)).collect()
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

/// A plist as XML text (binary plists are converted with `plutil`).
fn read_plist(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.starts_with(b"bplist") {
        let out = Command::new("plutil")
            .args(["-convert", "xml1", "-o", "-"])
            .arg(path)
            .output()
            .ok()?;
        return out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned());
    }
    String::from_utf8(bytes).ok()
}

/// The string value of `key` in plist XML, or the first string of its
/// array value.
fn plist_string(xml: &str, key: &str) -> Option<String> {
    let tag = format!("<key>{key}</key>");
    let rest = xml[xml.find(&tag)? + tag.len()..].trim_start();
    let rest = if let Some(array) = rest.strip_prefix("<array>") {
        let end = array.find("</array>")?;
        array[..end].trim_start()
    } else {
        rest
    };
    let value = rest.strip_prefix("<string>")?;
    let value = &value[..value.find("</string>")?];
    Some(
        value
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&amp;", "&"),
    )
}

// ---------------------------------------------------------------------------
// Which process serves the port
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Serving {
    pub pid: u32,
    /// The listening process's executable, symlinks resolved.
    pub executable: PathBuf,
}

/// The process listening on `port`, from `lsof` and `ps`.  None when
/// nothing listens or the process belongs to another user (`lsof` only
/// sees one's own processes without root).
pub fn serving_process(port: u16) -> Option<Serving> {
    let out = Command::new("lsof")
        .args(["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN", "-Fp"])
        .output()
        .ok()?;
    let pid = parse_lsof_pid(&String::from_utf8_lossy(&out.stdout))?;
    let out = Command::new("ps").args(["-p", &pid.to_string(), "-o", "comm="]).output().ok()?;
    let exe = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if exe.is_empty() {
        return None;
    }
    Some(Serving { pid, executable: canonical(Path::new(&exe)) })
}

/// The first pid of `lsof -F p` output (`p1234` lines).
pub fn parse_lsof_pid(out: &str) -> Option<u32> {
    out.lines().find_map(|l| l.strip_prefix('p')?.trim().parse().ok())
}

// ---------------------------------------------------------------------------
// The configured host
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Host {
    pub url: String,
    /// On this Mac: installs, the listening process and memory apply.
    pub local: bool,
    pub port: u16,
}

/// `[vlm] ollama_host` (the worker's setting), else the default.
pub fn parse_host(url: &str) -> Host {
    let url = url.trim().trim_end_matches('/');
    let url = if url.is_empty() { ollama::BASE_URL } else { url };
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    let authority = rest.split('/').next().unwrap_or("");
    let (name, port) = if let Some(v6) = authority.strip_prefix('[') {
        let (name, after) = v6.split_once(']').unwrap_or((v6, ""));
        (name, after.strip_prefix(':'))
    } else {
        match authority.rsplit_once(':') {
            Some((n, p)) => (n, Some(p)),
            None => (authority, None),
        }
    };
    let local = matches!(
        name.to_ascii_lowercase().as_str(),
        "localhost" | "127.0.0.1" | "::1" | "0.0.0.0" | ""
    );
    Host {
        url: url.to_string(),
        local,
        port: port.and_then(|p| p.parse().ok()).unwrap_or(11434),
    }
}

// ---------------------------------------------------------------------------
// Findings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Pass,
    Warn,
    Fail,
}

/// One check's outcome: a one-line title and the explanation and fix.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Finding {
    pub level: Level,
    pub title: String,
    pub lines: Vec<String>,
}

fn finding(level: Level, title: impl Into<String>, lines: Vec<String>) -> Finding {
    Finding { level, title: title.into(), lines }
}

/// What is known about the Ollama service.
pub struct ServiceFacts<'a> {
    pub host: &'a Host,
    /// `GET /api/version`; None when the service does not answer.
    pub version: Option<&'a str>,
    pub minimum: &'a str,
    pub installs: &'a [Install],
    pub launch_items: &'a [LaunchItem],
    pub serving: Option<&'a Serving>,
}

impl ServiceFacts<'_> {
    /// The install serving the port: the listening executable's owner, else
    /// the one install whose version the service reports.
    fn serving_install(&self) -> Option<&Install> {
        if let Some(s) = self.serving {
            if let Some(i) = self.installs.iter().find(|i| i.owns(&s.executable)) {
                return Some(i);
            }
        }
        let v = self.version?;
        let mut same = self.installs.iter().filter(|i| i.version.as_deref() == Some(v));
        match (same.next(), same.next()) {
            (Some(i), None) => Some(i),
            _ => None,
        }
    }

    /// "Ollama.app 0.30.6 (pid 812, /Applications/Ollama.app/Contents/Resources/ollama)".
    fn serving_description(&self) -> String {
        let who = self.serving_install().map(Install::name);
        match (who, self.serving) {
            (Some(who), Some(s)) => format!("{who} (pid {}, {})", s.pid, s.executable.display()),
            (Some(who), None) => who,
            (None, Some(s)) => format!("pid {}, {}", s.pid, s.executable.display()),
            (None, None) => "a process that could not be identified".to_string(),
        }
    }
}

/// A plist `brew services` wrote (`homebrew.mxcl.ollama`, `sh.brew.ollama`).
fn is_brew_service(item: &LaunchItem) -> bool {
    item.label.starts_with("homebrew.mxcl.") || item.label.starts_with("sh.brew.")
}

/// How to leave only `keep` installed and serving.
fn keep_only(keep: &Install, facts: &ServiceFacts) -> String {
    let mut steps: Vec<String> = Vec::new();
    for other in facts.installs.iter().filter(|i| *i != keep) {
        steps.push(match other.kind {
            HOMEBREW => "brew services stop ollama && brew uninstall ollama".to_string(),
            _ => format!(
                "quit {} (menu bar icon > Quit), remove it from System Settings > General > \
                 Login Items and move {} to the Trash",
                APP,
                other.path.display()
            ),
        });
    }
    for item in facts.launch_items {
        // `brew services` owns its own plist: kept with Homebrew, removed
        // by the `brew services stop` above otherwise.
        let brew_installed = facts.installs.iter().any(|i| i.kind == HOMEBREW);
        if is_brew_service(item) && brew_installed {
            continue;
        }
        let runs_keep = item.program.as_deref().is_some_and(|p| keep.owns(&canonical(p)));
        if !runs_keep {
            steps.push(format!(
                "launchctl bootout gui/$(id -u) {0} and delete {0}",
                item.plist.display()
            ));
        }
    }
    if keep.kind == HOMEBREW {
        steps.push("brew services restart ollama".to_string());
    } else {
        steps.push(format!("open {}", keep.path.display()));
    }
    format!("To keep {}: {}.", keep.name(), steps.join("; then "))
}

/// The service's findings: whether it answers, its version against the
/// minimum, and (on this Mac) more than one install or launch item.
pub fn diagnose_service(facts: &ServiceFacts) -> Vec<Finding> {
    let mut out = Vec::new();
    let host = facts.host;
    match facts.version {
        None => {
            let mut lines = vec![format!("Nothing answered at {}.", host.url)];
            if !host.local {
                lines.push(
                    "Fix: start Ollama on that host, or change [vlm] ollama_host in config.toml."
                        .into(),
                );
            } else if facts.installs.is_empty() {
                lines.push(
                    "Fix: install Ollama from https://ollama.com/download and open it \
                     (or: brew install ollama && brew services start ollama)."
                        .into(),
                );
            } else {
                for i in facts.installs {
                    lines.push(format!("Installed: {} at {}.", i.name(), i.path.display()));
                }
                let fix = match facts.installs[0].kind {
                    HOMEBREW => "Fix: brew services start ollama".to_string(),
                    _ => format!("Fix: open {}", facts.installs[0].path.display()),
                };
                lines.push(fix);
            }
            out.push(finding(Level::Fail, "Ollama service", lines));
        }
        Some(v) if ollama::version_at_least(v, facts.minimum) => {
            let title = if host.local && (facts.serving.is_some() || !facts.installs.is_empty()) {
                format!("Ollama service {v} at {} ({})", host.url, facts.serving_description())
            } else {
                format!("Ollama service {v} at {}", host.url)
            };
            out.push(finding(Level::Pass, title, vec![]));
        }
        Some(v) => {
            let mut lines = vec![format!(
                "AgentHandover's models need Ollama {} or later; {} answers with {v}.",
                facts.minimum, host.url
            )];
            let serving = facts.serving_install().filter(|_| host.local);
            let newer = facts.installs.iter().find(|i| {
                Some(*i) != serving
                    && i.version.as_deref().is_some_and(|iv| ollama::version_at_least(iv, facts.minimum))
            });
            if !host.local {
                lines.push(format!(
                    "Fix: update Ollama on that host to {} or later (https://ollama.com/download).",
                    facts.minimum
                ));
            } else if let Some(newer) = newer.filter(|_| facts.installs.len() > 1) {
                lines.push(format!(
                    "{} is installed, but {} is serving port {}.",
                    newer.name(),
                    facts.serving_description(),
                    host.port
                ));
                lines.push(format!("Fix: {}", keep_only(newer, facts)));
            } else {
                lines.push(match serving.map(|i| i.kind) {
                    Some(HOMEBREW) => {
                        "Fix: brew upgrade ollama && brew services restart ollama".to_string()
                    }
                    Some(_) => "Fix: update Ollama.app (menu bar icon > Restart to update), or \
                                download the current version from https://ollama.com/download"
                        .to_string(),
                    None => format!(
                        "Fix: update the Ollama serving port {} ({}) to {} or later.",
                        host.port,
                        facts.serving_description(),
                        facts.minimum
                    ),
                });
            }
            out.push(finding(
                Level::Fail,
                format!("Ollama {v} is older than {}", facts.minimum),
                lines,
            ));
        }
    }

    if host.local && (facts.installs.len() > 1 || facts.launch_items.len() > 1) {
        let mut lines = Vec::new();
        for i in facts.installs {
            lines.push(format!("Installed: {} at {}", i.name(), i.path.display()));
        }
        for item in facts.launch_items {
            lines.push(format!(
                "Started at login by {} ({}){}",
                item.label,
                item.plist.display(),
                item.program
                    .as_ref()
                    .map(|p| format!(", runs {}", p.display()))
                    .unwrap_or_default()
            ));
        }
        if facts.version.is_some() || facts.serving.is_some() {
            lines.push(format!(
                "Serving port {}: {}.",
                host.port,
                facts.serving_description()
            ));
        }
        lines.push(format!(
            "Only one server can listen on port {}; whichever starts first after login \
             wins, so the version you get can change at every restart. Keep one:",
            host.port
        ));
        for keep in facts.installs {
            lines.push(format!("  {}", keep_only(keep, facts)));
        }
        let title = if facts.installs.len() > 1 {
            format!("Ollama is installed {} times", facts.installs.len())
        } else {
            format!("Ollama is started by {} launch items", facts.launch_items.len())
        };
        out.push(finding(Level::Warn, title, lines));
    }
    out
}

// ---------------------------------------------------------------------------
// Memory headroom
// ---------------------------------------------------------------------------

/// Bytes in free, inactive and speculative pages of `vm_stat` output (what
/// the system can hand to a model load), as the worker counts them.
pub fn parse_vm_stat(text: &str) -> Option<u64> {
    let page = text
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    let pages = |name: &str| -> Option<u64> {
        let line = text.lines().find(|l| l.starts_with(&format!("Pages {name}:")))?;
        line.split(':').nth(1)?.trim().trim_end_matches('.').parse().ok()
    };
    Some(page * (pages("free")? + pages("inactive")? + pages("speculative")?))
}

/// Bytes of swap in use from `sysctl -n vm.swapusage`
/// (`total = 2048.00M  used = 971.81M  free = 1076.19M  (encrypted)`).
pub fn parse_swapusage(text: &str) -> Option<u64> {
    let after = text.split("used = ").nth(1)?;
    let token = after.split_whitespace().next()?;
    let (number, unit) = token.split_at(token.len().checked_sub(1)?);
    let scale: u64 = match unit {
        "K" => 1 << 10,
        "M" => 1 << 20,
        "G" => 1 << 30,
        "T" => 1 << 40,
        _ => return None,
    };
    Some((number.parse::<f64>().ok()? * scale as f64) as u64)
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn free_memory_now() -> Option<u64> {
    parse_vm_stat(&command_stdout("vm_stat", &[])?)
}

pub fn swap_used_now() -> Option<u64> {
    parse_swapusage(&command_stdout("sysctl", &["-n", "vm.swapusage"])?)
}

/// Free space on the startup disk (where macOS keeps swap).
pub fn free_disk_now() -> Option<u64> {
    let path = std::ffi::CString::new("/").ok()?;
    // SAFETY: statvfs only writes into the zeroed struct we pass.
    unsafe {
        let mut stat: libc::statvfs = std::mem::zeroed();
        (libc::statvfs(path.as_ptr(), &mut stat) == 0)
            .then(|| stat.f_bavail as u64 * stat.f_frsize as u64)
    }
}

/// One tier's memory need against what is available now.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TierNeed {
    pub id: String,
    pub label: String,
    pub need_bytes: u64,
    pub fits: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Headroom {
    /// Free, inactive and speculative memory plus the models Ollama has
    /// loaded (it unloads them for another model).
    pub available_bytes: u64,
    pub free_bytes: u64,
    pub loaded_bytes: u64,
    pub swap_used_bytes: Option<u64>,
    pub free_disk_bytes: Option<u64>,
    /// The configured models' need.
    pub need_bytes: u64,
    /// Every tier this Mac's RAM allows, ascending.
    pub tiers: Vec<TierNeed>,
}

/// What memory is known now.
pub struct MemoryFacts<'a> {
    pub free: u64,
    /// `/api/ps`: `(name, bytes)` of each loaded model.
    pub loaded: &'a [(String, u64)],
    pub swap_used: Option<u64>,
    pub free_disk: Option<u64>,
    pub ram_gb: u64,
}

fn model_bytes(table: &TierTable, loaded: &[(String, u64)], model: &str) -> u64 {
    let names = [model.to_string()];
    loaded
        .iter()
        .find(|(name, size)| *size > 0 && ollama::model_installed(&names, name))
        .map(|(_, size)| *size)
        .or_else(|| table.model_sizes_gb.get(model).map(|gb| (gb * GB as f64) as u64))
        .unwrap_or(0)
}

/// Memory the models need at their peak: the worker loads the annotation
/// and the Skills model one at a time, next to the embedding model.
pub fn need_bytes(table: &TierTable, loaded: &[(String, u64)], annotation: &str, sop: &str) -> u64 {
    model_bytes(table, loaded, annotation).max(model_bytes(table, loaded, sop))
        + model_bytes(table, loaded, &table.embedding_model)
        + MEMORY_MARGIN_BYTES
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / GB as f64)
}

/// Memory findings for the configured tier (`tier_label`, its models).
pub fn diagnose_headroom(
    table: &TierTable,
    mem: &MemoryFacts,
    tier_label: &str,
    annotation: &str,
    sop: &str,
) -> (Headroom, Vec<Finding>) {
    let loaded_bytes: u64 = mem.loaded.iter().map(|(_, s)| s).sum();
    let available = mem.free + loaded_bytes;
    let need = need_bytes(table, mem.loaded, annotation, sop);
    let tiers: Vec<TierNeed> = table
        .tiers
        .iter()
        .filter(|t| t.min_ram_gb <= mem.ram_gb)
        .map(|t| {
            let need = need_bytes(table, mem.loaded, &t.annotation_model, &t.sop_model);
            TierNeed { id: t.id.clone(), label: t.label.clone(), need_bytes: need, fits: need <= available }
        })
        .collect();

    let mut out = Vec::new();
    let title = format!(
        "Memory for the {tier_label} tier: needs {}, {} available now",
        gb(need),
        gb(available)
    );
    if need <= available {
        out.push(finding(Level::Pass, title, vec![]));
    } else {
        let mut lines = vec![format!(
            "Available is free, inactive and speculative memory ({}) plus the models Ollama \
             has loaded ({}); other apps use the rest.",
            gb(mem.free),
            gb(loaded_bytes)
        )];
        let fitting = tiers.iter().rev().find(|t| t.fits && t.need_bytes < need);
        lines.push(match fitting {
            Some(t) => format!(
                "Fix: close memory-heavy apps, or choose the {} tier (needs {}): \
                 agenthandover setup --vlm --tier {}",
                t.label,
                gb(t.need_bytes),
                t.id
            ),
            None => "Fix: close memory-heavy apps; no tier fits in the memory free now.".into(),
        });
        out.push(finding(Level::Warn, title, lines));
    }

    if let Some(swap) = mem.swap_used {
        if swap > SWAP_LIMIT_BYTES {
            out.push(finding(
                Level::Warn,
                format!("Swap in use: {} (more than {})", gb(swap), gb(SWAP_LIMIT_BYTES)),
                vec![
                    "The Mac is short of memory; the worker does not load the Skills model \
                     while this lasts."
                        .into(),
                    "Fix: close memory-heavy apps, or restart the Mac.".into(),
                ],
            ));
        } else {
            out.push(finding(Level::Pass, format!("Swap in use: {}", gb(swap)), vec![]));
        }
    }
    if let Some(disk) = mem.free_disk {
        if disk < DISK_FLOOR_BYTES {
            out.push(finding(
                Level::Warn,
                format!("Free disk: {} (less than {})", gb(disk), gb(DISK_FLOOR_BYTES)),
                vec![
                    "macOS keeps swap on the startup disk: with this little free it cannot \
                     grow, and a model load can run the Mac out of memory."
                        .into(),
                    "Fix: free up disk space.".into(),
                ],
            ));
        }
    }

    let headroom = Headroom {
        available_bytes: available,
        free_bytes: mem.free,
        loaded_bytes,
        swap_used_bytes: mem.swap_used,
        free_disk_bytes: mem.free_disk,
        need_bytes: need,
        tiers,
    };
    (headroom, out)
}

// ---------------------------------------------------------------------------
// The whole report
// ---------------------------------------------------------------------------

/// Everything `doctor` checks about Ollama and memory; the app's
/// onboarding reads it as JSON.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub host: Host,
    pub version: Option<String>,
    pub minimum: String,
    pub installs: Vec<Install>,
    pub launch_items: Vec<LaunchItem>,
    pub serving: Option<Serving>,
    pub findings: Vec<Finding>,
    /// None for a remote host, or when memory cannot be read.
    pub headroom: Option<Headroom>,
}

/// The configured tier and models.
pub struct ConfiguredModels {
    pub tier_label: String,
    pub annotation: String,
    pub sop: String,
    /// False with `[vlm] mode = "remote"` (a cloud API writes the Skills).
    pub local: bool,
}

/// Probe this Mac and the service at `host_url`.
pub fn gather(host_url: &str, models: &ConfiguredModels) -> Report {
    let table = agenthandover_common::tiers::table();
    let host = parse_host(host_url);
    let version = ollama::version(&host.url);
    let (installs, launch_items, serving) = if host.local {
        let roots = Roots::this_mac();
        (find_installs(&roots), find_launch_items(&roots), serving_process(host.port))
    } else {
        (Vec::new(), Vec::new(), None)
    };
    let mut findings = diagnose_service(&ServiceFacts {
        host: &host,
        version: version.as_deref(),
        minimum: &table.minimum_ollama_version_for_gemma4,
        installs: &installs,
        launch_items: &launch_items,
        serving: serving.as_ref(),
    });
    let mut headroom = None;
    if host.local {
        if let Some(free) = free_memory_now() {
            let loaded = ollama::loaded_models(&host.url).unwrap_or_default();
            let (h, f) = diagnose_headroom(
                table,
                &MemoryFacts {
                    free,
                    loaded: &loaded,
                    swap_used: swap_used_now(),
                    free_disk: free_disk_now(),
                    ram_gb: agenthandover_common::tiers::machine_ram_gb(),
                },
                &models.tier_label,
                &models.annotation,
                &models.sop,
            );
            headroom = Some(h);
            // With `[vlm] mode = "remote"` the tier's models do not run
            // here: the figures stay in the report (onboarding offers the
            // local tiers), but they are no finding.
            if models.local {
                findings.extend(f);
            }
        }
    }
    Report {
        host,
        version,
        minimum: table.minimum_ollama_version_for_gemma4.clone(),
        installs,
        launch_items,
        serving,
        findings,
        headroom,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed tier table (the shipped one changes with the tier plan; the
    /// memory arithmetic here must not).
    fn tiers() -> TierTable {
        serde_json::from_str(
            r#"{"schema_version": 1, "basis": "", "minimum_ollama_version_for_gemma4": "0.34.0",
                "embedding_model": "nomic-embed-text",
                "model_sizes_gb": {"qwen3.5:4b": 3.4, "gemma4:12b-it-qat": 7.2,
                                   "gemma4:31b-it-qat": 18.0, "nomic-embed-text": 0.27},
                "tiers": [
                  {"id": "standard", "label": "Standard", "subtitle": "", "min_ram_gb": 0,
                   "annotation_model": "qwen3.5:4b", "sop_model": "qwen3.5:4b"},
                  {"id": "recommended", "label": "Recommended", "subtitle": "", "min_ram_gb": 16,
                   "annotation_model": "qwen3.5:4b", "sop_model": "gemma4:12b-it-qat"},
                  {"id": "max_quality", "label": "Max Quality", "subtitle": "", "min_ram_gb": 48,
                   "annotation_model": "qwen3.5:4b", "sop_model": "gemma4:31b-it-qat"}
                ]}"#,
        )
        .unwrap()
    }

    const MIN: &str = "0.34.0";

    fn app_plist(version: &str) -> String {
        format!(
            "<?xml version=\"1.0\"?>\n<plist version=\"1.0\">\n<dict>\n\
             \t<key>CFBundleIdentifier</key>\n\t<string>com.electron.ollama</string>\n\
             \t<key>CFBundleShortVersionString</key>\n\t<string>{version}</string>\n\
             </dict>\n</plist>\n"
        )
    }

    fn agent_plist(label: &str, program: &str) -> String {
        format!(
            "<?xml version=\"1.0\"?>\n<plist version=\"1.0\">\n<dict>\n\
             \t<key>Label</key>\n\t<string>{label}</string>\n\
             \t<key>ProgramArguments</key>\n\t<array>\n\
             \t\t<string>{program}</string>\n\t\t<string>serve</string>\n\t</array>\n\
             </dict>\n</plist>\n"
        )
    }

    /// A fake filesystem: `/Applications`, a Homebrew prefix and a
    /// LaunchAgents directory under a temp root.
    struct Fake {
        dir: tempfile::TempDir,
    }

    impl Fake {
        fn new() -> Self {
            Fake { dir: tempfile::tempdir().unwrap() }
        }
        fn root(&self) -> &Path {
            self.dir.path()
        }
        fn app(&self, version: &str) -> PathBuf {
            let app = self.root().join("Applications/Ollama.app");
            std::fs::create_dir_all(app.join("Contents/Resources")).unwrap();
            std::fs::write(app.join("Contents/Info.plist"), app_plist(version)).unwrap();
            std::fs::write(app.join("Contents/Resources/ollama"), "").unwrap();
            app
        }
        fn keg(&self, version: &str) -> PathBuf {
            let keg = self.root().join("homebrew/Cellar/ollama").join(version);
            std::fs::create_dir_all(keg.join("bin")).unwrap();
            std::fs::write(keg.join("bin/ollama"), "").unwrap();
            keg
        }
        fn link_opt(&self, keg: &Path) {
            let opt = self.root().join("homebrew/opt");
            std::fs::create_dir_all(&opt).unwrap();
            std::os::unix::fs::symlink(keg, opt.join("ollama")).unwrap();
        }
        fn agent(&self, file: &str, label: &str, program: &Path) {
            let dir = self.root().join("LaunchAgents");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(file), agent_plist(label, &program.to_string_lossy())).unwrap();
        }
        fn roots(&self) -> Roots {
            Roots {
                apps: vec![self.root().join("Applications/Ollama.app")],
                brew_prefixes: vec![self.root().join("homebrew"), self.root().join("missing")],
                launch_dirs: vec![self.root().join("LaunchAgents")],
            }
        }
    }

    fn local() -> Host {
        parse_host("http://localhost:11434")
    }

    fn facts<'a>(
        host: &'a Host,
        version: Option<&'a str>,
        installs: &'a [Install],
        items: &'a [LaunchItem],
        serving: Option<&'a Serving>,
    ) -> ServiceFacts<'a> {
        ServiceFacts { host, version, minimum: MIN, installs, launch_items: items, serving }
    }

    fn text(f: &Finding) -> String {
        format!("{}\n{}", f.title, f.lines.join("\n"))
    }

    #[test]
    fn finds_the_app_the_formula_and_launch_items() {
        let fake = Fake::new();
        let app = fake.app("0.30.6");
        fake.keg("0.33.1");
        let keg = fake.keg("0.34.2_1");
        fake.link_opt(&keg);
        fake.agent("com.ollama.ollama.plist", "com.ollama.ollama", &app.join("Contents/Resources/ollama"));
        fake.agent("homebrew.mxcl.ollama.plist", "homebrew.mxcl.ollama", &fake.root().join("homebrew/opt/ollama/bin/ollama"));
        std::fs::write(fake.root().join("LaunchAgents/com.other.plist"), "").unwrap();

        let installs = find_installs(&fake.roots());
        assert_eq!(
            installs,
            vec![
                Install { kind: APP, path: app.clone(), version: Some("0.30.6".into()) },
                Install { kind: HOMEBREW, path: keg.clone(), version: Some("0.34.2".into()) },
            ]
        );
        let items = find_launch_items(&fake.roots());
        let labels: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, ["com.ollama.ollama", "homebrew.mxcl.ollama"]);
        assert_eq!(items[0].program.as_deref(), Some(app.join("Contents/Resources/ollama").as_path()));
    }

    #[test]
    fn an_unlinked_cellar_reports_its_highest_keg() {
        let fake = Fake::new();
        fake.keg("0.9.0");
        let newest = fake.keg("0.34.2");
        let installs = find_installs(&fake.roots());
        assert_eq!(installs, vec![Install { kind: HOMEBREW, path: newest, version: Some("0.34.2".into()) }]);
        assert!(find_launch_items(&fake.roots()).is_empty());
    }

    #[test]
    fn nothing_installed_is_an_empty_list() {
        let fake = Fake::new();
        assert!(find_installs(&fake.roots()).is_empty());
    }

    #[test]
    fn a_symlinked_executable_belongs_to_its_install() {
        // /opt/homebrew/bin/ollama -> /Applications/Ollama.app/Contents/Resources/ollama
        // is the app's binary, not Homebrew's (seen on a real Mac).
        let fake = Fake::new();
        let app = fake.app("0.30.9");
        let link = fake.root().join("homebrew/bin/ollama");
        std::fs::create_dir_all(link.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(app.join("Contents/Resources/ollama"), &link).unwrap();
        let installs = find_installs(&fake.roots());
        assert!(installs[0].owns(&canonical(&link)));
    }

    #[test]
    fn plist_values() {
        assert_eq!(plist_string(&app_plist("0.30.6"), "CFBundleShortVersionString").as_deref(), Some("0.30.6"));
        let agent = agent_plist("x", "/a &amp; b/ollama");
        assert_eq!(plist_string(&agent, "ProgramArguments").as_deref(), Some("/a & b/ollama"));
        assert_eq!(plist_string(&agent, "Program"), None);
        assert_eq!(plist_string("", "Label"), None);
    }

    #[test]
    fn lsof_pid() {
        assert_eq!(parse_lsof_pid("p19850\nf3\nn127.0.0.1:11434\n"), Some(19850));
        assert_eq!(parse_lsof_pid(""), None);
    }

    #[test]
    fn hosts() {
        assert_eq!(parse_host(""), Host { url: "http://localhost:11434".into(), local: true, port: 11434 });
        assert_eq!(parse_host("http://127.0.0.1:11500/").port, 11500);
        assert!(parse_host("http://[::1]:11434").local);
        let remote = parse_host("http://gpu-box.lan:11434");
        assert!(!remote.local);
        assert!(!parse_host("https://ollama.example.com").local);
    }

    #[test]
    fn a_current_version_passes_and_names_the_server() {
        let host = local();
        let installs = [Install { kind: APP, path: "/Applications/Ollama.app".into(), version: Some("0.34.2".into()) }];
        let f = diagnose_service(&facts(&host, Some("0.34.2"), &installs, &[], None));
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].level, Level::Pass);
        assert_eq!(f[0].title, "Ollama service 0.34.2 at http://localhost:11434 (Ollama.app 0.34.2)");
    }

    #[test]
    fn an_old_app_fails_with_the_update_fix() {
        let host = local();
        let installs = [Install { kind: APP, path: "/Applications/Ollama.app".into(), version: Some("0.30.6".into()) }];
        let f = diagnose_service(&facts(&host, Some("0.30.6"), &installs, &[], None));
        assert_eq!(f[0].level, Level::Fail);
        assert_eq!(f[0].title, "Ollama 0.30.6 is older than 0.34.0");
        assert!(text(&f[0]).contains("Fix: update Ollama.app"), "{}", text(&f[0]));
    }

    #[test]
    fn an_old_formula_fails_with_brew_upgrade() {
        let host = local();
        let installs = [Install { kind: HOMEBREW, path: "/opt/homebrew/Cellar/ollama/0.30.0".into(), version: Some("0.30.0".into()) }];
        let f = diagnose_service(&facts(&host, Some("0.30.0"), &installs, &[], None));
        assert!(text(&f[0]).contains("Fix: brew upgrade ollama && brew services restart ollama"));
    }

    /// The test Mac after a reboot: Ollama.app 0.30.6 took the port before
    /// Homebrew's 0.34.2 service.
    #[test]
    fn the_old_app_serving_ahead_of_a_current_formula() {
        let fake = Fake::new();
        let app = fake.app("0.30.6");
        let keg = fake.keg("0.34.2");
        fake.link_opt(&keg);
        fake.agent("com.ollama.ollama.plist", "com.ollama.ollama", &app.join("Contents/Resources/ollama"));
        fake.agent("homebrew.mxcl.ollama.plist", "homebrew.mxcl.ollama", &fake.root().join("homebrew/opt/ollama/bin/ollama"));
        let installs = find_installs(&fake.roots());
        let items = find_launch_items(&fake.roots());
        let serving = Serving { pid: 812, executable: canonical(&app.join("Contents/Resources/ollama")) };
        let host = local();
        let f = diagnose_service(&facts(&host, Some("0.30.6"), &installs, &items, Some(&serving)));

        assert_eq!(f.len(), 2);
        assert_eq!(f[0].level, Level::Fail);
        let fail = text(&f[0]);
        assert!(fail.contains("Homebrew 0.34.2 is installed, but Ollama.app 0.30.6 (pid 812"), "{fail}");
        assert!(fail.contains("is serving port 11434"), "{fail}");
        assert!(fail.contains("Fix: To keep Homebrew 0.34.2: quit Ollama.app"), "{fail}");
        assert!(fail.contains(&format!("launchctl bootout gui/$(id -u) {}", fake.root().join("LaunchAgents/com.ollama.ollama.plist").display())), "{fail}");
        assert!(fail.ends_with("brew services restart ollama."), "{fail}");
        // Homebrew's own service is kept.
        assert!(!fail.contains("homebrew.mxcl.ollama.plist and delete"), "{fail}");

        assert_eq!(f[1].level, Level::Warn);
        assert_eq!(f[1].title, "Ollama is installed 2 times");
        let warn = text(&f[1]);
        assert!(warn.contains("Serving port 11434: Ollama.app 0.30.6 (pid 812"), "{warn}");
        assert!(warn.contains("To keep Ollama.app 0.30.6: brew services stop ollama && brew uninstall ollama"), "{warn}");
        assert!(warn.contains("To keep Homebrew 0.34.2: quit Ollama.app"), "{warn}");
    }

    #[test]
    fn without_lsof_the_version_names_the_server() {
        let host = local();
        let installs = [
            Install { kind: APP, path: "/Applications/Ollama.app".into(), version: Some("0.34.2".into()) },
            Install { kind: HOMEBREW, path: "/opt/homebrew/Cellar/ollama/0.30.0".into(), version: Some("0.30.0".into()) },
        ];
        let f = diagnose_service(&facts(&host, Some("0.30.0"), &installs, &[], None));
        let fail = text(&f[0]);
        assert!(fail.contains("Ollama.app 0.34.2 is installed, but Homebrew 0.30.0 is serving"), "{fail}");
        assert!(fail.contains("Fix: To keep Ollama.app 0.34.2: brew services stop ollama && brew uninstall ollama; then open /Applications/Ollama.app."), "{fail}");
    }

    #[test]
    fn two_launch_items_for_one_install_warn() {
        let host = local();
        let installs = [Install { kind: APP, path: "/Applications/Ollama.app".into(), version: Some("0.34.2".into()) }];
        let items = [
            LaunchItem { label: "com.ollama.ollama".into(), plist: "/L/com.ollama.ollama.plist".into(), program: None },
            LaunchItem { label: "com.ollama.serve".into(), plist: "/L/com.ollama.serve.plist".into(), program: Some("/opt/homebrew/bin/ollama".into()) },
        ];
        let f = diagnose_service(&facts(&host, Some("0.34.2"), &installs, &items, None));
        assert_eq!(f[1].title, "Ollama is started by 2 launch items");
    }

    #[test]
    fn a_remote_host_gets_only_the_version_check() {
        let host = parse_host("http://gpu-box.lan:11434");
        let installs = [
            Install { kind: APP, path: "/Applications/Ollama.app".into(), version: Some("0.30.6".into()) },
            Install { kind: HOMEBREW, path: "/x".into(), version: Some("0.34.2".into()) },
        ];
        let ok = diagnose_service(&facts(&host, Some("0.34.2"), &installs, &[], None));
        assert_eq!(ok.len(), 1);
        assert_eq!(ok[0].title, "Ollama service 0.34.2 at http://gpu-box.lan:11434");
        let old = diagnose_service(&facts(&host, Some("0.30.6"), &installs, &[], None));
        assert_eq!(old.len(), 1);
        assert!(text(&old[0]).contains("Fix: update Ollama on that host"), "{}", text(&old[0]));
        let down = diagnose_service(&facts(&host, None, &[], &[], None));
        assert!(text(&down[0]).contains("[vlm] ollama_host"));
    }

    #[test]
    fn nothing_answering() {
        let host = local();
        let f = diagnose_service(&facts(&host, None, &[], &[], None));
        assert_eq!(f[0].level, Level::Fail);
        assert!(text(&f[0]).contains("https://ollama.com/download"));
        let installs = [Install { kind: HOMEBREW, path: "/x".into(), version: Some("0.34.2".into()) }];
        let f = diagnose_service(&facts(&host, None, &installs, &[], None));
        assert!(text(&f[0]).contains("Fix: brew services start ollama"));
    }

    // -- memory ---------------------------------------------------------

    const VM_STAT: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
        Pages free:                                     4148.\n\
        Pages active:                                 443075.\n\
        Pages inactive:                               433804.\n\
        Pages speculative:                              7794.\n\
        Pages throttled:                                   0.\n";

    #[test]
    fn vm_stat_and_swapusage() {
        assert_eq!(parse_vm_stat(VM_STAT), Some(16384 * (4148 + 433804 + 7794)));
        assert_eq!(parse_vm_stat("Pages free: 1.\n"), None);
        assert_eq!(
            parse_swapusage("total = 2048.00M  used = 971.81M  free = 1076.19M  (encrypted)\n"),
            Some((971.81 * (1u64 << 20) as f64) as u64)
        );
        assert_eq!(
            parse_swapusage("total = 45056.00M  used = 44.10G  free = 1.00G  (encrypted)"),
            Some((44.10 * (1u64 << 30) as f64) as u64)
        );
        assert_eq!(parse_swapusage("unknown oid"), None);
    }

    /// The worker refuses a load at the same limits doctor warns at.
    #[test]
    fn limits_match_the_worker() {
        let py = include_str!("../../../worker/src/agenthandover_worker/model_scheduler.py");
        for (name, value) in [
            ("SWAP_LIMIT_BYTES", SWAP_LIMIT_BYTES),
            ("DISK_FLOOR_BYTES", DISK_FLOOR_BYTES),
            ("MEMORY_MARGIN_BYTES", MEMORY_MARGIN_BYTES),
        ] {
            let line = format!("{name} = {} * GB\n", value / GB);
            assert!(py.contains(&line), "model_scheduler.py lacks `{}`", line.trim());
        }
    }

    #[test]
    fn need_is_the_larger_model_plus_embedding_and_margin() {
        let t = &tiers();
        // Table sizes: 7.2 GB gemma, 0.27 GB embedding, 1 GB margin.
        assert_eq!(need_bytes(t, &[], "qwen3.5:4b", "gemma4:12b-it-qat"), 8_470_000_000);
        // A loaded model's /api/ps size (with context) wins over the table.
        let loaded = [("gemma4:12b-it-qat".to_string(), 9_500_000_000)];
        assert_eq!(need_bytes(t, &loaded, "qwen3.5:4b", "gemma4:12b-it-qat"), 10_770_000_000);
        // A model the table does not know counts as nothing (the load decides).
        assert_eq!(need_bytes(t, &[], "custom:1b", "custom:1b"), 1_270_000_000);
    }

    fn mem(free: u64, loaded: &[(String, u64)], swap: Option<u64>, disk: Option<u64>) -> MemoryFacts<'_> {
        MemoryFacts { free, loaded, swap_used: swap, free_disk: disk, ram_gb: 16 }
    }

    #[test]
    fn a_tier_that_fits_passes() {
        let t = &tiers();
        let (h, f) = diagnose_headroom(t, &mem(12 * GB, &[], Some(GB), Some(100 * GB)), "Recommended", "qwen3.5:4b", "gemma4:12b-it-qat");
        assert_eq!(f[0].level, Level::Pass);
        assert_eq!(f[0].title, "Memory for the Recommended tier: needs 8.5 GB, 12.0 GB available now");
        assert_eq!(f[1].title, "Swap in use: 1.0 GB");
        assert_eq!(f.len(), 2);
        assert_eq!(h.need_bytes, 8_470_000_000);
        // Tiers above the Mac's 16 GB are not offered.
        let ids: Vec<&str> = h.tiers.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids.first(), Some(&"standard"));
        assert!(h.tiers.iter().all(|x| t.get(&x.id).unwrap().min_ram_gb <= 16));
    }

    #[test]
    fn a_tier_that_does_not_fit_suggests_the_one_that_does() {
        let t = &tiers();
        // 3 GB free + a loaded 1.6 GB model: 4.6 GB available.
        let loaded = [("qwen3.5:4b".to_string(), 1_600_000_000)];
        let (h, f) = diagnose_headroom(t, &mem(3 * GB, &loaded, None, None), "Recommended", "qwen3.5:4b", "gemma4:12b-it-qat");
        assert_eq!(h.available_bytes, 4_600_000_000);
        assert_eq!(f[0].level, Level::Warn);
        let warn = text(&f[0]);
        assert!(warn.contains("needs 8.5 GB, 4.6 GB available now"), "{warn}");
        let standard = h.tiers.iter().find(|x| x.id == "standard").unwrap();
        assert!(standard.fits);
        assert!(
            warn.contains(&format!(
                "choose the Standard tier (needs {}): agenthandover setup --vlm --tier standard",
                gb(standard.need_bytes)
            )),
            "{warn}"
        );
        // Unknown swap and disk: no findings for them.
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn nothing_fits() {
        let t = &tiers();
        let (_, f) = diagnose_headroom(t, &mem(GB / 2, &[], None, None), "Standard", "qwen3.5:4b", "qwen3.5:4b");
        assert!(text(&f[0]).contains("no tier fits in the memory free now"), "{}", text(&f[0]));
    }

    #[test]
    fn heavy_swap_and_a_full_disk_warn() {
        let t = &tiers();
        let (_, f) = diagnose_headroom(t, &mem(12 * GB, &[], Some(44 * GB), Some(4 * GB)), "Standard", "qwen3.5:4b", "qwen3.5:4b");
        assert_eq!(f[1].level, Level::Warn);
        assert_eq!(f[1].title, "Swap in use: 44.0 GB (more than 8.0 GB)");
        assert_eq!(f[2].level, Level::Warn);
        assert_eq!(f[2].title, "Free disk: 4.0 GB (less than 10.0 GB)");
    }

    /// The app's onboarding decodes this exact JSON
    /// (app/.../OllamaReportTests.swift reads the same fixture).
    #[test]
    fn the_report_matches_the_apps_fixture() {
        let host = local();
        let installs = [
            Install { kind: APP, path: "/Applications/Ollama.app".into(), version: Some("0.30.6".into()) },
            Install { kind: HOMEBREW, path: "/opt/homebrew/Cellar/ollama/0.34.2".into(), version: Some("0.34.2".into()) },
        ];
        let items = [LaunchItem {
            label: "homebrew.mxcl.ollama".into(),
            plist: "/Users/me/Library/LaunchAgents/homebrew.mxcl.ollama.plist".into(),
            program: Some("/opt/homebrew/opt/ollama/bin/ollama".into()),
        }];
        let serving = Serving { pid: 812, executable: "/Applications/Ollama.app/Contents/Resources/ollama".into() };
        let mut findings = diagnose_service(&facts(&host, Some("0.30.6"), &installs, &items, Some(&serving)));
        let loaded = [("qwen3.5:4b".to_string(), 4_100_000_000)];
        let (headroom, more) = diagnose_headroom(
            &tiers(),
            &mem(3 * GB, &loaded, Some(44 * GB), Some(40 * GB)),
            "Recommended",
            "qwen3.5:4b",
            "gemma4:12b-it-qat",
        );
        findings.extend(more);
        let report = Report {
            host,
            version: Some("0.30.6".into()),
            minimum: MIN.into(),
            installs: installs.to_vec(),
            launch_items: items.to_vec(),
            serving: Some(serving),
            findings,
            headroom: Some(headroom),
        };
        let json = serde_json::to_string_pretty(&report).unwrap() + "\n";
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/fixtures/ollama_report/app_serving_ahead_of_formula.json");
        if std::env::var_os("AH_WRITE_FIXTURES").is_some() {
            std::fs::write(path, &json).unwrap();
        }
        assert_eq!(json, std::fs::read_to_string(path).unwrap(), "rerun with AH_WRITE_FIXTURES=1 to update");
    }

    #[test]
    fn the_report_serializes_for_the_app() {
        let host = local();
        let installs = [Install { kind: APP, path: "/Applications/Ollama.app".into(), version: Some("0.30.6".into()) }];
        let findings = diagnose_service(&facts(&host, Some("0.30.6"), &installs, &[], None));
        let report = Report {
            host: host.clone(),
            version: Some("0.30.6".into()),
            minimum: MIN.into(),
            installs: installs.to_vec(),
            launch_items: vec![],
            serving: None,
            findings,
            headroom: None,
        };
        let v: serde_json::Value = serde_json::to_value(&report).unwrap();
        assert_eq!(v["findings"][0]["level"], "fail");
        assert_eq!(v["installs"][0]["kind"], "Ollama.app");
        assert_eq!(v["host"]["local"], true);
        assert!(v["headroom"].is_null());
    }
}
