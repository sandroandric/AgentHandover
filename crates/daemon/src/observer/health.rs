use std::path::{Path, PathBuf};
use tracing::warn;

/// Default artifact directory size threshold: 10 GB.
const DEFAULT_ARTIFACT_MAX_BYTES: u64 = 10 * 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub accessibility_permitted: bool,
    pub screen_recording_permitted: bool,
    pub disk_space_ok: bool,
    pub free_disk_gb: u64,
    pub daemon_memory_mb: u64,
    pub artifact_dir_bytes: u64,
    pub artifact_dir_ok: bool,
}

impl HealthStatus {
    pub fn is_healthy(&self) -> bool {
        self.accessibility_permitted && self.disk_space_ok && self.artifact_dir_ok
    }
}

pub struct HealthWatcher {
    min_free_disk_gb: u64,
    max_memory_mb: u64,
    artifact_path: Option<PathBuf>,
    artifact_max_bytes: u64,
}

impl HealthWatcher {
    pub fn new(min_free_disk_gb: u64, max_memory_mb: u64) -> Self {
        Self {
            min_free_disk_gb,
            max_memory_mb,
            artifact_path: None,
            artifact_max_bytes: DEFAULT_ARTIFACT_MAX_BYTES,
        }
    }

    pub fn with_artifact_path(mut self, path: PathBuf) -> Self {
        self.artifact_path = Some(path);
        self
    }

    /// Run a health check and return current status.
    pub fn check(&self) -> HealthStatus {
        let free_disk_gb = get_free_disk_gb().unwrap_or(0);
        let daemon_memory_mb = get_process_memory_mb().unwrap_or(0);

        let disk_space_ok = free_disk_gb >= self.min_free_disk_gb;
        if !disk_space_ok {
            warn!(free_disk_gb, min = self.min_free_disk_gb, "Low disk space");
        }

        if daemon_memory_mb > self.max_memory_mb {
            warn!(
                daemon_memory_mb,
                max = self.max_memory_mb,
                "High memory usage"
            );
        }

        let artifact_dir_bytes = self
            .artifact_path
            .as_ref()
            .map(|p| get_dir_size_shallow(p))
            .unwrap_or(0);
        let artifact_dir_ok = artifact_dir_bytes <= self.artifact_max_bytes;
        if !artifact_dir_ok {
            warn!(
                artifact_dir_mb = artifact_dir_bytes / (1024 * 1024),
                max_mb = self.artifact_max_bytes / (1024 * 1024),
                "Artifact directory exceeds size threshold"
            );
        }

        HealthStatus {
            accessibility_permitted: check_accessibility(),
            screen_recording_permitted: check_screen_recording(),
            disk_space_ok,
            free_disk_gb,
            daemon_memory_mb,
            artifact_dir_bytes,
            artifact_dir_ok,
        }
    }
}

/// Sum file sizes in a directory, traversing one level of subdirectories.
fn get_dir_size_shallow(dir: &Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                total += path.metadata().map(|m| m.len()).unwrap_or(0);
            } else if path.is_dir() {
                // One level deep (date directories like yyyy/mm/dd)
                if let Ok(sub_entries) = std::fs::read_dir(&path) {
                    for sub_entry in sub_entries.flatten() {
                        let sub_path = sub_entry.path();
                        if sub_path.is_file() {
                            total += sub_path.metadata().map(|m| m.len()).unwrap_or(0);
                        }
                    }
                }
            }
        }
    }
    total
}

fn get_free_disk_gb() -> Option<u64> {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::mem;

        let c_path = CString::new("/").ok()?;
        unsafe {
            let mut stat: libc::statvfs = mem::zeroed();
            if libc::statvfs(c_path.as_ptr(), &mut stat) == 0 {
                Some(stat.f_bavail as u64 * stat.f_frsize as u64 / (1024 * 1024 * 1024))
            } else {
                None
            }
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn get_process_memory_mb() -> Option<u64> {
    let output = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()?;

    let rss_kb: u64 = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse()
        .ok()?;

    Some(rss_kb / 1024)
}

#[cfg(target_os = "macos")]
fn check_accessibility() -> bool {
    unsafe { accessibility_sys::AXIsProcessTrusted() }
}

#[cfg(not(target_os = "macos"))]
fn check_accessibility() -> bool {
    true
}

fn check_screen_recording() -> bool {
    // Screen recording permission is checked by attempting a CGDisplay capture
    // On macOS 10.15+, CGDisplayCreateImage returns NULL without permission
    #[cfg(target_os = "macos")]
    {
        use core_graphics::display::CGDisplay;
        CGDisplay::main().image().is_some()
    }
    #[cfg(not(target_os = "macos"))]
    {
        true
    }
}
