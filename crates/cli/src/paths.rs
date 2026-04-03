//! Shared path resolution utilities for finding AgentHandover install artifacts.
//!
//! Used by `doctor`, `setup`, and other CLI commands that need to locate
//! the extension directory, Python venv, or Homebrew libexec.

use std::path::PathBuf;

/// Resolve the Homebrew `libexec` directory by tracing the running binary
/// back through its symlink chain.  Returns `None` when not a Homebrew install.
///
/// Homebrew layout:  `.../Cellar/agenthandover/HEAD-xxx/bin/agenthandover`
///                   `.../Cellar/agenthandover/HEAD-xxx/libexec/`
/// The `opt` symlink:  `.../opt/agenthandover` → `.../Cellar/agenthandover/HEAD-xxx`
pub fn find_homebrew_libexec() -> Option<PathBuf> {
    // 1. Resolve our own binary through any symlinks (opt → cellar)
    if let Ok(exe) = std::env::current_exe() {
        if let Ok(real) = exe.canonicalize() {
            // real is e.g. /usr/local/Cellar/agenthandover/HEAD-abc/bin/agenthandover
            if let Some(bin_dir) = real.parent() {
                if let Some(version_dir) = bin_dir.parent() {
                    let libexec = version_dir.join("libexec");
                    if libexec.is_dir() {
                        return Some(libexec);
                    }
                }
            }
        }
    }
    // 2. Fall back to well-known opt paths (Intel + Apple Silicon)
    for prefix in &[
        "/usr/local/opt/agenthandover/libexec",
        "/opt/homebrew/opt/agenthandover/libexec",
    ] {
        let p = PathBuf::from(prefix);
        if p.is_dir() {
            return Some(p);
        }
    }
    None
}

/// Find the Chrome extension directory in any known install location.
///
/// Returns the path to the directory that should be loaded as an unpacked
/// extension in Chrome, or `None` if not found.
///
/// Search order:
/// 1. Pkg installer: `/usr/local/lib/agenthandover/extension/dist`
/// 2. Homebrew libexec: `<libexec>/extension/` (flat layout with manifest.json)
/// 3. Well-known Homebrew opt paths
pub fn find_extension_dir() -> Option<PathBuf> {
    // Pkg installer path (flat layout: manifest.json + JS files at root)
    let pkg_path = PathBuf::from("/usr/local/lib/agenthandover/extension");
    if pkg_path.join("manifest.json").exists() {
        return Some(pkg_path);
    }
    // Homebrew libexec path (flat — manifest.json alongside dist contents)
    if let Some(libexec) = find_homebrew_libexec() {
        let brew_ext = libexec.join("extension");
        if brew_ext.join("manifest.json").exists() {
            return Some(brew_ext);
        }
    }
    None
}

/// Find the extension dist directory relative to the source repo.
///
/// Checks ancestors of the running binary and the current working directory
/// for `extension/dist/`. Used for dev/source builds.
pub fn find_local_extension_dist() -> Option<PathBuf> {
    // Check relative to the binary's location
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            // e.g. target/debug/ -> repo root is ../../
            for ancestor in parent.ancestors().take(5) {
                let candidate = ancestor.join("extension/dist");
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }
    }
    // Check current working directory
    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("extension/dist");
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Find any valid extension path (installed or local dev build).
pub fn find_any_extension_path() -> Option<PathBuf> {
    find_extension_dir().or_else(find_local_extension_dist)
}

/// Check whether the worker Python venv exists in any known install location.
///
/// Returns the path to the venv's `python` binary, or `None` if not found.
pub fn find_venv_python() -> Option<PathBuf> {
    // Pkg installer path
    let pkg_python = PathBuf::from("/usr/local/lib/agenthandover/venv/bin/python");
    if pkg_python.exists() {
        return Some(pkg_python);
    }
    // Homebrew libexec path
    if let Some(libexec) = find_homebrew_libexec() {
        let brew_python = libexec.join("venv/bin/python");
        if brew_python.exists() {
            return Some(brew_python);
        }
    }
    None
}

/// Find the daemon binary (`agenthandover-daemon`) in known install locations.
///
/// Search order:
/// 1. Packaged install: `/usr/local/bin/agenthandover-daemon`
/// 2. Homebrew libexec: `<libexec>/bin/agenthandover-daemon`
/// 3. Cargo build: walk up from current exe to find `target/{release,debug}/agenthandover-daemon`
///
/// Returns the first path that exists and is executable.
pub fn find_daemon_binary() -> Option<PathBuf> {
    // 1. Packaged install
    let pkg = PathBuf::from("/usr/local/bin/agenthandover-daemon");
    if is_executable(&pkg) {
        return Some(pkg);
    }

    // 2. Homebrew libexec
    if let Some(libexec) = find_homebrew_libexec() {
        let brew = libexec.join("bin/agenthandover-daemon");
        if is_executable(&brew) {
            return Some(brew);
        }
    }

    // 3. Cargo build: walk up from current exe
    if let Ok(exe) = std::env::current_exe() {
        if let Ok(real) = exe.canonicalize() {
            for ancestor in real.ancestors().take(6) {
                for profile in &["release", "debug"] {
                    let candidate =
                        ancestor.join("target").join(profile).join("agenthandover-daemon");
                    if is_executable(&candidate) {
                        return Some(candidate);
                    }
                }
                // Also check universal-release (from just build-all)
                let universal = ancestor.join("target/universal-release/agenthandover-daemon");
                if is_executable(&universal) {
                    return Some(universal);
                }
            }
        }
    }

    None
}

/// Return all Native Messaging Hosts directories for supported Chromium browsers.
///
/// Supports Chrome, Chromium, Brave, and Edge on macOS and Linux.
pub fn native_messaging_hosts_dirs() -> Vec<PathBuf> {
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => return vec![],
    };

    let mut dirs = Vec::new();

    if cfg!(target_os = "macos") {
        for browser_dir in &[
            "Google/Chrome/NativeMessagingHosts",
            "Chromium/NativeMessagingHosts",
            "BraveSoftware/Brave-Browser/NativeMessagingHosts",
            "Microsoft Edge/NativeMessagingHosts",
            "Comet/NativeMessagingHosts",
        ] {
            dirs.push(
                PathBuf::from(&home)
                    .join("Library/Application Support")
                    .join(browser_dir),
            );
        }
    } else {
        // Linux
        for browser_dir in &[
            ".config/google-chrome/NativeMessagingHosts",
            ".config/chromium/NativeMessagingHosts",
            ".config/BraveSoftware/Brave-Browser/NativeMessagingHosts",
            ".config/microsoft-edge/NativeMessagingHosts",
        ] {
            dirs.push(PathBuf::from(&home).join(browser_dir));
        }
    }

    dirs
}

/// Check if a path exists and is executable.
pub(crate) fn is_executable(path: &std::path::Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = path.metadata() {
            return meta.permissions().mode() & 0o111 != 0;
        }
        false
    }
    #[cfg(not(unix))]
    {
        path.exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_messaging_hosts_dirs_returns_multiple_browsers() {
        let dirs = native_messaging_hosts_dirs();
        // Should return at least Chrome dir on any platform
        assert!(!dirs.is_empty(), "Should return at least one browser dir");

        // Should include Chrome
        let has_chrome = dirs.iter().any(|d| {
            d.to_string_lossy().contains("Chrome")
                || d.to_string_lossy().contains("google-chrome")
        });
        assert!(has_chrome, "Should include Chrome directory");

        // All dirs should have the correct suffix
        for dir in &dirs {
            let s = dir.to_string_lossy();
            assert!(
                s.ends_with("NativeMessagingHosts"),
                "Dir should end with NativeMessagingHosts: {}",
                s
            );
        }
    }

    #[test]
    fn native_messaging_hosts_dirs_supports_cross_browser() {
        let dirs = native_messaging_hosts_dirs();
        // Check we have dirs for multiple browsers
        let dir_strs: Vec<String> = dirs
            .iter()
            .map(|d| d.to_string_lossy().to_string())
            .collect();

        // Should include at least Chrome and one other browser
        let browsers = ["Chrome", "Chromium", "Brave", "Edge"];
        let found: Vec<&str> = browsers
            .iter()
            .filter(|b| {
                dir_strs
                    .iter()
                    .any(|d| d.contains(*b) || d.contains(&b.to_lowercase()))
            })
            .copied()
            .collect();
        assert!(
            found.len() >= 2,
            "Should support at least 2 browsers, found: {:?}",
            found
        );
    }

    #[test]
    fn is_executable_returns_false_for_nonexistent() {
        assert!(!is_executable(std::path::Path::new(
            "/nonexistent/binary/foo"
        )));
    }

    #[test]
    fn find_daemon_binary_returns_option() {
        // This may or may not find a binary depending on build state,
        // but it should not panic.
        let result = find_daemon_binary();
        if let Some(ref path) = result {
            assert!(path.exists(), "Found binary should exist");
            assert!(
                path.to_string_lossy().contains("agenthandover-daemon"),
                "Path should contain daemon binary name"
            );
        }
    }
}
