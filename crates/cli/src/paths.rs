//! Shared path resolution utilities for finding OpenMimic install artifacts.
//!
//! Used by `doctor`, `setup`, and other CLI commands that need to locate
//! the extension directory, Python venv, or Homebrew libexec.

use std::path::PathBuf;

/// Resolve the Homebrew `libexec` directory by tracing the running binary
/// back through its symlink chain.  Returns `None` when not a Homebrew install.
///
/// Homebrew layout:  `.../Cellar/openmimic/HEAD-xxx/bin/openmimic`
///                   `.../Cellar/openmimic/HEAD-xxx/libexec/`
/// The `opt` symlink:  `.../opt/openmimic` → `.../Cellar/openmimic/HEAD-xxx`
pub fn find_homebrew_libexec() -> Option<PathBuf> {
    // 1. Resolve our own binary through any symlinks (opt → cellar)
    if let Ok(exe) = std::env::current_exe() {
        if let Ok(real) = exe.canonicalize() {
            // real is e.g. /usr/local/Cellar/openmimic/HEAD-abc/bin/openmimic
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
        "/usr/local/opt/openmimic/libexec",
        "/opt/homebrew/opt/openmimic/libexec",
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
/// 1. Pkg installer: `/usr/local/lib/openmimic/extension/dist`
/// 2. Homebrew libexec: `<libexec>/extension/` (flat layout with manifest.json)
/// 3. Well-known Homebrew opt paths
pub fn find_extension_dir() -> Option<PathBuf> {
    // Pkg installer path (flat layout: manifest.json + JS files at root)
    let pkg_path = PathBuf::from("/usr/local/lib/openmimic/extension");
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
    let pkg_python = PathBuf::from("/usr/local/lib/openmimic/venv/bin/python");
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
