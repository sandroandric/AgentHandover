//! `agenthandover setup --shell`: the opt-in zsh hook that records the
//! commands run in interactive shells.
//!
//! The hook is a zsh `preexec` function in `<data dir>/shell-hook.zsh` that
//! appends one JSON line `{"ts": <epoch seconds>, "cwd": ..., "cmd": ...}`
//! per command to `<data dir>/shell.jsonl`; one `source` line in `.zshrc`
//! loads it.  The daemon ingests the file as `ShellCommand` events (its
//! redactor strips tokens and passwords before anything is stored).
//! Commands typed with a leading space are skipped, as zsh's
//! `HIST_IGNORE_SPACE` skips them from history.

use anyhow::{Context, Result};
use colored::Colorize;
use std::path::{Path, PathBuf};

/// Marks the `.zshrc` line this command owns.
const RC_MARKER: &str = "# agenthandover shell hook";

/// `<data dir>/shell-hook.zsh`.
pub fn hook_path(data_dir: &Path) -> PathBuf {
    data_dir.join("shell-hook.zsh")
}

/// `$ZDOTDIR/.zshrc`, else `$HOME/.zshrc`.
pub fn zshrc_path() -> PathBuf {
    let dir = std::env::var_os("ZDOTDIR")
        .filter(|d| !d.is_empty())
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"));
    dir.join(".zshrc")
}

/// Single-quote `text` for zsh.
fn zsh_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
}

/// The hook script writing to `shell_log`.
pub fn hook_script(shell_log: &Path) -> String {
    let log = zsh_quote(&shell_log.to_string_lossy());
    format!(
        r#"# AgentHandover shell hook: appends each command run in this shell to
# shell.jsonl for AgentHandover to learn from.  Installed by
# `agenthandover setup --shell`; removed by `agenthandover setup --shell-remove`.
# A command typed with a leading space is not recorded.
zmodload zsh/datetime 2>/dev/null || return 0
autoload -Uz add-zsh-hook

typeset -g _agenthandover_shell_log={log}

# REPLY = $1 as the inside of a JSON string.
_agenthandover_json() {{
  REPLY=${{1//\\/\\\\}}
  REPLY=${{REPLY//\"/\\\"}}
  REPLY=${{REPLY//$'\n'/\\n}}
  REPLY=${{REPLY//$'\r'/\\r}}
  REPLY=${{REPLY//$'\t'/\\t}}
}}

_agenthandover_preexec() {{
  [[ $1 == ' '* ]] && return 0
  [[ -d ${{_agenthandover_shell_log:h}} ]] || return 0
  local cwd cmd term
  _agenthandover_json "$PWD"; cwd=$REPLY
  _agenthandover_json "$1"; cmd=$REPLY
  _agenthandover_json "$TERM_PROGRAM"; term=$REPLY
  print -r -- "{{\"ts\":$EPOCHREALTIME,\"cwd\":\"$cwd\",\"cmd\":\"$cmd\",\"term\":\"$term\"}}" >> "$_agenthandover_shell_log"
}}

add-zsh-hook preexec _agenthandover_preexec
"#
    )
}

/// The `.zshrc` line that loads the hook.
fn rc_line(hook: &Path) -> String {
    let hook = zsh_quote(&hook.to_string_lossy());
    format!("[[ -r {hook} ]] && source {hook}  {RC_MARKER}")
}

/// Write the hook into `data_dir` and add its `source` line to `zshrc`
/// (once).  Returns whether `zshrc` changed.
pub fn install(data_dir: &Path, zshrc: &Path) -> Result<bool> {
    std::fs::create_dir_all(data_dir)
        .with_context(|| format!("creating {}", data_dir.display()))?;
    let hook = hook_path(data_dir);
    std::fs::write(&hook, hook_script(&data_dir.join("shell.jsonl")))
        .with_context(|| format!("writing {}", hook.display()))?;

    let rc = std::fs::read_to_string(zshrc).unwrap_or_default();
    if rc.lines().any(|l| l.contains(RC_MARKER)) {
        return Ok(false);
    }
    let mut updated = rc;
    if !updated.is_empty() && !updated.ends_with('\n') {
        updated.push('\n');
    }
    updated.push_str(&rc_line(&hook));
    updated.push('\n');
    std::fs::write(zshrc, updated).with_context(|| format!("writing {}", zshrc.display()))?;
    Ok(true)
}

/// Remove the `source` line from `zshrc` and the hook file.  Recorded
/// commands already in `shell.jsonl` are left for the daemon to ingest.
/// Returns whether anything was removed.
pub fn remove(data_dir: &Path, zshrc: &Path) -> Result<bool> {
    let mut removed = false;
    if let Ok(rc) = std::fs::read_to_string(zshrc) {
        if rc.lines().any(|l| l.contains(RC_MARKER)) {
            let kept: String = rc
                .lines()
                .filter(|l| !l.contains(RC_MARKER))
                .map(|l| format!("{l}\n"))
                .collect();
            std::fs::write(zshrc, kept)
                .with_context(|| format!("writing {}", zshrc.display()))?;
            removed = true;
        }
    }
    let hook = hook_path(data_dir);
    if hook.exists() {
        std::fs::remove_file(&hook).with_context(|| format!("removing {}", hook.display()))?;
        removed = true;
    }
    Ok(removed)
}

/// `agenthandover setup --shell` / `--shell-remove`.
pub fn run(remove_hook: bool) -> Result<()> {
    let data_dir = agenthandover_common::status::data_dir();
    let zshrc = zshrc_path();
    if remove_hook {
        if remove(&data_dir, &zshrc)? {
            println!("  {} Shell hook removed from {}", "✓".green(), zshrc.display());
        } else {
            println!("  {} No shell hook installed", "✓".green());
        }
        return Ok(());
    }
    let changed = install(&data_dir, &zshrc)?;
    println!(
        "  {} Shell hook {} ({})",
        "✓".green(),
        if changed { "installed" } else { "updated" },
        hook_path(&data_dir).display()
    );
    println!(
        "    New zsh sessions record each command, its directory and time to {}.",
        data_dir.join("shell.jsonl").display()
    );
    println!(
        "    {}",
        "Commands typed with a leading space are skipped. Remove with `agenthandover setup --shell-remove`."
            .dimmed()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_is_idempotent_and_remove_restores_the_rc() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("Application Support/agenthandover");
        let zshrc = dir.path().join(".zshrc");
        std::fs::write(&zshrc, "export PATH=/opt/bin:$PATH").unwrap();

        assert!(install(&data, &zshrc).unwrap());
        assert!(!install(&data, &zshrc).unwrap(), "second install adds nothing");
        let rc = std::fs::read_to_string(&zshrc).unwrap();
        assert_eq!(rc.matches(RC_MARKER).count(), 1);
        assert!(rc.starts_with("export PATH=/opt/bin:$PATH\n[[ -r '"), "{rc}");
        assert!(hook_path(&data).exists());

        assert!(remove(&data, &zshrc).unwrap());
        assert_eq!(std::fs::read_to_string(&zshrc).unwrap(), "export PATH=/opt/bin:$PATH\n");
        assert!(!hook_path(&data).exists());
        assert!(!remove(&data, &zshrc).unwrap());
    }

    /// Runs the hook in a real zsh and parses what it wrote.
    #[test]
    fn hook_writes_one_json_line_per_command_with_exact_text() {
        if std::process::Command::new("zsh").arg("-fc").arg("true").status().is_err() {
            eprintln!("zsh not available; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("Application Support/agenthandover");
        let work = dir.path().join("my \"repo\"");
        std::fs::create_dir_all(&work).unwrap();
        install(&data, &dir.path().join(".zshrc")).unwrap();

        let commands = [
            r#"git commit -m "fix: \"quoted\" path\with\backslash""#,
            "printf 'a\tb'\necho second-line",
            " echo skipped-by-leading-space",
            "cargo test -p agenthandover-storage",
        ];
        let mut script = format!("source {}\n", zsh_quote(&hook_path(&data).to_string_lossy()));
        for cmd in commands {
            script.push_str(&format!("_agenthandover_preexec {}\n", zsh_quote(cmd)));
        }
        let status = std::process::Command::new("zsh")
            .arg("-fc")
            .arg(&script)
            .current_dir(&work)
            .env("HOME", dir.path())
            .env("TERM_PROGRAM", "iTerm.app")
            .status()
            .unwrap();
        assert!(status.success());

        let log = std::fs::read_to_string(data.join("shell.jsonl")).unwrap();
        let lines: Vec<serde_json::Value> =
            log.lines().map(|l| serde_json::from_str(l).expect(l)).collect();
        let recorded: Vec<&str> = lines.iter().map(|l| l["cmd"].as_str().unwrap()).collect();
        assert_eq!(recorded, [commands[0], commands[1], commands[3]]);
        let cwd = std::fs::canonicalize(&work).unwrap();
        assert_eq!(
            std::fs::canonicalize(lines[0]["cwd"].as_str().unwrap()).unwrap(),
            cwd
        );
        assert!(lines[0]["ts"].as_f64().unwrap() > 1.7e9);
        // The terminal, so the daemon can apply its privacy zone.
        assert!(lines.iter().all(|l| l["term"] == "iTerm.app"));
    }

    #[test]
    fn hook_is_silent_when_the_data_dir_is_gone() {
        if std::process::Command::new("zsh").arg("-fc").arg("true").status().is_err() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("agenthandover");
        install(&data, &dir.path().join(".zshrc")).unwrap();
        let hook = std::fs::read_to_string(hook_path(&data)).unwrap();
        let moved = dir.path().join("hook.zsh");
        std::fs::write(&moved, hook).unwrap();
        std::fs::remove_dir_all(&data).unwrap();

        let out = std::process::Command::new("zsh")
            .arg("-fc")
            .arg(format!("source {}; _agenthandover_preexec 'ls'", zsh_quote(&moved.to_string_lossy())))
            .output()
            .unwrap();
        assert!(out.status.success());
        assert!(out.stderr.is_empty(), "{}", String::from_utf8_lossy(&out.stderr));
        assert!(!data.exists(), "the hook never recreates the data dir");
    }
}
