//! The docs name only commands this CLI has (WP-S3).
//!
//! - README's CLI reference table is generated from the `clap` definition:
//!   a changed command fails here until the README is rewritten with
//!   `AH_UPDATE_DOCS=1 cargo test -p agenthandover-cli docs` and the diff
//!   reviewed.
//! - Every `agenthandover <subcommand> [--flag]` that README.md or the
//!   app's FAQ mentions is a real subcommand path, and its flags exist.

use std::path::{Path, PathBuf};

use clap::{Command, CommandFactory};

use crate::Cli;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn visible_subcommands(cmd: &Command) -> Vec<&Command> {
    cmd.get_subcommands()
        .filter(|c| !c.is_hide_set() && c.get_name() != "help")
        .collect()
}

fn about(cmd: &Command) -> String {
    cmd.get_about()
        .map(|a| a.to_string())
        .unwrap_or_default()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// `<TITLE>` / `[SERVICE]` for each positional argument.
fn usage(cmd: &Command) -> String {
    cmd.get_positionals()
        .filter(|a| !a.is_hide_set())
        .map(|a| {
            let name = a.get_id().as_str().to_uppercase();
            let many = if a.is_trailing_var_arg_set() { "..." } else { "" };
            if a.is_required_set() {
                format!(" <{name}>{many}")
            } else {
                format!(" [{name}]{many}")
            }
        })
        .collect()
}

fn options(cmd: &Command) -> String {
    cmd.get_arguments()
        .filter(|a| !a.is_positional() && !a.is_hide_set())
        .filter_map(|a| a.get_long())
        .filter(|l| *l != "help" && *l != "version")
        .map(|l| format!("`--{l}`"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn row(path: &str, cmd: &Command) -> String {
    format!(
        "| `agenthandover {path}{}` | {} | {} |\n",
        usage(cmd),
        about(cmd),
        options(cmd)
    )
}

/// The README's CLI table: one row per command (per subcommand where a
/// command has them), with its help text and its long options.
fn cli_table() -> String {
    let cli = Cli::command();
    let mut out = String::from("| Command | What it does | Options |\n|---------|--------------|---------|\n");
    for sub in visible_subcommands(&cli) {
        let children = visible_subcommands(sub);
        if children.is_empty() {
            out += &row(sub.get_name(), sub);
        } else {
            for child in children {
                out += &row(&format!("{} {}", sub.get_name(), child.get_name()), child);
            }
        }
    }
    out
}

const START: &str = "<!-- cli-table:start -->\n";
const END: &str = "<!-- cli-table:end -->";

#[test]
fn docs_readme_cli_table_is_generated_from_clap() {
    let path = repo_root().join("README.md");
    let text = std::fs::read_to_string(&path).unwrap();
    let i = text.find(START).expect("README has no cli-table:start marker") + START.len();
    let j = text.find(END).expect("README has no cli-table:end marker");
    let generated = cli_table();
    if std::env::var("AH_UPDATE_DOCS").as_deref() == Ok("1") {
        std::fs::write(&path, format!("{}{}{}", &text[..i], generated, &text[j..])).unwrap();
        return;
    }
    assert_eq!(
        &text[i..j],
        generated,
        "README's CLI table differs from clap; rewrite it with AH_UPDATE_DOCS=1 and review the diff"
    );
}

fn find_subcommand<'a>(cmd: &'a Command, name: &str) -> Option<&'a Command> {
    cmd.get_subcommands()
        .find(|c| c.get_name() == name || c.get_all_aliases().any(|a| a == name))
}

/// Why `agenthandover <words>` is not a real command line, or None.
fn check_mention(words: &[&str]) -> Option<String> {
    let cli = Cli::command();
    let mut cmd = &cli;
    let mut path = String::from("agenthandover");
    for (n, word) in words.iter().enumerate() {
        let word = word.trim_matches(|c| c == '[' || c == ']' || c == ',' || c == '.');
        if word.is_empty() {
            continue;
        }
        let takes_rest = cmd.get_positionals().any(|a| a.is_trailing_var_arg_set());
        if let Some(flag) = word.strip_prefix("--") {
            let flag = flag.split('=').next().unwrap_or(flag);
            if !takes_rest && !cmd.get_arguments().any(|a| a.get_long() == Some(flag)) {
                return Some(format!("{path} has no --{flag}"));
            }
        } else if let Some(short) = word.strip_prefix('-') {
            let known = short.chars().all(|c| cmd.get_arguments().any(|a| a.get_short() == Some(c)));
            if !takes_rest && !known {
                return Some(format!("{path} has no -{short}"));
            }
        } else if cmd.has_subcommands() {
            match find_subcommand(cmd, word) {
                Some(sub) => {
                    cmd = sub;
                    path = format!("{path} {word}");
                }
                None if n == 0 || word.chars().all(|c| c.is_ascii_lowercase() || c == '-') => {
                    return Some(format!("{path} has no subcommand {word:?}"));
                }
                None => return None,
            }
        }
    }
    None
}

/// Every `agenthandover <word> ...` in *text*, cut at the end of the
/// command (a backtick, quote, bracket-free comment, escape or line end).
fn mentions(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("agenthandover ") {
        let preceded = rest[..i].chars().last();
        rest = &rest[i + "agenthandover ".len()..];
        if matches!(preceded, Some(c) if c.is_alphanumeric() || c == '-' || c == '/' || c == '.') {
            continue;
        }
        if !rest.starts_with(|c: char| c.is_ascii_lowercase()) {
            continue;
        }
        let end = rest
            .find(|c: char| matches!(c, '`' | '"' | '\\' | '\n' | '#' | ')' | '|' | '<' | '*'))
            .unwrap_or(rest.len());
        let end = rest[..end].find(" (").unwrap_or(end);
        out.push(rest[..end].trim().to_string());
    }
    out
}

#[test]
fn docs_every_command_the_readme_and_faq_mention_exists() {
    let root = repo_root();
    let mut wrong = Vec::new();
    let mut seen = 0;
    for rel in [
        "README.md",
        "app/AgentHandoverApp/Sources/AgentHandoverApp/Views/FAQView.swift",
    ] {
        let text = std::fs::read_to_string(root.join(rel)).unwrap();
        for mention in mentions(&text) {
            seen += 1;
            let words: Vec<&str> = mention.split_whitespace().collect();
            if let Some(why) = check_mention(&words) {
                wrong.push(format!("{rel}: `agenthandover {mention}`: {why}"));
            }
        }
    }
    assert!(seen > 20, "only {seen} command mentions found; is the scan broken?");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn docs_the_mention_check_rejects_what_does_not_exist() {
    assert_eq!(check_mention(&["skills", "list"]), None);
    assert_eq!(check_mention(&["sops", "failed"]), None);
    assert_eq!(check_mention(&["uninstall", "--purge-data"]), None);
    assert_eq!(check_mention(&["connect", "codex", "--append"]), None);
    assert_eq!(check_mention(&["logs", "worker", "-f"]), None);
    assert!(check_mention(&["gates"]).is_some());
    assert!(check_mention(&["skills", "publish"]).is_some());
    assert!(check_mention(&["status", "--verbose"]).is_some());
    assert_eq!(
        mentions("run `agenthandover setup --vlm [--tier ID]` or agenthandover doctor (it checks)\n"),
        vec!["setup --vlm [--tier ID]".to_string(), "doctor".to_string()]
    );
}
