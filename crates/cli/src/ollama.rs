//! The local Ollama service, asked over its HTTP API.
//!
//! `setup` and `doctor` both list installed models through here: one way,
//! `GET /api/tags`. The `ollama` CLI is not used because it is not on the
//! PATH of a GUI-launched or non-interactive process (no /opt/homebrew/bin),
//! and a service can run without it (the Ollama app bundles its own), so
//! `ollama list` reported installed models as missing. HTTP goes through
//! `curl`, which macOS ships, as the rest of the CLI does.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

/// Where the worker's local models are served.
pub const BASE_URL: &str = "http://localhost:11434";

/// The body of `GET <base><path>`, or None when the service did not answer
/// with a 2xx within five seconds.
fn get(base: &str, path: &str) -> Option<String> {
    let out = Command::new("curl")
        .args(["-sf", "--max-time", "5", &format!("{base}{path}")])
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The names of the installed models; None when the service is not
/// reachable or its answer is not a tag list.
pub fn installed_models(base: &str) -> Option<Vec<String>> {
    tag_names(&get(base, "/api/tags")?)
}

/// Names of the models in a `GET /api/tags` response; None when the
/// response is not one.
pub fn tag_names(tags_json: &str) -> Option<Vec<String>> {
    let value: serde_json::Value = serde_json::from_str(tags_json).ok()?;
    let models = value.get("models")?.as_array()?;
    Some(
        models
            .iter()
            .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(str::to_string))
            .collect(),
    )
}

/// Whether `wanted` is among `installed`, where `name` and `name:latest`
/// are the same model (Ollama's default tag).
pub fn model_installed(installed: &[String], wanted: &str) -> bool {
    fn canonical(name: &str) -> String {
        let name = name.trim();
        if name.contains(':') {
            name.to_string()
        } else {
            format!("{name}:latest")
        }
    }
    let wanted = canonical(wanted);
    installed.iter().any(|n| canonical(n) == wanted)
}

/// The service's version (`GET /api/version`), e.g. "0.34.2".
pub fn version(base: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(&get(base, "/api/version")?).ok()?;
    value.get("version")?.as_str().map(str::to_string)
}

/// A Gemma 4 tag (`gemma4:12b-it-qat`): the models that need the tier
/// table's `minimum_ollama_version_for_gemma4`.
pub fn is_gemma4(model: &str) -> bool {
    model.starts_with("gemma4")
}

/// `version` ("0.34.2-rc1") >= `minimum` ("0.34.0"); false for a version
/// that is not dotted numbers ("dev").
pub fn version_at_least(version: &str, minimum: &str) -> bool {
    let parse = |s: &str| -> Option<Vec<u64>> {
        s.split('-').next()?.split('.').map(|p| p.parse().ok()).collect()
    };
    match (parse(version), parse(minimum)) {
        (Some(have), Some(need)) => have >= need,
        _ => false,
    }
}

/// The models the service has loaded now (`GET /api/ps`), each with the
/// bytes it occupies (weights plus its context); None when the service is
/// not reachable or the answer is not a model list.
pub fn loaded_models(base: &str) -> Option<Vec<(String, u64)>> {
    ps_models(&get(base, "/api/ps")?)
}

/// `(name, size)` of each model in a `GET /api/ps` response.
pub fn ps_models(ps_json: &str) -> Option<Vec<(String, u64)>> {
    let value: serde_json::Value = serde_json::from_str(ps_json).ok()?;
    let models = value.get("models")?.as_array()?;
    Some(
        models
            .iter()
            .filter_map(|m| {
                let name = m.get("name")?.as_str()?.to_string();
                Some((name, m.get("size").and_then(|s| s.as_u64()).unwrap_or(0)))
            })
            .collect(),
    )
}

/// One line of the `POST /api/pull` stream.
#[derive(Debug, PartialEq)]
pub enum PullEvent {
    /// A status, with the download's progress in percent when it has one.
    Status(String, Option<u8>),
    Success,
    Error(String),
}

pub fn parse_pull_line(line: &str) -> Option<PullEvent> {
    let value: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    if let Some(err) = value.get("error").and_then(|e| e.as_str()) {
        return Some(PullEvent::Error(err.to_string()));
    }
    let status = value.get("status")?.as_str()?;
    if status == "success" {
        return Some(PullEvent::Success);
    }
    let percent = match (
        value.get("completed").and_then(|c| c.as_u64()),
        value.get("total").and_then(|t| t.as_u64()),
    ) {
        (Some(done), Some(total)) if total > 0 => Some((done.min(total) * 100 / total) as u8),
        _ => None,
    };
    Some(PullEvent::Status(status.to_string(), percent))
}

/// Pull `model` through the service, passing each progress event to
/// `on_event`. Ok only when the service reported success.
pub fn pull(base: &str, model: &str, mut on_event: impl FnMut(&PullEvent)) -> Result<(), String> {
    let body = serde_json::json!({ "model": model }).to_string();
    let mut child = Command::new("curl")
        .args([
            "-sN",
            "-X",
            "POST",
            &format!("{base}/api/pull"),
            "-d",
            &body,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("could not run curl: {e}"))?;
    let mut outcome: Option<Result<(), String>> = None;
    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let Some(event) = parse_pull_line(&line) else {
                continue;
            };
            on_event(&event);
            match event {
                PullEvent::Success => outcome = Some(Ok(())),
                PullEvent::Error(e) => outcome = Some(Err(e)),
                PullEvent::Status(..) => {}
            }
        }
    }
    let exit = child.wait().map_err(|e| format!("curl: {e}"))?;
    outcome.unwrap_or_else(|| {
        Err(match exit.code() {
            Some(0) => "the service ended the pull without reporting success".to_string(),
            code => format!("the service did not answer (curl exit {code:?})"),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    const TAGS: &str = r#"{"models":[
        {"name":"qwen3.5:4b","model":"qwen3.5:4b","size":1},
        {"name":"nomic-embed-text:latest","model":"nomic-embed-text:latest","size":2},
        {"name":"gemma4:12b-it-qat","size":3}
    ]}"#;

    #[test]
    fn models_come_from_api_tags() {
        let installed = tag_names(TAGS).unwrap();
        assert_eq!(
            installed,
            vec!["qwen3.5:4b", "nomic-embed-text:latest", "gemma4:12b-it-qat"]
        );
        assert!(model_installed(&installed, "qwen3.5:4b"));
        assert!(model_installed(&installed, "gemma4:12b-it-qat"));
        // `name` and `name:latest` are the same model, in both directions.
        assert!(model_installed(&installed, "nomic-embed-text"));
        assert!(model_installed(
            &["nomic-embed-text".into()],
            "nomic-embed-text:latest"
        ));
        // A different tag or a name prefix is not a match.
        assert!(!model_installed(&installed, "qwen3.5:2b"));
        assert!(!model_installed(&installed, "qwen3.5"));
        assert!(!model_installed(&["qwen3.5:4b-extra".into()], "qwen3.5:4b"));
    }

    #[test]
    fn garbled_tags_are_not_a_model_list() {
        assert_eq!(tag_names(""), None);
        assert_eq!(tag_names("<html>"), None);
        assert_eq!(tag_names(r#"{"error":"x"}"#), None);
        assert_eq!(tag_names(r#"{"models":[]}"#), Some(vec![]));
    }

    #[test]
    fn compares_versions() {
        assert!(version_at_least("0.34.2", "0.34.0"));
        assert!(version_at_least("0.34.0-rc1", "0.34.0"));
        assert!(version_at_least("0.34.0", "0.34.0"));
        assert!(version_at_least("1.0.0", "0.34.0"));
        assert!(!version_at_least("0.30.6", "0.34.0"));
        assert!(!version_at_least("0.9.99", "0.34.0"));
        assert!(!version_at_least("dev", "0.34.0"));
    }

    #[test]
    fn loaded_models_come_from_api_ps() {
        let ps = r#"{"models":[{"name":"qwen3.5:4b","size":4100000000,"size_vram":0},
                               {"name":"nomic-embed-text:latest","size":300000000}]}"#;
        assert_eq!(
            ps_models(ps).unwrap(),
            vec![
                ("qwen3.5:4b".to_string(), 4_100_000_000),
                ("nomic-embed-text:latest".to_string(), 300_000_000)
            ]
        );
        assert_eq!(ps_models(r#"{"models":[]}"#), Some(vec![]));
        assert_eq!(ps_models("<html>"), None);
    }

    #[test]
    fn pull_lines() {
        assert_eq!(
            parse_pull_line(r#"{"status":"pulling manifest"}"#),
            Some(PullEvent::Status("pulling manifest".into(), None))
        );
        assert_eq!(
            parse_pull_line(
                r#"{"status":"pulling 2a65","digest":"sha256:2a65","total":200,"completed":50}"#
            ),
            Some(PullEvent::Status("pulling 2a65".into(), Some(25)))
        );
        assert_eq!(
            parse_pull_line(r#"{"status":"success"}"#),
            Some(PullEvent::Success)
        );
        assert_eq!(
            parse_pull_line(r#"{"error":"pull model manifest: file does not exist"}"#),
            Some(PullEvent::Error(
                "pull model manifest: file does not exist".into()
            ))
        );
        assert_eq!(parse_pull_line(""), None);
    }

    /// A one-connection-per-request stand-in for the Ollama service:
    /// answers each request line with the body `route` gives for it.
    fn serve(route: fn(&str) -> (u16, String), requests: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming().take(requests) {
                let mut stream = stream.unwrap();
                let mut buf = [0u8; 4096];
                let n = stream.read(&mut buf).unwrap();
                let request = String::from_utf8_lossy(&buf[..n]).into_owned();
                let (code, body) = route(request.lines().next().unwrap_or(""));
                let _ = write!(
                    stream,
                    "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        base
    }

    /// The service answers without any `ollama` binary involved: this is
    /// how `setup` works when the CLI is not on the PATH.
    #[test]
    fn a_running_service_is_asked_over_http() {
        let base = serve(
            |line| match line {
                l if l.starts_with("GET /api/tags ") => (200, TAGS.to_string()),
                l if l.starts_with("GET /api/version ") => (200, r#"{"version":"0.34.2"}"#.into()),
                l if l.starts_with("POST /api/pull ") => (
                    200,
                    "{\"status\":\"pulling manifest\"}\n\
                     {\"status\":\"pulling 2a65\",\"total\":10,\"completed\":10}\n\
                     {\"status\":\"success\"}\n"
                        .into(),
                ),
                _ => (404, "{}".into()),
            },
            3,
        );
        let installed = installed_models(&base).unwrap();
        assert!(model_installed(&installed, "nomic-embed-text"));
        assert_eq!(version(&base).as_deref(), Some("0.34.2"));
        let mut events = Vec::new();
        assert_eq!(
            pull(&base, "qwen3.5:4b", |e| events.push(format!("{e:?}"))),
            Ok(())
        );
        assert_eq!(events.len(), 3, "{events:?}");
    }

    #[test]
    fn a_failed_pull_says_why() {
        let base = serve(
            |_| {
                (
                    500,
                    r#"{"error":"pull model manifest: file does not exist"}"#.into(),
                )
            },
            1,
        );
        assert_eq!(
            pull(&base, "nope:1b", |_| {}),
            Err("pull model manifest: file does not exist".to_string())
        );
    }

    #[test]
    fn no_service_means_no_model_list() {
        // Bind then drop: nothing listens on the port any more.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let base = format!("http://127.0.0.1:{port}");
        assert_eq!(installed_models(&base), None);
        assert_eq!(version(&base), None);
        assert!(pull(&base, "qwen3.5:4b", |_| {}).is_err());
    }
}
