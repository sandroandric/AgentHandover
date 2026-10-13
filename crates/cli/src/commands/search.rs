//! `agenthandover search` (Commonplace) and `agenthandover recall`, and the
//! client of the worker's local query API (127.0.0.1:9477) every command
//! uses.
//!
//! `search` asks `POST /commonplace/search`: Skills, Memory (only while
//! agents may search it), daily summaries and, when asked for, the
//! profile, one group per kind.  It needs the running worker (the query
//! embeds with its models).  `recall` asks `POST /recall`, and only when
//! nothing is listening there (the API is disabled in config.toml, or the
//! worker is starting) falls back to the trigger spool and polls for the
//! worker's result file, which waits for the worker's next tick.
//!
//! The API refuses a POST without the `X-AgentHandover-Token` header, whose
//! value the worker keeps in `<data dir>/query-api-token` (mode 0600).

use anyhow::{bail, Result};
use colored::Colorize;
use agenthandover_common::status::data_dir;
use crate::commands::memory::{iso, parse_when, write_moment, Api, WORKER_DOWN};
use crate::trigger::write_trigger;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::time::Duration;

const RECALL_RESULT_FILE: &str = "recall-query-result.json";

/// The worker's query API (`knowledge.query_api_port`, default 9477).
pub(crate) const QUERY_API_PORT: u16 = 9477;
/// A Commonplace search can wait on two embeddings (the text model's and
/// Memory's, up to 8 s and 15 s while Ollama is busy) before it answers.
const QUERY_API_TIMEOUT: Duration = Duration::from_secs(30);
/// File in the data dir holding the token every POST must carry.
const QUERY_API_TOKEN_FILE: &str = "query-api-token";
const QUERY_API_TOKEN_HEADER: &str = "X-AgentHandover-Token";

/// The worker's API token, or `None` when it has not written one yet.
pub(crate) fn read_query_api_token(dir: &Path) -> Option<String> {
    let token = std::fs::read_to_string(dir.join(QUERY_API_TOKEN_FILE)).ok()?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_string())
}

/// POST `body` as JSON to the local query API.
///
/// Returns `Ok(None)` when nothing is listening (connection refused), so
/// the caller can fall back to the trigger spool; any other failure is an
/// error.  An HTTP error status still returns its JSON body, which carries
/// an `error` field.  `token` goes in the `X-AgentHandover-Token` header.
fn post_query_api(
    port: u16,
    path: &str,
    token: Option<&str>,
    body: &serde_json::Value,
) -> Result<Option<serde_json::Value>> {
    Ok(call_query_api(port, "POST", path, token, Some(body))?.map(|(_, reply)| reply))
}

/// Send `method path` (with `body` as JSON, if any) to the local query API
/// and return the HTTP status and the JSON reply; `Ok(None)` when nothing
/// is listening (connection refused).
pub(crate) fn call_query_api(
    port: u16,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<&serde_json::Value>,
) -> Result<Option<(u16, serde_json::Value)>> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = match TcpStream::connect_timeout(&addr, Duration::from_secs(1)) {
        Ok(stream) => stream,
        Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    stream.set_read_timeout(Some(QUERY_API_TIMEOUT))?;
    stream.set_write_timeout(Some(QUERY_API_TIMEOUT))?;

    let payload = match body {
        Some(body) => serde_json::to_vec(body)?,
        None => Vec::new(),
    };
    let token_header = token
        .map(|t| format!("{QUERY_API_TOKEN_HEADER}: {t}\r\n"))
        .unwrap_or_default();
    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         {token_header}Connection: close\r\n\r\n",
        payload.len()
    )
    .into_bytes();
    request.extend_from_slice(&payload);
    stream.write_all(&request)?;

    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    let split = response
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| anyhow::anyhow!("Malformed response from the query API"))?;
    let status_line = String::from_utf8_lossy(&response[..split]);
    let Some(status) = status_line
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
    else {
        bail!("Malformed response from the query API");
    };
    Ok(Some((status, serde_json::from_slice(&response[split + 4..])?)))
}

/// Ask the query API; fall back to the trigger spool when it is not up.
fn query_worker(
    api_path: &str,
    kind: &str,
    result_file: &str,
    request: &serde_json::Value,
) -> Result<serde_json::Value> {
    let token = read_query_api_token(&data_dir());
    if let Some(reply) = post_query_api(QUERY_API_PORT, api_path, token.as_deref(), request)? {
        return Ok(reply);
    }
    // A stale result from an earlier run must not be shown as this one's.
    let result_path = data_dir().join(result_file);
    let _ = std::fs::remove_file(&result_path);
    let trigger_path = write_trigger(kind, request)?;
    poll_result(&trigger_path, &result_path, 15)
}

/// Poll for a result file with a timeout, then parse and clean up.
fn poll_result(
    trigger_path: &Path,
    result_path: &Path,
    timeout_secs: u64,
) -> Result<serde_json::Value> {
    let start = std::time::Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    while start.elapsed() < timeout {
        if result_path.exists() {
            let content = std::fs::read_to_string(result_path)?;
            let parsed: serde_json::Value = serde_json::from_str(&content)?;
            let _ = std::fs::remove_file(result_path);
            return Ok(parsed);
        }
        std::thread::sleep(Duration::from_millis(250));
    }

    // Timeout — withdraw the request if the worker has not taken it
    let _ = std::fs::remove_file(trigger_path);
    bail!(
        "Timed out waiting for worker response. Is the worker running?\n\
         Check with: agenthandover status"
    );
}

pub struct SearchArgs<'a> {
    pub query: &'a str,
    /// Empty: the worker's default (skills, memory, days).
    pub kinds: &'a [String],
    pub since: Option<&'a str>,
    pub until: Option<&'a str>,
    pub limit: u32,
    pub json: bool,
}

/// `agenthandover search "query"` — search Commonplace (`POST
/// /commonplace/search`): one group per kind, each ranked on its own.
pub fn search<Tz: TimeZone>(
    api: &Api,
    out: &mut dyn Write,
    args: &SearchArgs,
    now: DateTime<Utc>,
    tz: &Tz,
) -> Result<()>
where
    Tz::Offset: std::fmt::Display,
{
    let mut body = json!({"query": args.query, "limit": args.limit});
    if !args.kinds.is_empty() {
        body["kinds"] = json!(args.kinds);
    }
    if let Some(since) = args.since {
        body["since"] = json!(iso(parse_when(since, now, tz, false)?));
    }
    if let Some(until) = args.until {
        body["until"] = json!(iso(parse_when(until, now, tz, true)?));
    }
    let Some((status, reply)) = api("POST", "/commonplace/search", Some(&body))? else {
        bail!(WORKER_DOWN);
    };
    if status != 200 {
        bail!("{}", reply.get("error").and_then(Value::as_str).unwrap_or("unknown error"));
    }
    if args.json {
        writeln!(out, "{}", serde_json::to_string_pretty(&reply)?)?;
        return Ok(());
    }
    writeln!(out, "Commonplace for \"{}\":", args.query)?;
    for (kind, heading) in [("skills", "Skills"), ("memory", "Memory"), ("days", "Days"), ("profile", "Profile")] {
        let Some(group) = reply.get(kind) else { continue };
        let mode = group["mode"].as_str().unwrap_or("");
        let reason = group["reason"].as_str().unwrap_or("");
        let results = group["results"].as_array().cloned().unwrap_or_default();
        writeln!(out)?;
        match mode {
            "not_allowed" => {
                writeln!(out, "{heading}: not searched. {reason}")?;
                if kind == "memory" {
                    writeln!(out, "  Search it yourself with: agenthandover memory search \"{}\"", args.query)?;
                }
                continue;
            }
            "keyword_fallback" if !reason.is_empty() => {
                writeln!(out, "{heading} (by words; {reason}):")?
            }
            "keyword_fallback" => writeln!(out, "{heading} (by words):")?,
            "semantic" => writeln!(out, "{heading} (by meaning):")?,
            _ => writeln!(out, "{heading}:")?,
        }
        if results.is_empty() {
            writeln!(out, "  Nothing found.")?;
            continue;
        }
        for item in &results {
            match kind {
                "skills" => write_skill(out, item)?,
                "memory" => write_moment(out, item, tz)?,
                "days" => write_day(out, item)?,
                _ => {
                    for line in serde_json::to_string_pretty(item)?.lines() {
                        writeln!(out, "  {line}")?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn write_skill(out: &mut dyn Write, item: &Value) -> Result<()> {
    let text = |key: &str| item[key].as_str().unwrap_or("").to_string();
    let handoff = match text("handoff").as_str() {
        "ready" => "ready for agents".to_string(),
        "held_back" => {
            let reasons: Vec<&str> = item["reasons"]
                .as_array()
                .map(|r| r.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            format!("held back: {}", reasons.join("; "))
        }
        _ => "not approved".to_string(),
    };
    writeln!(out, "  {} ({}) — {handoff}{}", text("title"), text("slug"), super::memory::match_label(item))?;
    let description = text("description");
    if !description.is_empty() {
        writeln!(out, "      {description}")?;
    }
    Ok(())
}

fn write_day(out: &mut dyn Write, item: &Value) -> Result<()> {
    let date = item["date"].as_str().unwrap_or("?");
    let day = NaiveDate::parse_from_str(date, "%Y-%m-%d")
        .map(|d| d.format("%a %-d %b %Y").to_string())
        .unwrap_or_else(|_| date.to_string());
    let mut line = format!("  {day}");
    if let Some(tasks) = item["task_count"].as_u64() {
        line += &format!("  {tasks} task{}", if tasks == 1 { "" } else { "s" });
    }
    if let Some(hours) = item["active_hours"].as_f64() {
        line += &format!(", {hours:.1} active hours");
    }
    let apps: Vec<&str> = item["apps"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if !apps.is_empty() {
        line += &format!(" ({})", apps.join(", "));
    }
    writeln!(out, "{line}{}", super::memory::match_label(item))?;
    let excerpt = item["excerpt"].as_str().unwrap_or("");
    if !excerpt.is_empty() {
        writeln!(out, "      {excerpt}")?;
    }
    Ok(())
}

/// `agenthandover recall` — reconstruct what you were doing at a given time.
pub fn recall(
    date: Option<&str>,
    app: Option<&str>,
    start_time: Option<&str>,
    end_time: Option<&str>,
) -> Result<()> {
    let mut trigger = serde_json::json!({
        "requested_at": chrono::Utc::now().to_rfc3339(),
    });

    if let Some(d) = date {
        trigger["date"] = serde_json::json!(d);
    }
    if let Some(a) = app {
        trigger["app"] = serde_json::json!(a);
    }
    if let Some(s) = start_time {
        trigger["start_time"] = serde_json::json!(s);
    }
    if let Some(e) = end_time {
        trigger["end_time"] = serde_json::json!(e);
    }

    let date_display = date.unwrap_or("today");
    println!("Recalling activity for: {}", date_display.bold());

    let parsed = query_worker("/recall", "recall-query", RECALL_RESULT_FILE, &trigger)?;

    // Display timeline
    if let Some(entries) = parsed.get("entries").and_then(|e| e.as_array()) {
        let total_minutes = parsed
            .get("total_active_minutes")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let apps = parsed
            .get("apps_used")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        if entries.is_empty() {
            println!("{} No activity recorded.", "·".dimmed());
            return Ok(());
        }

        println!(
            "\n{}  Active: {} min  |  Apps: {}",
            "Timeline".bold(),
            total_minutes,
            apps,
        );
        println!("{}", "─".repeat(70));

        for entry in entries {
            let timestamp = entry
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let app_name = entry
                .get("app")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let what_doing = entry
                .get("what_doing")
                .and_then(|v| v.as_str())
                .unwrap_or("?");

            let time_part = if let Some(pos) = timestamp.find('T') {
                &timestamp[pos + 1..std::cmp::min(pos + 9, timestamp.len())]
            } else {
                timestamp
            };

            println!(
                "  {} │ {:<15} │ {}",
                time_part.cyan(),
                app_name,
                what_doing,
            );
        }

        println!(
            "\n{} {} entries, {} active minutes",
            "✓".green(),
            entries.len(),
            total_minutes,
        );
    } else if let Some(error) = parsed.get("error").and_then(|e| e.as_str()) {
        println!("{} {}", "✗".red(), error);
    } else {
        println!("{}", serde_json::to_string_pretty(&parsed)?);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    /// Read one HTTP request (head plus Content-Length body) from `conn`.
    fn read_request(conn: &mut TcpStream) -> String {
        let mut data = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = conn.read(&mut buf).unwrap();
            assert!(n > 0, "client closed before sending the whole request");
            data.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&data).to_string();
            if let Some(split) = text.find("\r\n\r\n") {
                let length: usize = text[..split]
                    .lines()
                    .find_map(|l| l.strip_prefix("Content-Length: "))
                    .map(|v| v.trim().parse().unwrap())
                    .unwrap_or(0);
                if data.len() >= split + 4 + length {
                    return text;
                }
            }
        }
    }

    #[test]
    fn nothing_listening_means_fall_back() {
        // Port 1: privileged, so no test can bind it, and nothing listens on
        // it. (Binding port 0 and dropping the listener raced with the tests
        // running beside this one: the OS could hand the freed port to
        // another test's server before the connect.)
        let port = 1;

        let reply = post_query_api(port, "/search", None, &serde_json::json!({"query": "x"})).unwrap();
        assert!(reply.is_none());
    }

    #[test]
    fn answers_from_the_query_api_without_the_spool() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let request = read_request(&mut conn);
            let body = r#"{"results": [{"app": "Chrome"}], "count": 1}"#;
            write!(
                conn,
                "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
            request
        });

        let started = std::time::Instant::now();
        let reply = post_query_api(port, "/search", Some("tok-123"), &serde_json::json!({"query": "domains"}))
            .unwrap()
            .expect("API reply");
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(reply["results"][0]["app"], "Chrome");

        let request = server.join().unwrap();
        assert!(request.starts_with("POST /search HTTP/1.1\r\n"));
        assert!(request.contains("\r\nX-AgentHandover-Token: tok-123\r\n"));
        assert!(request.ends_with(r#"{"query":"domains"}"#));
    }

    #[test]
    fn an_error_status_still_returns_its_body() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let _ = read_request(&mut conn);
            let body = r#"{"error": "Search is not available"}"#;
            write!(conn, "HTTP/1.0 501 Not Implemented\r\n\r\n{}", body).unwrap();
        });

        let reply = post_query_api(port, "/recall", None, &serde_json::json!({})).unwrap().unwrap();
        assert_eq!(reply["error"], "Search is not available");
        server.join().unwrap();
    }

    #[test]
    fn a_get_without_a_body_returns_the_status_and_reply() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let request = read_request(&mut conn);
            let body = r#"{"error": "memory_off"}"#;
            write!(conn, "HTTP/1.0 503 Service Unavailable\r\n\r\n{}", body).unwrap();
            request
        });
        let (status, reply) = call_query_api(port, "GET", "/memory/status", Some("t"), None)
            .unwrap()
            .unwrap();
        assert_eq!(status, 503);
        assert_eq!(reply["error"], "memory_off");
        let request = server.join().unwrap();
        assert!(request.starts_with("GET /memory/status HTTP/1.1\r\n"));
        assert!(request.contains("\r\nContent-Length: 0\r\n"));
        assert!(request.contains("\r\nX-AgentHandover-Token: t\r\n"));
    }

    use chrono::FixedOffset;
    use std::cell::RefCell;

    type Call = (String, String, Option<Value>);

    /// `search` against a fake API answering `reply`; the calls and the output.
    fn run_search(args: &SearchArgs, reply: Option<(u16, Value)>) -> (Result<()>, Vec<Call>, String) {
        let calls: RefCell<Vec<Call>> = RefCell::new(Vec::new());
        let api = |m: &str, p: &str, b: Option<&Value>| {
            calls.borrow_mut().push((m.into(), p.into(), b.cloned()));
            Ok(reply.clone())
        };
        let tz = FixedOffset::east_opt(2 * 3600).unwrap();
        let now = Utc.with_ymd_and_hms(2026, 10, 12, 12, 0, 0).unwrap();
        let mut out = Vec::new();
        let result = search(&api, &mut out, args, now, &tz);
        (result, calls.into_inner(), String::from_utf8(out).unwrap())
    }

    fn args<'a>(query: &'a str, kinds: &'a [String]) -> SearchArgs<'a> {
        SearchArgs { query, kinds, since: None, until: None, limit: 5, json: false }
    }

    #[test]
    fn search_sends_the_commonplace_contract() {
        let kinds = vec!["memory".to_string(), "days".to_string()];
        let mut a = args("acme invoice", &kinds);
        a.since = Some("2026-10-12");
        a.until = Some("13:00");
        a.limit = 3;
        let (result, calls, _) = run_search(&a, Some((200, json!({"query": "acme invoice"}))));
        result.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!((calls[0].0.as_str(), calls[0].1.as_str()), ("POST", "/commonplace/search"));
        assert_eq!(calls[0].2, Some(json!({
            "query": "acme invoice", "limit": 3, "kinds": ["memory", "days"],
            "since": "2026-10-11T22:00:00Z", "until": "2026-10-12T11:00:00Z",
        })));
        // No kinds: the worker's default, so the body names none.
        let (_, calls, _) = run_search(&args("x", &[]), Some((200, json!({}))));
        assert_eq!(calls[0].2, Some(json!({"query": "x", "limit": 5})));
    }

    #[test]
    fn search_prints_each_kind_in_its_own_group() {
        let reply = json!({
            "query": "domain",
            "skills": {"mode": "semantic", "reason": "", "results": [
                {"slug": "check-domain", "title": "Check Domain Availability",
                 "description": "Is the domain free?", "score": 0.71, "handoff": "ready"},
                {"slug": "old", "title": "Old Skill", "description": "", "score": 0.5,
                 "handoff": "held_back", "reasons": ["stale", "trust is observe"]},
            ]},
            "memory": {"mode": "not_allowed", "reason": "The user has not let agents search their Memory.", "results": []},
            "days": {"mode": "keyword_fallback", "reason": "embedding model unavailable: down", "results": [
                {"date": "2026-10-12", "task_count": 3, "active_hours": 2.5,
                 "apps": ["Google Chrome", "Slack"], "excerpt": "Checked domains", "score": 0.5},
            ]},
            "profile": {"mode": "whole", "reason": "", "results": [{"tools": {"editor": "Zed"}}]},
        });
        let (result, _, out) = run_search(&args("domain", &[]), Some((200, reply)));
        result.unwrap();
        assert_eq!(out, "Commonplace for \"domain\":\n\
            \nSkills (by meaning):\n  \
            Check Domain Availability (check-domain) — ready for agents  · match 0.71\n      \
            Is the domain free?\n  \
            Old Skill (old) — held back: stale; trust is observe  · match 0.50\n\
            \nMemory: not searched. The user has not let agents search their Memory.\n  \
            Search it yourself with: agenthandover memory search \"domain\"\n\
            \nDays (by words; embedding model unavailable: down):\n  \
            Mon 12 Oct 2026  3 tasks, 2.5 active hours (Google Chrome, Slack)  · match 0.50\n      \
            Checked domains\n\
            \nProfile:\n  {\n    \"tools\": {\n      \"editor\": \"Zed\"\n    }\n  }\n");
    }

    #[test]
    fn search_prints_memory_moments_in_local_time_and_says_when_nothing_matched() {
        let reply = json!({"query": "q",
            "memory": {"mode": "semantic", "reason": "", "results": [
                {"app": "Google Chrome", "window_title": "Invoice INV-2041", "url_host": "billing.acme.com",
                 "description": "Paying the Acme invoice", "first_ts": "2026-10-12T09:30:00.000Z",
                 "last_ts": "2026-10-12T09:35:00.000Z", "score": 0.8}]},
            "skills": {"mode": "keyword_fallback", "reason": "", "results": []}});
        let (result, _, out) = run_search(&args("q", &[]), Some((200, reply)));
        result.unwrap();
        assert_eq!(out, "Commonplace for \"q\":\n\
            \nSkills (by words):\n  Nothing found.\n\
            \nMemory (by meaning):\n  \
            Mon 12 Oct 11:30–11:35  Google Chrome — Invoice INV-2041 (billing.acme.com)  · match 0.80\n      \
            Paying the Acme invoice\n");
    }

    #[test]
    fn search_json_errors_and_a_stopped_worker() {
        let mut a = args("q", &[]);
        a.json = true;
        let reply = json!({"query": "q", "days": {"mode": "semantic", "reason": "", "results": []}});
        let (result, _, out) = run_search(&a, Some((200, reply.clone())));
        result.unwrap();
        assert_eq!(serde_json::from_str::<Value>(&out).unwrap(), reply);

        let (result, _, _) = run_search(&a, Some((400, json!({"error": "unknown kind(s): x"}))));
        assert_eq!(result.unwrap_err().to_string(), "unknown kind(s): x");
        let (result, _, _) = run_search(&a, None);
        assert_eq!(result.unwrap_err().to_string(), WORKER_DOWN);
        let mut bad = args("q", &[]);
        bad.since = Some("yesterday-ish");
        let (result, calls, _) = run_search(&bad, Some((200, json!({}))));
        assert!(result.is_err() && calls.is_empty());
    }

    #[test]
    fn token_is_read_from_the_data_dir_file() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_query_api_token(dir.path()), None);
        std::fs::write(dir.path().join(QUERY_API_TOKEN_FILE), "").unwrap();
        assert_eq!(read_query_api_token(dir.path()), None);
        std::fs::write(dir.path().join(QUERY_API_TOKEN_FILE), "abc-DEF_1\n").unwrap();
        assert_eq!(read_query_api_token(dir.path()).as_deref(), Some("abc-DEF_1"));
    }
}
