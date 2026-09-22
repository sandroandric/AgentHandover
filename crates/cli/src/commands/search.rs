//! CLI commands for searching activity and recalling sessions.
//!
//! Asks the worker's local query API first (`POST /search`, `POST /recall`
//! on 127.0.0.1:9477), which answers immediately.  Only when nothing is
//! listening there (the API is disabled in config.toml, or the worker is
//! starting) does it fall back to the trigger spool and poll for the
//! worker's result file, which waits for the worker's next tick.

use anyhow::{bail, Result};
use colored::Colorize;
use agenthandover_common::status::data_dir;
use crate::trigger::write_trigger;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::Path;
use std::time::Duration;

const SEARCH_RESULT_FILE: &str = "search-query-result.json";
const RECALL_RESULT_FILE: &str = "recall-query-result.json";

/// The worker's query API (`knowledge.query_api_port`, default 9477).
const QUERY_API_PORT: u16 = 9477;
const QUERY_API_TIMEOUT: Duration = Duration::from_secs(15);

/// POST `body` as JSON to the local query API.
///
/// Returns `Ok(None)` when nothing is listening (connection refused), so
/// the caller can fall back to the trigger spool; any other failure is an
/// error.  An HTTP error status still returns its JSON body, which carries
/// an `error` field.
fn post_query_api(port: u16, path: &str, body: &serde_json::Value) -> Result<Option<serde_json::Value>> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = match TcpStream::connect_timeout(&addr, Duration::from_secs(1)) {
        Ok(stream) => stream,
        Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    stream.set_read_timeout(Some(QUERY_API_TIMEOUT))?;
    stream.set_write_timeout(Some(QUERY_API_TIMEOUT))?;

    let payload = serde_json::to_vec(body)?;
    let mut request = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n",
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
    let status_ok = status_line
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|code| code.parse::<u16>().ok())
        .is_some();
    if !status_ok {
        bail!("Malformed response from the query API");
    }
    Ok(Some(serde_json::from_slice(&response[split + 4..])?))
}

/// Ask the query API; fall back to the trigger spool when it is not up.
fn query_worker(
    api_path: &str,
    kind: &str,
    result_file: &str,
    request: &serde_json::Value,
) -> Result<serde_json::Value> {
    if let Some(reply) = post_query_api(QUERY_API_PORT, api_path, request)? {
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

/// `agenthandover search "query"` — full-text search over activity annotations.
pub fn search(
    query: &str,
    date: Option<&str>,
    app: Option<&str>,
    limit: usize,
) -> Result<()> {
    let mut trigger = serde_json::json!({
        "query": query,
        "limit": limit,
        "requested_at": chrono::Utc::now().to_rfc3339(),
    });

    if let Some(d) = date {
        trigger["date"] = serde_json::json!(d);
    }
    if let Some(a) = app {
        trigger["app"] = serde_json::json!(a);
    }

    println!("Searching for: {}", query.bold());

    let parsed = query_worker("/search", "search-query", SEARCH_RESULT_FILE, &trigger)?;

    if let Some(results) = parsed.get("results").and_then(|r| r.as_array()) {
        if results.is_empty() {
            println!("{} No results found.", "·".dimmed());
            return Ok(());
        }

        println!(
            "\n{:<20}  {:<15}  {}",
            "Time".bold(),
            "App".bold(),
            "Activity".bold(),
        );
        println!("{}", "─".repeat(70));

        for result in results {
            let timestamp = result
                .get("timestamp")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let app_name = result
                .get("app")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let what_doing = result
                .get("what_doing")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let score = result
                .get("relevance_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);

            // Truncate timestamp to time only if it has a T
            let time_part = if let Some(pos) = timestamp.find('T') {
                &timestamp[pos + 1..std::cmp::min(pos + 9, timestamp.len())]
            } else {
                timestamp
            };

            let score_display = format!("{:.0}%", score * 100.0);
            println!(
                "  {:<20}  {:<15}  {} {}",
                time_part,
                app_name,
                what_doing,
                score_display.dimmed(),
            );
        }

        println!(
            "\n{} {} result(s)",
            "✓".green(),
            results.len()
        );
    } else if let Some(error) = parsed.get("error").and_then(|e| e.as_str()) {
        println!("{} {}", "✗".red(), error);
    } else {
        println!("{}", serde_json::to_string_pretty(&parsed)?);
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
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let reply = post_query_api(port, "/search", &serde_json::json!({"query": "x"})).unwrap();
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
        let reply = post_query_api(port, "/search", &serde_json::json!({"query": "domains"}))
            .unwrap()
            .expect("API reply");
        assert!(started.elapsed() < Duration::from_secs(1));
        assert_eq!(reply["results"][0]["app"], "Chrome");

        let request = server.join().unwrap();
        assert!(request.starts_with("POST /search HTTP/1.1\r\n"));
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

        let reply = post_query_api(port, "/recall", &serde_json::json!({})).unwrap().unwrap();
        assert_eq!(reply["error"], "Search is not available");
        server.join().unwrap();
    }
}
