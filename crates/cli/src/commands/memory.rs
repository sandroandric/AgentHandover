//! `agenthandover memory`: search, forget, pause, resume, status, agents.
//!
//! Everything but `agents` goes through the worker's query API
//! (`/memory/*` on 127.0.0.1:9477, the contract the app uses too); when
//! nothing listens there the command says the worker isn't running and
//! exits 1.  `agents on|off` writes `[memory] agents_access` in
//! config.toml, which the MCP server reads on every `search_memory` call.

use crate::commands::config::config_path;
use crate::commands::search::{call_query_api, read_query_api_token, QUERY_API_PORT};
use agenthandover_common::status::data_dir;
use anyhow::{anyhow, bail, Result};
use chrono::{DateTime, Duration, Local, NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

const WORKER_DOWN: &str = "AgentHandover's worker isn't running; open the app.";

/// `(method, path, body)` → `(HTTP status, JSON reply)`, or `None` when
/// nothing listens.
pub type Api<'a> = dyn Fn(&str, &str, Option<&Value>) -> Result<Option<(u16, Value)>> + 'a;

/// The worker's query API with its token.
pub fn worker_api() -> impl Fn(&str, &str, Option<&Value>) -> Result<Option<(u16, Value)>> {
    let token = read_query_api_token(&data_dir());
    move |method, path, body| call_query_api(QUERY_API_PORT, method, path, token.as_deref(), body)
}

/// Call the API: the reply of a 200, else an error in plain words.
fn ask(api: &Api, method: &str, path: &str, body: Option<&Value>) -> Result<Value> {
    let Some((status, reply)) = api(method, path, body)? else {
        bail!(WORKER_DOWN);
    };
    if status == 200 {
        return Ok(reply);
    }
    let error = reply.get("error").and_then(Value::as_str).unwrap_or("unknown error");
    let detail = reply.get("detail").and_then(Value::as_str).unwrap_or("");
    match error {
        "memory_off" => bail!("Memory is off. Turn it on in AgentHandover's Settings."),
        "model_unavailable" => bail!(
            "Memory's model isn't available ({detail}). Open AgentHandover to download it; \
             Memory needs Ollama 0.40 or later."
        ),
        "busy" => bail!("AgentHandover is busy; try again in a minute."),
        _ => bail!("{error}"),
    }
}

// ---------------------------------------------------------------------------
// Times
// ---------------------------------------------------------------------------

/// `1h`, `30m`, `90s`, `2d`, or several (`1h30m`).
pub fn parse_duration(text: &str) -> Result<Duration> {
    let bad = || anyhow!("'{text}' is not a duration (like 30m, 1h, 2d or 1h30m)");
    let mut total = Duration::zero();
    let mut digits = String::new();
    for c in text.trim().chars() {
        if c.is_ascii_digit() {
            digits.push(c);
            continue;
        }
        let n: i64 = digits.parse().map_err(|_| bad())?;
        digits.clear();
        total += match c {
            's' => Duration::seconds(n),
            'm' => Duration::minutes(n),
            'h' => Duration::hours(n),
            'd' => Duration::days(n),
            _ => return Err(bad()),
        };
    }
    if !digits.is_empty() || total <= Duration::zero() {
        return Err(bad());
    }
    Ok(total)
}

fn local_to_utc<Tz: TimeZone>(tz: &Tz, naive: NaiveDateTime) -> Result<DateTime<Utc>> {
    tz.from_local_datetime(&naive)
        .earliest()
        .map(|t| t.with_timezone(&Utc))
        .ok_or_else(|| anyhow!("{naive} does not exist in the local time zone"))
}

/// A time on the command line: RFC 3339, a local `YYYY-MM-DD HH:MM[:SS]`
/// (or with `T`), a local date (its start, or with `end_of_day` its last
/// second), or `HH:MM` today.
pub fn parse_when<Tz: TimeZone>(
    text: &str,
    now: DateTime<Utc>,
    tz: &Tz,
    end_of_day: bool,
) -> Result<DateTime<Utc>> {
    let text = text.trim();
    if let Ok(t) = DateTime::parse_from_rfc3339(text) {
        return Ok(t.with_timezone(&Utc));
    }
    for format in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"] {
        if let Ok(naive) = NaiveDateTime::parse_from_str(text, format) {
            return local_to_utc(tz, naive);
        }
    }
    if let Ok(day) = NaiveDate::parse_from_str(text, "%Y-%m-%d") {
        let time = if end_of_day {
            NaiveTime::from_hms_opt(23, 59, 59).unwrap()
        } else {
            NaiveTime::MIN
        };
        return local_to_utc(tz, day.and_time(time));
    }
    if let Ok(time) = NaiveTime::parse_from_str(text, "%H:%M") {
        let today = now.with_timezone(tz).date_naive();
        return local_to_utc(tz, today.and_time(time));
    }
    bail!("'{text}' is not a time (like 2026-10-12 14:30, 14:30, or 2026-10-12)")
}

fn iso(t: DateTime<Utc>) -> String {
    t.format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn parse_iso(text: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text).ok().map(|t| t.with_timezone(&Utc))
}

/// `Mon 12 Oct 14:03`, in the local zone.
fn when<Tz: TimeZone>(tz: &Tz, t: DateTime<Utc>) -> String
where
    Tz::Offset: std::fmt::Display,
{
    t.with_timezone(tz).format("%a %-d %b %H:%M").to_string()
}

/// `HH:MM` today, else the day and time.
fn short_when<Tz: TimeZone>(tz: &Tz, t: DateTime<Utc>, now: DateTime<Utc>) -> String
where
    Tz::Offset: std::fmt::Display,
{
    if t.with_timezone(tz).date_naive() == now.with_timezone(tz).date_naive() {
        t.with_timezone(tz).format("%H:%M").to_string()
    } else {
        when(tz, t)
    }
}

// ---------------------------------------------------------------------------
// search
// ---------------------------------------------------------------------------

pub struct SearchArgs<'a> {
    pub query: &'a str,
    pub since: Option<&'a str>,
    pub until: Option<&'a str>,
    pub app: Option<&'a str>,
    pub limit: u32,
    pub json: bool,
}

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
    if let Some(since) = args.since {
        body["since"] = json!(iso(parse_when(since, now, tz, false)?));
    }
    if let Some(until) = args.until {
        body["until"] = json!(iso(parse_when(until, now, tz, true)?));
    }
    if let Some(app) = args.app {
        body["app"] = json!(app);
    }
    let reply = ask(api, "POST", "/memory/search", Some(&body))?;
    if args.json {
        writeln!(out, "{}", serde_json::to_string_pretty(&reply)?)?;
        return Ok(());
    }
    let moments = reply["moments"].as_array().cloned().unwrap_or_default();
    if moments.is_empty() {
        writeln!(out, "Nothing in Memory matches \"{}\".", args.query)?;
        return Ok(());
    }
    let noun = if moments.len() == 1 { "moment" } else { "moments" };
    writeln!(out, "{} {noun} for \"{}\":\n", moments.len(), args.query)?;
    for m in &moments {
        let text = |key: &str| m[key].as_str().unwrap_or("").to_string();
        let first = parse_iso(&text("first_ts"));
        let last = parse_iso(&text("last_ts"));
        let span = match (first, last) {
            (Some(a), Some(b)) if b.with_timezone(tz).format("%H:%M").to_string()
                != a.with_timezone(tz).format("%H:%M").to_string() =>
            {
                format!("{}–{}", when(tz, a), b.with_timezone(tz).format("%H:%M"))
            }
            (Some(a), _) => when(tz, a),
            _ => "?".to_string(),
        };
        let host = text("url_host");
        let host = if host.is_empty() { String::new() } else { format!(" ({host})") };
        let title = text("window_title");
        let window = if title.is_empty() { text("app") } else { format!("{} — {title}", text("app")) };
        writeln!(out, "  {span}  {window}{host}")?;
        let description = text("description");
        if !description.is_empty() {
            writeln!(out, "      {description}")?;
        }
        writeln!(out, "      id: {}", text("id"))?;
    }
    writeln!(out, "\nForget one with: agenthandover memory forget --moment <id>")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// forget
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct ForgetArgs<'a> {
    pub last: Option<&'a str>,
    pub today: bool,
    pub since: Option<&'a str>,
    pub until: Option<&'a str>,
    pub app: Option<&'a str>,
    pub moment: Option<&'a str>,
    pub all: bool,
    pub yes: bool,
}

/// The `POST /memory/forget` body and how to say it to the user.
pub fn forget_selector<Tz: TimeZone>(
    args: &ForgetArgs,
    now: DateTime<Utc>,
    tz: &Tz,
) -> Result<(Value, String)>
where
    Tz::Offset: std::fmt::Display,
{
    let ranges = [args.last.is_some(), args.today, args.since.is_some()]
        .iter()
        .filter(|x| **x)
        .count();
    let whole = usize::from(args.moment.is_some()) + usize::from(args.all);
    if ranges + whole > 1 || (whole == 1 && args.app.is_some()) {
        bail!("Give one of --last, --today, --since/--until, --app, --moment or --all \
               (--app can go with a time range).");
    }
    if args.until.is_some() && args.since.is_none() {
        bail!("--until needs --since.");
    }
    if let Some(id) = args.moment {
        return Ok((json!({"moment_id": id}), format!("the moment {id}")));
    }
    if args.all {
        return Ok((json!({"all": true}), "everything Memory and your activity history hold".into()));
    }
    let range = if let Some(last) = args.last {
        Some((now - parse_duration(last)?, now))
    } else if args.today {
        let midnight = now.with_timezone(tz).date_naive().and_time(NaiveTime::MIN);
        Some((local_to_utc(tz, midnight)?, now))
    } else if let Some(since) = args.since {
        let until = match args.until {
            Some(until) => parse_when(until, now, tz, true)?,
            None => now,
        };
        Some((parse_when(since, now, tz, false)?, until))
    } else {
        None
    };
    let apps = args.app.map(|a| format!(" in {a}")).unwrap_or_else(|| " (all apps)".into());
    match (range, args.app) {
        (Some((since, until)), app) => {
            if since > until {
                bail!("The start is after the end.");
            }
            let mut body = json!({"since": iso(since), "until": iso(until)});
            if let Some(app) = app {
                body["app"] = json!(app);
            }
            let words = format!(
                "{} to {}{apps}",
                when(tz, since),
                short_when(tz, until, since)
            );
            Ok((body, words))
        }
        (None, Some(app)) => Ok((json!({"app": app}), format!("everything{apps}, on every day"))),
        (None, None) => bail!(
            "Say what to forget: --last 1h, --today, --since T --until T, --app NAME, \
             --moment ID or --all."
        ),
    }
}

pub fn forget<Tz: TimeZone>(
    api: &Api,
    out: &mut dyn Write,
    input: &mut dyn BufRead,
    args: &ForgetArgs,
    now: DateTime<Utc>,
    tz: &Tz,
) -> Result<()>
where
    Tz::Offset: std::fmt::Display,
{
    let (body, words) = forget_selector(args, now, tz)?;
    if !args.yes {
        write!(
            out,
            "This deletes what AgentHandover kept from {words}: Memory moments, \
             screenshots, thumbnails, activity history, search entries and the daily \
             summaries of those days. Approved Skills are not touched.\nForget it? [y/N] "
        )?;
        out.flush()?;
        let mut answer = String::new();
        input.read_line(&mut answer)?;
        if !matches!(answer.trim().to_lowercase().as_str(), "y" | "yes") {
            writeln!(out, "Nothing was forgotten.")?;
            return Ok(());
        }
    }
    let reply = ask(api, "POST", "/memory/forget", Some(&body))?;
    let n = |key: &str| reply[key].as_u64().unwrap_or(0);
    let plural = |count: u64, one: &str, many: &str| {
        format!("{count} {}", if count == 1 { one } else { many })
    };
    writeln!(
        out,
        "Forgotten: {}, {}, {}, {}, {}, {}.",
        plural(n("events"), "captured frame", "captured frames"),
        plural(n("memory_moments"), "Memory moment", "Memory moments"),
        plural(n("thumbnails"), "thumbnail", "thumbnails"),
        plural(n("screenshots"), "screenshot", "screenshots"),
        plural(n("vectors"), "search entry", "search entries"),
        plural(n("daily_summaries"), "daily summary", "daily summaries"),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// pause, resume, status
// ---------------------------------------------------------------------------

pub fn pause<Tz: TimeZone>(
    api: &Api,
    out: &mut dyn Write,
    duration: Option<&str>,
    now: DateTime<Utc>,
    tz: &Tz,
) -> Result<()>
where
    Tz::Offset: std::fmt::Display,
{
    let minutes = match duration {
        Some(text) => {
            let d = parse_duration(text)?;
            json!((d.num_seconds() + 59) / 60)
        }
        None => Value::Null,
    };
    let status = ask(api, "POST", "/memory/pause", Some(&json!({"minutes": minutes})))?;
    match status["paused_until"].as_str().and_then(parse_iso) {
        Some(until) => writeln!(out, "Memory paused until {}.", short_when(tz, until, now))?,
        None => writeln!(
            out,
            "Memory paused until you resume it (agenthandover memory resume)."
        )?,
    }
    writeln!(
        out,
        "Nothing on screen is added to Memory meanwhile. Skills still learn from what you \
         do; pause observing in the app to stop that too."
    )?;
    Ok(())
}

pub fn resume(api: &Api, out: &mut dyn Write) -> Result<()> {
    ask(api, "POST", "/memory/resume", None)?;
    writeln!(out, "Memory resumed.")?;
    Ok(())
}

fn megabytes(bytes: u64) -> String {
    if bytes < 1024 * 1024 {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn status<Tz: TimeZone>(
    api: &Api,
    out: &mut dyn Write,
    as_json: bool,
    now: DateTime<Utc>,
    tz: &Tz,
) -> Result<()>
where
    Tz::Offset: std::fmt::Display,
{
    let s = ask(api, "GET", "/memory/status", None)?;
    if as_json {
        writeln!(out, "{}", serde_json::to_string_pretty(&s)?)?;
        return Ok(());
    }
    let state = if !s["enabled"].as_bool().unwrap_or(false) {
        "off (turn it on in AgentHandover's Settings)".to_string()
    } else if s["paused"].as_bool().unwrap_or(false) {
        match s["paused_until"].as_str().and_then(parse_iso) {
            Some(until) => format!("on, paused until {}", short_when(tz, until, now)),
            None => "on, paused until you resume it".to_string(),
        }
    } else {
        "on".to_string()
    };
    writeln!(out, "Memory: {state}")?;
    let agents = if s["agents_access"].as_bool().unwrap_or(false) {
        "yes (descriptions, apps, titles and times; never images)"
    } else {
        "no (agenthandover memory agents on)"
    };
    writeln!(out, "Agents may search Memory: {agents}")?;
    writeln!(out, "Model: {}", s["model"].as_str().unwrap_or("?"))?;
    let moments = s["moments"].as_u64().unwrap_or(0);
    let span = match (
        s["oldest"].as_str().and_then(parse_iso),
        s["newest"].as_str().and_then(parse_iso),
    ) {
        (Some(a), Some(b)) => format!(", {} to {}", when(tz, a), when(tz, b)),
        _ => String::new(),
    };
    writeln!(
        out,
        "Kept: {moments} moment{} ({} frames){span}",
        if moments == 1 { "" } else { "s" },
        s["frames"].as_u64().unwrap_or(0),
    )?;
    writeln!(out, "Thumbnails: {}", s["thumbnails"].as_u64().unwrap_or(0))?;
    writeln!(out, "Disk: {}", megabytes(s["disk_bytes"].as_u64().unwrap_or(0)))?;
    if let Some(waiting) = s["waiting"].as_str() {
        writeln!(out, "Waiting: {waiting}")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// agents on|off
// ---------------------------------------------------------------------------

/// `config` with `key = value` in `[section]`: the key's line replaced in
/// place, else added right after the header, else the section appended
/// (the edit `tiers::apply_vlm_block` makes for `[vlm]`).
pub fn set_section_key(config: &str, section: &str, key: &str, value: &str) -> String {
    let line = format!("{key} = {value}\n");
    let lines: Vec<&str> = config.split_inclusive('\n').collect();
    let header = format!("[{section}]");
    let Some(start) = lines.iter().position(|l| l.trim() == header) else {
        let mut out = config.to_string();
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&format!("{header}\n{line}"));
        return out;
    };
    let end = lines[start + 1..]
        .iter()
        .position(|l| l.trim_start().starts_with('['))
        .map_or(lines.len(), |i| start + 1 + i);
    let existing = (start + 1..end).find(|&i| {
        lines[i]
            .split_once('=')
            .is_some_and(|(k, _)| k.trim() == key && !lines[i].trim_start().starts_with('#'))
    });
    let mut out: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
    match existing {
        Some(i) => out[i] = line,
        None => {
            if !out[start].ends_with('\n') {
                out[start].push('\n');
            }
            out.insert(start + 1, line);
        }
    }
    out.concat()
}

pub fn agents(out: &mut dyn Write, path: &std::path::Path, on: bool) -> Result<()> {
    let current = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e.into()),
    };
    let updated = set_section_key(&current, "memory", "agents_access", if on { "true" } else { "false" });
    if updated != current {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, &updated)?;
        std::fs::rename(&tmp, path)?;
    }
    if on {
        writeln!(
            out,
            "Agents may now search your Memory over MCP (search_memory): descriptions, apps, \
             window titles and times, never images. Turn it off with: \
             agenthandover memory agents off"
        )?;
    } else {
        writeln!(out, "Agents can no longer search your Memory.")?;
    }
    Ok(())
}

/// `agents on|off` on the real config file.
pub fn agents_switch(on: bool) -> Result<()> {
    agents(&mut std::io::stdout(), &config_path(), on)
}

/// The real clock and zone, for `main`.
pub fn now() -> (DateTime<Utc>, Local) {
    (Utc::now(), Local)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;
    use std::cell::RefCell;

    fn tz() -> FixedOffset {
        FixedOffset::east_opt(2 * 3600).unwrap()
    }

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 10, 12, 12, 0, 0).unwrap()
    }

    /// A fake API answering `reply` and recording each call.
    struct Fake {
        calls: RefCell<Vec<(String, String, Option<Value>)>>,
        reply: Option<(u16, Value)>,
    }

    impl Fake {
        fn new(reply: Option<(u16, Value)>) -> Self {
            Fake { calls: RefCell::new(Vec::new()), reply }
        }
        fn api(&self) -> impl Fn(&str, &str, Option<&Value>) -> Result<Option<(u16, Value)>> + '_ {
            move |m, p, b| {
                self.calls.borrow_mut().push((m.into(), p.into(), b.cloned()));
                Ok(self.reply.clone())
            }
        }
    }

    fn text(out: Vec<u8>) -> String {
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn durations() {
        assert_eq!(parse_duration("30m").unwrap(), Duration::minutes(30));
        assert_eq!(parse_duration("1h").unwrap(), Duration::hours(1));
        assert_eq!(parse_duration("1h30m").unwrap(), Duration::minutes(90));
        assert_eq!(parse_duration("2d").unwrap(), Duration::days(2));
        for bad in ["", "h", "10", "5w", "0m", "-1h", "1.5h"] {
            assert!(parse_duration(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn times_are_local_unless_they_say_otherwise() {
        let t = |s: &str, end: bool| iso(parse_when(s, now(), &tz(), end).unwrap());
        assert_eq!(t("2026-10-12T09:00:00Z", false), "2026-10-12T09:00:00Z");
        assert_eq!(t("2026-10-12T09:00:00+02:00", false), "2026-10-12T07:00:00Z");
        assert_eq!(t("2026-10-12 14:30", false), "2026-10-12T12:30:00Z");
        assert_eq!(t("2026-10-12T14:30:15", false), "2026-10-12T12:30:15Z");
        assert_eq!(t("2026-10-12", false), "2026-10-11T22:00:00Z");
        assert_eq!(t("2026-10-12", true), "2026-10-12T21:59:59Z");
        assert_eq!(t("09:15", false), "2026-10-12T07:15:00Z");
        assert!(parse_when("yesterday", now(), &tz(), false).is_err());
    }

    #[test]
    fn worker_not_running_is_a_plain_error() {
        let fake = Fake::new(None);
        let err = status(&fake.api(), &mut Vec::new(), false, now(), &tz()).unwrap_err();
        assert_eq!(err.to_string(), "AgentHandover's worker isn't running; open the app.");
        let err = resume(&fake.api(), &mut Vec::new()).unwrap_err();
        assert!(err.to_string().contains("worker isn't running"));
    }

    #[test]
    fn api_errors_are_said_in_plain_words() {
        let off = Fake::new(Some((503, json!({"error": "memory_off"}))));
        let args = SearchArgs { query: "x", since: None, until: None, app: None, limit: 5, json: false };
        let err = search(&off.api(), &mut Vec::new(), &args, now(), &tz()).unwrap_err();
        assert!(err.to_string().starts_with("Memory is off."));
        let model = Fake::new(Some((503, json!({"error": "model_unavailable", "detail": "Ollama down"}))));
        let err = search(&model.api(), &mut Vec::new(), &args, now(), &tz()).unwrap_err();
        assert!(err.to_string().contains("(Ollama down)"));
        let bad = Fake::new(Some((400, json!({"error": "'limit' must be an integer from 1 to 100"}))));
        let err = search(&bad.api(), &mut Vec::new(), &args, now(), &tz()).unwrap_err();
        assert_eq!(err.to_string(), "'limit' must be an integer from 1 to 100");
    }

    #[test]
    fn search_sends_the_contract_and_renders_moments_in_local_time() {
        let fake = Fake::new(Some((200, json!({
            "model": "embeddinggemma-2:440m",
            "moments": [{
                "id": "ev-1", "first_ts": "2026-10-12T09:30:00.000Z",
                "last_ts": "2026-10-12T09:35:12.000Z", "frames": 4,
                "app": "Google Chrome", "window_title": "Invoice INV-2041",
                "url_host": "billing.acme.com", "description": "Paying the Acme invoice",
                "score": 0.81, "thumb_id": "ev-1"
            }]
        }))));
        let args = SearchArgs {
            query: "acme invoice", since: Some("2026-10-12"), until: Some("13:00"),
            app: Some("Google Chrome"), limit: 7, json: false,
        };
        let mut out = Vec::new();
        search(&fake.api(), &mut out, &args, now(), &tz()).unwrap();
        let calls = fake.calls.borrow();
        assert_eq!(calls[0].0, "POST");
        assert_eq!(calls[0].1, "/memory/search");
        assert_eq!(calls[0].2, Some(json!({
            "query": "acme invoice", "limit": 7, "app": "Google Chrome",
            "since": "2026-10-11T22:00:00Z", "until": "2026-10-12T11:00:00Z",
        })));
        assert_eq!(text(out), "1 moment for \"acme invoice\":\n\n  \
            Mon 12 Oct 11:30–11:35  Google Chrome — Invoice INV-2041 (billing.acme.com)\n      \
            Paying the Acme invoice\n      id: ev-1\n\n\
            Forget one with: agenthandover memory forget --moment <id>\n");
    }

    #[test]
    fn search_with_nothing_found_and_as_json() {
        let fake = Fake::new(Some((200, json!({"moments": [], "model": "m"}))));
        let mut args = SearchArgs { query: "nope", since: None, until: None, app: None, limit: 20, json: false };
        let mut out = Vec::new();
        search(&fake.api(), &mut out, &args, now(), &tz()).unwrap();
        assert_eq!(text(out), "Nothing in Memory matches \"nope\".\n");
        args.json = true;
        let mut out = Vec::new();
        search(&fake.api(), &mut out, &args, now(), &tz()).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&out).unwrap(), json!({"moments": [], "model": "m"}));
    }

    #[test]
    fn forget_selectors() {
        let sel = |args: ForgetArgs| forget_selector(&args, now(), &tz()).map(|(b, _)| b);
        assert_eq!(
            sel(ForgetArgs { last: Some("1h"), ..Default::default() }).unwrap(),
            json!({"since": "2026-10-12T11:00:00Z", "until": "2026-10-12T12:00:00Z"})
        );
        assert_eq!(
            sel(ForgetArgs { today: true, app: Some("Slack"), ..Default::default() }).unwrap(),
            json!({"since": "2026-10-11T22:00:00Z", "until": "2026-10-12T12:00:00Z", "app": "Slack"})
        );
        assert_eq!(
            sel(ForgetArgs { since: Some("2026-10-10"), until: Some("2026-10-10"), ..Default::default() }).unwrap(),
            json!({"since": "2026-10-09T22:00:00Z", "until": "2026-10-10T21:59:59Z"})
        );
        assert_eq!(
            sel(ForgetArgs { since: Some("10:00"), ..Default::default() }).unwrap(),
            json!({"since": "2026-10-12T08:00:00Z", "until": "2026-10-12T12:00:00Z"})
        );
        assert_eq!(sel(ForgetArgs { app: Some("Slack"), ..Default::default() }).unwrap(), json!({"app": "Slack"}));
        assert_eq!(sel(ForgetArgs { moment: Some("ev-1"), ..Default::default() }).unwrap(), json!({"moment_id": "ev-1"}));
        assert_eq!(sel(ForgetArgs { all: true, ..Default::default() }).unwrap(), json!({"all": true}));
        for bad in [
            ForgetArgs::default(),
            ForgetArgs { last: Some("1h"), today: true, ..Default::default() },
            ForgetArgs { all: true, app: Some("Slack"), ..Default::default() },
            ForgetArgs { moment: Some("x"), all: true, ..Default::default() },
            ForgetArgs { until: Some("10:00"), ..Default::default() },
            ForgetArgs { since: Some("13:00"), until: Some("12:00"), ..Default::default() },
            ForgetArgs { last: Some("soon"), ..Default::default() },
        ] {
            assert!(sel(bad).is_err());
        }
    }

    #[test]
    fn forget_asks_first_and_does_nothing_without_a_yes() {
        let fake = Fake::new(Some((200, json!({}))));
        let args = ForgetArgs { last: Some("1h"), ..Default::default() };
        let mut out = Vec::new();
        forget(&fake.api(), &mut out, &mut "n\n".as_bytes(), &args, now(), &tz()).unwrap();
        let said = text(out);
        assert!(said.contains("from Mon 12 Oct 13:00 to 14:00 (all apps)"), "{said}");
        assert!(said.contains("Approved Skills are not touched."));
        assert!(said.ends_with("Forget it? [y/N] Nothing was forgotten.\n"));
        assert!(fake.calls.borrow().is_empty());
        // End of input (no terminal) is a no.
        let all = ForgetArgs { all: true, ..Default::default() };
        forget(&fake.api(), &mut Vec::new(), &mut "".as_bytes(), &all, now(), &tz()).unwrap();
        assert!(fake.calls.borrow().is_empty());
    }

    #[test]
    fn forget_with_yes_or_after_a_yes_reports_the_counts() {
        let fake = Fake::new(Some((200, json!({
            "events": 3, "memory_moments": 1, "thumbnails": 2, "screenshots": 0,
            "vectors": 6, "daily_summaries": 1
        }))));
        let all = ForgetArgs { all: true, yes: true, ..Default::default() };
        let mut out = Vec::new();
        forget(&fake.api(), &mut out, &mut "".as_bytes(), &all, now(), &tz()).unwrap();
        assert_eq!(text(out), "Forgotten: 3 captured frames, 1 Memory moment, 2 thumbnails, \
            0 screenshots, 6 search entries, 1 daily summary.\n");
        assert_eq!(fake.calls.borrow()[0].2, Some(json!({"all": true})));
        let moment = ForgetArgs { moment: Some("ev-1"), ..Default::default() };
        forget(&fake.api(), &mut Vec::new(), &mut "yes\n".as_bytes(), &moment, now(), &tz()).unwrap();
        assert_eq!(fake.calls.borrow()[1].1, "/memory/forget");
        assert_eq!(fake.calls.borrow()[1].2, Some(json!({"moment_id": "ev-1"})));
    }

    #[test]
    fn pause_for_a_while_or_until_resumed() {
        let timed = Fake::new(Some((200, json!({"paused": true, "paused_until": "2026-10-12T13:30:00Z"}))));
        let mut out = Vec::new();
        pause(&timed.api(), &mut out, Some("90m"), now(), &tz()).unwrap();
        assert_eq!(timed.calls.borrow()[0].2, Some(json!({"minutes": 90})));
        assert!(text(out).starts_with("Memory paused until 15:30.\n"));
        let open = Fake::new(Some((200, json!({"paused": true, "paused_until": null}))));
        let mut out = Vec::new();
        pause(&open.api(), &mut out, None, now(), &tz()).unwrap();
        assert_eq!(open.calls.borrow()[0].2, Some(json!({"minutes": null})));
        assert!(text(out).starts_with("Memory paused until you resume it"));
        // Seconds round up to a whole minute.
        let fake = Fake::new(Some((200, json!({"paused_until": null}))));
        pause(&fake.api(), &mut Vec::new(), Some("90s"), now(), &tz()).unwrap();
        assert_eq!(fake.calls.borrow()[0].2, Some(json!({"minutes": 2})));
        let fake = Fake::new(Some((200, json!({}))));
        let mut out = Vec::new();
        resume(&fake.api(), &mut out).unwrap();
        assert_eq!(fake.calls.borrow()[0].0, "POST");
        assert_eq!(fake.calls.borrow()[0].1, "/memory/resume");
        assert_eq!(text(out), "Memory resumed.\n");
    }

    #[test]
    fn status_in_plain_words_and_as_json() {
        let reply = json!({
            "enabled": true, "paused": true, "paused_until": "2026-10-12T13:30:00Z",
            "agents_access": false, "model": "embeddinggemma-2:440m", "moments": 1234,
            "frames": 5678, "oldest": "2026-10-05T07:12:00Z", "newest": "2026-10-12T09:50:00Z",
            "thumbnails": 42, "disk_bytes": 12_900_000u64, "waiting": null
        });
        let fake = Fake::new(Some((200, reply.clone())));
        let mut out = Vec::new();
        status(&fake.api(), &mut out, false, now(), &tz()).unwrap();
        assert_eq!(fake.calls.borrow()[0].0, "GET");
        assert_eq!(fake.calls.borrow()[0].1, "/memory/status");
        assert_eq!(text(out), "Memory: on, paused until 15:30\n\
            Agents may search Memory: no (agenthandover memory agents on)\n\
            Model: embeddinggemma-2:440m\n\
            Kept: 1234 moments (5678 frames), Mon 5 Oct 09:12 to Mon 12 Oct 11:50\n\
            Thumbnails: 42\nDisk: 12.3 MB\n");
        let mut out = Vec::new();
        status(&fake.api(), &mut out, true, now(), &tz()).unwrap();
        assert_eq!(serde_json::from_slice::<Value>(&out).unwrap(), reply);
        let off = Fake::new(Some((200, json!({"enabled": false, "waiting": "the model is loading"}))));
        let mut out = Vec::new();
        status(&off.api(), &mut out, false, now(), &tz()).unwrap();
        let said = text(out);
        assert!(said.starts_with("Memory: off (turn it on in AgentHandover's Settings)\n"));
        assert!(said.ends_with("Waiting: the model is loading\n"));
    }

    #[test]
    fn the_section_updater_sets_one_key_and_keeps_the_rest() {
        assert_eq!(set_section_key("", "memory", "agents_access", "true"),
                   "[memory]\nagents_access = true\n");
        assert_eq!(
            set_section_key("[vlm]\ntier = \"x\"", "memory", "agents_access", "true"),
            "[vlm]\ntier = \"x\"\n\n[memory]\nagents_access = true\n"
        );
        assert_eq!(
            set_section_key("[memory]\nenabled = true\n[vlm]\nagents_access = 1\n", "memory", "agents_access", "false"),
            "[memory]\nagents_access = false\nenabled = true\n[vlm]\nagents_access = 1\n"
        );
        assert_eq!(
            set_section_key("[memory]\nenabled = true\nagents_access = false # was\nthumbnail_days = 7\n",
                            "memory", "agents_access", "true"),
            "[memory]\nenabled = true\nagents_access = true\nthumbnail_days = 7\n"
        );
        assert_eq!(
            set_section_key("[memory]\n# agents_access = maybe\n", "memory", "agents_access", "true"),
            "[memory]\nagents_access = true\n# agents_access = maybe\n"
        );
        assert_eq!(set_section_key("[memory]", "memory", "agents_access", "true"),
                   "[memory]\nagents_access = true\n");
    }

    #[test]
    fn agents_switch_writes_the_config_the_worker_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sub/config.toml");
        let mut out = Vec::new();
        agents(&mut out, &path, true).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "[memory]\nagents_access = true\n");
        assert!(text(out).contains("never images"));
        std::fs::write(&path, "[memory]\nenabled = true\nagents_access = true\n").unwrap();
        agents(&mut Vec::new(), &path, false).unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        assert_eq!(written, "[memory]\nenabled = true\nagents_access = false\n");
        let parsed = agenthandover_common::config::AppConfig::from_toml_str(&written).unwrap();
        assert!(!parsed.memory.agents_access);
        assert!(parsed.memory.enabled);
    }
}
