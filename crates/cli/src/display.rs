//! Shared display formatting helpers for CLI commands.

use chrono::{DateTime, Utc};

/// Format a number with comma separators (e.g. 1247 -> "1,247").
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Format a relative time string like "2s ago", "5m ago", "2h ago".
pub fn format_relative_time(timestamp: &str) -> String {
    match timestamp.parse::<DateTime<Utc>>() {
        Ok(dt) => {
            let now = Utc::now();
            let diff = now.signed_duration_since(dt);
            let secs = diff.num_seconds();
            if secs < 0 {
                "just now".to_string()
            } else if secs < 60 {
                format!("{}s ago", secs)
            } else if secs < 3600 {
                format!("{}m ago", secs / 60)
            } else if secs < 86400 {
                format!("{}h ago", secs / 3600)
            } else {
                format!("{}d ago", secs / 86400)
            }
        }
        Err(_) => timestamp.to_string(),
    }
}

/// Check if a heartbeat timestamp is stale (older than 2 minutes).
///
/// Returns `true` if the timestamp cannot be parsed — an unparseable
/// heartbeat is treated as stale because freshness cannot be verified.
pub fn is_heartbeat_stale(timestamp: &str) -> bool {
    match timestamp.parse::<DateTime<Utc>>() {
        Ok(dt) => {
            let now = Utc::now();
            now.signed_duration_since(dt).num_seconds() > 120
        }
        Err(_) => true,
    }
}

/// Format seconds as human-readable uptime ("5m 30s", "2h 15m", "3d 4h").
pub fn format_uptime(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else if secs < 86400 {
        format!("{}h {}m", secs / 3600, (secs % 3600) / 60)
    } else {
        format!("{}d {}h", secs / 86400, (secs % 86400) / 3600)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_number_basic() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(42), "42");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1_247), "1,247");
        assert_eq!(format_number(1_000_000), "1,000,000");
    }

    #[test]
    fn format_uptime_ranges() {
        assert_eq!(format_uptime(0), "0s");
        assert_eq!(format_uptime(59), "59s");
        assert_eq!(format_uptime(60), "1m 0s");
        assert_eq!(format_uptime(3661), "1h 1m");
        assert_eq!(format_uptime(86400), "1d 0h");
        assert_eq!(format_uptime(90061), "1d 1h");
    }

    #[test]
    fn stale_returns_true_on_unparseable() {
        assert!(is_heartbeat_stale("not-a-timestamp"));
        assert!(is_heartbeat_stale(""));
    }

    #[test]
    fn stale_returns_false_for_recent_timestamp() {
        let now = Utc::now().to_rfc3339();
        assert!(!is_heartbeat_stale(&now));
    }

    #[test]
    fn stale_returns_true_for_old_timestamp() {
        let old = (Utc::now() - chrono::Duration::seconds(300)).to_rfc3339();
        assert!(is_heartbeat_stale(&old));
    }

    #[test]
    fn relative_time_for_recent() {
        let now = Utc::now().to_rfc3339();
        let result = format_relative_time(&now);
        assert!(result.contains("ago") || result == "just now");
    }

    #[test]
    fn relative_time_for_unparseable() {
        assert_eq!(format_relative_time("garbage"), "garbage");
    }
}
