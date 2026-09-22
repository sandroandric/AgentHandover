//! Browser transitions handed from the native-messaging bridge to the
//! observer.
//!
//! Chrome launches the bridge as its own process, so a click, navigation or
//! form submit it receives cannot reach the observer loop in memory. The
//! bridge overwrites one small file in the data dir with the latest
//! transition; the observer reads it each tick and fires the shutter when
//! its timestamp is newer than the last one it saw. Only the kind and the
//! time are written: the event itself (URL, target) goes to SQLite as
//! before.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// File name under the data dir.
pub const BROWSER_TRANSITION_FILE: &str = "browser-transition.json";

/// What happened in the active tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrowserTransitionKind {
    Navigation,
    Click,
    FormSubmit,
}

/// The latest browser transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrowserTransition {
    pub kind: BrowserTransitionKind,
    pub at: DateTime<Utc>,
}

/// Replace the transition file in `dir` (write to a temp file, then rename,
/// so a reader never sees a partial file). No fsync: a transition lost to a
/// crash only costs one frame.
pub fn write_browser_transition(dir: &Path, transition: &BrowserTransition) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let tmp = dir.join(format!(".{BROWSER_TRANSITION_FILE}.tmp"));
    let json = serde_json::to_vec(transition)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, dir.join(BROWSER_TRANSITION_FILE))
}

/// Read the transition file in `dir`; `None` when it is missing or unreadable.
pub fn read_browser_transition(dir: &Path) -> Option<BrowserTransition> {
    let bytes = std::fs::read(dir.join(BROWSER_TRANSITION_FILE)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Tracks the last transition the observer consumed.
#[derive(Debug, Default)]
pub struct BrowserTransitionWatch {
    last_seen: Option<DateTime<Utc>>,
}

impl BrowserTransitionWatch {
    /// A watch that ignores transitions at or before `since` (the observer
    /// passes its start time so a file left by an earlier run never fires).
    pub fn since(since: DateTime<Utc>) -> Self {
        Self { last_seen: Some(since) }
    }

    /// The transition in `dir` if it is newer than the last one returned.
    pub fn poll(&mut self, dir: &Path) -> Option<BrowserTransitionKind> {
        let transition = read_browser_transition(dir)?;
        if self.last_seen.is_some_and(|seen| transition.at <= seen) {
            return None;
        }
        self.last_seen = Some(transition.at);
        Some(transition.kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn roundtrip_and_poll_fires_once_per_transition() {
        let dir = tempfile::tempdir().unwrap();
        let start = Utc::now();
        let mut watch = BrowserTransitionWatch::since(start);
        assert_eq!(watch.poll(dir.path()), None, "no file yet");

        let first = BrowserTransition {
            kind: BrowserTransitionKind::Navigation,
            at: start + Duration::milliseconds(10),
        };
        write_browser_transition(dir.path(), &first).unwrap();
        assert_eq!(read_browser_transition(dir.path()), Some(first));
        assert_eq!(watch.poll(dir.path()), Some(BrowserTransitionKind::Navigation));
        assert_eq!(watch.poll(dir.path()), None, "same transition twice");

        let second = BrowserTransition {
            kind: BrowserTransitionKind::FormSubmit,
            at: start + Duration::milliseconds(20),
        };
        write_browser_transition(dir.path(), &second).unwrap();
        assert_eq!(watch.poll(dir.path()), Some(BrowserTransitionKind::FormSubmit));
    }

    #[test]
    fn a_file_from_before_the_watch_started_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let old = BrowserTransition {
            kind: BrowserTransitionKind::Click,
            at: Utc::now() - Duration::seconds(30),
        };
        write_browser_transition(dir.path(), &old).unwrap();
        let mut watch = BrowserTransitionWatch::since(Utc::now());
        assert_eq!(watch.poll(dir.path()), None);
    }

    #[test]
    fn a_corrupt_file_reads_as_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(BROWSER_TRANSITION_FILE), b"{not json").unwrap();
        assert_eq!(read_browser_transition(dir.path()), None);
    }
}
