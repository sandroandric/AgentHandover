//! Paste matching: a paste the browser extension saw is linked to the copy
//! it came from.
//!
//! The extension reports a `paste` DOM event (Cmd-V, Edit > Paste or the
//! context menu) as a `PasteDetected` event with no hash.  The daemon then
//! hashes what is on the pasteboard at that moment, with the same function
//! the clipboard poller uses for `ClipboardChange` events, and sets
//! `matched_copy_hash` when a copy with that hash was recorded within
//! [`PASTE_MATCH_WINDOW`].  The worker's `ClipboardLinker` pairs the two
//! events by that hash.

use agenthandover_common::event::{Event, EventKind};
use agenthandover_storage::EventStore;
use std::path::Path;
use tracing::warn;

/// How far back a copy may lie from its paste.
pub const PASTE_MATCH_WINDOW: chrono::Duration = chrono::Duration::minutes(30);

/// Fill `matched_copy_hash` of a `PasteDetected` event from `pasted_hash`
/// (the hash of the pasteboard content at the paste) when the database at
/// `db_path` holds a copy with that hash within [`PASTE_MATCH_WINDOW`] of
/// the event.  Other events and unmatched pastes are left as they are.
pub fn match_paste(event: &mut Event, pasted_hash: Option<&str>, db_path: &Path) {
    let EventKind::PasteDetected { matched_copy_hash } = &mut event.kind else {
        return;
    };
    let Some(hash) = pasted_hash.filter(|h| !h.is_empty()) else {
        return;
    };
    let since = event.timestamp - PASTE_MATCH_WINDOW;
    match EventStore::open(db_path).and_then(|store| store.has_copy_since(hash, since)) {
        Ok(true) => *matched_copy_hash = Some(hash.to_string()),
        Ok(false) => {}
        Err(e) => warn!(error = %e, "Paste matching: could not read the event store"),
    }
}

/// The hash of the pasteboard's current content, as the clipboard poller
/// computes it (`capture_clipboard_meta`).
#[cfg(target_os = "macos")]
fn pasteboard_hash() -> Option<String> {
    crate::platform::clipboard_monitor::capture_clipboard_meta(None)
        .map(|meta| meta.content_hash)
        .filter(|hash| !hash.is_empty())
}

#[cfg(not(target_os = "macos"))]
fn pasteboard_hash() -> Option<String> {
    None
}

/// A copy made just before the paste may not be stored yet (the poller
/// runs every 500 ms and the storage writer after it): one retry after this.
const COPY_SETTLE: std::time::Duration = std::time::Duration::from_millis(1500);

/// [`match_paste`] with the live pasteboard, off the async executor.
/// A no-op for every event but `PasteDetected`.  The pasteboard is hashed
/// once, when the paste arrives; the store is asked again after
/// [`COPY_SETTLE`] when the copy is not there yet.
pub async fn resolve_paste(event: Event, db_path: &Path) -> Event {
    if !matches!(event.kind, EventKind::PasteDetected { .. }) {
        return event;
    }
    let hash = tokio::task::spawn_blocking(pasteboard_hash).await.ok().flatten();
    if hash.is_none() {
        return event;
    }
    let mut event = event;
    for attempt in 0..2 {
        if attempt > 0 {
            tokio::time::sleep(COPY_SETTLE).await;
        }
        let (db, h, mut candidate) = (db_path.to_path_buf(), hash.clone(), event.clone());
        let result = tokio::task::spawn_blocking(move || {
            match_paste(&mut candidate, h.as_deref(), &db);
            candidate
        })
        .await;
        if let Ok(resolved) = result {
            event = resolved;
        }
        if matches!(&event.kind, EventKind::PasteDetected { matched_copy_hash: Some(_) }) {
            break;
        }
    }
    event
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn event(kind: EventKind, minutes_ago: i64) -> Event {
        Event {
            id: Uuid::new_v4(),
            timestamp: Utc::now() - chrono::Duration::minutes(minutes_ago),
            kind,
            window: None,
            display_topology: vec![],
            primary_display_id: String::new(),
            cursor_global_px: None,
            ui_scale: None,
            artifact_ids: vec![],
            metadata: serde_json::json!({}),
            display_ids_spanned: None,
            focused_element: None,
        }
    }

    fn copy(hash: &str, minutes_ago: i64) -> Event {
        event(
            EventKind::ClipboardChange {
                content_types: vec!["public.utf8-plain-text".into()],
                byte_size: 5,
                high_entropy: false,
                content_hash: hash.into(),
                content_preview: None,
            },
            minutes_ago,
        )
    }

    fn paste() -> Event {
        event(EventKind::PasteDetected { matched_copy_hash: None }, 0)
    }

    fn matched(event: &Event) -> Option<&str> {
        match &event.kind {
            EventKind::PasteDetected { matched_copy_hash } => matched_copy_hash.as_deref(),
            other => panic!("not a paste: {other:?}"),
        }
    }

    #[test]
    fn paste_of_a_recent_copy_carries_its_hash() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("events.db");
        EventStore::open(&db).unwrap().insert_event(&copy("hash-a", 3)).unwrap();

        let mut pasted = paste();
        match_paste(&mut pasted, Some("hash-a"), &db);
        assert_eq!(matched(&pasted), Some("hash-a"));
    }

    #[test]
    fn paste_without_a_matching_copy_stays_unmatched() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("events.db");
        let store = EventStore::open(&db).unwrap();
        store.insert_event(&copy("hash-a", 3)).unwrap();
        store.insert_event(&copy("hash-old", 45)).unwrap();

        let mut other = paste();
        match_paste(&mut other, Some("hash-b"), &db);
        assert_eq!(matched(&other), None, "content never copied while observed");

        let mut stale = paste();
        match_paste(&mut stale, Some("hash-old"), &db);
        assert_eq!(matched(&stale), None, "copy outside the 30-minute window");

        let mut unknown = paste();
        match_paste(&mut unknown, None, &db);
        assert_eq!(matched(&unknown), None, "pasteboard unreadable");
    }

    #[test]
    fn other_events_are_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("events.db");
        let mut click = event(EventKind::ClickIntent { target_description: "x".into() }, 0);
        match_paste(&mut click, Some("hash-a"), &db);
        assert_eq!(click.kind, EventKind::ClickIntent { target_description: "x".into() });
        assert!(!db.exists(), "no store is opened for a non-paste event");
    }
}
