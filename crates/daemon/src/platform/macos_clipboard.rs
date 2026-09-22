//! macOS clipboard monitoring via NSPasteboard.
//!
//! Polls NSPasteboard.generalPasteboard's changeCount every 500ms.
//! On change, captures content types, byte size, SHA-256 hash, and entropy score.
//! Emits ClipboardChange events and tracks recent hashes for paste matching.

use crate::capture::clipboard::{clipboard_preview, hash_content, is_high_entropy};
use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::ffi::CStr;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

// ── C FFI types matching objc_try_catch.m pasteboard helpers ──────────

#[repr(C)]
struct PasteboardInfo {
    change_count: i64,
    types: *mut *mut i8, // NULL-terminated array of C strings
    type_count: i32,
    data: *mut u8,
    data_len: usize,
    success: i32,
}

extern "C" {
    fn pasteboard_change_count_safe() -> i64;
    fn pasteboard_get_info_safe() -> PasteboardInfo;
    fn free_pasteboard_info(info: *mut PasteboardInfo);
}

/// Represents a clipboard change detected by the monitor.
#[derive(Debug, Clone)]
pub struct ClipboardChangeEvent {
    pub content_types: Vec<String>,
    pub byte_size: u64,
    pub high_entropy: bool,
    pub content_hash: String,
    pub timestamp: DateTime<Utc>,
    /// Leading chars of plaintext clipboard content, present only when the
    /// user enabled `[privacy] enable_clipboard_preview`.  None for non-text
    /// types or high-entropy content (potential secrets).
    pub content_preview: Option<String>,
}

/// Tracks recent clipboard hashes for paste detection.
/// Keeps hashes for the last 30 minutes.
pub struct ClipboardHashTracker {
    entries: VecDeque<(String, DateTime<Utc>)>,
    max_age: Duration,
}

impl ClipboardHashTracker {
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            max_age: Duration::from_secs(30 * 60), // 30 minutes
        }
    }

    /// Record a new clipboard hash.
    pub fn record(&mut self, hash: String) {
        self.prune();
        self.entries.push_back((hash, Utc::now()));
    }

    /// Check if a paste matches a recent copy by hash.
    pub fn find_match(&self, hash: &str) -> Option<String> {
        self.entries
            .iter()
            .rev()
            .find(|(h, _)| h == hash)
            .map(|(h, _)| h.clone())
    }

    /// Remove entries older than max_age.
    fn prune(&mut self) {
        let cutoff = Utc::now() - chrono::Duration::from_std(self.max_age).unwrap_or_default();
        while let Some((_, ts)) = self.entries.front() {
            if *ts < cutoff {
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }
}

impl Default for ClipboardHashTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the current NSPasteboard.generalPasteboard changeCount.
/// Returns None if the Objective-C call fails or an exception was caught.
pub fn get_pasteboard_change_count() -> Option<i64> {
    let count = unsafe { pasteboard_change_count_safe() };
    if count < 0 { None } else { Some(count) }
}

/// Get the content types (UTIs) currently on the pasteboard.
fn get_pasteboard_types() -> Vec<String> {
    let mut info = unsafe { pasteboard_get_info_safe() };
    if info.success == 0 || info.types.is_null() {
        unsafe { free_pasteboard_info(&mut info) };
        return vec![];
    }

    let mut result = Vec::with_capacity(info.type_count as usize);
    for i in 0..info.type_count {
        let cstr_ptr = unsafe { *info.types.add(i as usize) };
        if !cstr_ptr.is_null() {
            if let Ok(s) = unsafe { CStr::from_ptr(cstr_ptr) }.to_str() {
                result.push(s.to_string());
            }
        }
    }

    unsafe { free_pasteboard_info(&mut info) };
    result
}

/// Get the raw data for the first available pasteboard type.
fn get_pasteboard_data_for_type(_type_name: &str) -> Option<Vec<u8>> {
    let mut info = unsafe { pasteboard_get_info_safe() };
    if info.success == 0 || info.data.is_null() || info.data_len == 0 {
        unsafe { free_pasteboard_info(&mut info) };
        return None;
    }

    let data = unsafe { std::slice::from_raw_parts(info.data, info.data_len) }.to_vec();
    unsafe { free_pasteboard_info(&mut info) };
    Some(data)
}

/// Capture current clipboard metadata.  A text preview is attached only
/// when `preview_max_chars` is `Some` (the user opted in via
/// `[privacy] enable_clipboard_preview`); see [`clipboard_preview`].
pub fn capture_clipboard_meta(preview_max_chars: Option<usize>) -> Option<ClipboardChangeEvent> {
    let types = get_pasteboard_types();
    if types.is_empty() {
        return None;
    }

    // Get data for the first available type to compute hash and size.
    // Prefer plaintext types for entropy analysis.
    let plaintext_types = [
        "public.utf8-plain-text",
        "public.plain-text",
        "NSStringPboardType",
    ];

    let data_type = plaintext_types
        .iter()
        .find(|&&t| types.iter().any(|existing| existing == t))
        .copied()
        .or(types.first().map(|s| s.as_str()));

    let is_plaintext = data_type
        .map(|dt| plaintext_types.contains(&dt))
        .unwrap_or(false);

    let (byte_size, content_hash, high_entropy, content_preview) = if let Some(dt) = data_type {
        match get_pasteboard_data_for_type(dt) {
            Some(data) => {
                let size = data.len() as u64;
                let hash = hash_content(&data);
                let entropy = is_high_entropy(&data);

                let preview = clipboard_preview(&data, is_plaintext, entropy, preview_max_chars);

                (size, hash, entropy, preview)
            }
            None => (0, String::new(), false, None),
        }
    } else {
        (0, String::new(), false, None)
    };

    Some(ClipboardChangeEvent {
        content_types: types,
        byte_size,
        high_entropy,
        content_hash,
        timestamp: Utc::now(),
        content_preview,
    })
}

/// Message type for clipboard events sent to the observer.
#[derive(Debug)]
pub enum ClipboardMessage {
    Change(ClipboardChangeEvent),
    Shutdown,
}

/// Run the clipboard monitoring loop as an async task.
///
/// Polls NSPasteboard.generalPasteboard changeCount every 500ms.
/// On change: captures metadata and sends a ClipboardMessage::Change.
/// `preview_max_chars` is passed to [`capture_clipboard_meta`].
pub async fn run_clipboard_monitor(
    tx: mpsc::Sender<ClipboardMessage>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    preview_max_chars: Option<usize>,
) {
    let poll_interval = Duration::from_millis(500);
    let mut interval = tokio::time::interval(poll_interval);

    // Read initial change count.
    let mut last_change_count = tokio::task::spawn_blocking(get_pasteboard_change_count)
        .await
        .unwrap_or(None)
        .unwrap_or(0);

    debug!(
        change_count = last_change_count,
        "Clipboard monitor started"
    );

    loop {
        tokio::select! {
            _ = interval.tick() => {
                if *shutdown_rx.borrow() {
                    debug!("Clipboard monitor received shutdown");
                    let _ = tx.send(ClipboardMessage::Shutdown).await;
                    break;
                }

                // Check change count in a blocking thread. Runs to completion
                // (no timeout) because the ObjC call inside
                // pasteboard_change_count_safe() is wrapped in
                // @try/@catch/@autoreleasepool and a Tokio-level timeout would
                // orphan the blocking thread — when the thread eventually
                // finishes, ObjC cleanup runs outside the @try/@catch scope
                // and an uncaught exception triggers abort(). Same bug class
                // as the v0.2.8 OCR timeout crash (hikoae issue #1).
                let current_count = match tokio::task::spawn_blocking(get_pasteboard_change_count).await {
                    Ok(Some(c)) => c,
                    Ok(None) => {
                        warn!("Failed to read pasteboard changeCount");
                        continue;
                    }
                    Err(e) => {
                        error!(error = %e, "Pasteboard changeCount task panicked");
                        continue;
                    }
                };

                if current_count != last_change_count {
                    last_change_count = current_count;

                    // Capture clipboard metadata in a blocking thread.
                    // No timeout — same reasoning as changeCount above.
                    let meta = match tokio::task::spawn_blocking(move || capture_clipboard_meta(preview_max_chars)).await {
                        Ok(Some(m)) => m,
                        Ok(None) => continue,
                        Err(e) => {
                            error!(error = %e, "Clipboard meta capture panicked");
                            continue;
                        }
                    };

                    debug!(
                        types = ?meta.content_types,
                        byte_size = meta.byte_size,
                        high_entropy = meta.high_entropy,
                        "Clipboard changed"
                    );

                    if tx.send(ClipboardMessage::Change(meta)).await.is_err() {
                        debug!("Clipboard monitor: receiver dropped, stopping");
                        break;
                    }
                }
            }
            _ = shutdown_rx.changed() => {
                debug!("Clipboard monitor: shutdown watch triggered");
                let _ = tx.send(ClipboardMessage::Shutdown).await;
                break;
            }
        }
    }

    // Grace period for any in-flight blocking tasks
    tokio::time::sleep(Duration::from_millis(100)).await;
    info!("Clipboard monitor stopped");
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial(macos_ffi)]
    fn test_get_pasteboard_change_count() {
        // Should return a non-negative value on a desktop macOS system.
        let count = get_pasteboard_change_count();
        if let Some(c) = count {
            assert!(c >= 0, "changeCount should be non-negative");
        }
    }

    #[test]
    #[serial(macos_ffi)]
    fn test_get_pasteboard_types() {
        // On a desktop macOS system with anything on the clipboard,
        // this should return at least one type.
        let types = get_pasteboard_types();
        // May be empty if clipboard has just been cleared, that's OK.
        let _ = types;
    }

    #[test]
    #[serial(macos_ffi)]
    fn test_capture_clipboard_meta() {
        let meta = capture_clipboard_meta(None);
        if let Some(ref m) = meta {
            assert!(m.content_preview.is_none(), "preview must stay off unless opted in");
        }
        if let Some(m) = meta {
            assert!(!m.content_types.is_empty());
            assert!(!m.content_hash.is_empty());
        }
    }

    #[test]
    fn test_clipboard_hash_tracker() {
        let mut tracker = ClipboardHashTracker::new();
        tracker.record("abc123".to_string());
        tracker.record("def456".to_string());

        assert_eq!(tracker.find_match("abc123"), Some("abc123".to_string()));
        assert_eq!(tracker.find_match("def456"), Some("def456".to_string()));
        assert_eq!(tracker.find_match("nonexistent"), None);
    }

    #[test]
    fn test_clipboard_hash_tracker_prune() {
        let mut tracker = ClipboardHashTracker {
            entries: VecDeque::new(),
            max_age: Duration::from_secs(0), // Expire immediately.
        };
        tracker.entries.push_back((
            "old_hash".to_string(),
            Utc::now() - chrono::Duration::seconds(1),
        ));

        // Pruning should remove the expired entry.
        tracker.prune();
        assert!(tracker.entries.is_empty());
    }

    #[tokio::test]
    async fn test_clipboard_monitor_shutdown() {
        let (tx, mut rx) = mpsc::channel(16);
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

        let handle = tokio::spawn(run_clipboard_monitor(tx, shutdown_rx, None));

        // Signal shutdown immediately.
        shutdown_tx.send(true).unwrap();

        // The monitor should stop and we should get a Shutdown message.
        handle.await.unwrap();

        // Drain messages -- we should find a Shutdown message.
        let mut got_shutdown = false;
        while let Ok(msg) = rx.try_recv() {
            if matches!(msg, ClipboardMessage::Shutdown) {
                got_shutdown = true;
            }
        }
        assert!(got_shutdown, "Should receive Shutdown message");
    }
}
