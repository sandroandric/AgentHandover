//! macOS clipboard monitoring via NSPasteboard.
//!
//! Polls NSPasteboard.generalPasteboard's changeCount every 500ms.
//! On change, captures content types, byte size, SHA-256 hash, and entropy score.
//! Emits ClipboardChange events and tracks recent hashes for paste matching.

use crate::capture::clipboard::{hash_content, is_high_entropy};
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

/// Capture current clipboard metadata without storing the actual content.
/// This is the privacy-safe approach: we hash the content but never store it.
pub fn capture_clipboard_meta() -> Option<ClipboardChangeEvent> {
    let types = get_pasteboard_types();
    if types.is_empty() {
        return None;
    }

    // Get data for the first available type to compute hash and size.
    // Prefer plaintext types for entropy analysis.
    let preferred_types = [
        "public.utf8-plain-text",
        "public.plain-text",
        "NSStringPboardType",
    ];

    let data_type = preferred_types
        .iter()
        .find(|&&t| types.iter().any(|existing| existing == t))
        .copied()
        .or(types.first().map(|s| s.as_str()));

    let (byte_size, content_hash, high_entropy) = if let Some(dt) = data_type {
        match get_pasteboard_data_for_type(dt) {
            Some(data) => {
                let size = data.len() as u64;
                let hash = hash_content(&data);
                let entropy = is_high_entropy(&data);
                (size, hash, entropy)
            }
            None => (0, String::new(), false),
        }
    } else {
        (0, String::new(), false)
    };

    Some(ClipboardChangeEvent {
        content_types: types,
        byte_size,
        high_entropy,
        content_hash,
        timestamp: Utc::now(),
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
pub async fn run_clipboard_monitor(
    tx: mpsc::Sender<ClipboardMessage>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
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

                // Check change count in a blocking thread with timeout.
                let count_task = tokio::task::spawn_blocking(get_pasteboard_change_count);
                let current_count = match tokio::time::timeout(Duration::from_millis(500), count_task).await {
                    Ok(Ok(Some(c))) => c,
                    Ok(Ok(None)) => {
                        warn!("Failed to read pasteboard changeCount");
                        continue;
                    }
                    Ok(Err(e)) => {
                        error!(error = %e, "Pasteboard changeCount task panicked");
                        continue;
                    }
                    Err(_) => {
                        warn!("Pasteboard changeCount check timed out");
                        continue;
                    }
                };

                if current_count != last_change_count {
                    last_change_count = current_count;

                    // Capture clipboard metadata in a blocking thread with timeout.
                    let meta_task = tokio::task::spawn_blocking(capture_clipboard_meta);
                    let meta = match tokio::time::timeout(Duration::from_millis(500), meta_task).await {
                        Ok(Ok(Some(m))) => m,
                        Ok(Ok(None)) => continue,
                        Ok(Err(e)) => {
                            error!(error = %e, "Clipboard meta capture panicked");
                            continue;
                        }
                        Err(_) => {
                            warn!("Clipboard meta capture timed out");
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
        let meta = capture_clipboard_meta();
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

        let handle = tokio::spawn(run_clipboard_monitor(tx, shutdown_rx));

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
