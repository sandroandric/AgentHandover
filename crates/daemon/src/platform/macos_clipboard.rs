//! macOS clipboard monitoring via NSPasteboard.
//!
//! Polls NSPasteboard.generalPasteboard's changeCount every 500ms.
//! On change, captures content types, byte size, SHA-256 hash, and entropy score.
//! Emits ClipboardChange events and tracks recent hashes for paste matching.

use crate::capture::clipboard::{hash_content, is_high_entropy};
use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use std::ffi::{c_long, c_void};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

// Objective-C runtime FFI bindings (libobjc.dylib ships with macOS).
#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const u8) -> *mut c_void;
    fn sel_registerName(name: *const u8) -> *mut c_void;
    fn objc_msgSend(obj: *mut c_void, sel: *mut c_void, ...) -> *mut c_void;
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
/// Returns None if the Objective-C call fails.
fn get_pasteboard_change_count() -> Option<i64> {
    unsafe {
        let class = objc_getClass(b"NSPasteboard\0".as_ptr());
        if class.is_null() {
            return None;
        }
        let sel_general = sel_registerName(b"generalPasteboard\0".as_ptr());
        let pasteboard = objc_msgSend(class, sel_general);
        if pasteboard.is_null() {
            return None;
        }
        let sel_count = sel_registerName(b"changeCount\0".as_ptr());
        let count = objc_msgSend(pasteboard, sel_count) as c_long;
        Some(count as i64)
    }
}

/// Get the content types (UTIs) currently on the pasteboard.
fn get_pasteboard_types() -> Vec<String> {
    unsafe {
        let class = objc_getClass(b"NSPasteboard\0".as_ptr());
        if class.is_null() {
            return vec![];
        }
        let sel_general = sel_registerName(b"generalPasteboard\0".as_ptr());
        let pasteboard = objc_msgSend(class, sel_general);
        if pasteboard.is_null() {
            return vec![];
        }
        let sel_types = sel_registerName(b"types\0".as_ptr());
        let types_array = objc_msgSend(pasteboard, sel_types);
        if types_array.is_null() {
            return vec![];
        }

        let sel_count = sel_registerName(b"count\0".as_ptr());
        let count = objc_msgSend(types_array, sel_count) as usize;

        let sel_object_at = sel_registerName(b"objectAtIndex:\0".as_ptr());
        let sel_utf8 = sel_registerName(b"UTF8String\0".as_ptr());

        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let obj = objc_msgSend(types_array, sel_object_at, i as c_long);
            if !obj.is_null() {
                let cstr = objc_msgSend(obj, sel_utf8) as *const u8;
                if !cstr.is_null() {
                    let s = std::ffi::CStr::from_ptr(cstr as *const _);
                    if let Ok(rust_str) = s.to_str() {
                        result.push(rust_str.to_string());
                    }
                }
            }
        }
        result
    }
}

/// Get the raw data for a specific pasteboard type.
fn get_pasteboard_data_for_type(type_name: &str) -> Option<Vec<u8>> {
    unsafe {
        let class = objc_getClass(b"NSPasteboard\0".as_ptr());
        if class.is_null() {
            return None;
        }
        let sel_general = sel_registerName(b"generalPasteboard\0".as_ptr());
        let pasteboard = objc_msgSend(class, sel_general);
        if pasteboard.is_null() {
            return None;
        }

        // Create NSString for the type name.
        let ns_string_class = objc_getClass(b"NSString\0".as_ptr());
        if ns_string_class.is_null() {
            return None;
        }
        let type_cstr = std::ffi::CString::new(type_name).ok()?;
        let sel_string_with = sel_registerName(b"stringWithUTF8String:\0".as_ptr());
        let ns_type = objc_msgSend(ns_string_class, sel_string_with, type_cstr.as_ptr());
        if ns_type.is_null() {
            return None;
        }

        // Get NSData for this type.
        let sel_data_for = sel_registerName(b"dataForType:\0".as_ptr());
        let ns_data = objc_msgSend(pasteboard, sel_data_for, ns_type);
        if ns_data.is_null() {
            return None;
        }

        // Get bytes and length from NSData.
        let sel_bytes = sel_registerName(b"bytes\0".as_ptr());
        let sel_length = sel_registerName(b"length\0".as_ptr());
        let bytes_ptr = objc_msgSend(ns_data, sel_bytes) as *const u8;
        let length = objc_msgSend(ns_data, sel_length) as usize;

        if bytes_ptr.is_null() || length == 0 {
            return Some(vec![]);
        }

        let slice = std::slice::from_raw_parts(bytes_ptr, length);
        Some(slice.to_vec())
    }
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

    #[test]
    fn test_get_pasteboard_change_count() {
        // Should return a non-negative value on a desktop macOS system.
        let count = get_pasteboard_change_count();
        if let Some(c) = count {
            assert!(c >= 0, "changeCount should be non-negative");
        }
    }

    #[test]
    fn test_get_pasteboard_types() {
        // On a desktop macOS system with anything on the clipboard,
        // this should return at least one type.
        let types = get_pasteboard_types();
        // May be empty if clipboard has just been cleared, that's OK.
        let _ = types;
    }

    #[test]
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
