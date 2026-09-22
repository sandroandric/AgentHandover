use agenthandover_daemon::capture::clipboard::{clipboard_preview, ClipboardMeta, hash_content, is_high_entropy};

#[test]
fn test_clipboard_preview_off_by_default() {
    // `enable_clipboard_preview = false` maps to `None`: no text is kept.
    assert_eq!(clipboard_preview(b"meeting notes", true, false, None), None);
}

#[test]
fn test_clipboard_preview_honours_max_chars() {
    assert_eq!(
        clipboard_preview(b"  meeting notes for monday  ", true, false, Some(7)),
        Some("meeting".to_string())
    );
    // Char-based truncation never splits a multi-byte character.
    assert_eq!(
        clipboard_preview("héllo wörld".as_bytes(), true, false, Some(4)),
        Some("héll".to_string())
    );
    assert_eq!(clipboard_preview(b"notes", true, false, Some(0)), None);
}

#[test]
fn test_clipboard_preview_second_line_gates() {
    // Opted in, but high-entropy content, non-text types, invalid UTF-8
    // and whitespace-only text still get no preview.
    assert_eq!(clipboard_preview(b"notes", true, true, Some(200)), None);
    assert_eq!(clipboard_preview(b"notes", false, false, Some(200)), None);
    assert_eq!(clipboard_preview(&[0xff, 0xfe], true, false, Some(200)), None);
    assert_eq!(clipboard_preview(b"   ", true, false, Some(200)), None);
}

#[test]
fn test_hash_content_deterministic() {
    let a = hash_content(b"hello world");
    let b = hash_content(b"hello world");
    assert_eq!(a, b);
}

#[test]
fn test_hash_content_different() {
    let a = hash_content(b"hello");
    let b = hash_content(b"world");
    assert_ne!(a, b);
}

#[test]
fn test_clipboard_meta_creation() {
    let meta = ClipboardMeta {
        content_types: vec!["text/plain".into()],
        byte_size: 42,
        high_entropy: false,
        content_hash: hash_content(b"test"),
    };
    assert_eq!(meta.byte_size, 42);
    assert!(!meta.high_entropy);
}

#[test]
fn test_high_entropy_detection() {
    // Random-looking bytes should be high entropy
    let random_data: Vec<u8> = (0..256).map(|i| (i * 37 + 13) as u8).collect();
    assert!(is_high_entropy(&random_data));

    // Repeated data should be low entropy
    let repeated = vec![0x41u8; 256];
    assert!(!is_high_entropy(&repeated));
}
