use oc_apprentice_daemon::capture::clipboard::{ClipboardMeta, hash_content, is_high_entropy};

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
