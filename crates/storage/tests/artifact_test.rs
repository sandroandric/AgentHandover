use agenthandover_storage::artifact_store::ArtifactStore;
use tempfile::TempDir;

#[test]
fn test_store_and_retrieve_artifact() {
    let tmp = TempDir::new().unwrap();
    let store = ArtifactStore::new(tmp.path().to_path_buf(), [0u8; 32]);

    let data = b"Hello, this is test artifact data for a DOM snapshot.";
    let meta = store.store(data, "dom_snapshot").unwrap();

    let retrieved = store.retrieve(&meta.artifact_id).unwrap();
    assert_eq!(retrieved, data);
}

#[test]
fn test_artifact_is_compressed_and_encrypted() {
    let tmp = TempDir::new().unwrap();
    let store = ArtifactStore::new(tmp.path().to_path_buf(), [42u8; 32]);

    let data = b"Repeated data for compression test. ".repeat(100);
    let meta = store.store(&data, "screenshot").unwrap();

    // Read raw file — should NOT contain plaintext
    let raw = std::fs::read(store.artifact_path(&meta.artifact_id)).unwrap();
    assert!(!raw.windows(10).any(|w| w == b"Repeated d"));

    // Stored size should be smaller due to compression
    assert!(raw.len() < data.len());

    // ArtifactMeta should report correct sizes
    assert_eq!(meta.original_size_bytes, data.len() as u64);
    assert_eq!(meta.stored_size_bytes, raw.len() as u64);
    assert_eq!(meta.artifact_type, "screenshot");
}

#[test]
fn test_artifact_path_uses_date_hierarchy() {
    let tmp = TempDir::new().unwrap();
    let store = ArtifactStore::new(tmp.path().to_path_buf(), [0u8; 32]);

    let data = b"test";
    let meta = store.store(data, "test").unwrap();
    let path = store.artifact_path(&meta.artifact_id);

    // Path should contain yyyy/mm/dd structure
    let path_str = path.to_string_lossy();
    let re = regex::Regex::new(r"\d{4}/\d{2}/\d{2}").unwrap();
    assert!(re.is_match(&path_str), "Path should contain date hierarchy: {}", path_str);

    // ArtifactMeta.file_path should match
    assert_eq!(meta.file_path, path);
}
