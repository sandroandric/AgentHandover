use agenthandover_test_harness::artifact::{ArtifactRecorder, ArtifactReplayer};
use tempfile::TempDir;
use uuid::Uuid;

#[test]
fn test_artifact_record_and_replay_round_trip() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("artifacts");

    let event_id = Uuid::new_v4();
    let data = b"screenshot pixel data here";

    // Record
    let mut recorder = ArtifactRecorder::new(&dir).unwrap();
    let artifact_id = recorder.record(event_id, "screenshot", data).unwrap();
    recorder.flush().unwrap();
    assert_eq!(recorder.artifact_count(), 1);

    // Replay
    let replayer = ArtifactReplayer::from_dir(&dir).unwrap();
    assert_eq!(replayer.artifact_count(), 1);

    let entry = &replayer.entries()[0];
    assert_eq!(entry.artifact_id, artifact_id);
    assert_eq!(entry.event_id, event_id);
    assert_eq!(entry.artifact_type, "screenshot");
    assert_eq!(entry.original_size, data.len());

    // Read raw data back
    let read_data = replayer.read_artifact(entry).unwrap();
    assert_eq!(read_data, data);
}

#[test]
fn test_multiple_artifacts_for_same_event() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("artifacts");

    let event_id = Uuid::new_v4();
    let screenshot_data = b"screenshot bytes";
    let dom_data = b"<html>dom snapshot</html>";

    let mut recorder = ArtifactRecorder::new(&dir).unwrap();
    recorder.record(event_id, "screenshot", screenshot_data).unwrap();
    recorder.record(event_id, "dom_snapshot", dom_data).unwrap();
    recorder.flush().unwrap();
    assert_eq!(recorder.artifact_count(), 2);

    let replayer = ArtifactReplayer::from_dir(&dir).unwrap();
    assert_eq!(replayer.artifact_count(), 2);

    let event_artifacts = replayer.artifacts_for_event(event_id);
    assert_eq!(event_artifacts.len(), 2);

    // Verify types
    let types: Vec<&str> = event_artifacts.iter().map(|e| e.artifact_type.as_str()).collect();
    assert!(types.contains(&"screenshot"));
    assert!(types.contains(&"dom_snapshot"));
}

#[test]
fn test_artifacts_for_different_events() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("artifacts");

    let event_a = Uuid::new_v4();
    let event_b = Uuid::new_v4();

    let mut recorder = ArtifactRecorder::new(&dir).unwrap();
    recorder.record(event_a, "screenshot", b"data_a").unwrap();
    recorder.record(event_b, "screenshot", b"data_b").unwrap();
    recorder.record(event_a, "dom", b"dom_a").unwrap();
    recorder.flush().unwrap();

    let replayer = ArtifactReplayer::from_dir(&dir).unwrap();
    assert_eq!(replayer.artifact_count(), 3);

    // Event A has 2 artifacts
    assert_eq!(replayer.artifacts_for_event(event_a).len(), 2);
    // Event B has 1 artifact
    assert_eq!(replayer.artifacts_for_event(event_b).len(), 1);
    // Unknown event has 0
    assert_eq!(replayer.artifacts_for_event(Uuid::new_v4()).len(), 0);
}

#[test]
fn test_empty_artifact_data() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("artifacts");

    let event_id = Uuid::new_v4();
    let mut recorder = ArtifactRecorder::new(&dir).unwrap();
    recorder.record(event_id, "empty", b"").unwrap();
    recorder.flush().unwrap();

    let replayer = ArtifactReplayer::from_dir(&dir).unwrap();
    let entry = &replayer.entries()[0];
    assert_eq!(entry.original_size, 0);

    let data = replayer.read_artifact(entry).unwrap();
    assert!(data.is_empty());
}

#[test]
fn test_large_artifact_data() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path().join("artifacts");

    let event_id = Uuid::new_v4();
    let large_data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

    let mut recorder = ArtifactRecorder::new(&dir).unwrap();
    recorder.record(event_id, "large_screenshot", &large_data).unwrap();
    recorder.flush().unwrap();

    let replayer = ArtifactReplayer::from_dir(&dir).unwrap();
    let entry = &replayer.entries()[0];
    assert_eq!(entry.original_size, 100_000);

    let read_data = replayer.read_artifact(entry).unwrap();
    assert_eq!(read_data, large_data);
}
