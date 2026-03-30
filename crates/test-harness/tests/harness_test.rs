use agenthandover_test_harness::recorder::EventRecorder;
use agenthandover_test_harness::replayer::EventReplayer;
use agenthandover_common::event::*;
use chrono::Utc;
use uuid::Uuid;
use tempfile::TempDir;

fn make_test_event(kind: EventKind) -> Event {
    Event {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        kind,
        window: Some(WindowInfo {
            window_id: "win_1".into(),
            app_id: "com.test.App".into(),
            title: "Test".into(),
            bounds_global_px: [0, 0, 800, 600],
            z_order: 0,
            is_fullscreen: false,
        }),
        display_topology: vec![DisplayInfo {
            display_id: "d1".into(),
            bounds_global_px: [0, 0, 2560, 1440],
            scale_factor: 2.0,
            orientation: 0,
        }],
        primary_display_id: "d1".into(),
        cursor_global_px: Some(CursorPosition { x: 100, y: 200 }),
        ui_scale: Some(2.0),
        artifact_ids: vec![],
        metadata: serde_json::json!({}),
        display_ids_spanned: None,
    }
}

#[test]
fn test_record_and_replay_round_trip() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("events.jsonl");

    // Record
    let mut recorder = EventRecorder::new(&path).unwrap();
    let events: Vec<Event> = vec![
        make_test_event(EventKind::FocusChange),
        make_test_event(EventKind::DwellSnapshot),
        make_test_event(EventKind::AppSwitch { from_app: "A".into(), to_app: "B".into() }),
    ];
    for event in &events {
        recorder.record(event).unwrap();
    }
    recorder.flush().unwrap();
    assert_eq!(recorder.event_count(), 3);

    // Replay
    let replayer = EventReplayer::from_file(&path).unwrap();
    assert_eq!(replayer.event_count(), 3);

    let replayed: Vec<&Event> = replayer.iter().collect();
    assert_eq!(replayed[0].id, events[0].id);
    assert_eq!(replayed[1].kind, EventKind::DwellSnapshot);
    assert_eq!(replayed[2].id, events[2].id);
}

#[test]
fn test_empty_file_replays_zero_events() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("empty.jsonl");
    std::fs::write(&path, "").unwrap();

    let replayer = EventReplayer::from_file(&path).unwrap();
    assert_eq!(replayer.event_count(), 0);
}

#[test]
fn test_recorder_count_tracks_events() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("count.jsonl");
    let mut recorder = EventRecorder::new(&path).unwrap();

    assert_eq!(recorder.event_count(), 0);
    recorder.record(&make_test_event(EventKind::FocusChange)).unwrap();
    assert_eq!(recorder.event_count(), 1);
    recorder.record(&make_test_event(EventKind::DwellSnapshot)).unwrap();
    assert_eq!(recorder.event_count(), 2);
}
