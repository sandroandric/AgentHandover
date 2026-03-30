use agenthandover_common::event::*;
use agenthandover_common::redaction::Redactor;
use agenthandover_test_harness::pipeline::{EventPattern, PipelineRule, PipelineRunner};
use chrono::Utc;
use uuid::Uuid;

fn make_event(kind: EventKind) -> Event {
    Event {
        id: Uuid::new_v4(),
        timestamp: Utc::now(),
        kind,
        window: Some(WindowInfo {
            window_id: "win_1".into(),
            app_id: "com.test.App".into(),
            title: "Test Window".into(),
            bounds_global_px: [0, 0, 800, 600],
            z_order: 0,
            is_fullscreen: false,
        }),
        display_topology: vec![],
        primary_display_id: "d1".into(),
        cursor_global_px: None,
        ui_scale: None,
        artifact_ids: vec![],
        metadata: serde_json::json!({}),
        display_ids_spanned: None,
    }
}

fn make_event_with_title(title: &str) -> Event {
    let mut event = make_event(EventKind::FocusChange);
    if let Some(ref mut w) = event.window {
        w.title = title.into();
    }
    event
}

fn make_event_with_metadata(key: &str, value: &str) -> Event {
    let mut event = make_event(EventKind::DwellSnapshot);
    event.metadata = serde_json::json!({ key: value });
    event
}

// ===== All Rule Tests =====

#[test]
fn test_all_rule_passes_when_all_match() {
    let events = vec![
        make_event(EventKind::FocusChange),
        make_event(EventKind::FocusChange),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::All(
        "all_have_window".into(),
        Box::new(|e| e.window.is_some()),
    ));

    let result = runner.verify(&events);
    assert!(result.is_ok());
    assert_eq!(result.total_events, 2);
}

#[test]
fn test_all_rule_fails_when_one_mismatches() {
    let mut event_no_window = make_event(EventKind::FocusChange);
    event_no_window.window = None;

    let events = vec![
        make_event(EventKind::FocusChange),
        event_no_window,
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::All(
        "all_have_window".into(),
        Box::new(|e| e.window.is_some()),
    ));

    let result = runner.verify(&events);
    assert!(!result.is_ok());
    assert_eq!(result.violations.len(), 1);
    assert_eq!(result.violations[0].event_index, 1);
}

// ===== Any Rule Tests =====

#[test]
fn test_any_rule_passes_when_one_matches() {
    let events = vec![
        make_event(EventKind::FocusChange),
        make_event(EventKind::DwellSnapshot),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::Any(
        "has_dwell".into(),
        Box::new(|e| matches!(e.kind, EventKind::DwellSnapshot)),
    ));

    let result = runner.verify(&events);
    assert!(result.is_ok());
}

#[test]
fn test_any_rule_fails_when_none_match() {
    let events = vec![
        make_event(EventKind::FocusChange),
        make_event(EventKind::FocusChange),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::Any(
        "has_dwell".into(),
        Box::new(|e| matches!(e.kind, EventKind::DwellSnapshot)),
    ));

    let result = runner.verify(&events);
    assert!(!result.is_ok());
    assert_eq!(result.violations.len(), 1);
}

// ===== None Rule Tests =====

#[test]
fn test_none_rule_passes_when_no_match() {
    let events = vec![
        make_event(EventKind::FocusChange),
        make_event(EventKind::DwellSnapshot),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::None(
        "no_secure_field".into(),
        Box::new(|e| matches!(e.kind, EventKind::SecureFieldFocus)),
    ));

    let result = runner.verify(&events);
    assert!(result.is_ok());
}

#[test]
fn test_none_rule_fails_when_match_found() {
    let events = vec![
        make_event(EventKind::FocusChange),
        make_event(EventKind::SecureFieldFocus),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::None(
        "no_secure_field".into(),
        Box::new(|e| matches!(e.kind, EventKind::SecureFieldFocus)),
    ));

    let result = runner.verify(&events);
    assert!(!result.is_ok());
    assert_eq!(result.violations[0].event_index, 1);
}

// ===== OrderedSequence Tests =====

#[test]
fn test_ordered_sequence_passes_in_order() {
    let events = vec![
        make_event(EventKind::FocusChange),
        make_event(EventKind::DwellSnapshot),
        make_event(EventKind::AppSwitch { from_app: "A".into(), to_app: "B".into() }),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::OrderedSequence(
        "focus_then_dwell_then_switch".into(),
        vec![
            EventPattern::KindIs(|k| matches!(k, EventKind::FocusChange)),
            EventPattern::KindIs(|k| matches!(k, EventKind::DwellSnapshot)),
            EventPattern::KindIs(|k| matches!(k, EventKind::AppSwitch { .. })),
        ],
    ));

    let result = runner.verify(&events);
    assert!(result.is_ok());
}

#[test]
fn test_ordered_sequence_fails_when_out_of_order() {
    let events = vec![
        make_event(EventKind::DwellSnapshot),
        make_event(EventKind::FocusChange),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::OrderedSequence(
        "focus_then_dwell".into(),
        vec![
            EventPattern::KindIs(|k| matches!(k, EventKind::FocusChange)),
            EventPattern::KindIs(|k| matches!(k, EventKind::DwellSnapshot)),
        ],
    ));

    // FocusChange is found at index 1 (after DwellSnapshot), but then DwellSnapshot
    // was already consumed at index 0. The second pattern (DwellSnapshot) is never
    // found after FocusChange.
    let result = runner.verify(&events);
    assert!(!result.is_ok());
}

#[test]
fn test_ordered_sequence_with_gaps() {
    let events = vec![
        make_event(EventKind::FocusChange),
        make_event(EventKind::ScrollReadSnapshot),
        make_event(EventKind::DwellSnapshot),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::OrderedSequence(
        "focus_then_dwell".into(),
        vec![
            EventPattern::KindIs(|k| matches!(k, EventKind::FocusChange)),
            EventPattern::KindIs(|k| matches!(k, EventKind::DwellSnapshot)),
        ],
    ));

    // Should pass even with ScrollReadSnapshot in between
    let result = runner.verify(&events);
    assert!(result.is_ok());
}

// ===== MetadataExcludes Tests =====

#[test]
fn test_metadata_excludes_passes_when_absent() {
    let events = vec![
        make_event_with_metadata("key", "normal_value"),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::MetadataExcludes(
        "no_secret".into(),
        "AKIAIOSFODNN7EXAMPLE".into(),
    ));

    let result = runner.verify(&events);
    assert!(result.is_ok());
}

#[test]
fn test_metadata_excludes_fails_when_present() {
    let events = vec![
        make_event_with_metadata("aws_key", "AKIAIOSFODNN7EXAMPLE"),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::MetadataExcludes(
        "no_aws_key".into(),
        "AKIAIOSFODNN7EXAMPLE".into(),
    ));

    let result = runner.verify(&events);
    assert!(!result.is_ok());
}

// ===== TitleExcludes Tests =====

#[test]
fn test_title_excludes_passes_when_absent() {
    let events = vec![
        make_event_with_title("Google Chrome - Search"),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::TitleExcludes(
        "no_credit_card".into(),
        "4111-1111-1111-1111".into(),
    ));

    let result = runner.verify(&events);
    assert!(result.is_ok());
}

#[test]
fn test_title_excludes_fails_when_present() {
    let events = vec![
        make_event_with_title("Payment: 4111-1111-1111-1111"),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::TitleExcludes(
        "no_credit_card".into(),
        "4111-1111-1111-1111".into(),
    ));

    let result = runner.verify(&events);
    assert!(!result.is_ok());
}

// ===== Combined Pipeline + Redaction Test =====

#[test]
fn test_pipeline_runner_verifies_redacted_events() {
    let redactor = Redactor::new();

    // Create events with secrets
    let mut events = vec![
        make_event_with_title("export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"),
        make_event_with_title("Card: 4111-1111-1111-1111"),
        make_event_with_title("Normal window title"),
    ];

    // Redact all event titles
    for event in &mut events {
        if let Some(ref mut w) = event.window {
            w.title = redactor.redact(&w.title);
        }
    }

    // Verify no secrets remain
    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::TitleExcludes(
        "no_aws_key".into(),
        "AKIAIOSFODNN7EXAMPLE".into(),
    ));
    runner.add_rule(PipelineRule::TitleExcludes(
        "no_cc_number".into(),
        "4111-1111-1111-1111".into(),
    ));
    // Normal text should still be there
    runner.add_rule(PipelineRule::Any(
        "has_normal_title".into(),
        Box::new(|e| {
            e.window.as_ref().map_or(false, |w| w.title == "Normal window title")
        }),
    ));

    let result = runner.verify(&events);
    assert!(result.is_ok(), "Violations: {:?}", result.violations);
}

// ===== Multiple Rules Test =====

#[test]
fn test_multiple_rules_all_checked() {
    let events = vec![
        make_event(EventKind::FocusChange),
        make_event(EventKind::DwellSnapshot),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::All(
        "all_have_window".into(),
        Box::new(|e| e.window.is_some()),
    ));
    runner.add_rule(PipelineRule::Any(
        "has_focus".into(),
        Box::new(|e| matches!(e.kind, EventKind::FocusChange)),
    ));
    runner.add_rule(PipelineRule::None(
        "no_secure".into(),
        Box::new(|e| matches!(e.kind, EventKind::SecureFieldFocus)),
    ));

    let result = runner.verify(&events);
    assert!(result.is_ok());
    assert_eq!(result.matched_events, 4); // 2 from All + 1 from Any + 1 from None
}

// ===== EventPattern::Predicate Test =====

#[test]
fn test_predicate_pattern_in_sequence() {
    let events = vec![
        make_event_with_title("Login Page"),
        make_event_with_title("Dashboard"),
    ];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::OrderedSequence(
        "login_then_dashboard".into(),
        vec![
            EventPattern::Predicate(Box::new(|e| {
                e.window.as_ref().map_or(false, |w| w.title.contains("Login"))
            })),
            EventPattern::Predicate(Box::new(|e| {
                e.window.as_ref().map_or(false, |w| w.title.contains("Dashboard"))
            })),
        ],
    ));

    let result = runner.verify(&events);
    assert!(result.is_ok());
}

// ===== Empty Events Test =====

#[test]
fn test_empty_events_all_passes() {
    let events: Vec<Event> = vec![];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::All(
        "all_have_window".into(),
        Box::new(|e| e.window.is_some()),
    ));

    // All on empty is vacuously true
    let result = runner.verify(&events);
    assert!(result.is_ok());
}

#[test]
fn test_empty_events_any_fails() {
    let events: Vec<Event> = vec![];

    let mut runner = PipelineRunner::new();
    runner.add_rule(PipelineRule::Any(
        "has_something".into(),
        Box::new(|_| true),
    ));

    let result = runner.verify(&events);
    assert!(!result.is_ok());
}
