use agenthandover_common::event::{Event, EventKind};

/// Result of a pipeline verification run.
#[derive(Debug)]
pub struct PipelineResult {
    pub total_events: usize,
    pub matched_events: usize,
    pub violations: Vec<PipelineViolation>,
}

impl PipelineResult {
    pub fn is_ok(&self) -> bool {
        self.violations.is_empty()
    }
}

/// A single pattern violation found during pipeline verification.
#[derive(Debug)]
pub struct PipelineViolation {
    pub event_index: usize,
    pub event_id: String,
    pub rule: String,
    pub message: String,
}

/// A predicate for matching and verifying event patterns in a replayed stream.
pub enum EventPattern {
    /// Match events by their EventKind discriminant.
    KindIs(fn(&EventKind) -> bool),
    /// Match events where a predicate on the full Event is true.
    Predicate(Box<dyn Fn(&Event) -> bool>),
}

/// A rule that verifies a property across a stream of events.
pub enum PipelineRule {
    /// Every event in the stream must satisfy the predicate.
    All(String, Box<dyn Fn(&Event) -> bool>),
    /// At least one event must satisfy the predicate.
    Any(String, Box<dyn Fn(&Event) -> bool>),
    /// No event may satisfy the predicate (inverse of Any).
    None(String, Box<dyn Fn(&Event) -> bool>),
    /// Events matching `pattern` must appear in the given order.
    OrderedSequence(String, Vec<EventPattern>),
    /// The metadata field must not contain the given substring.
    MetadataExcludes(String, String),
    /// Window title must not contain the given substring.
    TitleExcludes(String, String),
}

/// Takes replayed events and verifies they follow expected patterns.
pub struct PipelineRunner {
    rules: Vec<PipelineRule>,
}

impl PipelineRunner {
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule to verify.
    pub fn add_rule(&mut self, rule: PipelineRule) {
        self.rules.push(rule);
    }

    /// Verify all rules against the given events.
    pub fn verify(&self, events: &[Event]) -> PipelineResult {
        let mut violations = Vec::new();
        let mut matched = 0;

        for rule in &self.rules {
            match rule {
                PipelineRule::All(name, predicate) => {
                    for (i, event) in events.iter().enumerate() {
                        if predicate(event) {
                            matched += 1;
                        } else {
                            violations.push(PipelineViolation {
                                event_index: i,
                                event_id: event.id.to_string(),
                                rule: name.clone(),
                                message: format!(
                                    "Event at index {} failed 'All' rule: {}",
                                    i, name
                                ),
                            });
                        }
                    }
                }
                PipelineRule::Any(name, predicate) => {
                    let found = events.iter().any(predicate);
                    if !found {
                        violations.push(PipelineViolation {
                            event_index: 0,
                            event_id: String::new(),
                            rule: name.clone(),
                            message: format!(
                                "No event matched 'Any' rule: {}",
                                name
                            ),
                        });
                    } else {
                        matched += 1;
                    }
                }
                PipelineRule::None(name, predicate) => {
                    for (i, event) in events.iter().enumerate() {
                        if predicate(event) {
                            violations.push(PipelineViolation {
                                event_index: i,
                                event_id: event.id.to_string(),
                                rule: name.clone(),
                                message: format!(
                                    "Event at index {} violated 'None' rule: {}",
                                    i, name
                                ),
                            });
                        }
                    }
                    if violations.is_empty() {
                        matched += 1;
                    }
                }
                PipelineRule::OrderedSequence(name, patterns) => {
                    let mut pattern_idx = 0;
                    for event in events {
                        if pattern_idx >= patterns.len() {
                            break;
                        }
                        let matches = match &patterns[pattern_idx] {
                            EventPattern::KindIs(f) => f(&event.kind),
                            EventPattern::Predicate(f) => f(event),
                        };
                        if matches {
                            pattern_idx += 1;
                        }
                    }
                    if pattern_idx == patterns.len() {
                        matched += 1;
                    } else {
                        violations.push(PipelineViolation {
                            event_index: 0,
                            event_id: String::new(),
                            rule: name.clone(),
                            message: format!(
                                "OrderedSequence '{}' only matched {}/{} patterns",
                                name,
                                pattern_idx,
                                patterns.len()
                            ),
                        });
                    }
                }
                PipelineRule::MetadataExcludes(name, substring) => {
                    for (i, event) in events.iter().enumerate() {
                        let meta_str = event.metadata.to_string();
                        if meta_str.contains(substring.as_str()) {
                            violations.push(PipelineViolation {
                                event_index: i,
                                event_id: event.id.to_string(),
                                rule: name.clone(),
                                message: format!(
                                    "Event at index {} metadata contains '{}' (rule: {})",
                                    i, substring, name
                                ),
                            });
                        }
                    }
                    if violations.iter().all(|v| v.rule != *name) {
                        matched += 1;
                    }
                }
                PipelineRule::TitleExcludes(name, substring) => {
                    for (i, event) in events.iter().enumerate() {
                        if let Some(ref w) = event.window {
                            if w.title.contains(substring.as_str()) {
                                violations.push(PipelineViolation {
                                    event_index: i,
                                    event_id: event.id.to_string(),
                                    rule: name.clone(),
                                    message: format!(
                                        "Event at index {} title contains '{}' (rule: {})",
                                        i, substring, name
                                    ),
                                });
                            }
                        }
                    }
                    if violations.iter().all(|v| v.rule != *name) {
                        matched += 1;
                    }
                }
            }
        }

        PipelineResult {
            total_events: events.len(),
            matched_events: matched,
            violations,
        }
    }
}

impl Default for PipelineRunner {
    fn default() -> Self {
        Self::new()
    }
}
