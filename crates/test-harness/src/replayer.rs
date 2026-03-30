use anyhow::Result;
use agenthandover_common::event::Event;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Replays events from a JSON Lines (.jsonl) file.
pub struct EventReplayer {
    events: Vec<Event>,
}

impl EventReplayer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let event: Event = serde_json::from_str(&line)?;
            events.push(event);
        }

        Ok(Self { events })
    }

    pub fn events(&self) -> &[Event] {
        &self.events
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Iterate events in order.
    pub fn iter(&self) -> impl Iterator<Item = &Event> {
        self.events.iter()
    }
}
