use anyhow::Result;
use agenthandover_common::event::Event;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Records events to a JSON Lines (.jsonl) file for replay.
pub struct EventRecorder {
    writer: BufWriter<File>,
    count: usize,
}

impl EventRecorder {
    pub fn new(path: &Path) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            count: 0,
        })
    }

    pub fn record(&mut self, event: &Event) -> Result<()> {
        let json = serde_json::to_string(event)?;
        writeln!(self.writer, "{}", json)?;
        self.count += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    pub fn event_count(&self) -> usize {
        self.count
    }
}
