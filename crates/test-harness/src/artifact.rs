use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Metadata about a recorded artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactEntry {
    pub artifact_id: Uuid,
    pub event_id: Uuid,
    pub artifact_type: String,
    pub original_size: usize,
    pub file_name: String,
}

/// Records artifact data alongside events to a directory.
///
/// Creates an index file (`artifacts.jsonl`) with metadata and
/// individual `.bin` files for each artifact's raw data.
pub struct ArtifactRecorder {
    dir: PathBuf,
    index_writer: BufWriter<File>,
    count: usize,
}

impl ArtifactRecorder {
    pub fn new(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let index_path = dir.join("artifacts.jsonl");
        let file = File::create(index_path)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            index_writer: BufWriter::new(file),
            count: 0,
        })
    }

    /// Record an artifact: writes raw data to a `.bin` file and appends
    /// an entry to the index.
    pub fn record(
        &mut self,
        event_id: Uuid,
        artifact_type: &str,
        data: &[u8],
    ) -> Result<Uuid> {
        let artifact_id = Uuid::new_v4();
        let file_name = format!("{}.bin", artifact_id);
        let data_path = self.dir.join(&file_name);

        // Write raw data
        fs::write(&data_path, data)?;

        // Write index entry
        let entry = ArtifactEntry {
            artifact_id,
            event_id,
            artifact_type: artifact_type.to_string(),
            original_size: data.len(),
            file_name,
        };
        let json = serde_json::to_string(&entry)?;
        writeln!(self.index_writer, "{}", json)?;
        self.count += 1;

        Ok(artifact_id)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.index_writer.flush()?;
        Ok(())
    }

    pub fn artifact_count(&self) -> usize {
        self.count
    }
}

/// Replays artifact data from a directory written by `ArtifactRecorder`.
pub struct ArtifactReplayer {
    dir: PathBuf,
    entries: Vec<ArtifactEntry>,
}

impl ArtifactReplayer {
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let index_path = dir.join("artifacts.jsonl");
        let file = File::open(index_path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let entry: ArtifactEntry = serde_json::from_str(&line)?;
            entries.push(entry);
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            entries,
        })
    }

    /// Return all artifact entries.
    pub fn entries(&self) -> &[ArtifactEntry] {
        &self.entries
    }

    pub fn artifact_count(&self) -> usize {
        self.entries.len()
    }

    /// Read the raw data for a given artifact entry.
    pub fn read_artifact(&self, entry: &ArtifactEntry) -> Result<Vec<u8>> {
        let path = self.dir.join(&entry.file_name);
        let data = fs::read(&path)?;
        Ok(data)
    }

    /// Find all artifact entries for a given event ID.
    pub fn artifacts_for_event(&self, event_id: Uuid) -> Vec<&ArtifactEntry> {
        self.entries
            .iter()
            .filter(|e| e.event_id == event_id)
            .collect()
    }
}
