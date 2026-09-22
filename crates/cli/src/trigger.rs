//! Trigger spool: how the CLI asks the worker to do something.
//!
//! Each request is its own file, `<data_dir>/triggers/<kind>-<uuid>.json`,
//! written via tmp + fsync + rename.  The worker drains the directory once
//! per tick, oldest first, so two requests made before the worker's next
//! tick are both honoured (the old single-slot `<kind>-trigger.json` file
//! kept only the last one).

use anyhow::Result;
use agenthandover_common::status::data_dir;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const SPOOL_DIR: &str = "triggers";

/// Queue one request of `kind` for the worker; returns the spool file.
pub fn write_trigger(kind: &str, payload: &serde_json::Value) -> Result<PathBuf> {
    write_trigger_in(&data_dir(), kind, payload)
}

pub fn write_trigger_in(
    state_dir: &Path,
    kind: &str,
    payload: &serde_json::Value,
) -> Result<PathBuf> {
    let dir = state_dir.join(SPOOL_DIR);
    std::fs::create_dir_all(&dir)?;
    let name = format!("{}-{}.json", kind, uuid::Uuid::new_v4());
    let target = dir.join(&name);
    let tmp = dir.join(format!(".{}.tmp", name));

    let json = serde_json::to_string_pretty(payload)?;
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(json.as_bytes())?;
    file.sync_all()?;
    std::fs::rename(&tmp, &target)?;

    Ok(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_request_gets_its_own_spool_file() {
        let dir = std::env::temp_dir().join(format!("ah-spool-{}", uuid::Uuid::new_v4()));
        let payload = serde_json::json!({"sop_id": "a", "action": "approve"});

        let first = write_trigger_in(&dir, "approve", &payload).unwrap();
        let second = write_trigger_in(&dir, "approve", &payload).unwrap();

        assert_ne!(first, second);
        assert_eq!(first.parent().unwrap(), dir.join(SPOOL_DIR));
        let name = first.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with("approve-") && name.ends_with(".json"));
        // "approve-" + 36-char UUID + ".json"
        assert_eq!(name.len(), "approve-".len() + 36 + ".json".len());

        let mut names: Vec<String> = std::fs::read_dir(dir.join(SPOOL_DIR))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names.len(), 2, "no tmp file left behind: {names:?}");

        let read: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&first).unwrap()).unwrap();
        assert_eq!(read, payload);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
