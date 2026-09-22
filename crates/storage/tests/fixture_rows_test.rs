//! The worker's journey fixtures (`worker/tests/journeys/fixtures/*/events.jsonl.gz`)
//! hold `events` rows in the daemon's format.  Each row must parse into an
//! `Event` through `EventStore::get_event`, and writing that `Event` back
//! with `EventStore::insert_event` must give the same column text: the
//! fixtures are what this crate writes, byte for byte.

use agenthandover_storage::EventStore;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;
use uuid::Uuid;

const COLUMNS: [&str; 12] = [
    "id",
    "timestamp",
    "kind_json",
    "window_json",
    "display_topology_json",
    "primary_display_id",
    "cursor_x",
    "cursor_y",
    "ui_scale",
    "artifact_ids_json",
    "metadata_json",
    "display_ids_spanned_json",
];

fn fixture_files() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../worker/tests/journeys/fixtures");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("read {}: {e}", root.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path().join("events.jsonl.gz");
            path.is_file().then_some(path)
        })
        .collect();
    files.sort();
    files
}

fn read_rows(path: &Path) -> Vec<serde_json::Map<String, serde_json::Value>> {
    let output = Command::new("gzip")
        .arg("-dc")
        .arg(path)
        .output()
        .expect("run gzip -dc");
    assert!(output.status.success(), "gzip -dc {} failed", path.display());
    String::from_utf8(output.stdout)
        .expect("fixture is UTF-8")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).expect("fixture row is a JSON object"))
        .collect()
}

fn to_sql(value: &serde_json::Value) -> rusqlite::types::Value {
    match value {
        serde_json::Value::Null => rusqlite::types::Value::Null,
        serde_json::Value::String(s) => rusqlite::types::Value::Text(s.clone()),
        serde_json::Value::Number(n) if n.is_i64() => {
            rusqlite::types::Value::Integer(n.as_i64().unwrap())
        }
        serde_json::Value::Number(n) => rusqlite::types::Value::Real(n.as_f64().unwrap()),
        other => panic!("unexpected column value {other}"),
    }
}

fn stored_row(store: &EventStore, id: &str) -> Vec<rusqlite::types::Value> {
    let sql = format!("SELECT {} FROM events WHERE id = ?1", COLUMNS.join(", "));
    store
        .connection()
        .query_row(&sql, [id], |row| {
            (0..COLUMNS.len()).map(|i| row.get(i)).collect()
        })
        .unwrap()
}

#[test]
fn fixture_rows_round_trip_through_event_store() {
    let files = fixture_files();
    assert!(!files.is_empty(), "no journey fixtures found");

    for file in files {
        let rows = read_rows(&file);
        assert!(!rows.is_empty(), "{} has no rows", file.display());

        let tmp = TempDir::new().unwrap();
        let loaded = EventStore::open(&tmp.path().join("loaded.db")).unwrap();
        let rewritten = EventStore::open(&tmp.path().join("rewritten.db")).unwrap();
        let placeholders = vec!["?"; COLUMNS.len()].join(", ");
        let insert = format!(
            "INSERT INTO events ({}) VALUES ({placeholders})",
            COLUMNS.join(", ")
        );

        for row in &rows {
            let values: Vec<rusqlite::types::Value> =
                COLUMNS.iter().map(|col| to_sql(&row[*col])).collect();
            loaded
                .connection()
                .execute(&insert, rusqlite::params_from_iter(values.iter()))
                .unwrap();

            let id_text = row["id"].as_str().unwrap();
            let event = loaded
                .get_event(Uuid::parse_str(id_text).unwrap())
                .unwrap_or_else(|e| panic!("{}: row {id_text} does not parse: {e}", file.display()))
                .expect("row just inserted");
            rewritten.insert_event(&event).unwrap();

            let original = stored_row(&loaded, id_text);
            let written = stored_row(&rewritten, id_text);
            for (i, col) in COLUMNS.iter().enumerate() {
                assert_eq!(
                    original[i], written[i],
                    "{}: row {id_text} column {col} differs from what EventStore writes",
                    file.display()
                );
            }
        }
    }
}
