CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY NOT NULL,
    timestamp TEXT NOT NULL,
    kind_json TEXT NOT NULL,
    window_json TEXT,
    display_topology_json TEXT NOT NULL,
    primary_display_id TEXT NOT NULL,
    cursor_x INTEGER,
    cursor_y INTEGER,
    ui_scale REAL,
    artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    processed INTEGER NOT NULL DEFAULT 0,
    episode_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_processed ON events(processed);
CREATE INDEX IF NOT EXISTS idx_events_episode_id ON events(episode_id);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY NOT NULL,
    event_id TEXT NOT NULL REFERENCES events(id),
    artifact_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    compression_algo TEXT NOT NULL DEFAULT 'zstd',
    encryption_algo TEXT NOT NULL DEFAULT 'xchacha20poly1305',
    original_size_bytes INTEGER NOT NULL,
    stored_size_bytes INTEGER NOT NULL,
    artifact_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_artifacts_event_id ON artifacts(event_id);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY NOT NULL,
    segment_id INTEGER NOT NULL DEFAULT 0,
    prev_segment_id INTEGER,
    thread_id TEXT,
    start_time TEXT NOT NULL,
    end_time TEXT,
    event_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'open',
    summary TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS vlm_queue (
    id TEXT PRIMARY KEY NOT NULL,
    event_id TEXT NOT NULL REFERENCES events(id),
    priority REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    processed_at TEXT,
    result_json TEXT,
    ttl_expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_vlm_queue_status ON vlm_queue(status, priority DESC);
