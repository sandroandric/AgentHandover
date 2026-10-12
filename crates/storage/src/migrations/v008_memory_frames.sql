-- v008: Memory, the searchable record of what was on screen.
--
-- One row per moment: consecutive near-identical frames of one window
-- (cosine >= 0.97, less than 10 minutes apart) share a row whose time
-- span grows.  `id` is the event id of the moment's first frame.
-- `embedding` is the moment's first frame (screenshot + "app — title")
-- embedded with `[memory] model`, float32 little-endian, unit length,
-- `dim` values.  `description` is redacted text: the annotation of a frame
-- of the moment (`described_by`, its event id) once one exists, else the
-- frame's screen text.  `thumb_id` names the encrypted thumbnail
-- (`thumbs/<thumb_id>.ahthumb`) while it is kept.  The worker writes and
-- purges these rows (`[memory] retention_days`); they are not part of
-- `vector_store`, whose row cap and purge never touch them.

CREATE TABLE IF NOT EXISTS memory_frames (
    id               TEXT PRIMARY KEY NOT NULL,
    first_ts         TEXT NOT NULL,
    last_ts          TEXT NOT NULL,
    frames           INTEGER NOT NULL DEFAULT 1,
    app              TEXT NOT NULL DEFAULT '',
    window_title     TEXT NOT NULL DEFAULT '',
    url_host         TEXT NOT NULL DEFAULT '',
    description      TEXT NOT NULL DEFAULT '',
    described_by     TEXT,
    embedding        BLOB NOT NULL,
    model            TEXT NOT NULL,
    dim              INTEGER NOT NULL,
    thumb_id         TEXT,
    focus_session_id TEXT,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_memory_frames_first_ts ON memory_frames(first_ts);
CREATE INDEX IF NOT EXISTS idx_memory_frames_last_ts ON memory_frames(last_ts);
CREATE INDEX IF NOT EXISTS idx_memory_frames_app ON memory_frames(app);
CREATE INDEX IF NOT EXISTS idx_memory_frames_created ON memory_frames(created_at);
