-- v003: Add scene annotation and frame diff columns for v2 VLM pipeline.
--
-- scene_annotation_json: Structured JSON from VLM scene annotator
-- annotation_status: Pipeline processing state for this event
-- frame_diff_json: Action diff between this frame and the previous annotation

ALTER TABLE events ADD COLUMN scene_annotation_json TEXT DEFAULT NULL;
ALTER TABLE events ADD COLUMN annotation_status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE events ADD COLUMN frame_diff_json TEXT DEFAULT NULL;

-- Index for the annotation worker polling loop:
-- SELECT ... WHERE annotation_status = 'pending' ORDER BY timestamp ASC LIMIT N
CREATE INDEX IF NOT EXISTS idx_events_annotation_status
    ON events(annotation_status);

-- Mark all pre-existing events as 'skipped' — they have no screenshot
-- for the v2 annotator to process retroactively.
UPDATE events SET annotation_status = 'skipped' WHERE annotation_status = 'pending';
