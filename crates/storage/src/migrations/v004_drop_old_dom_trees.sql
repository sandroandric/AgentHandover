-- v004: DOM snapshots store an anchor set, not the viewport tree.
--
-- The Chrome extension now sends `anchors` (the interacted element and its
-- ancestors up to the nearest landmark) and a text `digest` instead of the
-- `nodes` tree that earlier versions wrote (hundreds of kilobytes per
-- capture).  Trees older than the raw-event retention window (the default
-- `retention_days_raw`, 14 days) are removed from their rows once; trees
-- inside the window stay readable by the worker until the normal purge
-- deletes their rows.

UPDATE events
SET metadata_json = json_remove(metadata_json, '$.nodes')
WHERE json_type(metadata_json, '$.nodes') = 'array'
  AND datetime(timestamp) < datetime('now', '-14 days');
