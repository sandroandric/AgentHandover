-- v005: the focused UI element of an event (Accessibility role, title,
-- value under `[privacy] capture_input_values`, ancestor chain, window
-- document).  JSON of `agenthandover_common::event::FocusedElement`, NULL
-- when the event carries none.

ALTER TABLE events ADD COLUMN focused_element_json TEXT;
