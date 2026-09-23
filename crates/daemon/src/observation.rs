//! The observation socket client lives in `agenthandover_common` so the
//! CLI's `doctor` uses the same one; re-exported here for daemon callers.
//! The observer tick uses the bounded async `snapshot_at_async`; health and
//! control replies use the blocking `snapshot`.
pub use agenthandover_common::observation::{
    focused_element_at_async, snapshot, snapshot_at_async, ObservationSnapshot, SleepRecord,
    SnapshotError, WarnLimiter, SNAPSHOT_TIMEOUT,
};
