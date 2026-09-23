//! The observation socket client lives in `agenthandover_common` so the
//! CLI's `doctor` uses the same one; re-exported here for daemon callers.
//! The observer tick uses the bounded async `snapshot_async`; health and
//! control replies use the blocking `snapshot`.
pub use agenthandover_common::observation::{
    focused_element_async, snapshot, snapshot_async, ObservationSnapshot, SleepRecord,
    SnapshotError, SNAPSHOT_TIMEOUT,
};
