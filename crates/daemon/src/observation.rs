//! The observation socket client lives in `agenthandover_common` so the
//! CLI's `doctor` uses the same one; re-exported here for daemon callers.
pub use agenthandover_common::observation::{snapshot, ObservationSnapshot};
