//! What capture costs: frames taken, kept and gated (per reason), and the
//! milliseconds spent capturing and in OCR, since the daemon started and
//! over the last hour.
//!
//! The observer loop records into a [`SharedCaptureStats`]; the health
//! watcher writes [`CaptureStats::snapshot`] into `daemon-status.json`
//! under `compute` once a minute, where `agenthandover status --compute`
//! and the J-compute journey read it. The reasons are documented on
//! [`CaptureCounts`].

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use agenthandover_common::status::{CaptureCompute, CaptureCounts};

/// The last hour is kept as one bucket per minute.
const BUCKET: Duration = Duration::from_secs(60);
const BUCKETS_PER_HOUR: u64 = 60;

/// The observer loop's counters, shared with the health watcher.
pub type SharedCaptureStats = Arc<Mutex<CaptureStats>>;

/// Counters since start-up plus per-minute buckets for the last hour.
#[derive(Debug)]
pub struct CaptureStats {
    start: Instant,
    total: CaptureCounts,
    /// `(minute index since start, counts)`, oldest first.
    minutes: VecDeque<(u64, CaptureCounts)>,
}

impl CaptureStats {
    pub fn new(now: Instant) -> Self {
        Self { start: now, total: CaptureCounts::default(), minutes: VecDeque::new() }
    }

    pub fn shared() -> SharedCaptureStats {
        Arc::new(Mutex::new(Self::new(Instant::now())))
    }

    fn minute(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.start).as_secs() / BUCKET.as_secs()
    }

    fn prune(&mut self, minute: u64) {
        while self
            .minutes
            .front()
            .is_some_and(|(m, _)| m + BUCKETS_PER_HOUR <= minute)
        {
            self.minutes.pop_front();
        }
    }

    fn record(&mut self, now: Instant, change: impl Fn(&mut CaptureCounts)) {
        change(&mut self.total);
        let minute = self.minute(now);
        self.prune(minute);
        match self.minutes.back_mut() {
            Some((m, counts)) if *m == minute => change(counts),
            _ => {
                let mut counts = CaptureCounts::default();
                change(&mut counts);
                self.minutes.push_back((minute, counts));
            }
        }
    }

    /// The capture socket returned a frame after `took`.
    pub fn captured(&mut self, now: Instant, took: Duration) {
        let ms = took.as_millis() as u64;
        self.record(now, |c| {
            c.frames_captured += 1;
            c.capture_ms += ms;
        });
    }

    /// The capture socket gave no frame after `took`.
    pub fn capture_failed(&mut self, now: Instant, took: Duration) {
        let ms = took.as_millis() as u64;
        self.record(now, |c| {
            c.capture_ms += ms;
            *c.frames_gated_by_reason.entry("capture_failed".into()).or_default() += 1;
        });
    }

    /// One OCR pass took `took`.
    pub fn ocr(&mut self, now: Instant, took: Duration) {
        let ms = took.as_millis() as u64;
        self.record(now, |c| c.ocr_ms += ms);
    }

    pub fn kept(&mut self, now: Instant) {
        self.record(now, |c| c.frames_kept += 1);
    }

    /// A frame was not taken, or taken and dropped, for `reason`.
    pub fn gated(&mut self, now: Instant, reason: &str) {
        self.record(now, |c| {
            *c.frames_gated_by_reason.entry(reason.to_string()).or_default() += 1;
        });
    }

    /// Since start-up and the last hour (the current minute and the 59
    /// before it).
    pub fn snapshot(&mut self, now: Instant) -> CaptureCompute {
        self.prune(self.minute(now));
        let mut last_hour = CaptureCounts::default();
        for (_, counts) in &self.minutes {
            last_hour.add(counts);
        }
        CaptureCompute { since_start: self.total.clone(), last_hour }
    }
}

/// Run `f` on the shared stats; a poisoned lock (a panic elsewhere while
/// recording) still records.
pub fn with_stats(stats: &SharedCaptureStats, f: impl FnOnce(&mut CaptureStats)) {
    let mut guard = stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    f(&mut guard);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_since_start_and_last_hour() {
        let t0 = Instant::now();
        let mut stats = CaptureStats::new(t0);
        stats.captured(t0, Duration::from_millis(30));
        stats.ocr(t0, Duration::from_millis(120));
        stats.kept(t0);
        stats.gated(t0 + Duration::from_secs(5), "unchanged");
        stats.capture_failed(t0 + Duration::from_secs(70), Duration::from_millis(8));

        let snap = stats.snapshot(t0 + Duration::from_secs(90));
        assert_eq!(snap.since_start, snap.last_hour);
        let c = &snap.last_hour;
        assert_eq!((c.frames_captured, c.frames_kept), (1, 1));
        assert_eq!((c.capture_ms, c.ocr_ms), (38, 120));
        assert_eq!(c.frames_gated_by_reason.get("unchanged"), Some(&1));
        assert_eq!(c.frames_gated_by_reason.get("capture_failed"), Some(&1));
    }

    #[test]
    fn minutes_older_than_an_hour_leave_the_last_hour_only() {
        let t0 = Instant::now();
        let mut stats = CaptureStats::new(t0);
        stats.captured(t0, Duration::from_millis(10));
        stats.captured(t0 + Duration::from_secs(30 * 60), Duration::from_millis(20));

        // 60 minutes after the first frame's minute began, it has left the window.
        let snap = stats.snapshot(t0 + Duration::from_secs(60 * 60));
        assert_eq!(snap.since_start.frames_captured, 2);
        assert_eq!(snap.last_hour.frames_captured, 1);
        assert_eq!(snap.last_hour.capture_ms, 20);

        let snap = stats.snapshot(t0 + Duration::from_secs(91 * 60));
        assert_eq!(snap.last_hour, CaptureCounts::default());
        assert_eq!(snap.since_start.capture_ms, 30);
    }
}
