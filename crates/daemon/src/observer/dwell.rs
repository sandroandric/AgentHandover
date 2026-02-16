use std::time::{Duration, Instant};

/// NOTE: This type is designed for single-threaded use within the observer event loop.
/// It is NOT thread-safe — do not share across threads without synchronization.
pub struct DwellTracker {
    t_dwell: Duration,
    t_scroll_read: Duration,
    last_manipulation: Instant,
    last_navigation: Option<Instant>,
    first_navigation_since_manipulation: Option<Instant>,
    /// True once a dwell snapshot has fired for this dwell period.
    /// Reset on the next manipulation input.
    dwell_fired: bool,
    /// True once a scroll-read snapshot has fired for this navigation period.
    /// Reset on the next manipulation input.
    scroll_read_fired: bool,
}

impl DwellTracker {
    pub fn new(t_dwell: Duration, t_scroll_read: Duration) -> Self {
        Self {
            t_dwell,
            t_scroll_read,
            last_manipulation: Instant::now(),
            last_navigation: None,
            first_navigation_since_manipulation: None,
            dwell_fired: false,
            scroll_read_fired: false,
        }
    }

    pub fn on_manipulation_input(&mut self) {
        self.last_manipulation = Instant::now();
        self.first_navigation_since_manipulation = None;
        // Reset one-shot flags so the next dwell period can fire again.
        self.dwell_fired = false;
        self.scroll_read_fired = false;
    }

    pub fn on_navigation_input(&mut self) {
        let now = Instant::now();
        self.last_navigation = Some(now);
        if self.first_navigation_since_manipulation.is_none() {
            self.first_navigation_since_manipulation = Some(now);
        }
    }

    pub fn tick(&mut self) {
        // Called periodically to check state
    }

    /// Returns true exactly once when the dwell threshold is first exceeded.
    /// Subsequent calls return false until a manipulation input resets the tracker.
    pub fn is_dwelling(&mut self) -> bool {
        if !self.dwell_fired && self.last_manipulation.elapsed() >= self.t_dwell {
            self.dwell_fired = true;
            return true;
        }
        false
    }

    /// Returns true exactly once when the scroll-read threshold is first exceeded.
    /// Subsequent calls return false until a manipulation input resets the tracker.
    pub fn is_scroll_reading(&mut self) -> bool {
        if self.scroll_read_fired {
            return false;
        }
        if let Some(first_nav) = self.first_navigation_since_manipulation {
            if first_nav.elapsed() >= self.t_scroll_read
                && self.last_manipulation.elapsed() >= self.t_scroll_read
            {
                self.scroll_read_fired = true;
                return true;
            }
        }
        false
    }

    pub fn should_capture_dwell_snapshot(&mut self) -> bool {
        self.is_dwelling() || self.is_scroll_reading()
    }
}
