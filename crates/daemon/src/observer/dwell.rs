use std::time::{Duration, Instant};

pub struct DwellTracker {
    t_dwell: Duration,
    t_scroll_read: Duration,
    last_manipulation: Instant,
    last_navigation: Option<Instant>,
    first_navigation_since_manipulation: Option<Instant>,
}

impl DwellTracker {
    pub fn new(t_dwell: Duration, t_scroll_read: Duration) -> Self {
        Self {
            t_dwell,
            t_scroll_read,
            last_manipulation: Instant::now(),
            last_navigation: None,
            first_navigation_since_manipulation: None,
        }
    }

    pub fn on_manipulation_input(&mut self) {
        self.last_manipulation = Instant::now();
        self.first_navigation_since_manipulation = None;
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

    pub fn is_dwelling(&self) -> bool {
        self.last_manipulation.elapsed() >= self.t_dwell
    }

    pub fn is_scroll_reading(&self) -> bool {
        if let Some(first_nav) = self.first_navigation_since_manipulation {
            first_nav.elapsed() >= self.t_scroll_read
                && self.last_manipulation.elapsed() >= self.t_scroll_read
        } else {
            false
        }
    }

    pub fn should_capture_dwell_snapshot(&self) -> bool {
        self.is_dwelling() || self.is_scroll_reading()
    }
}
