//! When the observer fires the shutter, and which frames it keeps.
//!
//! Capture is event-driven: one frame per transition (app switch, title
//! change, browser navigation or form submit, click, clipboard change) plus
//! a heartbeat when nothing has fired for [`HEARTBEAT`]. During a focus
//! recording the shutter also fires every [`FOCUS_INTERVAL`].
//!
//! Per tick the cheap checks run first and in this order:
//! 1. [`CapturePolicy::shutter_trigger`]: did anything happen, and is there
//!    budget left this minute? No trigger, no capture.
//! 2. [`CapturePolicy::frame_decision`]: is the captured frame different from
//!    the last *kept* frame (block-mean signature, `capture::dhash`)?
//! 3. For a heartbeat frame that passed 2, fast OCR:
//!    [`CapturePolicy::text_changed`] against the last kept frame's words.
//! 4. Only a frame that is kept gets accurate OCR and is written to disk.
//!
//! Budgets: passive frames are capped at `screenshot_max_per_minute` (20 by
//! default); focus-recording frames have their own counter
//! (`focus_screenshot_max_per_minute`, 40) so passive capture never leaves a
//! recording blind. Budgets count kept frames.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use crate::capture::dhash::{self, FrameSignature};

/// Fire a heartbeat frame after this long without a shutter.
pub const HEARTBEAT: Duration = Duration::from_secs(60);
/// Shutter interval during a focus recording.
pub const FOCUS_INTERVAL: Duration = Duration::from_secs(2);
/// A heartbeat frame whose OCR words overlap the last kept frame's by at
/// least this Jaccard similarity shows the same text and is dropped.
pub const SAME_TEXT_JACCARD: f64 = 0.9;

/// Why the shutter fired. Stored as `metadata.capture_trigger`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureTrigger {
    AppSwitch,
    TitleChange,
    Navigation,
    FormSubmit,
    Click,
    Clipboard,
    FocusInterval,
    Heartbeat,
}

impl CaptureTrigger {
    pub fn as_str(self) -> &'static str {
        match self {
            CaptureTrigger::AppSwitch => "app_switch",
            CaptureTrigger::TitleChange => "title_change",
            CaptureTrigger::Navigation => "navigation",
            CaptureTrigger::FormSubmit => "form_submit",
            CaptureTrigger::Click => "click",
            CaptureTrigger::Clipboard => "clipboard",
            CaptureTrigger::FocusInterval => "focus_interval",
            CaptureTrigger::Heartbeat => "heartbeat",
        }
    }

    /// When several transitions land in one tick, the frame is attributed
    /// to the most specific one (lower = preferred).
    fn rank(self) -> u8 {
        match self {
            CaptureTrigger::AppSwitch => 0,
            CaptureTrigger::Navigation => 1,
            CaptureTrigger::FormSubmit => 2,
            CaptureTrigger::TitleChange => 3,
            CaptureTrigger::Click => 4,
            CaptureTrigger::Clipboard => 5,
            CaptureTrigger::FocusInterval => 6,
            CaptureTrigger::Heartbeat => 7,
        }
    }
}

/// What to do with a captured frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDecision {
    /// Keep it: accurate OCR, write the JPEG, emit the event.
    Keep,
    /// Same screen as the last kept frame: drop it.
    Unchanged,
    /// The pixels changed but it is only a heartbeat: keep it only if the
    /// fast-OCR text changed too ([`CapturePolicy::text_changed`]).
    CheckText,
}

/// Shutter timing, per-minute budgets and the last kept frame.
pub struct CapturePolicy {
    passive_cap: u32,
    focus_cap: u32,
    threshold: u32,
    minute_start: Instant,
    passive_kept: u32,
    focus_kept: u32,
    last_shutter: Instant,
    last_focus_shutter: Option<Instant>,
    last_kept: Option<FrameSignature>,
    last_kept_words: Option<HashSet<String>>,
}

impl CapturePolicy {
    /// `threshold` is `[observer] dhash_threshold`: the number of changed
    /// blocks that makes a frame new. The first heartbeat is due
    /// [`HEARTBEAT`] after `now`.
    pub fn new(passive_cap: u32, focus_cap: u32, threshold: u32, now: Instant) -> Self {
        Self {
            passive_cap,
            focus_cap,
            threshold: threshold.max(1),
            minute_start: now,
            passive_kept: 0,
            focus_kept: 0,
            last_shutter: now,
            last_focus_shutter: None,
            last_kept: None,
            last_kept_words: None,
        }
    }

    /// Decide whether to fire the shutter this tick, and why.
    ///
    /// `transitions` are what the observer saw since the previous tick;
    /// `focus_recording` is whether a focus recording is active. Returns
    /// `None` when nothing happened (and no heartbeat or focus interval is
    /// due) or when this minute's budget for the mode is spent.
    pub fn shutter_trigger(
        &mut self,
        transitions: &[CaptureTrigger],
        focus_recording: bool,
        now: Instant,
    ) -> Option<CaptureTrigger> {
        if now.duration_since(self.minute_start) >= Duration::from_secs(60) {
            self.minute_start = now;
            self.passive_kept = 0;
            self.focus_kept = 0;
        }
        if !focus_recording {
            self.last_focus_shutter = None;
        }

        let over_budget = if focus_recording {
            self.focus_kept >= self.focus_cap
        } else {
            self.passive_kept >= self.passive_cap
        };
        if over_budget {
            return None;
        }

        let focus_due = focus_recording
            && self
                .last_focus_shutter
                .is_none_or(|t| now.duration_since(t) >= FOCUS_INTERVAL);
        let trigger = transitions
            .iter()
            .copied()
            .min_by_key(|t| t.rank())
            .or(focus_due.then_some(CaptureTrigger::FocusInterval))
            .or((now.duration_since(self.last_shutter) >= HEARTBEAT)
                .then_some(CaptureTrigger::Heartbeat))?;

        self.last_shutter = now;
        if focus_recording {
            self.last_focus_shutter = Some(now);
        }
        Some(trigger)
    }

    /// Decide what to do with a frame captured for `trigger`.
    ///
    /// Focus-recording frames and app switches are always kept (a recording
    /// keeps every frame it pays for; a new app is a new context). Every
    /// other frame is compared with the last kept frame, not the last
    /// captured one, so a slow drift of small changes still adds up.
    pub fn frame_decision(
        &self,
        trigger: CaptureTrigger,
        focus_recording: bool,
        signature: &FrameSignature,
    ) -> FrameDecision {
        if focus_recording || trigger == CaptureTrigger::AppSwitch {
            return FrameDecision::Keep;
        }
        let Some(last) = &self.last_kept else {
            return FrameDecision::Keep;
        };
        if dhash::distance(last, signature) < self.threshold {
            return FrameDecision::Unchanged;
        }
        if trigger == CaptureTrigger::Heartbeat && self.last_kept_words.is_some() {
            return FrameDecision::CheckText;
        }
        FrameDecision::Keep
    }

    /// For a [`FrameDecision::CheckText`] frame: whether its words (from
    /// fast OCR) differ from the last kept frame's.
    pub fn text_changed(&self, words: &HashSet<String>) -> bool {
        match &self.last_kept_words {
            Some(last) => jaccard(last, words) < SAME_TEXT_JACCARD,
            None => true,
        }
    }

    /// Record a kept frame: it becomes the reference for the next decision
    /// and counts against the mode's budget. `words` are the kept frame's
    /// OCR words (`None` when OCR is unavailable).
    pub fn record_kept(
        &mut self,
        signature: FrameSignature,
        words: Option<HashSet<String>>,
        focus_recording: bool,
    ) {
        self.last_kept = Some(signature);
        self.last_kept_words = words;
        if focus_recording {
            self.focus_kept += 1;
        } else {
            self.passive_kept += 1;
        }
    }
}

/// The words of an OCR text: lowercase alphanumeric runs of two or more
/// characters (single characters are mostly icon and punctuation noise).
pub fn ocr_words(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 2)
        .map(str::to_lowercase)
        .collect()
}

/// Jaccard similarity of two word sets; two empty sets are identical.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    let union = a.union(b).count();
    if union == 0 {
        return 1.0;
    }
    a.intersection(b).count() as f64 / union as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::dhash::BLOCKS;

    fn signature(value: u8) -> FrameSignature {
        FrameSignature([value; BLOCKS])
    }

    fn policy(now: Instant) -> CapturePolicy {
        CapturePolicy::new(20, 40, 1, now)
    }

    #[test]
    fn nothing_happened_means_no_shutter_until_the_heartbeat() {
        let t0 = Instant::now();
        let mut p = policy(t0);
        assert_eq!(p.shutter_trigger(&[], false, t0 + Duration::from_secs(1)), None);
        assert_eq!(p.shutter_trigger(&[], false, t0 + Duration::from_secs(59)), None);
        assert_eq!(
            p.shutter_trigger(&[], false, t0 + Duration::from_secs(60)),
            Some(CaptureTrigger::Heartbeat)
        );
        // The heartbeat counts from the last shutter, whatever fired it.
        assert_eq!(
            p.shutter_trigger(&[CaptureTrigger::Click], false, t0 + Duration::from_secs(90)),
            Some(CaptureTrigger::Click)
        );
        assert_eq!(p.shutter_trigger(&[], false, t0 + Duration::from_secs(149)), None);
        assert_eq!(
            p.shutter_trigger(&[], false, t0 + Duration::from_secs(150)),
            Some(CaptureTrigger::Heartbeat)
        );
    }

    #[test]
    fn one_frame_per_tick_attributed_to_the_most_specific_transition() {
        let t0 = Instant::now();
        let mut p = policy(t0);
        let fired = p.shutter_trigger(
            &[CaptureTrigger::Click, CaptureTrigger::TitleChange, CaptureTrigger::Navigation],
            false,
            t0,
        );
        assert_eq!(fired, Some(CaptureTrigger::Navigation));
        let fired = p.shutter_trigger(
            &[CaptureTrigger::Clipboard, CaptureTrigger::AppSwitch],
            false,
            t0 + Duration::from_millis(500),
        );
        assert_eq!(fired, Some(CaptureTrigger::AppSwitch));
    }

    #[test]
    fn passive_cap_is_a_ceiling_on_kept_frames_per_minute() {
        let t0 = Instant::now();
        let mut p = CapturePolicy::new(3, 40, 1, t0);
        for i in 0..3u8 {
            let now = t0 + Duration::from_millis(500 * i as u64);
            assert!(p.shutter_trigger(&[CaptureTrigger::Click], false, now).is_some());
            p.record_kept(signature(i * 50), None, false);
        }
        let now = t0 + Duration::from_secs(10);
        assert_eq!(p.shutter_trigger(&[CaptureTrigger::Click], false, now), None);
        // A new minute resets the budget.
        let now = t0 + Duration::from_secs(61);
        assert!(p.shutter_trigger(&[CaptureTrigger::Click], false, now).is_some());
    }

    #[test]
    fn a_spent_passive_budget_never_blinds_a_focus_recording() {
        let t0 = Instant::now();
        let mut p = CapturePolicy::new(2, 40, 1, t0);
        for i in 0..2u8 {
            p.shutter_trigger(&[CaptureTrigger::Click], false, t0);
            p.record_kept(signature(i * 100), None, false);
        }
        assert_eq!(p.shutter_trigger(&[CaptureTrigger::Click], false, t0), None);

        // Recording starts: the focus interval fires at once and every 2 s,
        // up to its own 40 per minute.
        let mut kept = 0;
        for tick in 0..120u64 {
            let now = t0 + Duration::from_millis(500 * tick);
            if let Some(trigger) = p.shutter_trigger(&[], true, now) {
                assert_eq!(trigger, CaptureTrigger::FocusInterval);
                assert_eq!(p.frame_decision(trigger, true, &signature(7)), FrameDecision::Keep);
                p.record_kept(signature(7), None, true);
                kept += 1;
            }
        }
        // 60 s of ticks at one shutter per 2 s = 30 frames, within 40.
        assert_eq!(kept, 30);
    }

    #[test]
    fn focus_cap_is_its_own_ceiling() {
        let t0 = Instant::now();
        let mut p = CapturePolicy::new(20, 3, 1, t0);
        let mut kept = 0;
        for tick in 0..20u64 {
            let now = t0 + Duration::from_millis(500 * tick);
            if p.shutter_trigger(&[CaptureTrigger::Click], true, now).is_some() {
                p.record_kept(signature(1), None, true);
                kept += 1;
            }
        }
        assert_eq!(kept, 3);
        // Passive capture has its own budget once the recording stops.
        assert!(p
            .shutter_trigger(&[CaptureTrigger::Click], false, t0 + Duration::from_secs(11))
            .is_some());
    }

    #[test]
    fn frames_are_compared_with_the_last_kept_frame_not_the_last_captured() {
        let t0 = Instant::now();
        let mut p = policy(t0);
        let base = signature(100);
        assert_eq!(p.frame_decision(CaptureTrigger::Click, false, &base), FrameDecision::Keep);
        p.record_kept(base, None, false);

        // A drift of 3 levels per frame: every step is under the dead zone
        // relative to the previous capture, but the comparison is with the
        // kept frame, so the second step (6 levels in total) is a change.
        let mut kept_at = None;
        for step in 1..=4u8 {
            let frame = signature(100 + 3 * step);
            match p.frame_decision(CaptureTrigger::Click, false, &frame) {
                FrameDecision::Keep => {
                    kept_at = Some(step);
                    break;
                }
                FrameDecision::Unchanged => {}
                FrameDecision::CheckText => unreachable!("not a heartbeat"),
            }
        }
        assert_eq!(kept_at, Some(2), "drift of 6 levels exceeds the dead zone of 4");
    }

    #[test]
    fn app_switch_and_focus_frames_are_always_kept() {
        let t0 = Instant::now();
        let mut p = policy(t0);
        p.record_kept(signature(10), None, false);
        let same = signature(10);
        assert_eq!(p.frame_decision(CaptureTrigger::TitleChange, false, &same), FrameDecision::Unchanged);
        assert_eq!(p.frame_decision(CaptureTrigger::AppSwitch, false, &same), FrameDecision::Keep);
        assert_eq!(p.frame_decision(CaptureTrigger::Heartbeat, true, &same), FrameDecision::Keep);
    }

    #[test]
    fn a_heartbeat_with_new_pixels_but_the_same_text_is_dropped() {
        let t0 = Instant::now();
        let mut p = policy(t0);
        p.record_kept(signature(10), Some(ocr_words("Inbox — 3 unread\nQuarterly report draft")), false);

        let video_frame = signature(90);
        assert_eq!(
            p.frame_decision(CaptureTrigger::Heartbeat, false, &video_frame),
            FrameDecision::CheckText
        );
        assert!(!p.text_changed(&ocr_words("inbox 3 UNREAD quarterly report draft")));
        assert!(p.text_changed(&ocr_words("Inbox 4 unread Quarterly report final Invoice March")));
        // A click with the same pixels change is kept without the text check.
        assert_eq!(p.frame_decision(CaptureTrigger::Click, false, &video_frame), FrameDecision::Keep);
    }

    #[test]
    fn ocr_words_ignores_case_punctuation_and_single_characters() {
        let words = ocr_words("File  Edit | View > x 2026-09-23");
        let expected: HashSet<String> =
            ["file", "edit", "view", "2026", "09", "23"].iter().map(|s| s.to_string()).collect();
        assert_eq!(words, expected);
    }
}
