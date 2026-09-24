use core_graphics::event::CGEventType;
use core_graphics::event_source::CGEventSourceStateID;

// CGEventSourceSecondsSinceLastEventType is not wrapped by the core-graphics crate,
// so we declare the FFI binding directly. This function queries the HID system
// for how long ago the last event of a given type occurred, without intercepting
// any keystrokes or input.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGEventSourceSecondsSinceLastEventType(
        source_state: CGEventSourceStateID,
        event_type: CGEventType,
    ) -> f64;
    // How many events of a type the HID system has seen (a counter only).
    fn CGEventSourceCounterForEventType(
        source_state: CGEventSourceStateID,
        event_type: CGEventType,
    ) -> u32;
}

pub struct IdleDetector;

impl IdleDetector {
    pub fn new() -> Self {
        Self
    }

    /// Returns seconds since last HID event (keyboard/mouse/trackpad).
    /// Uses CGEventSourceSecondsSinceLastEventType -- no keystroke interception.
    /// Passing CGEventType::Null queries across all event types.
    pub fn seconds_since_last_input(&self) -> f64 {
        unsafe {
            CGEventSourceSecondsSinceLastEventType(
                CGEventSourceStateID::HIDSystemState,
                CGEventType::Null,
            )
        }
    }

    /// Seconds since the last left or right mouse-button press, by the same
    /// HID query (a timestamp, not an event tap: nothing is intercepted and
    /// no Input Monitoring permission is involved).
    pub fn seconds_since_last_click(&self) -> f64 {
        let since = |event_type| unsafe {
            CGEventSourceSecondsSinceLastEventType(CGEventSourceStateID::HIDSystemState, event_type)
        };
        since(CGEventType::LeftMouseDown).min(since(CGEventType::RightMouseDown))
    }

    /// How many key presses the login session has counted (it wraps at
    /// `u32::MAX`). A count only: no key code, no text, no event tap, no
    /// Input Monitoring permission. The observer records the difference
    /// between two kept frames.
    ///
    /// The combined session state, not the HID system state: the HID
    /// counter sees only hardware keyboards, so text typed by a text
    /// expander, a remote-control or screen-sharing session, or automation
    /// (measured on the test Mac: five System Events keystrokes moved the
    /// combined counter by five and the HID counter by none) would read as
    /// "nothing typed". The combined counter includes the hardware keys.
    pub fn key_down_count(&self) -> u32 {
        unsafe {
            CGEventSourceCounterForEventType(
                CGEventSourceStateID::CombinedSessionState,
                CGEventType::KeyDown,
            )
        }
    }

    pub fn is_idle(&self, threshold_seconds: f64) -> bool {
        self.seconds_since_last_input() >= threshold_seconds
    }
}

impl Default for IdleDetector {
    fn default() -> Self {
        Self::new()
    }
}
