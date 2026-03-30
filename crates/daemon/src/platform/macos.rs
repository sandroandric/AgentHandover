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

    pub fn is_idle(&self, threshold_seconds: f64) -> bool {
        self.seconds_since_last_input() >= threshold_seconds
    }
}

impl Default for IdleDetector {
    fn default() -> Self {
        Self::new()
    }
}
