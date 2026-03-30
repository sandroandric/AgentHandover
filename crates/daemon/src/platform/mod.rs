#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "macos")]
pub mod macos_accessibility;
#[cfg(target_os = "macos")]
pub mod macos_clipboard;
#[cfg(target_os = "macos")]
pub mod macos_windows;
#[cfg(target_os = "macos")]
pub mod macos_power;
#[cfg(target_os = "macos")]
pub mod macos_ocr;
#[cfg(target_os = "macos")]
pub mod applescript_bridge;
pub mod electron_detect;

#[cfg(target_os = "macos")]
pub use macos::IdleDetector;

#[cfg(target_os = "macos")]
pub mod accessibility {
    pub use super::macos_accessibility::*;
}

#[cfg(target_os = "macos")]
pub mod clipboard_monitor {
    pub use super::macos_clipboard::*;
}

#[cfg(target_os = "macos")]
pub mod window_capture {
    pub use super::macos_windows::*;
}

#[cfg(target_os = "macos")]
pub mod power {
    pub use super::macos_power::*;
}

#[cfg(target_os = "macos")]
pub mod ocr {
    pub use super::macos_ocr::*;
}

#[cfg(target_os = "macos")]
pub mod applescript {
    pub use super::applescript_bridge::*;
}
