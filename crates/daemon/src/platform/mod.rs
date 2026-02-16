#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "macos")]
pub mod macos_accessibility;

#[cfg(target_os = "macos")]
pub use macos::IdleDetector;

#[cfg(target_os = "macos")]
pub mod accessibility {
    pub use super::macos_accessibility::*;
}
