use tracing::warn;

/// Check if the app has macOS Accessibility permission.
/// Required for reading the AX tree of other applications.
pub fn check_accessibility_permission() -> bool {
    let trusted = unsafe { accessibility_sys::AXIsProcessTrusted() };
    if !trusted {
        warn!(
            "Accessibility permission not granted. \
             Request it in System Settings > Privacy & Security > Accessibility."
        );
    }
    trusted
}

/// Check if the currently focused UI element is a secure text field (password).
/// If so, the observer must NOT capture any content (per spec section 5.4
/// secure-field hard drop).
///
/// Full implementation requires AXUIElementCopyAttributeValue calls against
/// the focused element to check `kAXRoleAttribute == kAXSecureTextFieldRole`.
/// This will be completed when the AX tree capture module is built (Task 11).
pub fn is_secure_field_focused() -> bool {
    // Safe default: not a secure field. The full AX tree observer will
    // provide accurate detection when implemented.
    false
}
