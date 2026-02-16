use core_foundation::base::{CFType, TCFType};
use core_foundation::string::CFString;
use std::sync::atomic::{AtomicU32, Ordering};
use tracing::{debug, error, warn};

/// Tracks consecutive AX API timeouts for degradation visibility.
static AX_CONSECUTIVE_TIMEOUTS: AtomicU32 = AtomicU32::new(0);

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
/// Uses AXUIElementCreateSystemWide() to get the system-wide AX element, then
/// queries the kAXFocusedUIElementAttribute to get the focused element, and
/// checks its kAXSubroleAttribute for kAXSecureTextFieldSubrole.
///
/// Runs synchronously in a blocking context. The caller should wrap this in
/// tokio::task::spawn_blocking with a timeout to avoid AX API deadlocks.
pub fn is_secure_field_focused() -> bool {
    is_secure_field_focused_inner().unwrap_or(false)
}

/// Inner implementation that returns Option so we can use ? for early returns.
fn is_secure_field_focused_inner() -> Option<bool> {
    unsafe {
        // Get the system-wide accessibility element.
        let system_wide = accessibility_sys::AXUIElementCreateSystemWide();
        if system_wide.is_null() {
            return None;
        }

        // Set a short messaging timeout (100ms) to avoid hangs.
        accessibility_sys::AXUIElementSetMessagingTimeout(system_wide, 0.1);

        // Query the focused UI element.
        let focused_attr =
            CFString::new(accessibility_sys::kAXFocusedUIElementAttribute);
        let mut focused_value: core_foundation::base::CFTypeRef = std::ptr::null();
        let err = accessibility_sys::AXUIElementCopyAttributeValue(
            system_wide,
            focused_attr.as_concrete_TypeRef(),
            &mut focused_value,
        );

        // Release the system-wide element (it was created with Create rule).
        core_foundation::base::CFRelease(system_wide as *const _);

        if err != accessibility_sys::kAXErrorSuccess || focused_value.is_null() {
            return None;
        }

        // We now own focused_value. Wrap it to ensure it gets released.
        let focused_element = focused_value as accessibility_sys::AXUIElementRef;

        // Query the subrole of the focused element.
        let subrole_attr = CFString::new(accessibility_sys::kAXSubroleAttribute);
        let mut subrole_value: core_foundation::base::CFTypeRef = std::ptr::null();
        let err = accessibility_sys::AXUIElementCopyAttributeValue(
            focused_element,
            subrole_attr.as_concrete_TypeRef(),
            &mut subrole_value,
        );

        // Release the focused element.
        core_foundation::base::CFRelease(focused_value);

        if err != accessibility_sys::kAXErrorSuccess || subrole_value.is_null() {
            return Some(false);
        }

        // Wrap the subrole value and check if it matches kAXSecureTextFieldSubrole.
        let cf_type: CFType = TCFType::wrap_under_create_rule(subrole_value);
        if let Some(subrole_str) = cf_type.downcast::<CFString>() {
            let is_secure =
                subrole_str.to_string() == accessibility_sys::kAXSecureTextFieldSubrole;
            if is_secure {
                debug!("Secure text field detected (password field focused)");
            }
            return Some(is_secure);
        }

        Some(false)
    }
}

/// Async-safe wrapper for is_secure_field_focused that runs in a blocking thread
/// with a 100ms timeout to prevent AX API deadlocks.
pub async fn is_secure_field_focused_async() -> bool {
    use std::time::Duration;

    match tokio::time::timeout(
        Duration::from_millis(100),
        tokio::task::spawn_blocking(is_secure_field_focused),
    )
    .await
    {
        Ok(Ok(result)) => {
            AX_CONSECUTIVE_TIMEOUTS.store(0, Ordering::Relaxed);
            result
        }
        Ok(Err(e)) => {
            AX_CONSECUTIVE_TIMEOUTS.store(0, Ordering::Relaxed);
            warn!(error = %e, "AX secure field check task panicked");
            false
        }
        Err(_) => {
            let count = AX_CONSECUTIVE_TIMEOUTS.fetch_add(1, Ordering::Relaxed) + 1;
            if count >= 10 {
                error!(
                    consecutive_timeouts = count,
                    "AX API consistently timing out — capture is likely fully degraded"
                );
            } else if count >= 3 {
                warn!(
                    consecutive_timeouts = count,
                    "AX API consistently timing out — capture may be degraded"
                );
            }
            // On timeout, assume it's a secure field as a safety measure.
            // Better to skip capture than to leak password data.
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_accessibility_permission() {
        // This test just verifies the function doesn't panic.
        // The actual result depends on whether the test runner has AX permissions.
        let _result = check_accessibility_permission();
    }

    #[test]
    fn test_is_secure_field_focused_does_not_hang() {
        // Verify the function completes within a reasonable time.
        // It should return false in a normal test environment.
        let result = is_secure_field_focused();
        // In test context, likely no secure field is focused.
        // The important thing is it doesn't hang or panic.
        let _ = result;
    }

    #[tokio::test]
    async fn test_is_secure_field_focused_async_respects_timeout() {
        // Verify the async version completes within its 100ms timeout.
        let start = std::time::Instant::now();
        let _result = is_secure_field_focused_async().await;
        let elapsed = start.elapsed();
        // Should complete well within 1 second (the 100ms timeout + overhead).
        assert!(
            elapsed.as_secs() < 2,
            "Async secure field check took too long: {:?}",
            elapsed
        );
    }
}
