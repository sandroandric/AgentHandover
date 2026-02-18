import Cocoa
import CoreGraphics

/// Checks macOS system permissions required by OpenMimic.
enum PermissionChecker {

    // MARK: - Accessibility

    /// Check if Accessibility permission is granted (needed for window info).
    static func isAccessibilityGranted() -> Bool {
        AXIsProcessTrusted()
    }

    /// Prompt the user to grant Accessibility permission.
    /// Opens System Settings to the correct pane.
    static func requestAccessibility() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue(): true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
    }

    // MARK: - Screen Recording

    /// Check if Screen Recording permission is granted (needed for screenshots).
    /// Uses CGDisplayCreateImage as a probe — returns nil without permission.
    static func isScreenRecordingGranted() -> Bool {
        let image = CGDisplayCreateImage(CGMainDisplayID())
        return image != nil
    }

    /// Open System Settings to the Screen Recording privacy pane.
    static func openScreenRecordingSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
            NSWorkspace.shared.open(url)
        }
    }

    // MARK: - Composite

    /// Check all required permissions.
    static func allPermissionsGranted() -> Bool {
        isAccessibilityGranted() && isScreenRecordingGranted()
    }
}
