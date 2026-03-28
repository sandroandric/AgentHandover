import Cocoa
import CoreGraphics
import ScreenCaptureKit

/// Checks macOS system permissions required by AgentHandover.
enum PermissionChecker {
    struct ScreenRecordingStatus {
        let granted: Bool
        let captureReady: Bool
    }

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

    static func openAccessibilitySettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }

    /// Kick off the Accessibility grant flow from the main app.
    ///
    /// We intentionally avoid `AXIsProcessTrustedWithOptions(...prompt: true)`
    /// in onboarding/menu bar flows. On newer macOS releases that prompt path
    /// can surface broader Privacy UI and has proven too noisy in this app's
    /// setup sequence. The reliable UX here is a direct deep-link to the
    /// Accessibility pane, followed by an explicit recheck when the user
    /// returns.
    @MainActor
    @discardableResult
    static func requestAccessibilityAndOpenSettingsIfNeeded() async -> Bool {
        let granted = isAccessibilityGranted()
        if !granted {
            openAccessibilitySettings()
        }
        return granted
    }

    // MARK: - Screen Recording

    /// Check if Screen Recording permission is granted (needed for screenshots).
    /// Uses CGPreflightScreenCaptureAccess which checks the current state
    /// without triggering the system permission prompt or capturing anything.
    /// The old CGDisplayCreateImage probe triggered the prompt on first call
    /// and returned non-nil on macOS 15+ even without permission.
    static func isScreenRecordingGranted() -> Bool {
        CGPreflightScreenCaptureAccess()
    }

    /// Request Screen Recording permission from the main app bundle.
    /// This uses the same capture service the daemon depends on so the
    /// permission is exercised by the exact runtime principal users see.
    @discardableResult
    static func requestScreenRecording() async -> Bool {
        await ScreenCaptureService().requestPermission()
    }

    /// Resolve Screen Recording state after a user grant/regrant flow.
    ///
    /// macOS can lag between the user toggling the permission and the capture
    /// pipeline becoming live. We treat "TCC granted but capture not live yet"
    /// as a distinct state so the UI does not regress back to a misleading
    /// "missing permission" banner.
    @MainActor
    static func resolveScreenRecordingStatus(
        timeoutNanoseconds: UInt64 = 4_000_000_000
    ) async -> ScreenRecordingStatus {
        let service = ScreenCaptureService()
        var granted = isScreenRecordingGranted()

        if granted, await service.captureMainDisplay() != nil {
            return ScreenRecordingStatus(granted: true, captureReady: true)
        }

        let startedAt = DispatchTime.now().uptimeNanoseconds
        while DispatchTime.now().uptimeNanoseconds - startedAt < timeoutNanoseconds {
            try? await Task.sleep(nanoseconds: 250_000_000)
            granted = isScreenRecordingGranted() || granted
            if granted, await service.captureMainDisplay() != nil {
                return ScreenRecordingStatus(granted: true, captureReady: true)
            }
        }

        return ScreenRecordingStatus(granted: granted, captureReady: false)
    }

    /// Kick off the full Screen Recording grant flow from the main app.
    ///
    /// We first exercise the permission from the app principal. We do NOT
    /// auto-open System Settings from this call, because doing so while the
    /// macOS permission sheet is still unresolved can land the user on an
    /// empty Screen Recording pane with no app toggle visible yet.
    @MainActor
    @discardableResult
    static func requestScreenRecordingAndOpenSettingsIfNeeded() async -> Bool {
        ServiceController.prepareForAppOwnedPermissionRequest()
        NSApp.activate(ignoringOtherApps: true)
        return await requestScreenRecording()
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
