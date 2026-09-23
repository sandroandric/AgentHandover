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

    /// The system calls behind the "Open Settings" buttons, injectable so
    /// tests can check their order without touching TCC.
    struct SettingsRegistration {
        var isAccessibilityTrusted: () -> Bool = { PermissionChecker.isAccessibilityGranted() }
        /// `AXIsProcessTrustedWithOptions(prompt: true)`: adds AgentHandover
        /// to the Accessibility list (switched off) and shows macOS's prompt.
        var registerAccessibility: () -> Void = { PermissionChecker.requestAccessibility() }
        var openAccessibilityPane: () -> Void = { PermissionChecker.openAccessibilitySettings() }
        /// One ScreenCaptureKit capture attempt: adds AgentHandover to the
        /// Screen Recording list.
        var registerScreenRecording: () async -> Void = { _ = await ScreenCaptureService.runCaptureProbe() }
        var openScreenRecordingPane: () -> Void = { PermissionChecker.openScreenRecordingSettings() }
    }

    /// Accessibility "Open Settings": register the app with TCC, then open
    /// the pane. Opening the pane alone (as this used to) left nothing to
    /// switch on when the app was not in the list yet: a first run, or
    /// after the installer reset a grant whose code signature changed.
    @MainActor
    @discardableResult
    static func requestAccessibilityAndOpenSettingsIfNeeded(
        _ system: SettingsRegistration = SettingsRegistration()
    ) async -> Bool {
        let granted = system.isAccessibilityTrusted()
        if !granted {
            system.registerAccessibility()
            system.openAccessibilityPane()
        }
        return granted
    }

    /// Screen Recording "Open Settings" (the manual fallback): one capture
    /// attempt so the app is in the list, then open the pane.
    @MainActor
    static func registerAndOpenScreenRecordingSettings(
        _ system: SettingsRegistration = SettingsRegistration()
    ) async {
        await system.registerScreenRecording()
        system.openScreenRecordingPane()
    }

    /// Shown when the user asked for a recheck and the grant is still
    /// missing. After an upgrade that changed the app's code signature the
    /// old switch can stay ON while it no longer applies to this build;
    /// removing and re-adding the entry is what makes macOS apply it.
    static let staleGrantGuidance =
        "If AgentHandover is already switched on, select it, click \u{2212} to remove it, then add it again with +."

    /// Guidance after a user-requested recheck: nil when the grant is there.
    static func recheckGuidance(granted: Bool) -> String? {
        granted ? nil : staleGrantGuidance
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
        let probe = ScreenRecordingProbe.shared
        var granted = isScreenRecordingGranted()

        if granted, await probe.isCaptureReady() {
            return ScreenRecordingStatus(granted: true, captureReady: true)
        }

        let startedAt = DispatchTime.now().uptimeNanoseconds
        while DispatchTime.now().uptimeNanoseconds - startedAt < timeoutNanoseconds {
            try? await Task.sleep(nanoseconds: 250_000_000)
            granted = isScreenRecordingGranted() || granted
            if granted, await probe.isCaptureReady() {
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
