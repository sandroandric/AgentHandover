import ScreenCaptureKit
import CoreGraphics
import AppKit

/// Captures screenshots with ScreenCaptureKit.
/// This runs in the main app process, which is the Screen Recording TCC principal.
final class ScreenCaptureService {

    private let preflight: () -> Bool
    private let registrationProbe: () async -> Void
    private let pollIntervalNanoseconds: UInt64

    /// - Parameters:
    ///   - preflight: whether Screen Recording is granted (TCC preflight).
    ///   - registrationProbe: the capture attempt that makes macOS register
    ///     this app for Screen Recording (defaults to the ScreenCaptureKit
    ///     probe; injectable for tests).
    ///   - pollIntervalNanoseconds: how often to re-check after the probe.
    init(
        preflight: @escaping () -> Bool = { CGPreflightScreenCaptureAccess() },
        registrationProbe: @escaping () async -> Void = ScreenCaptureService.exerciseScreenCaptureKitRegistrationProbe,
        pollIntervalNanoseconds: UInt64 = 250_000_000
    ) {
        self.preflight = preflight
        self.registrationProbe = registrationProbe
        self.pollIntervalNanoseconds = pollIntervalNanoseconds
    }

    /// Capture the main display as raw BGRA pixel data.
    /// Returns (width, height, bgra_bytes) or nil if capture fails.
    func captureMainDisplay() async -> (Int, Int, Data)? {
        guard hasPermission() else { return nil }
        return await captureViaSCK()
    }

    /// Check if Screen Recording permission is granted.
    func hasPermission() -> Bool {
        preflight()
    }

    /// Request Screen Recording permission from the main app bundle.
    ///
    /// Keep this path minimal and deterministic: ask TCC from the foreground
    /// app principal, then let the UI decide whether Settings still needs to
    /// be opened manually.
    @MainActor
    @discardableResult
    func requestPermission() async -> Bool {
        guard !hasPermission() else { return true }

        let previousPolicy = NSApp.activationPolicy()
        if previousPolicy != .regular {
            NSApp.setActivationPolicy(.regular)
        }
        NSApp.activate(ignoringOtherApps: true)
        defer {
            if previousPolicy == .accessory {
                NSApp.setActivationPolicy(.accessory)
            }
        }
        return await registerAndAwaitGrant()
    }

    /// Make the capture attempt that registers the app with TCC, then poll
    /// the preflight for up to 5 s (Tahoe can lag between the attempt and
    /// the Settings pane showing the app) before reporting.
    ///
    /// On macOS 14+ a real ScreenCaptureKit capture attempt is what registers
    /// the app for Screen Recording. The previous trigger,
    /// `CGDisplayCreateImage`, is obsoleted in macOS 15. Do NOT use
    /// `CGRequestScreenCaptureAccess`: it triggers stale prompts for
    /// previously seen binaries from the same developer on Tahoe.
    func registerAndAwaitGrant() async -> Bool {
        await registrationProbe()
        for _ in 0..<20 {
            if hasPermission() { return true }
            try? await Task.sleep(nanoseconds: pollIntervalNanoseconds)
        }
        return hasPermission()
    }

    static func exerciseScreenCaptureKitRegistrationProbe() async {
        guard let content = try? await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: true
        ) else {
            return
        }
        guard let display = content.displays.first else {
            return
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])
        let config = SCStreamConfiguration()
        config.width = max(display.width, 1)
        config.height = max(display.height, 1)
        config.pixelFormat = kCVPixelFormatType_32BGRA

        _ = try? await SCScreenshotManager.captureImage(
            contentFilter: filter,
            configuration: config
        )
    }

    // MARK: - ScreenCaptureKit

    private func captureViaSCK() async -> (Int, Int, Data)? {
        guard let content = try? await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: true
        ) else {
            return nil
        }

        guard let display = content.displays.first else {
            return nil
        }

        let filter = SCContentFilter(display: display, excludingWindows: [])

        let config = SCStreamConfiguration()
        config.width = display.width
        config.height = display.height
        config.pixelFormat = kCVPixelFormatType_32BGRA

        guard let cgImage = try? await SCScreenshotManager.captureImage(
            contentFilter: filter,
            configuration: config
        ) else {
            return nil
        }

        return cgImageToBGRA(cgImage)
    }

    // MARK: - CGImage to BGRA conversion

    private func cgImageToBGRA(_ image: CGImage) -> (Int, Int, Data)? {
        let width = image.width
        let height = image.height
        let bytesPerRow = width * 4
        var pixels = Data(count: height * bytesPerRow)

        let success: Bool = pixels.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else { return false }
            guard let context = CGContext(
                data: baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue
                    | CGImageAlphaInfo.premultipliedFirst.rawValue
            ) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }

        guard success else { return nil }
        return (width, height, pixels)
    }
}
