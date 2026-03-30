import ScreenCaptureKit
import CoreGraphics
import AppKit

/// Captures screenshots using ScreenCaptureKit (macOS 14+).
/// Falls back to CGDisplayCreateImage on older systems.
/// This runs in the main app process, which is the Screen Recording TCC principal.
final class ScreenCaptureService {

    /// Capture the main display as raw BGRA pixel data.
    /// Returns (width, height, bgra_bytes) or nil if capture fails.
    func captureMainDisplay() async -> (Int, Int, Data)? {
        guard hasPermission() else { return nil }

        // Use SCScreenshotManager on macOS 14+
        if #available(macOS 14.0, *) {
            return await captureViaSCK()
        }
        // Fallback to CoreGraphics
        return captureViaCG()
    }

    /// Check if Screen Recording permission is granted.
    func hasPermission() -> Bool {
        CGPreflightScreenCaptureAccess()
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

        // TCC registers the app for Screen Recording after its first capture
        // attempt. CGDisplayCreateImage is the most reliable trigger — it
        // works even without prior permission (unlike SCShareableContent which
        // throws before reaching the capture call).
        // DO NOT use CGRequestScreenCaptureAccess — it triggers stale prompts
        // for previously-seen binaries from the same developer on Tahoe.
        _ = CGDisplayCreateImage(CGMainDisplayID())

        // Tahoe can lag between the request call and the Settings pane showing
        // the new app entry. Poll preflight for a short bounded window before
        // deciding we need to send the user to System Settings manually.
        for _ in 0..<20 {
            if hasPermission() {
                if previousPolicy == .accessory {
                    NSApp.setActivationPolicy(.accessory)
                }
                return true
            }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }

        if previousPolicy == .accessory {
            NSApp.setActivationPolicy(.accessory)
        }
        return false
    }

    @available(macOS 14.0, *)
    private func exerciseScreenCaptureKitRegistrationProbe() async {
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

    // MARK: - ScreenCaptureKit (macOS 14+)

    @available(macOS 14.0, *)
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

    // MARK: - CoreGraphics fallback

    private func captureViaCG() -> (Int, Int, Data)? {
        guard let cgImage = CGDisplayCreateImage(CGMainDisplayID()) else {
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
