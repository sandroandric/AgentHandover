import ScreenCaptureKit
import CoreGraphics
import AppKit
import ImageIO
import UniformTypeIdentifiers

/// Parameters of one capture request from the daemon.
///
/// Wire form: `{"command":"capture","scale":0.5,"format":"jpeg","quality":70}`.
/// `scale` applies to the native-pixel capture; `format` must be `jpeg`.
struct CaptureRequest: Equatable {
    static let defaultScale = 0.5
    static let defaultQuality = 70

    var scale: Double = CaptureRequest.defaultScale
    var quality: Int = CaptureRequest.defaultQuality

    static func parse(_ json: [String: Any]) -> Result<CaptureRequest, CaptureError> {
        guard let command = json["command"] as? String else {
            return .failure(.invalidRequest("missing command"))
        }
        guard command == "capture" else {
            return .failure(.invalidRequest("unknown command: \(command)"))
        }

        var request = CaptureRequest()
        if let format = json["format"] {
            guard let format = format as? String, format == "jpeg" else {
                return .failure(.invalidRequest("unsupported format: \(format)"))
            }
        }
        if let scale = json["scale"] {
            guard let scale = (scale as? NSNumber)?.doubleValue, scale > 0, scale <= 1 else {
                return .failure(.invalidRequest("scale must be in (0, 1]"))
            }
            request.scale = scale
        }
        if let quality = json["quality"] {
            guard let quality = (quality as? NSNumber)?.intValue, (1...100).contains(quality) else {
                return .failure(.invalidRequest("quality must be in 1...100"))
            }
            request.quality = quality
        }
        return .success(request)
    }
}

/// One JPEG-encoded frame, ready to go over `capture.sock`.
struct EncodedFrame {
    /// Per-frame JPEG budget on `capture.sock` (and on disk).
    static let maxBytes = 400_000
    /// Quality floor when stepping down to fit `maxBytes`; below this small
    /// UI text stops being legible, so a frame is sent over budget instead.
    static let minQuality = 40
    static let qualityStep = 10

    let width: Int
    let height: Int
    let jpeg: Data
    /// The quality the JPEG was actually encoded at.
    let quality: Int
}

enum CaptureError: Error, Equatable {
    case permissionDenied
    case noDisplay
    case captureFailed(String)
    case encodeFailed
    case timeout
    case stalled(pending: Int)
    case invalidRequest(String)

    /// Text sent to the daemon in `{"error": ...}`.
    var message: String {
        switch self {
        case .permissionDenied: return "screen recording permission not granted"
        case .noDisplay: return "no display to capture"
        case .captureFailed(let reason): return "capture failed: \(reason)"
        case .encodeFailed: return "jpeg encode failed"
        case .timeout: return "capture timeout"
        case .stalled(let pending):
            return "capture timeout: \(pending) earlier captures still pending"
        case .invalidRequest(let reason): return reason
        }
    }
}

/// Captures the focused window at native pixel resolution using
/// ScreenCaptureKit (macOS 14+, the package minimum).
/// This runs in the main app process, which is the Screen Recording TCC principal.
final class ScreenCaptureService {

    private let preflight: () -> Bool
    private let registrationProbe: () async -> Void
    private let pollIntervalNanoseconds: UInt64

    /// - Parameters:
    ///   - preflight: whether Screen Recording is granted (TCC preflight).
    ///   - registrationProbe: the capture attempt that makes macOS register
    ///     this app for Screen Recording (defaults to the tiny ScreenCaptureKit
    ///     readiness probe, `runCaptureProbe`; injectable for tests).
    ///   - pollIntervalNanoseconds: how often to re-check after the probe.
    init(
        preflight: @escaping () -> Bool = { CGPreflightScreenCaptureAccess() },
        registrationProbe: @escaping () async -> Void = { _ = await ScreenCaptureService.runCaptureProbe() },
        pollIntervalNanoseconds: UInt64 = 250_000_000
    ) {
        self.preflight = preflight
        self.registrationProbe = registrationProbe
        self.pollIntervalNanoseconds = pollIntervalNanoseconds
    }

    /// Capture the focused window (or the display under it), downscale by
    /// `request.scale` and JPEG-encode at `request.quality`.
    func captureFrame(_ request: CaptureRequest) async -> Result<EncodedFrame, CaptureError> {
        guard hasPermission() else { return .failure(.permissionDenied) }
        let image = await captureViaSCK()
        return image.flatMap { native in
            guard let frame = Self.encodeJPEG(native, scale: request.scale, quality: request.quality) else {
                return .failure(.encodeFailed)
            }
            return .success(frame)
        }
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
    /// TCC registers the app for Screen Recording after its first capture
    /// attempt, so by default this runs the same tiny ScreenCaptureKit
    /// capture the readiness probe uses (`runCaptureProbe`).
    /// `CGDisplayCreateImage` is obsoleted in macOS 15. Do NOT use
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

    // MARK: - Readiness probe

    /// Side length of the probe capture. The probe answers "can this process
    /// capture?" without materialising a display-sized buffer.
    static let probeSidePixels = 2

    /// One tiny capture of the main display. True when the capture pipeline
    /// returns an image. Nonisolated and async, so it never runs on the main actor.
    static func runCaptureProbe() async -> Bool {
        guard let content = try? await SCShareableContent.excludingDesktopWindows(
            false, onScreenWindowsOnly: true
        ) else {
            return false
        }
        let mainID = CGMainDisplayID()
        guard let display = content.displays.first(where: { $0.displayID == mainID }) else {
            return false
        }
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let image = try? await SCScreenshotManager.captureImage(
            contentFilter: filter,
            configuration: probeConfiguration()
        )
        return image != nil
    }

    static func probeConfiguration() -> SCStreamConfiguration {
        let config = SCStreamConfiguration()
        config.width = probeSidePixels
        config.height = probeSidePixels
        config.pixelFormat = kCVPixelFormatType_32BGRA
        return config
    }

    // MARK: - Target selection

    /// The frontmost normal window, as the observation snapshot reports it.
    static func focusedWindow() -> (id: CGWindowID, bounds: CGRect)? {
        guard let info = ObservationService().getFocusedWindow(displays: []),
              let id = CGWindowID(info.windowId),
              info.boundsGlobalPx.count == 4 else {
            return nil
        }
        let b = info.boundsGlobalPx
        return (id, CGRect(x: b[0], y: b[1], width: b[2], height: b[3]))
    }

    /// The point whose display should be captured when no window filter is
    /// possible: the focused window's centre, else the cursor.
    static func anchorPoint(focusedBounds: CGRect?) -> CGPoint? {
        if let bounds = focusedBounds, bounds.width > 0, bounds.height > 0 {
            return CGPoint(x: bounds.midX, y: bounds.midY)
        }
        return CGEvent(source: nil)?.location
    }

    /// Pick the display containing `point` (global top-left coordinates),
    /// else the main display. Never an arbitrary first display.
    static func displayID(
        containing point: CGPoint?,
        frames: [(id: CGDirectDisplayID, frame: CGRect)],
        mainID: CGDirectDisplayID
    ) -> CGDirectDisplayID? {
        if let point, let hit = frames.first(where: { $0.frame.contains(point) }) {
            return hit.id
        }
        return frames.first(where: { $0.id == mainID })?.id
    }

    /// Point-to-pixel scale of a display's current mode: 2 on a Retina panel
    /// (including "looks like" scaled modes), 1 on a standard display.
    /// Read from the display mode itself because the running app on a 2x
    /// panel produced frames at point resolution when the size came from
    /// `SCContentFilter.pointPixelScale` alone.
    static func displayPixelScale(_ id: CGDirectDisplayID) -> CGFloat? {
        guard let mode = CGDisplayCopyDisplayMode(id), mode.width > 0 else { return nil }
        return CGFloat(mode.pixelWidth) / CGFloat(mode.width)
    }

    /// The point-to-pixel scale to capture at: the larger of what the filter
    /// reports and the scale of the display the target is on, never below 1.
    static func capturePixelScale(filterScale: Float, displayScale: CGFloat?) -> CGFloat {
        max(CGFloat(filterScale), displayScale ?? 1, 1)
    }

    /// Native pixel size for a filter's content rect (points) at a
    /// point-to-pixel scale.
    static func nativePixelSize(contentRect: CGRect, pixelScale: CGFloat) -> (width: Int, height: Int) {
        let scale = max(pixelScale, 1)
        return (
            max(Int((contentRect.width * scale).rounded()), 1),
            max(Int((contentRect.height * scale).rounded()), 1)
        )
    }

    // MARK: - ScreenCaptureKit (macOS 14+)

    private func captureViaSCK() async -> Result<CGImage, CaptureError> {
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true
            )
        } catch {
            return .failure(.captureFailed("shareable content: \(error.localizedDescription)"))
        }

        let focused = Self.focusedWindow()
        // The display under the focused window (or the cursor): the capture
        // target when no window filter is possible, and the source of the
        // pixel scale on both paths.
        let displayID = Self.displayID(
            containing: Self.anchorPoint(focusedBounds: focused?.bounds),
            frames: content.displays.map { (id: $0.displayID, frame: $0.frame) },
            mainID: CGMainDisplayID()
        )
        let filter: SCContentFilter
        if let focused, let window = content.windows.first(where: { $0.windowID == focused.id }) {
            filter = SCContentFilter(desktopIndependentWindow: window)
        } else {
            // Desktop, full-screen space or unresolvable window.
            guard let displayID,
                  let display = content.displays.first(where: { $0.displayID == displayID }) else {
                return .failure(.noDisplay)
            }
            filter = SCContentFilter(display: display, excludingWindows: [])
        }

        let pixelScale = Self.capturePixelScale(
            filterScale: filter.pointPixelScale,
            displayScale: displayID.flatMap(Self.displayPixelScale))
        let size = Self.nativePixelSize(contentRect: filter.contentRect, pixelScale: pixelScale)
        let config = SCStreamConfiguration()
        config.width = size.width
        config.height = size.height
        // `.automatic` may capture at point resolution and resample up to
        // width x height; `.best` takes the backing store's pixels.
        config.captureResolution = .best
        config.pixelFormat = kCVPixelFormatType_32BGRA

        do {
            let image = try await SCScreenshotManager.captureImage(
                contentFilter: filter, configuration: config)
            if image.width != size.width || image.height != size.height {
                NSLog(
                    "ScreenCaptureService: asked for %dx%d (%.0fx%.0f pt, filter scale %.2f, capture scale %.2f), got %dx%d",
                    size.width, size.height, filter.contentRect.width, filter.contentRect.height,
                    Double(filter.pointPixelScale), Double(pixelScale), image.width, image.height)
            }
            return .success(image)
        } catch {
            return .failure(.captureFailed(error.localizedDescription))
        }
    }

    // MARK: - Downscale + JPEG

    /// Downscale `image` by `scale` and encode it as JPEG at `quality` (1-100),
    /// lowering the quality in steps (not below `EncodedFrame.minQuality`)
    /// while the JPEG exceeds `maxBytes`.
    static func encodeJPEG(
        _ image: CGImage, scale: Double, quality: Int, maxBytes: Int = EncodedFrame.maxBytes
    ) -> EncodedFrame? {
        let width = max(Int(Double(image.width) * scale), 1)
        let height = max(Int(Double(image.height) * scale), 1)

        let target: CGImage
        if width == image.width && height == image.height {
            target = image
        } else {
            guard let space = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(
                      data: nil,
                      width: width,
                      height: height,
                      bitsPerComponent: 8,
                      bytesPerRow: 0,
                      space: space,
                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                  ) else {
                return nil
            }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            guard let scaled = context.makeImage() else { return nil }
            target = scaled
        }

        // Dense screens (code, spreadsheets) can exceed the per-frame budget
        // at the requested quality; step the quality down until they fit.
        var q = quality
        while true {
            guard let jpeg = jpegData(target, quality: q) else { return nil }
            if jpeg.count <= maxBytes || q <= EncodedFrame.minQuality {
                return EncodedFrame(width: target.width, height: target.height, jpeg: jpeg, quality: q)
            }
            q = max(q - EncodedFrame.qualityStep, EncodedFrame.minQuality)
        }
    }

    private static func jpegData(_ image: CGImage, quality: Int) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.jpeg.identifier as CFString, 1, nil
        ) else {
            return nil
        }
        let options = [
            kCGImageDestinationLossyCompressionQuality: Double(quality) / 100.0
        ] as CFDictionary
        CGImageDestinationAddImage(destination, image, options)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }
}

/// Answers "is the capture pipeline live?" for `PermissionChecker` with one
/// tiny capture, cached on success so repeated UI checks are free.
///
/// Failures are not cached: the grant flow polls this while waiting for the
/// capture pipeline to come alive after the user toggles the permission.
actor ScreenRecordingProbe {
    static let shared = ScreenRecordingProbe()
    static let successTTL: TimeInterval = 60

    private let probe: @Sendable () async -> Bool
    private let now: @Sendable () -> Date
    private var lastSuccess: Date?

    init(
        probe: @escaping @Sendable () async -> Bool = { await ScreenCaptureService.runCaptureProbe() },
        now: @escaping @Sendable () -> Date = { Date() }
    ) {
        self.probe = probe
        self.now = now
    }

    func isCaptureReady() async -> Bool {
        if let lastSuccess, now().timeIntervalSince(lastSuccess) < Self.successTTL {
            return true
        }
        let ready = await probe()
        lastSuccess = ready ? now() : nil
        return ready
    }
}
