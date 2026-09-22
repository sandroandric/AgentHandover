import XCTest
import CoreGraphics
import ImageIO
import ScreenCaptureKit
@testable import AgentHandoverApp

final class ScreenCaptureTests: XCTestCase {

    // MARK: - Helpers

    private func syntheticImage(width: Int, height: Int) -> CGImage {
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        context.setFillColor(CGColor(red: 0.1, green: 0.2, blue: 0.8, alpha: 1))
        for row in stride(from: 0, to: height, by: 20) {
            context.fill(CGRect(x: 10, y: row, width: width / 2, height: 8))
        }
        return context.makeImage()!
    }

    private func sampleFrame() -> EncodedFrame {
        ScreenCaptureService.encodeJPEG(syntheticImage(width: 64, height: 48), scale: 1, quality: 70)!
    }

    private func temporarySocketPath() -> String {
        let name = "ahcap-\(UUID().uuidString.prefix(8)).sock"
        return (NSTemporaryDirectory() as NSString).appendingPathComponent(name)
    }

    /// Connect to `path`, send `request`, read until the server closes.
    private func roundTrip(_ path: String, request: String) throws -> Data {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        XCTAssertGreaterThanOrEqual(fd, 0)
        defer { close(fd) }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: 104) { dest in
                for (i, byte) in path.utf8CString.enumerated() { dest[i] = byte }
            }
        }
        let connected = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connected == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        _ = request.withCString { send(fd, $0, strlen($0), 0) }
        var response = Data()
        var buffer = [UInt8](repeating: 0, count: 65536)
        while true {
            let n = recv(fd, &buffer, buffer.count, 0)
            if n <= 0 { break }
            response.append(contentsOf: buffer[0..<n])
        }
        return response
    }

    private func u32(_ data: Data, at offset: Int) -> UInt32 {
        data.subdata(in: offset..<(offset + 4)).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) }
            .littleEndian
    }

    // MARK: - Request parsing

    func testRequestDefaultsWhenOnlyCommandGiven() {
        XCTAssertEqual(try CaptureRequest.parse(["command": "capture"]).get(), CaptureRequest())
    }

    func testRequestParsesScaleFormatQuality() throws {
        let request = try CaptureRequest.parse(
            ["command": "capture", "scale": 0.75, "format": "jpeg", "quality": 55]).get()
        XCTAssertEqual(request.scale, 0.75)
        XCTAssertEqual(request.quality, 55)
    }

    func testRequestRejectsBadValues() {
        let bad: [[String: Any]] = [
            ["command": "snapshot"],
            ["command": "capture", "format": "bgra"],
            ["command": "capture", "scale": 0],
            ["command": "capture", "scale": 1.5],
            ["command": "capture", "quality": 0],
            ["command": "capture", "quality": 101],
            ["scale": 0.5],
        ]
        for json in bad {
            guard case .failure(.invalidRequest) = CaptureRequest.parse(json) else {
                return XCTFail("accepted \(json)")
            }
        }
    }

    // MARK: - Resolution and target selection

    func testNativePixelSizeUsesBackingScale() {
        let size = ScreenCaptureService.nativePixelSize(
            contentRect: CGRect(x: 0, y: 0, width: 1728, height: 1117), pointPixelScale: 2)
        XCTAssertEqual(size.width, 3456)
        XCTAssertEqual(size.height, 2234)

        let window = ScreenCaptureService.nativePixelSize(
            contentRect: CGRect(x: 0, y: 0, width: 800.5, height: 600), pointPixelScale: 2)
        XCTAssertEqual(window.width, 1601)
        XCTAssertEqual(window.height, 1200)
    }

    func testDisplaySelectionFollowsFocusedWindowNotFirstDisplay() {
        let frames: [(id: CGDirectDisplayID, frame: CGRect)] = [
            (id: 1, frame: CGRect(x: 0, y: 0, width: 1728, height: 1117)),
            (id: 2, frame: CGRect(x: 1728, y: 0, width: 2560, height: 1440)),
        ]
        let onSecond = ScreenCaptureService.anchorPoint(
            focusedBounds: CGRect(x: 2000, y: 100, width: 800, height: 600))
        XCTAssertEqual(
            ScreenCaptureService.displayID(containing: onSecond, frames: frames, mainID: 1), 2)
        // Nothing under the point: the main display, even when it is not first.
        XCTAssertEqual(
            ScreenCaptureService.displayID(
                containing: CGPoint(x: -500, y: -500), frames: frames, mainID: 2), 2)
        // Main display missing from the shareable list: no guess.
        XCTAssertNil(
            ScreenCaptureService.displayID(
                containing: CGPoint(x: -500, y: -500), frames: frames, mainID: 9))
    }

    // MARK: - JPEG encoding

    func testEncodeJPEGDownscalesAndProducesJPEG() throws {
        let frame = try XCTUnwrap(
            ScreenCaptureService.encodeJPEG(syntheticImage(width: 400, height: 300), scale: 0.5, quality: 70))
        XCTAssertEqual(frame.width, 200)
        XCTAssertEqual(frame.height, 150)
        XCTAssertEqual(Array(frame.jpeg.prefix(2)), [0xFF, 0xD8], "payload must be JPEG (SOI)")

        let source = try XCTUnwrap(CGImageSourceCreateWithData(frame.jpeg as CFData, nil))
        let decoded = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual(decoded.width, 200)
        XCTAssertEqual(decoded.height, 150)
    }

    func testEncodeJPEGScaleOneKeepsNativePixels() throws {
        let frame = try XCTUnwrap(
            ScreenCaptureService.encodeJPEG(syntheticImage(width: 321, height: 123), scale: 1, quality: 70))
        XCTAssertEqual(frame.width, 321)
        XCTAssertEqual(frame.height, 123)
    }

    // MARK: - Bounded capture

    func testHungCaptureReturnsTimeoutWithinEightSeconds() {
        XCTAssertEqual(ScreenCaptureServer.defaultCaptureTimeout, 8)
        let server = ScreenCaptureServer(socketPath: temporarySocketPath()) { _ in
            try? await Task.sleep(nanoseconds: 60_000_000_000)
            return .failure(.captureFailed("unreachable"))
        }
        let started = Date()
        let result = server.performCapture(CaptureRequest())
        let elapsed = Date().timeIntervalSince(started)

        guard case .failure(let error) = result else { return XCTFail("expected timeout") }
        XCTAssertEqual(error, .timeout)
        XCTAssertEqual(error.message, "capture timeout")
        XCTAssertLessThan(elapsed, 8.5)
        XCTAssertGreaterThanOrEqual(elapsed, 7.9)
    }

    func testPendingHungCapturesFailFastInsteadOfPilingUp() {
        let server = ScreenCaptureServer(socketPath: temporarySocketPath(), captureTimeout: 0.2) { _ in
            try? await Task.sleep(nanoseconds: 60_000_000_000)
            return .failure(.captureFailed("unreachable"))
        }
        XCTAssertEqual(server.performCapture(CaptureRequest()).failureValue, .timeout)
        XCTAssertEqual(server.performCapture(CaptureRequest()).failureValue, .timeout)

        let started = Date()
        let third = server.performCapture(CaptureRequest())
        XCTAssertEqual(third.failureValue, .stalled(pending: 2))
        XCTAssertLessThan(Date().timeIntervalSince(started), 0.1)
    }

    func testCaptureRecoversAfterPendingCaptureFinishes() {
        let server = ScreenCaptureServer(socketPath: temporarySocketPath(), captureTimeout: 0.1) { _ in
            try? await Task.sleep(nanoseconds: 300_000_000)
            return .failure(.captureFailed("slow"))
        }
        XCTAssertEqual(server.performCapture(CaptureRequest()).failureValue, .timeout)
        Thread.sleep(forTimeInterval: 0.4)
        // The earlier task finished, so this one runs (and times out itself)
        // rather than being refused as stalled.
        XCTAssertEqual(server.performCapture(CaptureRequest()).failureValue, .timeout)
    }

    // MARK: - Socket protocol

    func testSocketCarriesJPEGFrameWithHeader() throws {
        let path = temporarySocketPath()
        let expected = sampleFrame()
        var seen: CaptureRequest?
        let server = ScreenCaptureServer(socketPath: path) { request in
            seen = request
            return .success(expected)
        }
        server.start()
        defer { server.stop() }

        let response = try roundTrip(
            path, request: #"{"command":"capture","scale":0.5,"format":"jpeg","quality":70}"# + "\n")

        XCTAssertEqual(Array(response.prefix(4)), ScreenCaptureServer.frameMagic)
        XCTAssertEqual(u32(response, at: 4), UInt32(expected.width))
        XCTAssertEqual(u32(response, at: 8), UInt32(expected.height))
        XCTAssertEqual(u32(response, at: 12), UInt32(expected.jpeg.count))
        XCTAssertEqual(response.count, 16 + expected.jpeg.count)
        XCTAssertEqual(response.subdata(in: 16..<response.count), expected.jpeg)
        XCTAssertEqual(seen, CaptureRequest(scale: 0.5, quality: 70))
    }

    func testSocketReportsErrorsAsJSON() throws {
        let path = temporarySocketPath()
        let server = ScreenCaptureServer(socketPath: path) { _ in .failure(.timeout) }
        server.start()
        defer { server.stop() }

        let timeout = try roundTrip(path, request: #"{"command":"capture"}"# + "\n")
        let json = try JSONSerialization.jsonObject(with: timeout) as? [String: String]
        XCTAssertEqual(json?["error"], "capture timeout")

        let badFormat = try roundTrip(path, request: #"{"command":"capture","format":"png"}"# + "\n")
        let badJSON = try JSONSerialization.jsonObject(with: badFormat) as? [String: String]
        XCTAssertEqual(badJSON?["error"], "unsupported format: png")
    }

    func testStopEndsAcceptLoopAndClosesSocket() throws {
        let path = temporarySocketPath()
        let server = ScreenCaptureServer(socketPath: path) { _ in .success(self.sampleFrame()) }
        server.start()
        let listener = try XCTUnwrap(server.currentListener)
        _ = try roundTrip(path, request: #"{"command":"capture"}"#)

        server.stop()
        XCTAssertEqual(listener.exited.wait(timeout: .now() + 1), .success,
                       "accept loop must exit (and close its fd) after stop()")
        XCTAssertFalse(FileManager.default.fileExists(atPath: path))
        XCTAssertThrowsError(try roundTrip(path, request: #"{"command":"capture"}"#))

        // Restart on the same path works.
        server.start()
        defer { server.stop() }
        XCTAssertFalse(try roundTrip(path, request: #"{"command":"capture"}"#).isEmpty)
    }

    // MARK: - Screen Recording readiness probe

    func testProbeConfigurationIsTinyNotDisplaySized() throws {
        guard #available(macOS 14.0, *) else { throw XCTSkip("SCScreenshotManager needs macOS 14") }
        let config = ScreenCaptureService.probeConfiguration()
        XCTAssertEqual(config.width, 2)
        XCTAssertEqual(config.height, 2)
    }

    func testTwentyPermissionChecksAreFastAndProbeOnce() async {
        let calls = Counter()
        let probe = ScreenRecordingProbe(probe: {
            await calls.increment()
            try? await Task.sleep(nanoseconds: 20_000_000)
            return true
        })
        let started = Date()
        for _ in 0..<20 {
            let ready = await probe.isCaptureReady()
            XCTAssertTrue(ready)
        }
        XCTAssertLessThan(Date().timeIntervalSince(started), 0.2)
        let count = await calls.value
        XCTAssertEqual(count, 1)
    }

    func testProbeFailureIsNotCachedAndSuccessExpires() async {
        let calls = Counter()
        let outcome = Flag()
        let clock = Clock()
        let probe = ScreenRecordingProbe(
            probe: { await calls.increment(); return await outcome.value },
            now: { clock.now })

        let first = await probe.isCaptureReady()
        let second = await probe.isCaptureReady()
        XCTAssertFalse(first)
        XCTAssertFalse(second)
        var count = await calls.value
        XCTAssertEqual(count, 2, "a failed probe must be retried so the grant flow sees the pipeline come alive")

        await outcome.set(true)
        let third = await probe.isCaptureReady()
        XCTAssertTrue(third)
        clock.advance(ScreenRecordingProbe.successTTL - 1)
        _ = await probe.isCaptureReady()
        count = await calls.value
        XCTAssertEqual(count, 3)

        clock.advance(2)
        _ = await probe.isCaptureReady()
        count = await calls.value
        XCTAssertEqual(count, 4, "cached success expires after the TTL")
    }

    // MARK: - Live capture (test Mac only: needs Screen Recording for the test runner)

    /// Set AH_LIVE_CAPTURE_TESTS=1 to run. Asserts the captured frame at
    /// scale 1.0 has the native pixel height of the captured target.
    func testLiveCaptureIsNativeResolution() async throws {
        guard ProcessInfo.processInfo.environment["AH_LIVE_CAPTURE_TESTS"] == "1" else {
            throw XCTSkip("live capture test; set AH_LIVE_CAPTURE_TESTS=1 on the test Mac")
        }
        let service = ScreenCaptureService()
        guard service.hasPermission() else { throw XCTSkip("Screen Recording not granted to runner") }

        let focused = ScreenCaptureService.focusedWindow()
        let frame = try await service.captureFrame(CaptureRequest(scale: 1.0, quality: 70)).get()

        let anchor = ScreenCaptureService.anchorPoint(focusedBounds: focused?.bounds)
        var displayID = CGMainDisplayID()
        if let anchor {
            var hit: CGDirectDisplayID = 0
            var count: UInt32 = 0
            if CGGetDisplaysWithPoint(anchor, 1, &hit, &count) == .success, count > 0 {
                displayID = hit
            }
        }
        let mode = try XCTUnwrap(CGDisplayCopyDisplayMode(displayID))
        let nativeHeight = mode.pixelHeight
        let scale = Double(mode.pixelHeight) / Double(mode.height)
        let expectedHeight = focused.map { Int(($0.bounds.height * scale).rounded()) } ?? nativeHeight
        XCTAssertEqual(Double(frame.height), Double(expectedHeight), accuracy: 2,
                       "frame height \(frame.height) must be native pixels (\(expectedHeight))")
        print("live capture: \(frame.width)x\(frame.height), \(frame.jpeg.count) bytes at scale 1.0")

        let half = try await service.captureFrame(CaptureRequest()).get()
        print("live capture: \(half.width)x\(half.height), \(half.jpeg.count) bytes at scale 0.5 q70")
        XCTAssertLessThanOrEqual(half.jpeg.count, 400_000)
    }
}

// MARK: - Test doubles

private actor Counter {
    private(set) var value = 0
    func increment() { value += 1 }
}

private actor Flag {
    private(set) var value = false
    func set(_ newValue: Bool) { value = newValue }
}

private final class Clock: @unchecked Sendable {
    private let lock = NSLock()
    private var current = Date(timeIntervalSince1970: 1_000_000)
    var now: Date { lock.lock(); defer { lock.unlock() }; return current }
    func advance(_ seconds: TimeInterval) { lock.lock(); current += seconds; lock.unlock() }
}

private extension Result {
    var failureValue: Failure? {
        if case .failure(let error) = self { return error }
        return nil
    }
}
