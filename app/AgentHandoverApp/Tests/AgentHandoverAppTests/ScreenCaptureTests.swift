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

    /// Rows of 12 pt monospaced text at 2x (24 px glyphs, 16 pt line pitch)
    /// covering `coverage` of each row's width: 1.0 is a terminal filled
    /// edge to edge, 0.4 an editor with gutter, sidebar and ragged lines.
    private func denseTextImage(width: Int, height: Int, coverage: Double = 1) -> CGImage {
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let source = "func captureFrame(_ request: CaptureRequest) async -> Result<EncodedFrame, CaptureError> { guard hasPermission() else { return .failure(.permissionDenied) } } // 0123456789 "
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 24, weight: .regular),
            .foregroundColor: NSColor(red: 0.15, green: 0.15, blue: 0.2, alpha: 1),
        ]
        var row = 0
        for y in stride(from: height - 32, to: 0, by: -32) {
            let offset = (row * 7) % source.count
            let text = String((String(source.dropFirst(offset)) + source + source + source)
                .prefix(Int(Double(width / 14) * coverage)))
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: text, attributes: attributes))
            context.textPosition = CGPoint(x: 12, y: y)
            CTLineDraw(line, context)
            row += 1
        }
        return context.makeImage()!
    }

    private func sampleFrame() -> EncodedFrame {
        ScreenCaptureService.encodeJPEG(syntheticImage(width: 64, height: 48), scale: 1, quality: 70)!
    }

    private func temporarySocketPath() -> String {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ahcap-\(UUID().uuidString.prefix(8))")
        return (dir as NSString).appendingPathComponent("capture.sock")
    }

    /// Lets this test process (and only it) connect.
    private var testRunnerPolicy: PeerPolicy {
        PeerPolicy(allowedPaths: [PeerPolicy.executablePath(of: getpid())!], identifiers: [], teamID: nil)
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
            contentRect: CGRect(x: 0, y: 0, width: 1728, height: 1117), pixelScale: 2)
        XCTAssertEqual(size.width, 3456)
        XCTAssertEqual(size.height, 2234)

        let window = ScreenCaptureService.nativePixelSize(
            contentRect: CGRect(x: 0, y: 0, width: 800.5, height: 600), pixelScale: 2)
        XCTAssertEqual(window.width, 1601)
        XCTAssertEqual(window.height, 1200)
    }

    /// WP-C8b: on the test Mac (3840x2160 panel at 2x, 1920x1080 points) the
    /// stored frame was 960x540. The capture size must come out native even
    /// when the filter reports a point-to-pixel scale of 1.
    func testCaptureScaleFallsBackToTheDisplayModeScale() {
        XCTAssertEqual(ScreenCaptureService.capturePixelScale(filterScale: 1, displayScale: 2), 2)
        XCTAssertEqual(ScreenCaptureService.capturePixelScale(filterScale: 2, displayScale: nil), 2)
        XCTAssertEqual(ScreenCaptureService.capturePixelScale(filterScale: 2, displayScale: 1), 2)
        XCTAssertEqual(ScreenCaptureService.capturePixelScale(filterScale: 0, displayScale: nil), 1)

        let testMac = ScreenCaptureService.nativePixelSize(
            contentRect: CGRect(x: 0, y: 0, width: 1920, height: 1080),
            pixelScale: ScreenCaptureService.capturePixelScale(filterScale: 1, displayScale: 2))
        XCTAssertEqual(testMac.width, 3840)
        XCTAssertEqual(testMac.height, 2160)
    }

    func testDisplayPixelScaleReadsTheMainDisplayMode() throws {
        let mode = try XCTUnwrap(CGDisplayCopyDisplayMode(CGMainDisplayID()))
        let scale = try XCTUnwrap(ScreenCaptureService.displayPixelScale(CGMainDisplayID()))
        XCTAssertEqual(scale, CGFloat(mode.pixelWidth) / CGFloat(mode.width))
        XCTAssertGreaterThanOrEqual(scale, 1)
    }

    /// Native 3840x2160 at the daemon's scale 0.5 is a 1920x1080 frame, and a
    /// text-heavy editor screen fits the 400 KB per-frame budget (quality
    /// stepped down from 70 where needed, never below the floor).
    func testNativeFrameAtDaemonScaleIsFullHDAndWithinBudget() throws {
        let native = denseTextImage(width: 3840, height: 2160, coverage: 0.4)
        let frame = try XCTUnwrap(ScreenCaptureService.encodeJPEG(
            native, scale: CaptureRequest.defaultScale, quality: CaptureRequest.defaultQuality))
        XCTAssertEqual(frame.width, 1920)
        XCTAssertEqual(frame.height, 1080)
        XCTAssertLessThanOrEqual(frame.jpeg.count, EncodedFrame.maxBytes)
        XCTAssertGreaterThanOrEqual(frame.quality, EncodedFrame.minQuality)
        print("editor text 1920x1080: \(frame.jpeg.count) bytes at q\(frame.quality)")
    }

    /// A display filled edge to edge with small text cannot reach 400 KB at a
    /// legible quality (about 560 KB at q40): the encoder stops at the floor
    /// and sends it rather than blurring the text further.
    func testWallOfTextStopsAtTheQualityFloor() throws {
        let native = denseTextImage(width: 3840, height: 2160, coverage: 1)
        let frame = try XCTUnwrap(ScreenCaptureService.encodeJPEG(
            native, scale: CaptureRequest.defaultScale, quality: CaptureRequest.defaultQuality))
        XCTAssertEqual(frame.width, 1920)
        XCTAssertEqual(frame.height, 1080)
        XCTAssertEqual(frame.quality, EncodedFrame.minQuality)
        print("wall of text 1920x1080: \(frame.jpeg.count) bytes at q\(frame.quality)")
    }

    func testEncodeJPEGStepsQualityDownToFitTheBudget() throws {
        let image = denseTextImage(width: 1200, height: 800)
        let full = try XCTUnwrap(
            ScreenCaptureService.encodeJPEG(image, scale: 1, quality: 90, maxBytes: .max))
        XCTAssertEqual(full.quality, 90)

        let budget = full.jpeg.count * 2 / 3
        let fitted = try XCTUnwrap(
            ScreenCaptureService.encodeJPEG(image, scale: 1, quality: 90, maxBytes: budget))
        XCTAssertLessThan(fitted.quality, 90)
        XCTAssertGreaterThanOrEqual(fitted.quality, EncodedFrame.minQuality)
        XCTAssertLessThanOrEqual(fitted.jpeg.count, budget)

        // An unreachable budget stops at the floor instead of destroying the text.
        let floor = try XCTUnwrap(
            ScreenCaptureService.encodeJPEG(image, scale: 1, quality: 90, maxBytes: 1))
        XCTAssertEqual(floor.quality, EncodedFrame.minQuality)
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
        let server = ScreenCaptureServer(socketPath: temporarySocketPath(), policy: testRunnerPolicy) { _ in
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
        let server = ScreenCaptureServer(socketPath: temporarySocketPath(), captureTimeout: 0.2, policy: testRunnerPolicy) { _ in
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
        let server = ScreenCaptureServer(socketPath: temporarySocketPath(), captureTimeout: 0.1, policy: testRunnerPolicy) { _ in
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
        let server = ScreenCaptureServer(socketPath: path, policy: testRunnerPolicy) { request in
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
        let server = ScreenCaptureServer(socketPath: path, policy: testRunnerPolicy) { _ in .failure(.timeout) }
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
        let server = ScreenCaptureServer(socketPath: path, policy: testRunnerPolicy) { _ in .success(self.sampleFrame()) }
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

    // MARK: - Concurrency and recycling (coverage audit #110)

    func testHungCaptureDoesNotBlockTheNextRequest() throws {
        let path = temporarySocketPath()
        let frame = sampleFrame()
        let calls = Counter()
        let server = ScreenCaptureServer(socketPath: path, captureTimeout: 1.5, policy: testRunnerPolicy) { _ in
            await calls.increment()
            if await calls.value == 1 {
                try? await Task.sleep(nanoseconds: 60_000_000_000)  // the hung call
                return .failure(.captureFailed("unreachable"))
            }
            return .success(frame)
        }
        server.start()
        let listener = try XCTUnwrap(server.currentListener)

        var first = Data()
        let firstDone = expectation(description: "first request answered")
        let firstStarted = Date()
        var firstElapsed: TimeInterval = 0
        DispatchQueue.global().async {
            first = (try? self.roundTrip(path, request: #"{"command":"capture"}"#)) ?? Data()
            firstElapsed = Date().timeIntervalSince(firstStarted)
            firstDone.fulfill()
        }
        Thread.sleep(forTimeInterval: 0.2)  // the first capture is now hung

        let started = Date()
        let second = try roundTrip(path, request: #"{"command":"capture"}"#)
        XCTAssertLessThan(Date().timeIntervalSince(started), 1.0,
                          "a hung capture must not park the acceptor")
        XCTAssertEqual(Array(second.prefix(4)), ScreenCaptureServer.frameMagic)

        wait(for: [firstDone], timeout: 5)
        let error = try JSONSerialization.jsonObject(with: first) as? [String: String]
        XCTAssertEqual(error?["error"], "capture timeout")
        XCTAssertGreaterThanOrEqual(firstElapsed, 1.4)

        server.stop()
        XCTAssertEqual(listener.exited.wait(timeout: .now() + 1), .success)
    }

    func testThreeTimeoutsRecycleTheCapturePathAndSurfaceAStall() {
        let built = Counter()
        let frame = sampleFrame()
        let server = ScreenCaptureServer(
            socketPath: temporarySocketPath(), captureTimeout: 0.1, minRecycleInterval: 60,
            policy: testRunnerPolicy,
            makeCapture: {
                let generation = built.incrementNow()
                return { _ in
                    if generation == 1 {  // the first ScreenCaptureService is wedged
                        try? await Task.sleep(nanoseconds: 60_000_000_000)
                        return .failure(.captureFailed("unreachable"))
                    }
                    return .success(frame)
                }
            })
        let stalled = expectation(forNotification: ScreenCaptureServer.stallStatusChanged, object: server)

        XCTAssertEqual(server.performCapture(CaptureRequest()).failureValue, .timeout)
        XCTAssertEqual(server.performCapture(CaptureRequest()).failureValue, .timeout)
        XCTAssertNil(server.stallStatus)
        XCTAssertEqual(server.recycleCount, 0)
        // Third: two hung captures pending, so it fails fast, and recycles.
        XCTAssertEqual(server.performCapture(CaptureRequest()).failureValue, .stalled(pending: 2))
        XCTAssertEqual(server.recycleCount, 1)
        XCTAssertEqual(built.valueNow, 2, "a fresh capture path was built")
        XCTAssertEqual(server.stallStatus, "Screen capture stalled; restarted the capture path")
        wait(for: [stalled], timeout: 1)

        // The fresh path works (the old hung tasks no longer hold slots).
        guard case .success = server.performCapture(CaptureRequest()) else {
            return XCTFail("the recycled capture path should answer")
        }
        XCTAssertNil(server.stallStatus, "a successful capture clears the stall")
    }

    func testRecyclingIsRateLimited() {
        let built = Counter()
        let server = ScreenCaptureServer(
            socketPath: temporarySocketPath(), captureTimeout: 0.05, minRecycleInterval: 60,
            policy: testRunnerPolicy,
            makeCapture: {
                _ = built.incrementNow()
                return { _ in
                    try? await Task.sleep(nanoseconds: 60_000_000_000)
                    return .failure(.captureFailed("unreachable"))
                }
            })
        for _ in 0..<9 { _ = server.performCapture(CaptureRequest()) }
        XCTAssertEqual(server.recycleCount, 1, "at most one recycle per minRecycleInterval")
        XCTAssertEqual(built.valueNow, 2)
    }

    // MARK: - Socket directory and peer checks (coverage audit #2)

    func testSocketDirectoryIsPrivate() throws {
        let path = temporarySocketPath()
        let dir = (path as NSString).deletingLastPathComponent
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o755])
        let server = ScreenCaptureServer(socketPath: path, policy: testRunnerPolicy) { _ in .failure(.timeout) }
        server.start()
        defer { server.stop() }
        let mode = try XCTUnwrap(
            FileManager.default.attributesOfItem(atPath: dir)[.posixPermissions] as? Int)
        XCTAssertEqual(mode, 0o700, "an existing socket directory is tightened to 0700")
        let sockMode = try XCTUnwrap(
            FileManager.default.attributesOfItem(atPath: path)[.posixPermissions] as? Int)
        XCTAssertEqual(sockMode & 0o077, 0)
    }

    func testSymlinkedSocketDirectoryIsRefused() throws {
        let base = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ahlink-\(UUID().uuidString.prefix(8))")
        let real = base + "-real"
        try FileManager.default.createDirectory(atPath: real, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(atPath: base, withDestinationPath: real)
        XCTAssertFalse(SocketDirectory.preparePrivate(base))
        let server = ScreenCaptureServer(
            socketPath: (base as NSString).appendingPathComponent("capture.sock"),
            policy: testRunnerPolicy) { _ in .failure(.timeout) }
        server.start()
        XCTAssertNil(server.currentListener, "no socket is bound in a symlinked directory")
    }

    /// A tiny Unix-socket client compiled for this test: `client <socket> <request>`
    /// sends the request and prints the reply. Stands in for the daemon.
    private func buildSocketClient(in dir: String) throws -> String {
        let source = (dir as NSString).appendingPathComponent("client.c")
        try """
        #include <string.h>
        #include <sys/socket.h>
        #include <sys/un.h>
        #include <unistd.h>
        int main(int argc, char **argv) {
            int fd = socket(AF_UNIX, SOCK_STREAM, 0);
            struct sockaddr_un a = {0};
            a.sun_family = AF_UNIX;
            strncpy(a.sun_path, argv[1], sizeof(a.sun_path) - 1);
            if (connect(fd, (struct sockaddr *)&a, sizeof(a)) != 0) return 2;
            write(fd, argv[2], strlen(argv[2]));
            char buf[65536];
            ssize_t n;
            while ((n = read(fd, buf, sizeof(buf))) > 0) write(1, buf, (size_t)n);
            return 0;
        }
        """.write(toFile: source, atomically: true, encoding: .utf8)
        let binary = (dir as NSString).appendingPathComponent("ah-observer")
        let cc = Process()
        cc.executableURL = URL(fileURLWithPath: "/usr/bin/cc")
        cc.arguments = ["-o", binary, source]
        try cc.run()
        cc.waitUntilExit()
        XCTAssertEqual(cc.terminationStatus, 0, "cc failed")
        return String(cString: try XCTUnwrap(realpath(binary, nil)))
    }

    private func runClient(_ executable: String, _ socketPath: String) throws -> Data {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = [socketPath, #"{"command":"capture"}"# + "\n"]
        let output = Pipe()
        process.standardOutput = output
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return data
    }

    func testOnlyTheAllowedExecutableMayConnect() throws {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("ahpeer-\(UUID().uuidString.prefix(8))")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let daemon = try buildSocketClient(in: dir)
        // A byte-identical copy of the allowed binary at another path.
        let copyDir = (dir as NSString).appendingPathComponent("elsewhere")
        try FileManager.default.createDirectory(atPath: copyDir, withIntermediateDirectories: true)
        let copy = (copyDir as NSString).appendingPathComponent("ah-observer")
        try FileManager.default.copyItem(atPath: daemon, toPath: copy)

        let path = temporarySocketPath()
        let policy = PeerPolicy(allowedPaths: [daemon], identifiers: [], teamID: nil)
        let server = ScreenCaptureServer(socketPath: path, policy: policy) { _ in .success(self.sampleFrame()) }
        server.start()
        defer { server.stop() }

        XCTAssertEqual(Array(try runClient(daemon, path).prefix(4)), ScreenCaptureServer.frameMagic)
        XCTAssertTrue(try runClient(copy, path).isEmpty,
                      "a copy of the allowed binary at a different path is refused")
        // This (unlisted) test process is refused too.
        XCTAssertTrue(try roundTrip(path, request: #"{"command":"capture"}"#).isEmpty)

        // Control: the copy works once its own path is the allowed one, so the
        // refusal above was the path check and nothing else.
        let path2 = temporarySocketPath()
        let server2 = ScreenCaptureServer(
            socketPath: path2,
            policy: PeerPolicy(allowedPaths: [String(cString: try XCTUnwrap(realpath(copy, nil)))],
                               identifiers: [], teamID: nil)
        ) { _ in .success(self.sampleFrame()) }
        server2.start()
        defer { server2.stop() }
        XCTAssertEqual(Array(try runClient(copy, path2).prefix(4)), ScreenCaptureServer.frameMagic)

        // With a Team ID pinned, the (ad hoc signed) helper at the allowed
        // path is refused by the code-signature check.
        let path3 = temporarySocketPath()
        let server3 = ScreenCaptureServer(
            socketPath: path3,
            policy: PeerPolicy(allowedPaths: [daemon], identifiers: [PeerPolicy.daemonIdentifier],
                               teamID: "ABCDE12345")
        ) { _ in .success(self.sampleFrame()) }
        server3.start()
        defer { server3.stop() }
        XCTAssertTrue(try runClient(daemon, path3).isEmpty,
                      "an unsigned helper fails the pinned code requirement")
    }

    func testCodeRequirementPinsTeamAndIdentifier() throws {
        let policy = PeerPolicy(
            allowedPaths: [], identifiers: ["com.agenthandover.observer", "agenthandover"],
            teamID: "ABCDE12345")
        XCTAssertEqual(
            policy.requirement,
            #"anchor apple generic and certificate leaf[subject.OU] = "ABCDE12345" and (identifier "com.agenthandover.observer" or identifier "agenthandover")"#)
        XCTAssertNil(PeerPolicy(allowedPaths: [], identifiers: ["x"], teamID: nil).requirement)

        // An unsigned / ad hoc helper (this test runner) fails the pinned requirement ...
        XCTAssertNotEqual(PeerPolicy.check(pid: getpid(), requirement: try XCTUnwrap(policy.requirement)),
                          errSecSuccess)
        // ... while a validly signed binary satisfies a requirement it meets.
        let sleeper = Process()
        sleeper.executableURL = URL(fileURLWithPath: "/bin/sleep")
        sleeper.arguments = ["5"]
        try sleeper.run()
        defer { sleeper.terminate() }
        XCTAssertEqual(PeerPolicy.check(pid: sleeper.processIdentifier, requirement: "anchor apple"),
                       errSecSuccess)
        XCTAssertNotEqual(PeerPolicy.check(pid: sleeper.processIdentifier,
                                           requirement: try XCTUnwrap(policy.requirement)),
                          errSecSuccess)
    }

    func testObservationServerUsesThePrivateListener() throws {
        let path = (temporarySocketPath() as NSString).deletingLastPathComponent + "/observation.sock"
        let server = ObservationServer(socketPath: path, policy: testRunnerPolicy)
        server.start()
        let listener = try XCTUnwrap(server.currentListener)
        let reply = try roundTrip(path, request: #"{"command":"nope"}"# + "\n")
        let json = try JSONSerialization.jsonObject(with: reply) as? [String: String]
        XCTAssertEqual(json?["error"], "unknown command: nope")
        server.stop()
        XCTAssertEqual(listener.exited.wait(timeout: .now() + 1), .success)
        XCTAssertThrowsError(try roundTrip(path, request: #"{"command":"snapshot"}"#))
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

        // The daemon's request (scale 0.5): half the native pixels, not half
        // the points. On a 3840x2160 panel at 2x this is 1920x1080, not 960x540.
        let half = try await service.captureFrame(CaptureRequest()).get()
        print("live capture: \(half.width)x\(half.height), \(half.jpeg.count) bytes at scale 0.5 q\(half.quality)")
        XCTAssertEqual(Double(half.height), Double(expectedHeight) * CaptureRequest.defaultScale, accuracy: 2,
                       "scale-0.5 frame height \(half.height) must be half the native \(expectedHeight)")
        XCTAssertLessThanOrEqual(half.jpeg.count, EncodedFrame.maxBytes)
    }
}

// MARK: - Test doubles

private actor Counter {
    private(set) var value = 0
    func increment() { value += 1 }

    /// Synchronous access for non-async test code (makeCapture factories).
    private let box = LockedInt()
    nonisolated func incrementNow() -> Int { box.increment() }
    nonisolated var valueNow: Int { box.value }
}

private final class LockedInt: @unchecked Sendable {
    private let lock = NSLock()
    private var current = 0
    func increment() -> Int { lock.lock(); defer { lock.unlock() }; current += 1; return current }
    var value: Int { lock.lock(); defer { lock.unlock() }; return current }
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
