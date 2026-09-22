import Foundation

/// Unix socket server that serves screenshot capture requests to the daemon.
///
/// Protocol:
/// - Client sends: `{"command":"capture","scale":0.5,"format":"jpeg","quality":70}\n`
///   (`scale`, `format`, `quality` optional; defaults 0.5, jpeg, 70)
/// - Server responds: magic `AHJ1` + width u32 LE + height u32 LE +
///   byte length u32 LE + JPEG bytes (the frame after downscaling)
/// - Or error: `{"error": "..."}\n`
final class ScreenCaptureServer {
    typealias CaptureFunction = (CaptureRequest) async -> Result<EncodedFrame, CaptureError>

    static let frameMagic: [UInt8] = Array("AHJ1".utf8)
    /// A capture that has not returned after this long is answered with
    /// `capture timeout`; the daemon's own read timeout (10 s) is longer.
    static let defaultCaptureTimeout: TimeInterval = 8
    /// Captures that timed out keep running in the background (ScreenCaptureKit
    /// calls cannot be cancelled). Once this many are still pending, new
    /// requests fail fast instead of piling up more hung tasks.
    static let maxPendingCaptures = 2

    private let socketPath: String
    private let captureTimeout: TimeInterval
    private let capture: CaptureFunction
    private var listener: Listener?

    private let pendingLock = NSLock()
    private var pendingCaptures = 0

    init(
        socketPath: String = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover/capture.sock").path,
        captureTimeout: TimeInterval = ScreenCaptureServer.defaultCaptureTimeout,
        capture: CaptureFunction? = nil
    ) {
        self.socketPath = socketPath
        self.captureTimeout = captureTimeout
        if let capture {
            self.capture = capture
        } else {
            let service = ScreenCaptureService()
            self.capture = { await service.captureFrame($0) }
        }
    }

    /// Start the server. Runs the accept loop in the background.
    func start() {
        guard listener == nil else { return }

        let dir = (socketPath as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true, attributes: nil)

        // Remove stale socket file if it exists
        unlink(socketPath)

        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            NSLog("ScreenCaptureServer: failed to create socket: errno=%d", errno)
            return
        }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            NSLog("ScreenCaptureServer: socket path too long (%d bytes)", pathBytes.count)
            Darwin.close(fd)
            return
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                for (i, byte) in pathBytes.enumerated() {
                    dest[i] = byte
                }
            }
        }

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.bind(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bindResult == 0 else {
            NSLog("ScreenCaptureServer: bind failed: errno=%d", errno)
            Darwin.close(fd)
            return
        }

        guard Darwin.listen(fd, 5) == 0 else {
            NSLog("ScreenCaptureServer: listen failed: errno=%d", errno)
            Darwin.close(fd)
            return
        }

        let listener = Listener(fd: fd)
        self.listener = listener
        NSLog("ScreenCaptureServer: listening on %@", socketPath)

        DispatchQueue.global(qos: .userInitiated).async { [self] in
            acceptLoop(listener)
        }
    }

    /// Stop the server. The accept loop notices within one poll interval and
    /// closes the listening fd itself after it has exited, so the fd number is
    /// never closed (and possibly reused) while the loop still uses it.
    func stop() {
        guard let listener else { return }
        self.listener = nil
        listener.requestStop()
        unlink(socketPath)
        NSLog("ScreenCaptureServer: stopped")
    }

    /// The running listener (tests wait on its `exited` semaphore).
    var currentListener: Listener? { listener }

    // MARK: - Accept Loop

    final class Listener {
        let fd: Int32
        let exited = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var stopRequested = false

        init(fd: Int32) { self.fd = fd }

        var isStopRequested: Bool {
            lock.lock(); defer { lock.unlock() }
            return stopRequested
        }

        func requestStop() {
            lock.lock(); stopRequested = true; lock.unlock()
        }
    }

    /// Poll interval of the accept loop; bounds how long `stop()` takes to land.
    static let acceptPollMilliseconds: Int32 = 200

    private func acceptLoop(_ listener: Listener) {
        defer {
            Darwin.close(listener.fd)
            listener.exited.signal()
        }
        while !listener.isStopRequested {
            var pfd = pollfd(fd: listener.fd, events: Int16(POLLIN), revents: 0)
            let ready = poll(&pfd, 1, Self.acceptPollMilliseconds)
            if ready < 0 {
                if errno == EINTR { continue }
                NSLog("ScreenCaptureServer: poll failed: errno=%d", errno)
                return
            }
            if ready == 0 || listener.isStopRequested { continue }

            let clientFD = accept(listener.fd, nil, nil)
            guard clientFD >= 0 else {
                if errno == EINTR || errno == EAGAIN || errno == ECONNABORTED { continue }
                NSLog("ScreenCaptureServer: accept failed: errno=%d", errno)
                return
            }

            var timeout = timeval(tv_sec: 10, tv_usec: 0)
            setsockopt(clientFD, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                       socklen_t(MemoryLayout<timeval>.size))
            setsockopt(clientFD, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                       socklen_t(MemoryLayout<timeval>.size))
            // A daemon that hung up must not kill the app with SIGPIPE.
            var noSigPipe: Int32 = 1
            setsockopt(clientFD, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe,
                       socklen_t(MemoryLayout<Int32>.size))

            handleConnection(clientFD)
            Darwin.close(clientFD)
        }
    }

    // MARK: - Connection Handler

    private func handleConnection(_ fd: Int32) {
        var buffer = [UInt8](repeating: 0, count: 4096)
        let bytesRead = recv(fd, &buffer, buffer.count, 0)
        guard bytesRead > 0 else {
            sendError(fd, message: "empty request")
            return
        }

        guard let json = try? JSONSerialization.jsonObject(
            with: Data(buffer[0..<bytesRead])) as? [String: Any] else {
            sendError(fd, message: "invalid JSON")
            return
        }

        let result = CaptureRequest.parse(json).flatMap { performCapture($0) }
        switch result {
        case .success(let frame):
            _ = sendAll(fd, Self.encodeFrameResponse(frame))
        case .failure(let error):
            if case .timeout = error {
                NSLog("ScreenCaptureServer: capture did not return within %.0f s", captureTimeout)
            }
            sendError(fd, message: error.message)
        }
    }

    /// Run one capture, bridging async to this blocking thread with a bounded wait.
    func performCapture(_ request: CaptureRequest) -> Result<EncodedFrame, CaptureError> {
        pendingLock.lock()
        if pendingCaptures >= Self.maxPendingCaptures {
            let pending = pendingCaptures
            pendingLock.unlock()
            return .failure(.stalled(pending: pending))
        }
        pendingCaptures += 1
        pendingLock.unlock()

        let box = ResultBox()
        let semaphore = DispatchSemaphore(value: 0)
        let capture = self.capture
        Task { [weak self] in
            let result = await capture(request)
            box.set(result)
            self?.capturesFinished()
            semaphore.signal()
        }

        guard semaphore.wait(timeout: .now() + captureTimeout) == .success,
              let result = box.get() else {
            return .failure(.timeout)
        }
        return result
    }

    private func capturesFinished() {
        pendingLock.lock()
        pendingCaptures -= 1
        pendingLock.unlock()
    }

    private final class ResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Result<EncodedFrame, CaptureError>?
        func set(_ result: Result<EncodedFrame, CaptureError>) {
            lock.lock(); value = result; lock.unlock()
        }
        func get() -> Result<EncodedFrame, CaptureError>? {
            lock.lock(); defer { lock.unlock() }
            return value
        }
    }

    // MARK: - Wire encoding

    static func encodeFrameResponse(_ frame: EncodedFrame) -> Data {
        var data = Data(frameMagic)
        for value in [UInt32(frame.width), UInt32(frame.height), UInt32(frame.jpeg.count)] {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(frame.jpeg)
        return data
    }

    static func encodeErrorResponse(_ message: String) -> Data {
        var data = (try? JSONSerialization.data(withJSONObject: ["error": message]))
            ?? Data(#"{"error":"unencodable error"}"#.utf8)
        data.append(UInt8(ascii: "\n"))
        return data
    }

    private func sendError(_ fd: Int32, message: String) {
        _ = sendAll(fd, Self.encodeErrorResponse(message))
    }

    /// Write every byte of `data`, looping over partial sends.
    private func sendAll(_ fd: Int32, _ data: Data) -> Bool {
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return true }
            var sent = 0
            while sent < data.count {
                let n = Darwin.send(fd, base.advanced(by: sent), data.count - sent, 0)
                if n < 0 && errno == EINTR { continue }
                if n <= 0 { return false }
                sent += n
            }
            return true
        }
    }
}
