import CryptoKit
import Foundation

/// Unix socket server that serves screenshot capture requests to the daemon.
///
/// Protocol:
/// - Client sends: `{"command":"capture","scale":0.5,"format":"jpeg","quality":70}\n`
///   (`scale`, `format`, `quality` optional; defaults 0.5, jpeg, 70)
/// - Server responds: magic `AHJ1` + width u32 LE + height u32 LE +
///   byte length u32 LE + JPEG bytes (the frame after downscaling)
/// - Or error: `{"error": "..."}\n`
///
/// Memory thumbnail key (2.0): `{"command":"thumbnail_key"}\n` is answered
/// with byte length u32 LE (32) + the raw key (`MemoryThumbnailKey`), or the
/// error line. Like every request it reaches only peers `policy` accepts:
/// the daemon at its installed path and, in a signed build, signed by this
/// team as `com.agenthandover.observer`.
///
/// Each connection is handled on a concurrent queue, so one hung
/// ScreenCaptureKit call never blocks the next request. After
/// `recycleAfterTimeouts` consecutive timeouts the capture path is replaced
/// with a fresh `ScreenCaptureService`, and `stallStatus` tells the menu bar.
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
    /// Consecutive timeouts (or fail-fast refusals) before the capture path
    /// is recycled.
    static let recycleAfterTimeouts = 3
    /// Recycle at most this often, so a permanently wedged ScreenCaptureKit
    /// cannot make us spawn hung tasks without bound.
    static let defaultMinRecycleInterval: TimeInterval = 60
    /// Posted (on the main queue) whenever `stallStatus` changes.
    static let stallStatusChanged = Notification.Name("AgentHandoverCaptureStallStatusChanged")

    private let socketPath: String
    private let captureTimeout: TimeInterval
    private let minRecycleInterval: TimeInterval
    private let makeCapture: () -> CaptureFunction
    private let policy: PeerPolicy
    private let thumbnailKey: () -> SymmetricKey?
    private let handlerQueue = DispatchQueue(
        label: "com.agenthandover.capture-connections", qos: .userInitiated, attributes: .concurrent)
    private var listener: UnixSocketListener?

    private let lock = NSLock()
    private var capture: CaptureFunction
    private var generation = 0
    private var pendingCaptures = 0
    private var consecutiveTimeouts = 0
    private var lastRecycle: Date?
    private var status: String?

    /// - Parameter makeCapture: builds a capture path; called once at init
    ///   and again on every recycle. Defaults to a `ScreenCaptureService`.
    init(
        socketPath: String = SocketDirectory.socketPath("capture.sock"),
        captureTimeout: TimeInterval = ScreenCaptureServer.defaultCaptureTimeout,
        minRecycleInterval: TimeInterval = ScreenCaptureServer.defaultMinRecycleInterval,
        policy: PeerPolicy = .production(
            paths: [PeerPolicy.daemonPath], identifiers: [PeerPolicy.daemonIdentifier]),
        thumbnailKey: @escaping () -> SymmetricKey? = MemoryThumbnailKey.appKey,
        makeCapture: @escaping () -> CaptureFunction = ScreenCaptureServer.serviceCapture
    ) {
        self.socketPath = socketPath
        self.captureTimeout = captureTimeout
        self.minRecycleInterval = minRecycleInterval
        self.policy = policy
        self.thumbnailKey = thumbnailKey
        self.makeCapture = makeCapture
        self.capture = makeCapture()
    }

    /// A single capture function for every generation (tests).
    convenience init(
        socketPath: String,
        captureTimeout: TimeInterval = ScreenCaptureServer.defaultCaptureTimeout,
        policy: PeerPolicy,
        thumbnailKey: @escaping () -> SymmetricKey? = MemoryThumbnailKey.appKey,
        capture: @escaping CaptureFunction
    ) {
        self.init(socketPath: socketPath, captureTimeout: captureTimeout, policy: policy,
                  thumbnailKey: thumbnailKey, makeCapture: { capture })
    }

    static func serviceCapture() -> CaptureFunction {
        let service = ScreenCaptureService()
        return { await service.captureFrame($0) }
    }

    /// One line for the menu bar while capture is stalled, else nil.
    var stallStatus: String? {
        lock.lock(); defer { lock.unlock() }
        return status
    }

    /// How many times the capture path has been replaced.
    var recycleCount: Int {
        lock.lock(); defer { lock.unlock() }
        return generation
    }

    /// Start the server. Runs the accept loop in the background.
    func start() {
        guard listener == nil else { return }
        let listener = UnixSocketListener(
            name: "ScreenCaptureServer", path: socketPath, policy: policy,
            handlerQueue: handlerQueue
        ) { [weak self] fd in self?.handleConnection(fd) }
        if listener.start() { self.listener = listener }
    }

    /// Stop the server. The accept loop exits within one poll interval and
    /// closes the listening fd itself; connections in flight finish.
    func stop() {
        guard let listener else { return }
        self.listener = nil
        listener.stop()
        NSLog("ScreenCaptureServer: stopped")
    }

    /// The running listener (tests wait on its `exited` semaphore).
    var currentListener: UnixSocketListener? { listener }

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

        if json["command"] as? String == "thumbnail_key" {
            guard let key = thumbnailKey() else {
                sendError(fd, message: "thumbnail key unavailable")
                return
            }
            sendAll(fd, Self.encodeKeyResponse(key))
            return
        }

        let result = CaptureRequest.parse(json).flatMap { performCapture($0) }
        switch result {
        case .success(let frame):
            sendAll(fd, Self.encodeFrameResponse(frame))
        case .failure(let error):
            if case .timeout = error {
                NSLog("ScreenCaptureServer: capture did not return within %.0f s", captureTimeout)
            }
            sendError(fd, message: error.message)
        }
    }

    /// Run one capture, bridging async to this blocking thread with a bounded wait.
    func performCapture(_ request: CaptureRequest) -> Result<EncodedFrame, CaptureError> {
        lock.lock()
        if pendingCaptures >= Self.maxPendingCaptures {
            let pending = pendingCaptures
            lock.unlock()
            recordOutcome(timedOut: true)
            return .failure(.stalled(pending: pending))
        }
        pendingCaptures += 1
        let capture = self.capture
        let myGeneration = generation
        lock.unlock()

        let box = ResultBox()
        let semaphore = DispatchSemaphore(value: 0)
        Task { [weak self] in
            let result = await capture(request)
            box.set(result)
            self?.captureFinished(generation: myGeneration)
            semaphore.signal()
        }

        if semaphore.wait(timeout: .now() + captureTimeout) == .timedOut {
            recordOutcome(timedOut: true)
            return .failure(.timeout)
        }
        recordOutcome(timedOut: false)
        return box.get() ?? .failure(.timeout)
    }

    /// A finished task frees its pending slot, unless the capture path has
    /// been recycled since it started (the new path starts with no slots used).
    private func captureFinished(generation finished: Int) {
        lock.lock()
        if finished == generation { pendingCaptures -= 1 }
        lock.unlock()
    }

    private func recordOutcome(timedOut: Bool) {
        lock.lock()
        let before = status
        if timedOut {
            consecutiveTimeouts += 1
            let now = Date()
            let recycleDue = lastRecycle.map { now.timeIntervalSince($0) >= minRecycleInterval } ?? true
            if consecutiveTimeouts >= Self.recycleAfterTimeouts && recycleDue {
                capture = makeCapture()
                generation += 1
                pendingCaptures = 0
                consecutiveTimeouts = 0
                lastRecycle = now
                status = "Screen capture stalled; restarted the capture path"
                NSLog("ScreenCaptureServer: %d captures timed out in a row; capture path recycled",
                      Self.recycleAfterTimeouts)
            }
        } else {
            consecutiveTimeouts = 0
            status = nil
        }
        let changed = before != status
        lock.unlock()
        if changed {
            DispatchQueue.main.async {
                NotificationCenter.default.post(name: Self.stallStatusChanged, object: self)
            }
        }
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

    static func encodeKeyResponse(_ key: SymmetricKey) -> Data {
        var data = Data()
        withUnsafeBytes(of: UInt32(key.bitCount / 8).littleEndian) { data.append(contentsOf: $0) }
        key.withUnsafeBytes { data.append(contentsOf: $0) }
        return data
    }

    static func encodeErrorResponse(_ message: String) -> Data {
        var data = (try? JSONSerialization.data(withJSONObject: ["error": message]))
            ?? Data(#"{"error":"unencodable error"}"#.utf8)
        data.append(UInt8(ascii: "\n"))
        return data
    }

    private func sendError(_ fd: Int32, message: String) {
        sendAll(fd, Self.encodeErrorResponse(message))
    }
}
