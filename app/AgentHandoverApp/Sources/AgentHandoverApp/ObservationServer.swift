import Foundation

/// Unix socket server that serves observation metadata to the daemon.
///
/// Protocol:
/// - Client sends: `{"command":"snapshot"}\n`, or
///   `{"command":"focused_element","capture_values":<bool>}\n`
/// - Server responds: the JSON snapshot, or `{"focused_element":...}`, + newline
///
/// Peers are the daemon and the CLI (`agenthandover doctor`); see
/// `PeerPolicy`. Requests are answered one at a time on a serial queue.
final class ObservationServer {
    private let observationService = ObservationService(sleepMonitor: .started())
    private let socketPath: String
    private let policy: PeerPolicy
    private let handlerQueue = DispatchQueue(
        label: "com.agenthandover.observation-connections", qos: .userInitiated)
    private var listener: UnixSocketListener?

    init(
        socketPath: String = SocketDirectory.socketPath("observation.sock"),
        policy: PeerPolicy = .production(
            paths: [PeerPolicy.daemonPath, PeerPolicy.cliPath],
            identifiers: [PeerPolicy.daemonIdentifier, PeerPolicy.cliIdentifier])
    ) {
        self.socketPath = socketPath
        self.policy = policy
    }

    func start() {
        guard listener == nil else { return }
        let listener = UnixSocketListener(
            name: "ObservationServer", path: socketPath, policy: policy,
            handlerQueue: handlerQueue
        ) { [weak self] fd in self?.handleConnection(fd) }
        if listener.start() { self.listener = listener }
    }

    func stop() {
        guard let listener else { return }
        self.listener = nil
        listener.stop()
        NSLog("ObservationServer: stopped")
    }

    /// The running listener (tests wait on its `exited` semaphore).
    var currentListener: UnixSocketListener? { listener }

    private func handleConnection(_ fd: Int32) {
        var buffer = [UInt8](repeating: 0, count: 4096)
        let bytesRead = recv(fd, &buffer, buffer.count, 0)
        guard bytesRead > 0 else {
            sendError(fd, message: "empty request")
            return
        }

        guard let requestStr = String(bytes: buffer[0..<bytesRead], encoding: .utf8),
              let requestData = requestStr
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: requestData) as? [String: Any],
              let command = json["command"] as? String else {
            sendError(fd, message: "invalid JSON")
            return
        }

        guard var data = Self.reply(to: command, request: json, service: observationService)
        else {
            sendError(fd, message: "unknown command: \(command)")
            return
        }
        data.append(UInt8(ascii: "\n"))
        sendAll(fd, data)
    }

    /// The encoded reply to `command`, nil for an unknown command.
    /// `focused_element` reads `AXValue` only when the request's
    /// `capture_values` is true (`[privacy] capture_input_values`).
    static func reply(
        to command: String, request: [String: Any], service: ObservationService
    ) -> Data? {
        let encoder = JSONEncoder()
        switch command {
        case "snapshot":
            return try? encoder.encode(service.snapshot())
        case "focused_element":
            let captureValues = request["capture_values"] as? Bool ?? false
            return try? encoder.encode(FocusedElementReply(
                focusedElement: service.focusedElement(captureValues: captureValues)))
        default:
            return nil
        }
    }

    private func sendError(_ fd: Int32, message: String) {
        sendAll(fd, ScreenCaptureServer.encodeErrorResponse(message))
    }
}

/// Reply to `{"command":"focused_element"}`; `focused_element` is JSON null
/// when there is none.
struct FocusedElementReply: Encodable {
    let focusedElement: ObservationFocusedElement?

    enum CodingKeys: String, CodingKey {
        case focusedElement = "focused_element"
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(focusedElement, forKey: .focusedElement)
    }
}
