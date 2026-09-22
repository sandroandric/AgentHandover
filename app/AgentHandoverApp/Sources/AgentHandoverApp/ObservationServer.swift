import Foundation

/// Unix socket server that serves observation metadata to the daemon.
///
/// Protocol:
/// - Client sends: `{"command":"snapshot"}\n`
/// - Server responds: JSON snapshot + newline
///
/// Peers are the daemon and the CLI (`agenthandover doctor`); see
/// `PeerPolicy`. Requests are answered one at a time on a serial queue.
final class ObservationServer {
    private let observationService = ObservationService()
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

        guard command == "snapshot" else {
            sendError(fd, message: "unknown command: \(command)")
            return
        }

        guard var data = try? JSONEncoder().encode(observationService.snapshot()) else {
            sendError(fd, message: "encode failed")
            return
        }
        data.append(UInt8(ascii: "\n"))
        sendAll(fd, data)
    }

    private func sendError(_ fd: Int32, message: String) {
        sendAll(fd, ScreenCaptureServer.encodeErrorResponse(message))
    }
}
