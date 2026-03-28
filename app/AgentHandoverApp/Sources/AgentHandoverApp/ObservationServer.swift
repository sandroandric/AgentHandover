import Foundation

/// Unix socket server that serves observation metadata to the daemon.
///
/// Protocol:
/// - Client sends: `{"command":"snapshot"}\n`
/// - Server responds: JSON snapshot + newline
final class ObservationServer {
    private let observationService = ObservationService()
    private var serverFD: Int32 = -1
    private var isRunning = false
    private let socketPath: String

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        socketPath = home
            .appendingPathComponent("Library/Application Support/agenthandover/observation.sock")
            .path
    }

    func start() {
        guard !isRunning else { return }

        let dir = (socketPath as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(
            atPath: dir,
            withIntermediateDirectories: true,
            attributes: nil
        )

        unlink(socketPath)

        serverFD = socket(AF_UNIX, SOCK_STREAM, 0)
        guard serverFD >= 0 else {
            NSLog("ObservationServer: failed to create socket: errno=%d", errno)
            return
        }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            NSLog("ObservationServer: socket path too long (%d bytes)", pathBytes.count)
            Darwin.close(serverFD)
            serverFD = -1
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
                Darwin.bind(serverFD, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard bindResult == 0 else {
            NSLog("ObservationServer: bind failed: errno=%d", errno)
            Darwin.close(serverFD)
            serverFD = -1
            return
        }

        guard Darwin.listen(serverFD, 5) == 0 else {
            NSLog("ObservationServer: listen failed: errno=%d", errno)
            Darwin.close(serverFD)
            serverFD = -1
            return
        }

        isRunning = true
        NSLog("ObservationServer: listening on %@", socketPath)

        let fd = serverFD
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.acceptLoop(serverFD: fd)
        }
    }

    func stop() {
        guard isRunning else { return }
        isRunning = false

        if serverFD >= 0 {
            Darwin.close(serverFD)
            serverFD = -1
        }

        unlink(socketPath)
        NSLog("ObservationServer: stopped")
    }

    private func acceptLoop(serverFD: Int32) {
        while isRunning {
            var clientAddr = sockaddr_un()
            var clientAddrLen = socklen_t(MemoryLayout<sockaddr_un>.size)

            let clientFD = withUnsafeMutablePointer(to: &clientAddr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                    accept(serverFD, sockPtr, &clientAddrLen)
                }
            }

            guard clientFD >= 0 else {
                if isRunning {
                    NSLog("ObservationServer: accept failed: errno=%d", errno)
                }
                break
            }

            var timeout = timeval(tv_sec: 5, tv_usec: 0)
            setsockopt(clientFD, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                       socklen_t(MemoryLayout<timeval>.size))
            setsockopt(clientFD, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                       socklen_t(MemoryLayout<timeval>.size))

            handleConnection(clientFD)
            Darwin.close(clientFD)
        }
    }

    private func handleConnection(_ fd: Int32) {
        var buffer = [UInt8](repeating: 0, count: 4096)
        let bytesRead = recv(fd, &buffer, buffer.count - 1, 0)
        guard bytesRead > 0 else {
            sendError(fd, message: "empty request")
            return
        }

        buffer[bytesRead] = 0
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

        let snapshot = observationService.snapshot()
        guard let data = try? JSONEncoder().encode(snapshot) else {
            sendError(fd, message: "encode failed")
            return
        }

        _ = data.withUnsafeBytes { rawPtr in
            guard let base = rawPtr.baseAddress else { return 0 }
            return Darwin.send(fd, base, rawPtr.count, 0)
        }
        _ = "\n".withCString { ptr in
            Darwin.send(fd, ptr, 1, 0)
        }
    }

    private func sendError(_ fd: Int32, message: String) {
        let errorJSON = "{\"error\":\"\(message)\"}\n"
        _ = errorJSON.withCString { ptr in
            Darwin.send(fd, ptr, strlen(ptr), 0)
        }
    }
}
