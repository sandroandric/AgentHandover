import Foundation

/// Unix socket server that serves screenshot capture requests to the daemon.
///
/// Protocol:
/// - Client sends: `{"command": "capture"}\n`
/// - Server responds: 8 bytes header (width u32 LE + height u32 LE) + raw BGRA pixel data
/// - Or error: `{"error": "..."}\n`
final class ScreenCaptureServer {
    private let captureService = ScreenCaptureService()
    private var serverFD: Int32 = -1
    private var isRunning = false
    private let socketPath: String

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        socketPath = home.appendingPathComponent(
            "Library/Application Support/agenthandover/capture.sock").path
    }

    /// Start the server. Runs accept loop in background.
    func start() {
        guard !isRunning else { return }

        // Ensure parent directory exists
        let dir = (socketPath as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true, attributes: nil)

        // Remove stale socket file if it exists
        unlink(socketPath)

        // Create Unix domain socket
        serverFD = socket(AF_UNIX, SOCK_STREAM, 0)
        guard serverFD >= 0 else {
            NSLog("ScreenCaptureServer: failed to create socket: errno=%d", errno)
            return
        }

        // Bind
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else {
            NSLog("ScreenCaptureServer: socket path too long (%d bytes)", pathBytes.count)
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
            NSLog("ScreenCaptureServer: bind failed: errno=%d", errno)
            Darwin.close(serverFD)
            serverFD = -1
            return
        }

        // Listen
        guard Darwin.listen(serverFD, 5) == 0 else {
            NSLog("ScreenCaptureServer: listen failed: errno=%d", errno)
            Darwin.close(serverFD)
            serverFD = -1
            return
        }

        isRunning = true
        NSLog("ScreenCaptureServer: listening on %@", socketPath)

        // Accept loop on background queue
        let fd = serverFD
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            self?.acceptLoop(serverFD: fd)
        }
    }

    /// Stop the server and clean up.
    func stop() {
        guard isRunning else { return }
        isRunning = false

        if serverFD >= 0 {
            Darwin.close(serverFD)
            serverFD = -1
        }

        unlink(socketPath)
        NSLog("ScreenCaptureServer: stopped")
    }

    // MARK: - Accept Loop

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
                // accept returns -1 when serverFD is closed during stop()
                if isRunning {
                    NSLog("ScreenCaptureServer: accept failed: errno=%d", errno)
                }
                break
            }

            // Set 10 second read/write timeout
            var timeout = timeval(tv_sec: 10, tv_usec: 0)
            setsockopt(clientFD, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                       socklen_t(MemoryLayout<timeval>.size))
            setsockopt(clientFD, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                       socklen_t(MemoryLayout<timeval>.size))

            handleConnection(clientFD)
            Darwin.close(clientFD)
        }
    }

    // MARK: - Connection Handler

    private func handleConnection(_ fd: Int32) {
        // Read command (up to 4096 bytes)
        var buffer = [UInt8](repeating: 0, count: 4096)
        let bytesRead = recv(fd, &buffer, buffer.count - 1, 0)
        guard bytesRead > 0 else {
            sendError(fd, message: "empty request")
            return
        }

        buffer[bytesRead] = 0
        guard let requestStr = String(bytes: buffer[0..<bytesRead], encoding: .utf8),
              let requestData = requestStr.trimmingCharacters(in: .whitespacesAndNewlines)
                  .data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: requestData) as? [String: Any],
              let command = json["command"] as? String else {
            sendError(fd, message: "invalid JSON")
            return
        }

        guard command == "capture" else {
            sendError(fd, message: "unknown command: \(command)")
            return
        }

        // Run async capture, bridging to sync with semaphore
        var result: (Int, Int, Data)?
        let semaphore = DispatchSemaphore(value: 0)
        Task {
            result = await captureService.captureMainDisplay()
            semaphore.signal()
        }
        semaphore.wait()

        guard let (width, height, pixelData) = result else {
            sendError(fd, message: "capture failed")
            return
        }

        // Write header: width (u32 LE) + height (u32 LE)
        var widthLE = UInt32(width).littleEndian
        var heightLE = UInt32(height).littleEndian

        let headerSent = withUnsafePointer(to: &widthLE) { wPtr in
            Darwin.send(fd, wPtr, 4, 0)
        }
        guard headerSent == 4 else { return }

        let heightSent = withUnsafePointer(to: &heightLE) { hPtr in
            Darwin.send(fd, hPtr, 4, 0)
        }
        guard heightSent == 4 else { return }

        // Write pixel data in chunks (large buffers may need multiple send calls)
        pixelData.withUnsafeBytes { rawPtr in
            guard let baseAddress = rawPtr.baseAddress else { return }
            var totalSent = 0
            let totalBytes = pixelData.count
            while totalSent < totalBytes {
                let remaining = totalBytes - totalSent
                let chunkSize = min(remaining, 1024 * 1024) // 1 MB chunks
                let sent = Darwin.send(
                    fd, baseAddress.advanced(by: totalSent), chunkSize, 0)
                if sent <= 0 { break }
                totalSent += sent
            }
        }
    }

    // MARK: - Error Response

    private func sendError(_ fd: Int32, message: String) {
        let errorJSON = "{\"error\": \"\(message)\"}\n"
        _ = errorJSON.withCString { ptr in
            Darwin.send(fd, ptr, strlen(ptr), 0)
        }
    }
}
