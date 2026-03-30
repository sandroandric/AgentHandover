import Foundation

/// Client for the AgentHandoverDaemon control socket.
///
/// Replaces file-based status polling with explicit RPC calls to the daemon's
/// Unix domain socket. The daemon remains the authority for capture state,
/// focus state, and Accessibility. Screen Recording is checked directly by
/// the app.
@MainActor
final class CaptureAgentClient {

    /// Path to the daemon's Unix domain socket.
    private var socketPath: String {
        Self.controlSocketPath()
    }

    /// Whether the daemon is reachable over its control socket.
    var isAvailable: Bool {
        Self.socketResponsive()
    }

    nonisolated static func socketAvailable() -> Bool {
        FileManager.default.fileExists(atPath: controlSocketPath())
    }

    /// Whether the daemon is accepting control RPCs.
    ///
    /// This is stronger than a raw file-exists check and avoids treating a
    /// stale socket path as a live daemon during onboarding.
    nonisolated static func socketResponsive() -> Bool {
        guard socketAvailable() else { return false }
        return sendSync(
            socketPath: controlSocketPath(),
            command: ["command": "get_status"]
        ) != nil
    }

    nonisolated private static func controlSocketPath() -> String {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover/control.sock")
            .path
    }

    // MARK: - Commands

    /// Get full daemon status: permissions, capture state, focus session.
    func getStatus() async -> DaemonControlStatus? {
        guard let response = await send(command: ["command": "get_status"]) else {
            return nil
        }
        return DaemonControlStatus(
            pid: response["pid"] as? Int ?? 0,
            version: response["version"] as? String ?? "unknown",
            uptimeSeconds: response["uptime_seconds"] as? Int ?? 0,
            accessibilityPermitted: response["accessibility_permitted"] as? Bool ?? false,
            screenRecordingPermitted: response["screen_recording_permitted"] as? Bool ?? false,
            captureActive: response["capture_active"] as? Bool ?? false,
            focusSession: response["focus_session"] as? [String: Any],
            eventsToday: response["events_today"] as? Int ?? 0
        )
    }

    /// Request accessibility permission (triggers macOS prompt).
    func requestAccessibility() async -> Bool {
        let response = await send(command: ["command": "request_accessibility"])
        return response?["accessibility_permitted"] as? Bool ?? false
    }

    /// Start a focus recording session.
    func startFocus(title: String, sessionId: String) async -> Bool {
        let response = await send(command: [
            "command": "start_focus",
            "title": title,
            "session_id": sessionId,
        ])
        return response?["ok"] as? Bool ?? false
    }

    /// Stop the current focus recording session.
    func stopFocus() async -> Bool {
        let response = await send(command: ["command": "stop_focus"])
        return response?["ok"] as? Bool ?? false
    }

    /// Pause passive capture.
    func pauseCapture() async -> Bool {
        let response = await send(command: ["command": "pause_capture"])
        return response?["ok"] as? Bool ?? false
    }

    /// Resume passive capture.
    func resumeCapture() async -> Bool {
        let response = await send(command: ["command": "resume_capture"])
        return response?["ok"] as? Bool ?? false
    }

    // MARK: - Socket Communication

    /// Send a JSON command to the daemon and receive the JSON response.
    private func send(command: [String: Any]) async -> [String: Any]? {
        // Run socket I/O off the main actor
        let path = socketPath
        return await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let result = Self.sendSync(socketPath: path, command: command)
                continuation.resume(returning: result)
            }
        }
    }

    /// Synchronous socket send/receive (runs on background thread).
    nonisolated private static func sendSync(
        socketPath: String,
        command: [String: Any]
    ) -> [String: Any]? {
        // Create Unix domain socket
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { return nil }
        defer { close(fd) }

        // Connect
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = socketPath.utf8CString
        guard pathBytes.count <= MemoryLayout.size(ofValue: addr.sun_path) else { return nil }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: pathBytes.count) { dest in
                for (i, byte) in pathBytes.enumerated() {
                    dest[i] = byte
                }
            }
        }

        let connectResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.connect(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard connectResult == 0 else { return nil }

        // Set timeout (5 seconds)
        var timeout = timeval(tv_sec: 5, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

        // Send command as JSON + newline
        guard let jsonData = try? JSONSerialization.data(withJSONObject: command),
              var jsonString = String(data: jsonData, encoding: .utf8) else {
            return nil
        }
        jsonString += "\n"
        let sent = jsonString.withCString { ptr in
            Darwin.send(fd, ptr, strlen(ptr), 0)
        }
        guard sent > 0 else { return nil }

        // Receive response
        var buffer = [UInt8](repeating: 0, count: 65536)
        let received = recv(fd, &buffer, buffer.count - 1, 0)
        guard received > 0 else { return nil }

        buffer[received] = 0
        guard let responseStr = String(bytes: buffer[0..<received], encoding: .utf8),
              let responseData = responseStr.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any] else {
            return nil
        }

        return json
    }
}

/// Parsed daemon status from the control socket.
struct DaemonControlStatus {
    let pid: Int
    let version: String
    let uptimeSeconds: Int
    let accessibilityPermitted: Bool
    let screenRecordingPermitted: Bool
    let captureActive: Bool
    let focusSession: [String: Any]?
    let eventsToday: Int
}
