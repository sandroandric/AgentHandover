import Foundation
import SwiftUI

/// Overall service health derived from daemon + worker status.
enum ServiceHealth: String {
    case healthy   // Both running, no issues
    case warning   // Running but with issues (permissions, stale heartbeat)
    case down      // One or both services not running
    case stopped   // User intentionally stopped services

    var color: Color {
        switch self {
        case .healthy: return .green
        case .warning: return .yellow
        case .down:    return .red
        case .stopped: return .gray
        }
    }

    var label: String {
        switch self {
        case .healthy: return "Healthy"
        case .warning: return "Warning"
        case .down:    return "Down"
        case .stopped: return "Stopped"
        }
    }
}

/// Decoded daemon-status.json
struct DaemonStatusFile: Codable {
    let pid: UInt32
    let version: String
    let started_at: String
    let heartbeat: String
    let events_today: UInt64
    let permissions_ok: Bool
    let accessibility_permitted: Bool
    let screen_recording_permitted: Bool
    let db_path: String
    let uptime_seconds: UInt64
    let last_extension_message: String?
}

/// Decoded extension-heartbeat.json
struct ExtensionHeartbeatFile: Codable {
    let pid: UInt32
    let last_message: String
    let messages_this_session: UInt64
    let session_started: String
}

/// Decoded worker-status.json
struct WorkerStatusFile: Codable {
    let pid: UInt32
    let version: String
    let started_at: String
    let heartbeat: String
    let events_processed_today: UInt64
    let sops_generated: UInt64
    let last_pipeline_duration_ms: UInt64?
    let consecutive_errors: UInt32
    let vlm_available: Bool
    let sop_inducer_available: Bool
    let vlm_queue_pending: UInt64?
    let vlm_jobs_today: UInt64?
    let vlm_dropped_today: UInt64?
    let vlm_mode: String?
    let vlm_provider: String?
}

@MainActor
final class AppState: ObservableObject {
    // MARK: - Published State

    @Published var daemonStatus: DaemonStatusFile?
    @Published var workerStatus: WorkerStatusFile?
    @Published var extensionHeartbeat: ExtensionHeartbeatFile?
    @Published var daemonRunning = false
    @Published var workerRunning = false
    @Published var extensionConnected = false
    @Published var health: ServiceHealth = .down
    @Published var userStopped = false

    // Permissions
    @Published var accessibilityGranted = false
    @Published var screenRecordingGranted = false

    // VLM
    @Published var vlmAvailable = false
    @Published var vlmMode: String = "local"
    @Published var vlmProvider: String?

    // MARK: - Computed

    var menuBarIcon: String {
        switch health {
        case .healthy: return "eye.circle.fill"
        case .warning: return "exclamationmark.triangle.fill"
        case .down:    return "eye.slash.circle.fill"
        case .stopped: return "pause.circle.fill"
        }
    }

    var eventsToday: UInt64 {
        daemonStatus?.events_today ?? 0
    }

    var sopsGenerated: UInt64 {
        workerStatus?.sops_generated ?? 0
    }

    var daemonVersion: String {
        daemonStatus?.version ?? "unknown"
    }

    var workerVersion: String {
        workerStatus?.version ?? "unknown"
    }

    var vlmQueuePending: UInt64 {
        workerStatus?.vlm_queue_pending ?? 0
    }

    /// True when VLM queue has significant backlog (>50 pending jobs).
    var vlmBacklogged: Bool {
        vlmQueuePending > 50
    }

    // MARK: - Polling

    private var pollTimer: Timer?
    private let statusDir: URL

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        self.statusDir = home
            .appendingPathComponent("Library/Application Support/oc-apprentice")

        startPolling()
    }

    func startPolling() {
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.refreshStatus()
            }
        }
        refreshStatus()
    }

    func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
    }

    // MARK: - Status Reading

    func refreshStatus() {
        readDaemonStatus()
        readWorkerStatus()
        readExtensionHeartbeat()
        updateHealth()
        checkPermissions()
        updateVLMStatus()
    }

    private func readDaemonStatus() {
        let path = statusDir.appendingPathComponent("daemon-status.json")
        guard let data = try? Data(contentsOf: path),
              let status = try? JSONDecoder().decode(DaemonStatusFile.self, from: data) else {
            daemonStatus = nil
            daemonRunning = false
            return
        }

        daemonStatus = status
        daemonRunning = isHeartbeatFresh(status.heartbeat) && isProcessRunning(pid: status.pid)
    }

    private func readWorkerStatus() {
        let path = statusDir.appendingPathComponent("worker-status.json")
        guard let data = try? Data(contentsOf: path),
              let status = try? JSONDecoder().decode(WorkerStatusFile.self, from: data) else {
            workerStatus = nil
            workerRunning = false
            return
        }

        workerStatus = status
        workerRunning = isHeartbeatFresh(status.heartbeat) && isProcessRunning(pid: status.pid)
    }

    private func readExtensionHeartbeat() {
        let path = statusDir.appendingPathComponent("extension-heartbeat.json")
        guard let data = try? Data(contentsOf: path),
              let heartbeat = try? JSONDecoder().decode(ExtensionHeartbeatFile.self, from: data) else {
            extensionHeartbeat = nil
            // Fall back to daemon's last_extension_message
            if let daemonExt = daemonStatus?.last_extension_message {
                extensionConnected = isHeartbeatFresh(daemonExt)
            } else {
                extensionConnected = false
            }
            return
        }

        extensionHeartbeat = heartbeat
        extensionConnected = isHeartbeatFresh(heartbeat.last_message)
    }

    private func updateVLMStatus() {
        vlmAvailable = workerStatus?.vlm_available ?? false
        // Read vlm mode from config.toml
        readVLMConfig()
    }

    private func readVLMConfig() {
        let configPath = statusDir.appendingPathComponent("config.toml")
        guard let content = try? String(contentsOf: configPath, encoding: .utf8) else {
            return
        }
        // Simple TOML parsing for vlm section
        var inVlmSection = false
        for line in content.components(separatedBy: "\n") {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("[") {
                inVlmSection = trimmed == "[vlm]"
                continue
            }
            if inVlmSection {
                if trimmed.hasPrefix("mode") {
                    let value = trimmed.components(separatedBy: "=").last?
                        .trimmingCharacters(in: .whitespaces)
                        .trimmingCharacters(in: CharacterSet(charactersIn: "\"")) ?? "local"
                    vlmMode = value
                }
                if trimmed.hasPrefix("provider") {
                    let value = trimmed.components(separatedBy: "=").last?
                        .trimmingCharacters(in: .whitespaces)
                        .trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                    vlmProvider = value
                }
            }
        }
    }

    private func updateHealth() {
        if userStopped {
            health = .stopped
            return
        }

        if !daemonRunning && !workerRunning {
            health = .down
            return
        }

        let hasWarnings = !(daemonStatus?.permissions_ok ?? true)
            || (workerStatus?.consecutive_errors ?? 0) > 0
            || !daemonRunning || !workerRunning
            || vlmBacklogged

        health = hasWarnings ? .warning : .healthy
    }

    // MARK: - Permissions

    private func checkPermissions() {
        accessibilityGranted = PermissionChecker.isAccessibilityGranted()
        screenRecordingGranted = PermissionChecker.isScreenRecordingGranted()
    }

    // MARK: - Helpers

    /// Check if a heartbeat timestamp is within the last 2 minutes.
    private func isHeartbeatFresh(_ isoString: String) -> Bool {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        guard let date = formatter.date(from: isoString) else {
            // Try without fractional seconds
            formatter.formatOptions = [.withInternetDateTime]
            guard let date = formatter.date(from: isoString) else {
                return false
            }
            return Date().timeIntervalSince(date) < 120
        }
        return Date().timeIntervalSince(date) < 120
    }

    /// Check if a process with given PID is running.
    private func isProcessRunning(pid: UInt32) -> Bool {
        kill(Int32(pid), 0) == 0
    }
}
