import Foundation
import SwiftUI

/// Overall service health derived from daemon + worker status.
enum ServiceHealth: String {
    case healthy   // Both running, no issues
    case warning   // Running but with issues (permissions, stale heartbeat)
    case down      // One or both services not running
    case stopped   // User intentionally stopped services
    case starting  // Started, waiting for both services to report

    var color: Color {
        switch self {
        case .healthy:  return .green
        case .warning:  return .yellow
        case .down:     return .red
        case .stopped:  return .gray
        case .starting: return .orange
        }
    }

    var label: String {
        switch self {
        case .healthy:  return "Healthy"
        case .warning:  return "Warning"
        case .down:     return "Down"
        case .stopped:  return "Stopped"
        case .starting: return "Starting"
        }
    }

    /// How long after a start the menu may say "Starting…" before a
    /// service that still has not reported counts as down.
    static let startGracePeriod: TimeInterval = 60

    /// Health from the service state. `startedAt` is when the user (or a
    /// launch) last started the services; until both report, or the grace
    /// period ends, that is `.starting`, never a red "Offline".
    static func derive(
        userStopped: Bool,
        startedAt: Date?,
        now: Date,
        daemonRunning: Bool,
        workerRunning: Bool,
        hasIssues: Bool
    ) -> ServiceHealth {
        if userStopped { return .stopped }
        if let startedAt, now.timeIntervalSince(startedAt) < startGracePeriod,
           !(daemonRunning && workerRunning) {
            return .starting
        }
        if !daemonRunning && !workerRunning { return .down }
        return hasIssues || !daemonRunning || !workerRunning ? .warning : .healthy
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

/// Decoded focus-session.json
struct FocusSessionSignalFile: Codable, Equatable {
    let session_id: String
    let title: String
    let started_at: String
    var status: String
}

/// A single question from focus-questions.json
struct FocusQuestion: Codable, Identifiable {
    let index: Int
    let question: String
    let category: String
    let context: String
    /// The model's guess: the answer field starts with it.
    let `default`: String
    /// The steps (0-based) the question is about; kept so the worker reads
    /// them back from the answered file.
    var step_indexes: [Int]? = nil

    var id: Int { index }
}

/// Decoded focus-questions.json (written by the worker's
/// `write_focus_questions`, answered here, read back by its
/// `parse_qa_result_from_file`).
struct FocusQuestionsFile: Codable {
    let session_id: String
    let slug: String
    let questions: [FocusQuestion]
    var status: String
    /// Question index -> the answer; a cleared field is left out.
    var answers: [String: String]?
    /// The session title as the user typed it.
    var title: String? = nil
    /// Question index -> `typed` or `accepted_guess` (the pre-filled
    /// guess submitted unchanged), one per entry of `answers`.
    var answer_sources: [String: String]? = nil

    enum CodingKeys: String, CodingKey {
        case session_id, slug, questions, status, answers, title, answer_sources
    }
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
    @Published var userStopped = UserDefaults.standard.bool(forKey: "observingPaused")
    /// Set when the app had to restart a crashed daemon (see DaemonSupervisor).
    @Published var daemonSupervisorStatus: String?
    /// Set while ScreenCaptureServer reports stalled captures.
    @Published var captureStallStatus: String?

    // Permissions — app-owned and checked directly by AgentHandover.app.
    @Published var accessibilityGranted = false
    @Published var screenRecordingGranted = false

    // VLM
    @Published var vlmAvailable = false
    @Published var vlmMode: String = "local"
    @Published var vlmProvider: String?

    // SOP Index
    @Published var sopDraftCount: Int = 0
    @Published var sopApprovedCount: Int = 0
    @Published var sopTotalCount: Int = 0
    @Published var sopAgentReadyCount: Int = 0

    // Focus Recording
    /// Mirrors focus-session.json, whoever wrote it (app or CLI).
    @Published private(set) var focusState: FocusRecordState = .idle

    // Focus Q&A
    @Published var focusQuestionsAvailable = false
    @Published var focusQuestionsSlug: String = ""

    // MARK: - Computed

    var menuBarIcon: String {
        switch health {
        case .healthy: return "binoculars.fill"        // Actively observing
        case .warning: return "binoculars"             // Watching but with issues
        case .down:
            if workerRunning && !daemonRunning {
                return "binoculars"                    // Processing but not watching
            }
            return "briefcase.fill"                    // Idle, not active
        case .stopped: return "briefcase.fill"         // Idle, not active
        case .starting: return "binoculars"            // Coming up
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
    /// Set by `servicesStarting()`, cleared once health leaves `.starting`.
    private var startedAt: Date?
    private var startupTimer: Timer?
    private var supervisorObserver: NSObjectProtocol?
    private var captureStallObserver: NSObjectProtocol?
    let statusDir: URL

    init() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        self.statusDir = home
            .appendingPathComponent("Library/Application Support/agenthandover")

        supervisorObserver = NotificationCenter.default.addObserver(
            forName: DaemonSupervisor.statusChanged, object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.daemonSupervisorStatus = DaemonSupervisor.shared.statusLine
            }
        }
        captureStallObserver = NotificationCenter.default.addObserver(
            forName: ScreenCaptureServer.stallStatusChanged, object: nil, queue: .main
        ) { [weak self] note in
            let line = (note.object as? ScreenCaptureServer)?.stallStatus
            Task { @MainActor [weak self] in
                self?.captureStallStatus = line
                self?.updateHealth()
            }
        }
        startPolling()
    }

    func startPolling() {
        pollTimer?.invalidate()
        // Poll every 30s in background — fast refresh happens on menu bar open
        pollTimer = Timer.scheduledTimer(withTimeInterval: 30.0, repeats: true) { [weak self] _ in
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
        readFocusSession()
        readFocusQuestions()
        readSOPIndex()
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

        let newRunning = isHeartbeatFresh(status.heartbeat) && isProcessRunning(pid: status.pid)
        // Only update @Published if values actually changed to avoid view redraws
        if daemonStatus?.heartbeat != status.heartbeat || daemonStatus?.events_today != status.events_today {
            daemonStatus = status
        }
        if daemonRunning != newRunning { daemonRunning = newRunning }
    }

    private func readWorkerStatus() {
        let path = statusDir.appendingPathComponent("worker-status.json")
        guard let data = try? Data(contentsOf: path),
              let status = try? JSONDecoder().decode(WorkerStatusFile.self, from: data) else {
            if workerStatus != nil { workerStatus = nil }
            if workerRunning { workerRunning = false }
            return
        }

        let newRunning = isHeartbeatFresh(status.heartbeat) && isProcessRunning(pid: status.pid)
        if workerStatus?.heartbeat != status.heartbeat || workerStatus?.events_processed_today != status.events_processed_today {
            workerStatus = status
        }
        if workerRunning != newRunning { workerRunning = newRunning }
    }

    private func readExtensionHeartbeat() {
        let path = statusDir.appendingPathComponent("extension-heartbeat.json")

        // Check 1: fresh heartbeat file
        if let data = try? Data(contentsOf: path),
           let heartbeat = try? JSONDecoder().decode(ExtensionHeartbeatFile.self, from: data) {
            extensionHeartbeat = heartbeat
            if isHeartbeatFresh(heartbeat.last_message) {
                extensionConnected = true
                return
            }
        }

        // Check 2: daemon reports extension connection
        if let daemonExt = daemonStatus?.last_extension_message,
           isHeartbeatFresh(daemonExt) {
            extensionConnected = true
            return
        }

        // Native messaging manifests can be installed by the pkg without the
        // browser extension actually being loaded. Treat only a live heartbeat
        // or fresh daemon message as a true browser-extension connection.
        extensionConnected = false
    }

    /// Re-reads focus-session.json. Called on every status refresh and each
    /// second while the popover is open, so a session started or stopped
    /// from the CLI shows up without waiting for the 30 s poll.
    func readFocusSession() {
        let state = FocusRecordState(signal: FocusSignal.read(in: statusDir))
        if state != focusState { focusState = state }
    }

    private func readFocusQuestions() {
        let path = statusDir.appendingPathComponent("focus-questions.json")
        guard let data = try? Data(contentsOf: path),
              let file = try? JSONDecoder().decode(FocusQuestionsFile.self, from: data) else {
            focusQuestionsAvailable = false
            focusQuestionsSlug = ""
            return
        }

        if file.status == "pending" {
            focusQuestionsAvailable = true
            focusQuestionsSlug = file.slug
            // Once per session: a notification when the user allowed them,
            // else the Q&A window itself (FocusQuestionsNotifier).
            FocusQuestionsNotifier.shared.questionsReady(file)
        } else {
            focusQuestionsAvailable = false
            focusQuestionsSlug = ""
            // Close the Q&A window if it's open and questions are done
            DispatchQueue.main.async {
                for window in NSApp.windows where window.title == "Focus Q&A" {
                    window.close()
                }
            }
        }
    }

    private func readSOPIndex() {
        let path = statusDir.appendingPathComponent("sops-index.json")
        guard let data = try? Data(contentsOf: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            sopDraftCount = 0
            sopApprovedCount = 0
            sopTotalCount = 0
            sopAgentReadyCount = 0
            return
        }
        sopDraftCount = json["draft_count"] as? Int ?? 0
        sopApprovedCount = json["approved_count"] as? Int ?? 0
        if let sops = json["sops"] as? [[String: Any]] {
            sopTotalCount = sops.count
            sopAgentReadyCount = sops.filter { ($0["lifecycle_state"] as? String) == "agent_ready" }.count
        }
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

    /// The user (or a launch) just started the services: show "Starting…"
    /// and re-read their status every second until both report.
    func servicesStarting() {
        startedAt = Date()
        updateHealth()
        startupTimer?.invalidate()
        startupTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            Task { @MainActor [weak self] in
                guard let self, self.startedAt != nil else {
                    timer.invalidate()
                    return
                }
                self.refreshStatus()
            }
        }
    }

    private func updateHealth() {
        let now = Date()
        let newHealth = ServiceHealth.derive(
            userStopped: userStopped,
            startedAt: startedAt,
            now: now,
            daemonRunning: daemonRunning,
            workerRunning: workerRunning,
            hasIssues: !accessibilityGranted
                || !screenRecordingGranted
                || (workerStatus?.consecutive_errors ?? 0) > 0
                || vlmBacklogged
                || captureStallStatus != nil
        )
        if newHealth != .starting, startedAt != nil {
            startedAt = nil
            startupTimer?.invalidate()
            startupTimer = nil
        }
        if health != newHealth { health = newHealth }
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
