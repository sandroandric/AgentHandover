import Foundation
import SwiftUI

/// Tells whether a file was written, created or removed since the last
/// look, from its modification date: a stat, so it can run every couple
/// of seconds.
struct FileStamp {
    /// Nil before the first look; `.some(nil)` when the file was missing.
    private var last: Date??

    /// True on the first look and whenever the file changed since.
    mutating func changed(_ url: URL) -> Bool {
        let modified = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
        defer { last = .some(modified) }
        guard let last else { return true }
        return last != modified
    }
}

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
    /// Set on stop (ISO 8601); absent while recording and in signals from
    /// older writers. Left out of the JSON when nil.
    var stopped_at: String? = nil
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
    /// The worker's id for the question (its row in `micro_questions`);
    /// kept so the worker closes it when the answers come back.
    var questionID: String? = nil
    /// Answers the recorded runs suggest; the window offers them as
    /// buttons next to the text field.
    var choices: [String]? = nil

    var id: Int { index }

    enum CodingKeys: String, CodingKey {
        case index, question, category, context, `default`, step_indexes, choices
        case questionID = "id"
    }
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
    /// The configured models Ollama does not serve; absent when Ollama did
    /// not answer (and in the early "starting" status).
    let missing_models: [String]?
    /// "Upgrade everything": the background re-embed of the Skills index
    /// with EmbeddingGemma 2; absent unless the user chose it.
    let embedding_upgrade: EmbeddingUpgradeStatus?
    /// `[skills] enabled` as the running worker read it: false is Memory
    /// only (`SkillsMode`); absent from a worker before 2.0.
    let skills_enabled: Bool?

    /// The worker runs Memory only: no generative model, no Skills learned.
    var memoryOnly: Bool { skills_enabled == false }
}

/// worker-status.json's `embedding_upgrade` (worker embedding_upgrade.py).
struct EmbeddingUpgradeStatus: Codable, Equatable {
    let state: String
    let done: Int
    let total: Int
    let dropped: Int
    let error: String?

    /// One plain line for Memory Settings.
    var line: String {
        switch state {
        case "running":
            return "Re-indexing your Skills with EmbeddingGemma 2: \(done) of \(total). Search keeps working meanwhile."
        case "done":
            return "Your Skills are indexed with EmbeddingGemma 2."
        case "failed":
            return "Re-indexing stopped: \(error ?? "unknown error"). Your Skills still work with the previous model."
        default:
            return "Re-indexing your Skills with EmbeddingGemma 2 starts shortly."
        }
    }
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
    /// Open review-card questions (micro-questions.json).
    @Published var microQuestionCount = 0

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

    // MARK: - Polling

    private var pollTimer: Timer?
    private var quickTimer: Timer?
    private var focusQuestionsStamp = FileStamp()
    private var focusResultStamp = FileStamp()
    private var microQuestionsStamp = FileStamp()
    /// Called when both grants are present after one of them was missing
    /// (set by the app delegate; see `OnboardingProgress.regrantAction`).
    var onPermissionsGranted: (() -> Void)?
    /// Both grants at the previous `checkPermissions`; nil before the first.
    private(set) var grantedAtLastCheck: Bool?
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
        // Two cheap checks every couple of seconds: new focus questions
        // (the Q&A shows within seconds, not at the next 30 s poll) and
        // grants switched on in System Settings.
        quickTimer?.invalidate()
        quickTimer = Timer.scheduledTimer(withTimeInterval: Self.quickCheckInterval, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.quickCheck()
            }
        }
        refreshStatus()
    }

    func stopPolling() {
        pollTimer?.invalidate()
        pollTimer = nil
        quickTimer?.invalidate()
        quickTimer = nil
    }

    nonisolated static let quickCheckInterval: TimeInterval = 2.0

    /// Re-reads focus-questions.json and focus-result.json when they
    /// changed, and the grants.
    func quickCheck() {
        if focusQuestionsStamp.changed(statusDir.appendingPathComponent("focus-questions.json")) {
            readFocusQuestions()
        }
        if focusResultStamp.changed(statusDir.appendingPathComponent(FocusResult.fileName)) {
            readFocusSession()
        }
        // The worker's review-card questions show within seconds of the
        // evidence that raised them.
        if microQuestionsStamp.changed(statusDir.appendingPathComponent(MicroQuestions.fileName)) {
            let questions = MicroQuestions.load(from: statusDir)
            microQuestionCount = questions.count
            NotificationCenter.default.post(name: MicroQuestions.changed, object: nil)
            FocusQuestionsNotifier.shared.microQuestionsReady(questions)
        }
        if Self.quickCheckReadsGrants(grantedAtLastCheck: grantedAtLastCheck) {
            checkPermissions()
        }
    }

    /// Whether the 2 s check reads the grants: only while one is missing
    /// (or not read yet), so switching it on in System Settings starts
    /// capture within seconds. Each reading re-verifies the app's code
    /// signature; done every 2 s it cost the idle app ~0.7 s of CPU a
    /// minute on the test Mac. With both grants in place, a revocation is
    /// seen by `refreshStatus` (the 30 s poll and every menu open).
    nonisolated static func quickCheckReadsGrants(grantedAtLastCheck: Bool?) -> Bool {
        grantedAtLastCheck != true
    }

    /// The launch's own reading of the grants, which decided whether setup
    /// opened: the next reading is compared with it (`grantsReturned`).
    func seedGrants(accessibility: Bool, screenRecording: Bool) {
        grantedAtLastCheck = accessibility && screenRecording
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

    /// Re-reads focus-session.json and focus-result.json. Called on every
    /// status refresh, when focus-result.json changes, and each
    /// second while the popover is open, so a session started or stopped
    /// from the CLI shows up without waiting for the 30 s poll.
    func readFocusSession() {
        let state = FocusRecordState(
            signal: FocusSignal.read(in: statusDir),
            result: FocusResult.read(in: statusDir)
        )
        if state != focusState { focusState = state }
    }

    /// The menu's Dismiss on "Nothing was recorded": deletes the worker's
    /// focus-result.json for that session.
    func dismissFocusResult() {
        guard case .nothingRecorded(let sessionId, _) = focusState else { return }
        FocusResult.dismiss(sessionID: sessionId, in: statusDir)
        readFocusSession()
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
        let accessibility = PermissionChecker.isAccessibilityGranted()
        let screenRecording = PermissionChecker.isScreenRecordingGranted()
        if accessibilityGranted != accessibility { accessibilityGranted = accessibility }
        if screenRecordingGranted != screenRecording { screenRecordingGranted = screenRecording }
        let now = accessibility && screenRecording
        // Compared with this function's previous reading, not with the
        // published flags: the permission screens set those themselves.
        let before = grantedAtLastCheck
        grantedAtLastCheck = now
        if Self.grantsReturned(before: before, now: now) {
            onPermissionsGranted?()
        }
    }

    /// Both grants are present now and one was missing at the previous
    /// check. The first check (`before` nil) is not a return: a launch
    /// with the grants in place is the launch's business.
    nonisolated static func grantsReturned(before: Bool?, now: Bool) -> Bool {
        before == false && now
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
