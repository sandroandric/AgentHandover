import SwiftUI

struct MenuBarView: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var delegate: AppDelegate
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    @Environment(\.openWindow) private var openWindow

    // Focus recording state
    @State private var isRecording = false
    @State private var focusSessionTitle: String = ""
    @State private var focusSessionId: UUID?
    @State private var recordingStartTime: Date?
    @State private var showTitlePrompt = false
    @State private var elapsedTimer: Timer?
    @State private var elapsedSeconds: Int = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header with status
            headerSection

            Divider()

            // Stats
            statsSection

            Divider()

            // Focus Recording
            focusRecordingSection

            Divider()

            // Service controls
            controlsSection

            Divider()

            // Quick actions
            actionsSection

            Divider()

            // Quit
            Button("Quit OpenMimic") {
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q")
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .frame(width: 300)
        .onChange(of: delegate.pendingOnboarding) { pending in
            if pending {
                delegate.pendingOnboarding = false
                openWindow(id: "onboarding")
            }
        }
        .onAppear {
            // Also check on first appear (when user clicks menu bar icon)
            if !hasCompletedOnboarding && delegate.pendingOnboarding {
                delegate.pendingOnboarding = false
                openWindow(id: "onboarding")
            }
            // Sync focus state from AppState (restarts timer if recording)
            syncFocusState()
        }
        .onDisappear {
            // Pause the elapsed timer when popover is dismissed to avoid
            // wasted CPU updates while the view isn't visible.  It will
            // be restarted by syncFocusState() on next .onAppear.
            elapsedTimer?.invalidate()
            elapsedTimer = nil
        }
    }

    // MARK: - Sections

    private var headerSection: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(appState.health.color)
                .frame(width: 10, height: 10)

            VStack(alignment: .leading, spacing: 2) {
                Text("OpenMimic")
                    .font(.headline)
                Text(appState.health.label)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            Text("v\(appState.daemonVersion)")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    private var statsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            StatRow(label: "Events Today", value: "\(appState.eventsToday)")
            StatRow(label: "SOPs Generated", value: "\(appState.sopsGenerated)")

            // VLM queue status (only shown when VLM is available)
            if appState.workerStatus?.vlm_available == true {
                let pending = appState.vlmQueuePending
                if pending > 0 {
                    StatRow(
                        label: "VLM Queue",
                        value: appState.vlmBacklogged
                            ? "\(pending) pending ⚠️"
                            : "\(pending) pending"
                    )
                }
            }

            HStack(spacing: 12) {
                ServicePill(
                    name: "Daemon",
                    running: appState.daemonRunning
                )
                ServicePill(
                    name: "Worker",
                    running: appState.workerRunning
                )
                ServicePill(
                    name: "Extension",
                    running: appState.extensionConnected
                )
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private var focusRecordingSection: some View {
        VStack(spacing: 4) {
            if isRecording {
                // Recording active — show status + stop button
                HStack(spacing: 8) {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 8, height: 8)
                        .opacity(pulsingOpacity)
                        .animation(
                            .easeInOut(duration: 0.8).repeatForever(autoreverses: true),
                            value: pulsingOpacity
                        )

                    VStack(alignment: .leading, spacing: 1) {
                        Text("Recording: \(focusSessionTitle)")
                            .font(.caption)
                            .fontWeight(.medium)
                            .lineLimit(1)
                        Text(formattedElapsed)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 6)

                Button(action: stopFocusSession) {
                    Label("Stop Recording", systemImage: "stop.circle.fill")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundColor(.red)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            } else if showTitlePrompt {
                // Title input prompt
                VStack(alignment: .leading, spacing: 6) {
                    Text("Workflow title:")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    TextField("e.g. Expense report filing", text: $focusSessionTitle)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption)

                    HStack {
                        Button("Cancel") {
                            showTitlePrompt = false
                            focusSessionTitle = ""
                        }
                        .buttonStyle(.plain)
                        .font(.caption)
                        .foregroundColor(.secondary)

                        Spacer()

                        Button("Start") {
                            startFocusSession(title: focusSessionTitle)
                        }
                        .buttonStyle(.plain)
                        .font(.caption)
                        .foregroundColor(.blue)
                        .disabled(focusSessionTitle.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            } else {
                // Idle — show record button
                Button(action: {
                    showTitlePrompt = true
                }) {
                    Label("Record Workflow", systemImage: "record.circle")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            }
        }
        .padding(.vertical, 4)
    }

    private var controlsSection: some View {
        VStack(spacing: 4) {
            if appState.daemonRunning || appState.workerRunning {
                Button(action: {
                    appState.userStopped = true
                    ServiceController.stopAll()
                }) {
                    Label("Stop All Services", systemImage: "stop.circle")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            } else {
                Button(action: {
                    appState.userStopped = false
                    ServiceController.startAll()
                }) {
                    Label("Start All Services", systemImage: "play.circle")
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            }

            Button(action: {
                ServiceController.restartAll()
            }) {
                Label("Restart Services", systemImage: "arrow.clockwise.circle")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)
        }
        .padding(.vertical, 4)
    }

    private var actionsSection: some View {
        VStack(spacing: 4) {
            // Workflow inbox
            Button(action: {
                openWindow(id: "workflows")
            }) {
                HStack {
                    Label("Workflows", systemImage: "tray.full")
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if appState.sopDraftCount > 0 {
                        Text("\(appState.sopDraftCount)")
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.orange.opacity(0.2))
                            .foregroundColor(.orange)
                            .cornerRadius(4)
                    }
                }
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)

            // Daily digest
            Button(action: {
                openWindow(id: "daily-digest")
            }) {
                Label("Daily Digest", systemImage: "calendar.badge.clock")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)

            // Review queue
            Button(action: {
                openWindow(id: "micro-review")
            }) {
                Label("Review Queue", systemImage: "checkmark.rectangle.stack")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)

            // Open config
            Button(action: openConfig) {
                Label("Edit Configuration", systemImage: "gearshape")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)

            // Open SOPs directory
            Button(action: openSOPsDir) {
                Label("Open SOPs Folder", systemImage: "folder")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)

            // View logs
            Button(action: openLogs) {
                Label("View Logs", systemImage: "doc.text")
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 16)
            .padding(.vertical, 6)

            // Setup / Permissions / Extension CTA
            if !hasCompletedOnboarding {
                Button(action: {
                    openWindow(id: "onboarding")
                }) {
                    Label("Complete Setup", systemImage: "sparkles")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundColor(.blue)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            } else if !appState.accessibilityGranted || !appState.screenRecordingGranted {
                Button(action: {
                    openWindow(id: "onboarding")
                }) {
                    Label("Fix Permissions", systemImage: "exclamationmark.shield")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundColor(.orange)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            } else if !appState.extensionConnected {
                Button(action: {
                    openWindow(id: "onboarding")
                }) {
                    Label("Connect Extension", systemImage: "puzzlepiece.extension")
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .foregroundColor(.orange)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 6)
            }
        }
        .padding(.vertical, 4)
    }

    // MARK: - Focus Recording Helpers

    private var pulsingOpacity: Double {
        isRecording ? 0.3 : 1.0
    }

    private var formattedElapsed: String {
        let minutes = elapsedSeconds / 60
        let seconds = elapsedSeconds % 60
        return String(format: "%dm %02ds", minutes, seconds)
    }

    private func syncFocusState() {
        if appState.focusSessionActive {
            isRecording = true
            focusSessionTitle = appState.focusSessionTitle
            focusSessionId = UUID(uuidString: appState.focusSessionId ?? "")
            // Restore recordingStartTime from the signal file's started_at
            if let startedStr = appState.focusSessionStartedAt {
                let fmt = ISO8601DateFormatter()
                if let restored = fmt.date(from: startedStr) {
                    recordingStartTime = restored
                    // Compute elapsed seconds from original start time
                    elapsedSeconds = Int(Date().timeIntervalSince(restored))
                    // Restart the elapsed timer
                    elapsedTimer?.invalidate()
                    elapsedTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                        elapsedSeconds += 1
                    }
                }
            }
        }
    }

    private func startFocusSession(title: String) {
        let sessionId = UUID()
        let signal: [String: Any] = [
            "session_id": sessionId.uuidString,
            "title": title,
            "started_at": ISO8601DateFormatter().string(from: Date()),
            "status": "recording"
        ]

        writeFocusSignalFile(signal)

        focusSessionId = sessionId
        focusSessionTitle = title
        recordingStartTime = Date()
        isRecording = true
        showTitlePrompt = false
        elapsedSeconds = 0

        // Start elapsed time counter
        elapsedTimer?.invalidate()
        elapsedTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            elapsedSeconds += 1
        }
    }

    private func stopFocusSession() {
        guard let sessionId = focusSessionId else { return }

        // Preserve original started_at: read from signal file if in-memory
        // value is missing (e.g., app was reopened during recording).
        var startedAt: String
        if let startTime = recordingStartTime {
            startedAt = ISO8601DateFormatter().string(from: startTime)
        } else if let existing = readExistingSignalStartedAt() {
            startedAt = existing
        } else {
            startedAt = ISO8601DateFormatter().string(from: Date())
        }

        let signal: [String: Any] = [
            "session_id": sessionId.uuidString,
            "title": focusSessionTitle,
            "started_at": startedAt,
            "status": "stopped"
        ]

        writeFocusSignalFile(signal)

        elapsedTimer?.invalidate()
        elapsedTimer = nil
        isRecording = false
        focusSessionId = nil
        recordingStartTime = nil
        focusSessionTitle = ""
        elapsedSeconds = 0
    }

    private func writeFocusSignalFile(_ signal: [String: Any]) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let dir = home.appendingPathComponent("Library/Application Support/oc-apprentice")
        let target = dir.appendingPathComponent("focus-session.json")
        let tmp = dir.appendingPathComponent(".focus-session.json.tmp")

        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: signal, options: .prettyPrinted)
            try data.write(to: tmp, options: .atomic)
            // Atomic rename
            if FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.removeItem(at: target)
            }
            try FileManager.default.moveItem(at: tmp, to: target)
        } catch {
            // Best-effort; don't crash the app over a signal file
            print("Failed to write focus-session.json: \(error)")
        }
    }

    private func readExistingSignalStartedAt() -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent(
            "Library/Application Support/oc-apprentice/focus-session.json"
        )
        guard let data = try? Data(contentsOf: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let startedAt = json["started_at"] as? String else {
            return nil
        }
        return startedAt
    }

    // MARK: - Actions

    private func openConfig() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let configPath = home
            .appendingPathComponent("Library/Application Support/oc-apprentice/config.toml")
        NSWorkspace.shared.open(configPath)
    }

    private func openSOPsDir() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let sopsPath = home
            .appendingPathComponent(".openclaw/workspace/memory/apprentice/sops")
        if FileManager.default.fileExists(atPath: sopsPath.path) {
            NSWorkspace.shared.open(sopsPath)
        } else {
            // Fall back to the apprentice data dir
            let dataDir = home
                .appendingPathComponent("Library/Application Support/oc-apprentice")
            NSWorkspace.shared.open(dataDir)
        }
    }

    private func openLogs() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let logsPath = home
            .appendingPathComponent("Library/Application Support/oc-apprentice/logs")
        NSWorkspace.shared.open(logsPath)
    }
}

// MARK: - Subviews

struct StatRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
    }
}

struct ServicePill: View {
    let name: String
    let running: Bool

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(running ? Color.green : Color.red)
                .frame(width: 6, height: 6)
            Text(name)
                .font(.caption2)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(
            RoundedRectangle(cornerRadius: 4)
                .fill(Color.secondary.opacity(0.1))
        )
    }
}
