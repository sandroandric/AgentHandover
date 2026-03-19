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
    @State private var showMoreActions = false

    // Record button idle pulse
    @State private var idlePulse = false
    // Prevent reopening Q&A window on every poll
    @AppStorage("lastOpenedQASlug") private var lastOpenedQASlug = ""

    // Contra design tokens
    private let darkNavy = Color(red: 0.09, green: 0.10, blue: 0.12)
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)
    private let goldenYellow = Color(red: 1.0, green: 0.74, blue: 0.07)
    private let cardRadius: CGFloat = 14
    private let contraBorder: CGFloat = 1.5

    var body: some View {
        VStack(spacing: 0) {
            // Status + brand
            statusHeader

            // Main content area
            VStack(spacing: 12) {
                // Today's progress card
                todayCard

                // Attention items (questions, drafts)
                if hasAttentionItems {
                    attentionSection
                }

                // Primary action: Record
                recordSection

                // Quick links grid
                quickLinksGrid

                // Footer: services + quit
                footerSection
            }
            .padding(.horizontal, 14)
            .padding(.bottom, 12)
        }
        .frame(width: 320)
        .onChange(of: delegate.pendingOnboarding) { pending in
            if pending {
                delegate.pendingOnboarding = false
                delegate.showOnboarding(appState: appState)
            }
        }
        .onChange(of: appState.focusQuestionsAvailable) { available in
            if available && appState.focusQuestionsSlug != lastOpenedQASlug {
                lastOpenedQASlug = appState.focusQuestionsSlug
                openWindow(id: "focus-qa")
            }
            if !available {
                lastOpenedQASlug = ""
            }
        }
        .onAppear {
            if !hasCompletedOnboarding && delegate.pendingOnboarding {
                delegate.pendingOnboarding = false
                delegate.showOnboarding(appState: appState)
            }
            syncFocusState()
        }
        .onDisappear {
            elapsedTimer?.invalidate()
            elapsedTimer = nil
        }
    }

    // MARK: - Status Header

    private var statusHeader: some View {
        HStack(spacing: 10) {
            // Status indicator with thick border
            ZStack {
                Circle()
                    .fill(appState.health.color)
                    .frame(width: 28, height: 28)
                    .overlay(
                        Circle()
                            .stroke(darkNavy, lineWidth: 2)
                    )
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(statusTitle)
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)
                Text(statusSubtitle)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
            }

            Spacer()

            // Setup needed indicator
            if !hasCompletedOnboarding || !appState.accessibilityGranted {
                Button(action: { delegate.showOnboarding(appState: appState) }) {
                    Image(systemName: "exclamationmark.circle.fill")
                        .font(.system(size: 16))
                        .foregroundColor(warmOrange)
                }
                .buttonStyle(.plain)
                .help("Setup needed")
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
        .background(Color.primary.opacity(0.03))
    }

    private var statusTitle: String {
        if isRecording { return "Recording..." }
        if appState.userStopped { return "Paused" }
        switch appState.health {
        case .healthy: return "Observing"
        case .warning: return "Observing"
        case .down: return "Offline"
        case .stopped: return "Stopped"
        }
    }

    private var statusSubtitle: String {
        if isRecording {
            return "\(focusSessionTitle) \u{00B7} \(formattedElapsed)"
        }
        if appState.userStopped { return "Tap Start to resume learning" }
        if !appState.daemonRunning && !appState.workerRunning {
            return "Services not running"
        }
        if appState.eventsToday > 0 {
            return "Learning from your work"
        }
        return "Waiting for activity"
    }

    // MARK: - Today Card

    private var todayCard: some View {
        VStack(spacing: 8) {
            HStack {
                Text("Today")
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundColor(.secondary)
                    .textCase(.uppercase)
                    .tracking(0.5)
                Spacer()
            }

            HStack(spacing: 16) {
                TodayStat(
                    icon: "camera.viewfinder",
                    value: "\(appState.eventsToday)",
                    label: "Captured",
                    color: .blue
                )
                TodayStat(
                    icon: "doc.text",
                    value: "\(appState.sopsGenerated)",
                    label: "Learned",
                    color: .purple
                )
                if appState.sopAgentReadyCount > 0 {
                    TodayStat(
                        icon: "checkmark.shield",
                        value: "\(appState.sopAgentReadyCount)",
                        label: "Ready",
                        color: .green
                    )
                } else {
                    TodayStat(
                        icon: "hourglass",
                        value: "\(appState.vlmQueuePending)",
                        label: "Processing",
                        color: .orange
                    )
                }
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: cardRadius)
                .fill(Color(nsColor: .controlBackgroundColor))
        )
        .overlay(
            RoundedRectangle(cornerRadius: cardRadius)
                .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
        )
    }

    // MARK: - Attention Section

    private var hasAttentionItems: Bool {
        appState.focusQuestionsAvailable || appState.sopDraftCount > 0
    }

    private var attentionSection: some View {
        VStack(spacing: 6) {
            // Focus Q&A pending
            if appState.focusQuestionsAvailable {
                Button(action: { openWindow(id: "focus-qa") }) {
                    HStack(spacing: 10) {
                        ZStack {
                            Circle()
                                .fill(goldenYellow)
                                .frame(width: 30, height: 30)
                                .overlay(Circle().stroke(darkNavy, lineWidth: 1.5))
                            Image(systemName: "questionmark.bubble.fill")
                                .font(.system(size: 13))
                                .foregroundColor(darkNavy)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            Text("Finish your workflow")
                                .font(.system(size: 12, weight: .bold, design: .rounded))
                                .foregroundColor(darkNavy)
                            Text("Answer a few questions to complete")
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Image(systemName: "chevron.right")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(darkNavy)
                    }
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: cardRadius)
                            .fill(goldenYellow.opacity(0.12))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: cardRadius)
                            .stroke(darkNavy.opacity(0.15), lineWidth: contraBorder)
                    )
                }
                .buttonStyle(.plain)
            }

            // Drafts to review
            if appState.sopDraftCount > 0 {
                Button(action: { openWindow(id: "micro-review") }) {
                    HStack(spacing: 10) {
                        ZStack {
                            Circle()
                                .fill(warmOrange)
                                .frame(width: 30, height: 30)
                                .overlay(Circle().stroke(darkNavy, lineWidth: 1.5))
                            Image(systemName: "checkmark.rectangle.stack.fill")
                                .font(.system(size: 13))
                                .foregroundColor(.white)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(appState.sopDraftCount) workflow\(appState.sopDraftCount == 1 ? "" : "s") to review")
                                .font(.system(size: 12, weight: .bold, design: .rounded))
                                .foregroundColor(darkNavy)
                            Text("Approve to make agent-ready")
                                .font(.system(size: 10))
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Image(systemName: "chevron.right")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundColor(darkNavy)
                    }
                    .padding(10)
                    .background(
                        RoundedRectangle(cornerRadius: cardRadius)
                            .fill(warmOrange.opacity(0.08))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: cardRadius)
                            .stroke(darkNavy.opacity(0.15), lineWidth: contraBorder)
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: - Record Section

    private var recordSection: some View {
        Group {
            if isRecording {
                // Active recording
                VStack(spacing: 8) {
                    HStack(spacing: 8) {
                        Circle()
                            .fill(Color.red)
                            .frame(width: 8, height: 8)
                            .opacity(pulsingOpacity)
                            .animation(
                                .easeInOut(duration: 0.8).repeatForever(autoreverses: true),
                                value: pulsingOpacity
                            )
                        Text(focusSessionTitle)
                            .font(.system(size: 12, weight: .medium))
                            .lineLimit(1)
                        Spacer()
                        Text(formattedElapsed)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.secondary)
                    }

                    Button(action: stopFocusSession) {
                        HStack {
                            Image(systemName: "stop.fill")
                                .font(.system(size: 10))
                            Text("Stop Recording")
                                .font(.system(size: 12, weight: .medium))
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 8)
                        .background(Color.red.opacity(0.1))
                        .foregroundColor(.red)
                        .cornerRadius(8)
                    }
                    .buttonStyle(.plain)
                }
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .fill(Color(nsColor: .controlBackgroundColor))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .stroke(Color.red.opacity(0.25), lineWidth: 1)
                )
                .shadow(color: Color.red.opacity(0.04), radius: 8, y: 2)
            } else if showTitlePrompt {
                // Title input
                VStack(alignment: .leading, spacing: 8) {
                    Text("What are you about to do?")
                        .font(.system(size: 12, weight: .medium))

                    TextField("e.g. File expense report", text: $focusSessionTitle)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12))

                    HStack {
                        Button("Cancel") {
                            showTitlePrompt = false
                            focusSessionTitle = ""
                        }
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)

                        Spacer()

                        Button(action: { startFocusSession(title: focusSessionTitle) }) {
                            HStack(spacing: 4) {
                                Image(systemName: "record.circle")
                                    .font(.system(size: 10))
                                Text("Start")
                                    .font(.system(size: 12, weight: .medium))
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 6)
                            .background(Color.red)
                            .foregroundColor(.white)
                            .cornerRadius(6)
                        }
                        .disabled(focusSessionTitle.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                }
                .buttonStyle(.plain)
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .fill(Color(nsColor: .controlBackgroundColor))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                )
            } else if appState.focusSessionProcessing {
                // Focus session stopped, worker is processing
                HStack(spacing: 10) {
                    ProgressView()
                        .controlSize(.small)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Analyzing \"\(appState.focusSessionTitle)\"")
                            .font(.system(size: 12, weight: .medium))
                            .lineLimit(1)
                        Text("AI is reviewing your screenshots - 2-5 min")
                            .font(.system(size: 10))
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .fill(Color.purple.opacity(0.04))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .stroke(Color.purple.opacity(0.15), lineWidth: 1)
                )
            } else {
                // Record button with idle pulse
                Button(action: { showTitlePrompt = true }) {
                    HStack(spacing: 8) {
                        ZStack {
                            // Pulse ring
                            Circle()
                                .fill(Color.red.opacity(0.15))
                                .frame(width: 24, height: 24)
                                .scaleEffect(idlePulse ? 1.4 : 1.0)
                                .opacity(idlePulse ? 0.0 : 0.5)

                            Image(systemName: "record.circle")
                                .font(.system(size: 15))
                                .foregroundColor(.red)
                        }
                        Text("Record a Workflow")
                            .font(.system(size: 13, weight: .medium))
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundColor(.secondary)
                    }
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: cardRadius)
                            .fill(Color(nsColor: .controlBackgroundColor))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: cardRadius)
                            .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                    )
                }
                .buttonStyle(.plain)
                .onAppear {
                    withAnimation(.easeInOut(duration: 2.0).repeatForever(autoreverses: false)) {
                        idlePulse = true
                    }
                }
            }
        }
    }

    // MARK: - Quick Links

    private var quickLinksGrid: some View {
        LazyVGrid(columns: [
            GridItem(.flexible()),
            GridItem(.flexible()),
        ], spacing: 8) {
            QuickLink(icon: "tray.full", label: "Workflows", badge: appState.sopTotalCount) {
                openWindow(id: "workflows")
            }
            QuickLink(icon: "calendar.badge.clock", label: "Digest") {
                openWindow(id: "daily-digest")
            }
            QuickLink(icon: "checkmark.rectangle.stack", label: "Review") {
                openWindow(id: "micro-review")
            }
            QuickLink(icon: "gearshape", label: "Settings") {
                openConfig()
            }
        }
    }

    // MARK: - Footer

    private var footerSection: some View {
        HStack(spacing: 0) {
            // Service toggle
            if appState.daemonRunning || appState.workerRunning {
                Button(action: {
                    appState.userStopped = true
                    ServiceController.stopAll()
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: "pause.fill")
                            .font(.system(size: 8))
                        Text("Pause")
                            .font(.system(size: 11))
                    }
                    .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            } else {
                Button(action: {
                    appState.userStopped = false
                    ServiceController.startAll()
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: "play.fill")
                            .font(.system(size: 8))
                        Text("Start")
                            .font(.system(size: 11))
                    }
                    .foregroundColor(.blue)
                }
                .buttonStyle(.plain)
            }

            Spacer()

            // Service pills (compact)
            HStack(spacing: 5) {
                serviceDot(running: appState.daemonRunning, label: "Daemon")
                serviceDot(running: appState.workerRunning, label: "Worker")
                serviceDot(running: appState.extensionConnected, label: "Extension")
            }

            Spacer()

            Button("Quit") {
                NSApplication.shared.terminate(nil)
            }
            .font(.system(size: 11))
            .foregroundColor(.secondary)
            .buttonStyle(.plain)
            .keyboardShortcut("q")
        }
        .padding(.top, 6)
        .padding(.horizontal, 2)
    }

    private func serviceDot(running: Bool, label: String) -> some View {
        Circle()
            .fill(running ? Color.green : Color.red.opacity(0.4))
            .frame(width: 6, height: 6)
            .help(label)
    }

    // MARK: - Focus Recording Helpers

    private var pulsingOpacity: Double {
        isRecording ? 0.3 : 1.0
    }

    private var formattedElapsed: String {
        let minutes = elapsedSeconds / 60
        let seconds = elapsedSeconds % 60
        return String(format: "%d:%02d", minutes, seconds)
    }

    private func syncFocusState() {
        if appState.focusSessionActive {
            isRecording = true
            focusSessionTitle = appState.focusSessionTitle
            focusSessionId = UUID(uuidString: appState.focusSessionId ?? "")
            if let startedStr = appState.focusSessionStartedAt {
                let fmt = ISO8601DateFormatter()
                if let restored = fmt.date(from: startedStr) {
                    recordingStartTime = restored
                    elapsedSeconds = Int(Date().timeIntervalSince(restored))
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

        elapsedTimer?.invalidate()
        elapsedTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            elapsedSeconds += 1
        }
    }

    private func stopFocusSession() {
        guard let sessionId = focusSessionId else { return }

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
        let dir = home.appendingPathComponent("Library/Application Support/agenthandover")
        let target = dir.appendingPathComponent("focus-session.json")
        let tmp = dir.appendingPathComponent(".focus-session.json.tmp")

        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let data = try JSONSerialization.data(withJSONObject: signal, options: .prettyPrinted)
            try data.write(to: tmp, options: .atomic)
            if FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.removeItem(at: target)
            }
            try FileManager.default.moveItem(at: tmp, to: target)
        } catch {
            print("Failed to write focus-session.json: \(error)")
        }
    }

    private func readExistingSignalStartedAt() -> String? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent(
            "Library/Application Support/agenthandover/focus-session.json"
        )
        guard let data = try? Data(contentsOf: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let startedAt = json["started_at"] as? String else {
            return nil
        }
        return startedAt
    }

    private func openConfig() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let configPath = home
            .appendingPathComponent("Library/Application Support/agenthandover/config.toml")
        NSWorkspace.shared.open(configPath)
    }
}

// MARK: - Components

struct TodayStat: View {
    let icon: String
    let value: String
    let label: String
    var color: Color = .primary

    var body: some View {
        VStack(spacing: 4) {
            Image(systemName: icon)
                .font(.system(size: 14))
                .foregroundColor(color)
            Text(value)
                .font(.system(size: 16, weight: .bold, design: .rounded))
            Text(label)
                .font(.system(size: 9))
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

struct QuickLink: View {
    let icon: String
    let label: String
    var badge: Int = 0
    let action: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                Text(label)
                    .font(.system(size: 11))
                if badge > 0 {
                    Spacer()
                    Text("\(badge)")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundColor(.secondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(isHovered ? Color.primary.opacity(0.06) : Color.primary.opacity(0.03))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.primary.opacity(0.06), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
    }
}
