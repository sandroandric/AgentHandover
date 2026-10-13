import SwiftUI

struct MenuBarView: View {
    private enum PermissionRepairKind {
        case accessibility
        case screenRecording
    }

    @EnvironmentObject var appState: AppState
    @EnvironmentObject var delegate: AppDelegate
    @EnvironmentObject var observationToggle: ObservationToggle
    @AppStorage("hasCompletedOnboarding") private var hasCompletedOnboarding = false
    @StateObject private var agentDetector = AgentDetector()
    @ObservedObject private var memory = MemoryCenter.shared
    @Environment(\.openWindow) private var openWindow

    // Focus recording: the session itself lives in focus-session.json
    // (appState.focusState); the view only holds the title being typed and
    // the clock for the elapsed time.
    @State private var newSessionTitle: String = ""
    @State private var showTitlePrompt = false
    /// Why the last Start did not record (shown in the title prompt).
    @State private var focusStartError: String?
    @State private var now = Date()
    @State private var showMoreActions = false
    @State private var wasPausedBeforeFocus = false
    @State private var qaWindowShown = false
    @State private var openedPermissionSettings: PermissionRepairKind?
    @State private var screenRecordingNeedsManualSettings = false

    // Record button idle pulse
    @State private var idlePulse = false

    // Contra design tokens
    private let darkNavy = Color.primary
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
                // Attention items (questions, drafts)
                if hasAttentionItems {
                    attentionSection
                }

                // Primary action: Record
                recordSection

                // Quick links grid
                quickLinksGrid

                // Memory: search, pause, forget
                memorySection

                // Connected agents
                agentsSection

                // Daemon supervision: why it was restarted, if it was
                if let supervisorLine = appState.daemonSupervisorStatus {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.clockwise.circle")
                            .font(.system(size: 9))
                        Text(supervisorLine)
                            .font(.system(size: 10))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                // Screen capture stalled (ScreenCaptureServer recycled its path)
                if let stallLine = appState.captureStallStatus {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.system(size: 9))
                        Text(stallLine)
                            .font(.system(size: 10))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .foregroundColor(.orange)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

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
                delegate.showOnboarding()
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            if let pendingPermission = openedPermissionSettings {
                openedPermissionSettings = nil
                Task {
                    switch pendingPermission {
                    case .accessibility:
                        appState.accessibilityGranted = PermissionChecker.isAccessibilityGranted()
                    case .screenRecording:
                        let granted = await refreshScreenRecordingGrant()
                        appState.screenRecordingGranted = granted
                        screenRecordingNeedsManualSettings = !granted
                    }
                }
            }
        }
        .onAppear {
            // Refresh immediately when user opens menu bar
            appState.refreshStatus()
            Task { await memory.refresh() }
            agentDetector.detect()
            if !hasCompletedOnboarding && delegate.pendingOnboarding {
                delegate.pendingOnboarding = false
                delegate.showOnboarding()
            }
        }
        // While the popover is open, follow focus-session.json each second so
        // `agenthandover focus start/stop` shows up here, and tick the clock.
        .task {
            while !Task.isCancelled {
                appState.readFocusSession()
                if isRecording { now = Date() }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    // MARK: - Status Header

    private var statusHeader: some View {
        VStack(spacing: 0) {
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
            if !hasCompletedOnboarding || !appState.accessibilityGranted || !appState.screenRecordingGranted {
                Button(action: { delegate.showOnboarding() }) {
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

        // Permission missing banner
        if hasCompletedOnboarding && (!appState.accessibilityGranted || !appState.screenRecordingGranted) {
            Button(action: { repairPermissions() }) {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 12))
                        .foregroundColor(warmOrange)
                    Text(permissionBannerText)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(darkNavy.opacity(0.7))
                    Spacer()
                    Text("Fix")
                        .font(.system(size: 11, weight: .bold))
                        .foregroundColor(warmOrange)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(warmOrange.opacity(0.08))
            }
            .buttonStyle(.plain)
        }

        // A model the worker needs is not pulled: nothing is processed.
        if hasCompletedOnboarding, appState.workerRunning,
           let row = ModelUpgrade.missingModelRow(
               vlmAvailable: appState.vlmAvailable,
               missingModels: appState.workerStatus?.missing_models) {
            Button(action: { delegate.showModelDownload() }) {
                HStack(spacing: 8) {
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 12))
                        .foregroundColor(warmOrange)
                    Text(row)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(darkNavy.opacity(0.7))
                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(warmOrange.opacity(0.08))
            }
            .buttonStyle(.plain)
        }
        } // VStack
    }

    private var statusTitle: String {
        if isRecording { return "Recording..." }
        if appState.userStopped { return "Paused" }
        if appState.health == .starting { return "Starting\u{2026}" }
        // Derive title from actual daemon state, not combined health
        if appState.daemonRunning {
            return "Observing"
        }
        if appState.workerRunning {
            return "Processing"
        }
        return "Offline"
    }

    private var statusSubtitle: String {
        if isRecording {
            return "\(recordingTitle) \u{00B7} \(formattedElapsed)"
        }
        if appState.userStopped { return "Tap Start to resume learning" }
        if appState.health == .starting { return "Waiting for the daemon and worker to report" }
        if appState.daemonRunning && appState.workerRunning {
            if appState.eventsToday > 0 {
                return "Learning from your work"
            }
            return "Waiting for activity"
        }
        if appState.workerRunning && !appState.daemonRunning {
            return "Processing captured data"
        }
        if appState.daemonRunning && !appState.workerRunning {
            return "Capturing - worker starting"
        }
        return "Services not running"
    }

    private var permissionBannerText: String {
        if !appState.accessibilityGranted && !appState.screenRecordingGranted {
            return "Accessibility and Screen Recording not granted"
        }
        if !appState.accessibilityGranted {
            return "Accessibility not granted - observation may fail"
        }
        return "Screen Recording not granted - recordings may fail"
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
                TodayStat(
                    icon: "checkmark.shield",
                    value: "\(appState.sopAgentReadyCount)",
                    label: "Ready",
                    color: .green
                )
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
        questionsActuallyPending || appState.sopDraftCount > 0
    }

    /// Check if questions file actually exists with pending status,
    /// not just the stale appState value which may lag behind.
    private var questionsActuallyPending: Bool {
        guard appState.focusQuestionsAvailable else { return false }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let path = home.appendingPathComponent(
            "Library/Application Support/agenthandover/focus-questions.json")
        guard let data = try? Data(contentsOf: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let status = json["status"] as? String else {
            return false
        }
        return status == "pending"
    }

    private var attentionSection: some View {
        VStack(spacing: 6) {
            // Focus Q&A pending
            if questionsActuallyPending {
                Button(action: { delegate.showFocusQA() }) {
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

            // Drafts to review — opens Workflows where user can see details + approve
            if appState.sopDraftCount > 0 {
                Button(action: { openAndActivate("workflows") }) {
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
                        Text(recordingTitle)
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
                // Title input + optional Screen Recording gate
                VStack(alignment: .leading, spacing: 8) {
                    Text("What are you about to do?")
                        .font(.system(size: 12, weight: .medium))

                    TextField("e.g. File expense report", text: $newSessionTitle)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 12))

                    // Permission gate
                    if !appState.accessibilityGranted || !appState.screenRecordingGranted {
                        HStack(spacing: 6) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 10))
                                .foregroundColor(warmOrange)
                            Text(recordPermissionMessage)
                                .font(.system(size: 10))
                                .foregroundColor(warmOrange)
                        }
                        .padding(.top, 2)
                    }

                    if let focusStartError {
                        Text(focusStartError)
                            .font(.system(size: 10))
                            .foregroundColor(.red)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    HStack {
                        Button("Cancel") {
                            showTitlePrompt = false
                            newSessionTitle = ""
                            focusStartError = nil
                        }
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)

                        Spacer()

                        if !appState.accessibilityGranted || !appState.screenRecordingGranted {
                            Button(action: { repairPermissions() }) {
                                Text("Grant Permission")
                                    .font(.system(size: 12, weight: .medium))
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 6)
                                    .background(warmOrange)
                                    .foregroundColor(.white)
                                    .cornerRadius(6)
                            }
                        } else {
                            Button(action: { startFocusSession(title: newSessionTitle) }) {
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
                            .disabled(newSessionTitle.trimmingCharacters(in: .whitespaces).isEmpty)
                        }
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
                        .stroke((appState.screenRecordingGranted && appState.accessibilityGranted) ? darkNavy.opacity(0.12) : warmOrange.opacity(0.3), lineWidth: contraBorder)
                    )
            } else if let processingTitle {
                // Focus session stopped, worker is processing
                HStack(spacing: 10) {
                    ProgressView()
                        .controlSize(.small)

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Analyzing \"\(processingTitle)\"")
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
              VStack(spacing: 8) {
                if let nothingRecordedMessage = appState.focusState.nothingRecordedMessage {
                // The worker closed the stopped session without events
                // (focus-result.json): say so instead of "Analyzing",
                // above the Record button (recording again needs no dismiss).
                HStack(alignment: .top, spacing: 10) {
                    Image(systemName: "exclamationmark.circle")
                        .font(.system(size: 14))
                        .foregroundColor(warmOrange)

                    Text(nothingRecordedMessage)
                        .font(.system(size: 12, weight: .medium))
                        .fixedSize(horizontal: false, vertical: true)

                    Spacer(minLength: 4)

                    Button("Dismiss") { appState.dismissFocusResult() }
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                        .buttonStyle(.plain)
                }
                .padding(12)
                .background(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .fill(warmOrange.opacity(0.05))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .stroke(warmOrange.opacity(0.3), lineWidth: 1)
                )
                }
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
                        Text("Record a Focus Session")
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
    }

    // MARK: - Quick Links

    private var quickLinksGrid: some View {
        HStack(spacing: 6) {
            compactLink("Skills", badge: appState.sopTotalCount) {
                openAndActivate("workflows")
            }
            compactLink("Digest") {
                openAndActivate("daily-digest")
            }
            compactLink("FAQ") {
                openAndActivate("faq")
            }
        }
    }

    private func compactLink(_ label: String, badge: Int = 0, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Text(label)
                    .font(.system(size: 12, weight: .medium))
                if badge > 0 {
                    Text("\(badge)")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundColor(.white)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Capsule().fill(Color.secondary))
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.primary.opacity(0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.primary.opacity(0.1), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }

    // MARK: - Memory

    /// "Memory · Recording" with Commonplace…, Pause Memory (or Resume
    /// Memory), Forget and Settings.  The state comes from the worker's
    /// `/memory/status`, read each time the menu opens.
    private var memorySection: some View {
        let state = memory.menuState
        return VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                Text("Memory")
                    .font(.system(size: 10, weight: .bold, design: .rounded))
                    .foregroundColor(.secondary)
                Text("\u{00B7} \(state.label(now: Date()))")
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                Spacer()
                Button {
                    openAndActivate("memory-settings")
                } label: {
                    Image(systemName: "gearshape")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                .help("Memory Settings")
            }
            .padding(.leading, 2)

            HStack(spacing: 6) {
                compactLink("Commonplace\u{2026}") {
                    openAndActivate("memory")
                }
                switch state {
                case .recording, .waiting:
                    Menu {
                        ForEach(MemoryPauseOption.allCases) { option in
                            Button(option.title) { Task { await memory.pause(option) } }
                        }
                    } label: {
                        Text("Pause Memory")
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                case .paused:
                    Button("Resume Memory") { Task { await memory.resume() } }
                        .buttonStyle(.plain)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.blue)
                case .off:
                    Button("Turn On\u{2026}") { openAndActivate("memory-settings") }
                        .buttonStyle(.plain)
                        .font(.system(size: 12, weight: .medium))
                        .foregroundColor(.blue)
                case .unavailable:
                    EmptyView()
                }
                if state.isOn || state.isPaused {
                    Menu {
                        Button("Forget Last Hour\u{2026}") { Task { await memory.forget(.lastHour) } }
                        Button("Forget Today\u{2026}") { Task { await memory.forget(.today) } }
                    } label: {
                        Text("Forget")
                    }
                    .menuStyle(.borderlessButton)
                    .fixedSize()
                }
            }

            if let notice = memory.notice {
                Text(notice)
                    .font(.system(size: 10))
                    .foregroundColor(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Connected Agents

    private var agentsSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Agents")
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundColor(.secondary)
                .padding(.leading, 2)

            ForEach(agentDetector.agents) { agent in
                HStack(spacing: 8) {
                    Image(systemName: agent.icon)
                        .font(.system(size: 11))
                        .foregroundColor(agent.isConnected ? .green : .secondary)
                        .frame(width: 16)
                    Text(agent.name)
                        .font(.system(size: 12))
                    Spacer()
                    if agent.isBusy {
                        ProgressView()
                            .controlSize(.mini)
                    } else if agent.isConnected {
                        Text("Connected")
                            .font(.system(size: 10))
                            .foregroundColor(.green)
                    } else {
                        if let failure = agent.failure {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .font(.system(size: 10))
                                .foregroundColor(.orange)
                                .help(failure)
                        }
                        Button("Connect") {
                            agentDetector.connect(agent)
                        }
                        .font(.system(size: 10, weight: .medium))
                        .buttonStyle(.plain)
                        .foregroundColor(.blue)
                    }
                }
                .padding(.vertical, 2)
            }
        }
        .padding(.top, 4)
    }

    // MARK: - Footer

    private var footerSection: some View {
        HStack(spacing: 0) {
            // Service toggle — based on user intent, not process state.
            // Daemon may be running for focus processing even when paused.
            // Start/stop run off the main thread (ObservationToggle); while
            // one is in flight the button shows it and is disabled.
            if let phase = observationToggle.phase {
                HStack(spacing: 4) {
                    ProgressView()
                        .controlSize(.mini)
                    Text(phase.label)
                        .font(.system(size: 11))
                }
                .foregroundColor(.secondary)
            } else if !appState.userStopped {
                Button(action: {
                    appState.userStopped = true
                    UserDefaults.standard.set(true, forKey: "observingPaused")
                    observationToggle.stop { appState.refreshStatus() }
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: "eye.slash.fill")
                            .font(.system(size: 8))
                        Text("Stop Observing")
                            .font(.system(size: 11))
                    }
                    .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
            } else if !appState.accessibilityGranted || !appState.screenRecordingGranted {
                Button(action: { repairPermissions() }) {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.system(size: 8))
                        Text("Fix Setup")
                            .font(.system(size: 11))
                    }
                    .foregroundColor(warmOrange)
                }
                .buttonStyle(.plain)
            } else {
                Button(action: {
                    appState.userStopped = false
                    UserDefaults.standard.set(false, forKey: "observingPaused")
                    appState.servicesStarting()
                    observationToggle.start { _ in appState.refreshStatus() }
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: "eye.fill")
                            .font(.system(size: 8))
                        Text("Observe Me")
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

            Button("Update") {
                delegate.updaterController.checkForUpdates(nil)
            }
            .font(.system(size: 11))
            .foregroundColor(.secondary)
            .buttonStyle(.plain)
            .padding(.trailing, 8)

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

    private var isRecording: Bool {
        if case .recording = appState.focusState { return true }
        return false
    }

    private var recordingTitle: String {
        if case .recording(_, let title, _) = appState.focusState { return title }
        return ""
    }

    private var processingTitle: String? {
        if case .processing(let title) = appState.focusState { return title }
        return nil
    }

    private var pulsingOpacity: Double {
        isRecording ? 0.3 : 1.0
    }

    private var formattedElapsed: String {
        guard case .recording(_, _, let startedAt?) = appState.focusState else {
            return "0:00"
        }
        let elapsed = max(0, Int(now.timeIntervalSince(startedAt)))
        return String(format: "%d:%02d", elapsed / 60, elapsed % 60)
    }

    private var recordPermissionMessage: String {
        if !appState.accessibilityGranted && !appState.screenRecordingGranted {
            return "Accessibility and Screen Recording are required"
        }
        if !appState.accessibilityGranted {
            return "Accessibility required for workflow capture"
        }
        return "Screen Recording required for workflow capture"
    }

    private func repairPermissions() {
        if !appState.accessibilityGranted {
            requestAccessibilityIfNeeded()
        } else if !appState.screenRecordingGranted {
            requestScreenRecordingIfNeeded()
        } else if !hasCompletedOnboarding {
            delegate.showOnboarding()
        }
    }

    private func requestAccessibilityIfNeeded() {
        Task {
            let granted = await PermissionChecker.requestAccessibilityAndOpenSettingsIfNeeded()
            appState.accessibilityGranted = granted
            openedPermissionSettings = granted ? nil : .accessibility
        }
    }

    /// Request Screen Recording from the app principal, then open Settings if
    /// the user explicitly asks for a manual fallback.
    private func requestScreenRecordingIfNeeded() {
        Task {
            if screenRecordingNeedsManualSettings {
                ServiceController.prepareForAppOwnedPermissionRequest()
                await PermissionChecker.registerAndOpenScreenRecordingSettings()
                openedPermissionSettings = .screenRecording
                return
            }

            let granted = await PermissionChecker.requestScreenRecordingAndOpenSettingsIfNeeded()
            if granted {
                let status = await PermissionChecker.resolveScreenRecordingStatus(
                    timeoutNanoseconds: 1_500_000_000
                )
                appState.screenRecordingGranted = status.granted
                screenRecordingNeedsManualSettings = !status.granted
                openedPermissionSettings = status.granted ? nil : .screenRecording
            } else {
                appState.screenRecordingGranted = false
                screenRecordingNeedsManualSettings = true
                openedPermissionSettings = .screenRecording
            }
        }
    }

    private func refreshScreenRecordingGrant() async -> Bool {
        let status = await PermissionChecker.resolveScreenRecordingStatus(
            timeoutNanoseconds: 5_000_000_000
        )
        return status.granted
    }

    private func startFocusSession(title: String) {
        // Remember user's paused state so we restore it after recording
        wasPausedBeforeFocus = appState.userStopped
        focusStartError = nil

        // Write focus signal BEFORE starting daemon so it picks up the
        // session on its first event loop iteration — no missed events.
        let signal: FocusSessionSignalFile
        do {
            signal = try FocusSignal.start(
                title: title.trimmingCharacters(in: .whitespacesAndNewlines),
                in: appState.statusDir
            )
        } catch {
            focusStartError = "Could not start the recording: \(error.localizedDescription)"
            return
        }
        appState.readFocusSession()
        now = Date()
        showTitlePrompt = false
        newSessionTitle = ""

        // Always ensure daemon is running (startDaemon is a no-op if already alive)
        // Then resume capture in case daemon was paused.
        let dir = appState.statusDir
        DispatchQueue.global(qos: .userInitiated).async {
            guard ServiceController.startDaemon() else {
                // Nothing would be captured: take the recording back and
                // say so, instead of a timer counting an empty session.
                DispatchQueue.main.async {
                    FocusSignal.discard(sessionID: signal.session_id, in: dir)
                    appState.readFocusSession()
                    newSessionTitle = signal.title
                    showTitlePrompt = true
                    focusStartError = FocusSignal.daemonDidNotStart
                    appState.refreshStatus()
                }
                return
            }
            // Give the daemon a moment to bind its control socket
            Thread.sleep(forTimeInterval: 0.5)
            // Resume capture so the observer loop emits events during recording,
            // even if the user was previously paused.
            let client = CaptureAgentClient()
            Task { @MainActor in
                let _ = await client.resumeCapture()
            }
        }
    }

    /// Stops the recording session, whether the app or the CLI started it:
    /// the signal keeps its session id, title and start time.
    private func stopFocusSession() {
        do {
            guard try FocusSignal.stop(in: appState.statusDir) != nil else {
                appState.readFocusSession()
                return
            }
        } catch {
            print("Failed to write focus-session.json: \(error)")
            return
        }
        // Shows "Analyzing..." immediately so the user knows it's working.
        appState.readFocusSession()

        // Start worker to process the session.
        // If user was paused before recording, send pause_capture to the
        // daemon so it stays alive (future recordings need it) but stops
        // emitting passive events.  If user was observing, keep capturing.
        let restorePaused = wasPausedBeforeFocus
        wasPausedBeforeFocus = false
        DispatchQueue.global(qos: .userInitiated).async {
            ServiceController.startWorker()

            if restorePaused {
                // User was paused — tell daemon to pause capture instead
                // of killing it.  The daemon stays alive for future focus
                // recordings but emits zero passive events.
                let client = CaptureAgentClient()
                Task { @MainActor in
                    let _ = await client.pauseCapture()
                }
                DispatchQueue.main.async {
                    self.appState.userStopped = true
                    UserDefaults.standard.set(true, forKey: "observingPaused")
                }
            }
        }
    }

    private static let windowIdToTitle: [String: String] = [
        "workflows": "Workflows",
        "daily-digest": "Daily Digest",
        "micro-review": "Review Queue",
        "faq": "FAQ",
        "memory": "Commonplace",
        "memory-settings": "Memory Settings",
    ]

    private func openAndActivate(_ windowId: String) {
        NSApp.setActivationPolicy(.regular)
        openWindow(id: windowId)
        let targetTitle = Self.windowIdToTitle[windowId]
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            NSApp.activate(ignoringOtherApps: true)
            // Bring the specific requested window to front, not just any app window
            if let targetTitle {
                for window in NSApp.windows where window.title == targetTitle {
                    window.makeKeyAndOrderFront(nil)
                    window.orderFrontRegardless()
                    return
                }
            }
            // Fallback: bring any known window to front
            let knownTitles: Set<String> = Set(Self.windowIdToTitle.values)
            for window in NSApp.windows where knownTitles.contains(window.title) {
                window.makeKeyAndOrderFront(nil)
                window.orderFrontRegardless()
                break
            }
        }
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
