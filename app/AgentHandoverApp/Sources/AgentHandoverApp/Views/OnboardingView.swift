import SwiftUI

/// Step-by-step onboarding for first-run setup.
///
/// 7 steps: Welcome → How It Works → Permissions → AI Model → Browser Extension →
/// Ready (First Recording)
struct OnboardingView: View {
    @EnvironmentObject var appState: AppState
    @State private var currentStep = 0
    @State private var extensionPath: String = ""
    @State private var chromeOpenError: String? = nil
    @State private var vlmPullInProgress = false
    @State private var vlmPullOutput = ""
    @State private var serviceStartFailed = false

    // Cloud VLM state
    enum VLMMode: String, CaseIterable {
        case local = "Local"
        case cloud = "Cloud"
    }

    enum RemoteProvider: String, CaseIterable, Identifiable {
        case openai = "openai"
        case anthropic = "anthropic"
        case google = "google"

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .openai: return "OpenAI"
            case .anthropic: return "Anthropic (Claude)"
            case .google: return "Google (Gemini)"
            }
        }

        var defaultModel: String {
            switch self {
            case .openai: return "gpt-4.1-mini"
            case .anthropic: return "claude-sonnet-4-6-20260320"
            case .google: return "gemini-2.5-flash"
            }
        }

        var envVar: String {
            switch self {
            case .openai: return "OPENAI_API_KEY"
            case .anthropic: return "ANTHROPIC_API_KEY"
            case .google: return "GOOGLE_API_KEY"
            }
        }
    }

    @State private var vlmMode: VLMMode = .local
    @State private var selectedProvider: RemoteProvider = .openai
    @State private var apiKeyInput: String = ""
    @State private var customModelName: String = ""
    @State private var apiKeyValidating = false
    @State private var apiKeyValid: Bool? = nil
    @State private var remoteConsentGiven = false

    // Focus recording from onboarding
    @State private var firstRecordingTitle: String = ""

    /// Called when onboarding completes (sets hasCompletedOnboarding).
    var onComplete: (() -> Void)?

    private let totalSteps = 7

    var body: some View {
        VStack(spacing: 0) {
            // Progress dots
            HStack(spacing: 8) {
                ForEach(0..<totalSteps, id: \.self) { index in
                    Circle()
                        .fill(index <= currentStep ? Color.accentColor : Color.secondary.opacity(0.3))
                        .frame(width: 8, height: 8)
                }
            }
            .padding(.top, 20)

            Spacer()

            // Current step content
            stepContent(for: currentStep)
                .padding(.horizontal, 40)

            Spacer()

            // Navigation
            navigationBar
                .padding(.horizontal, 40)
                .padding(.bottom, 24)
        }
        .onAppear {
            resolveExtensionPath()
        }
    }

    // MARK: - Navigation Bar

    private var navigationBar: some View {
        HStack {
            if currentStep > 0 {
                Button("Back") {
                    withAnimation { currentStep -= 1 }
                }
            }

            Spacer()

            switch currentStep {
            case 0:
                // Welcome — single CTA
                Button("Get Started") {
                    withAnimation { currentStep += 1 }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

            case 1:
                // How it works
                Button("Next -- Let's set up") {
                    withAnimation { currentStep += 1 }
                }
                .buttonStyle(.borderedProminent)

            case 2:
                // Permissions — blocked until both granted, with skip option
                VStack(spacing: 4) {
                    Button("Next") {
                        withAnimation { currentStep += 1 }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!appState.accessibilityGranted || !appState.screenRecordingGranted)

                    if !appState.accessibilityGranted || !appState.screenRecordingGranted {
                        Button("Skip for now") {
                            withAnimation { currentStep += 1 }
                        }
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .buttonStyle(.plain)
                    }
                }

            case 3:
                // VLM Setup — blocked until model ready
                VStack(spacing: 2) {
                    Button("Next") {
                        withAnimation { currentStep += 1 }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!appState.vlmAvailable)

                    if !appState.vlmAvailable {
                        Text("Set up an AI model above to continue")
                            .font(.caption2)
                            .foregroundColor(.orange)
                    }
                }

            case 4:
                // Browser extension — optional
                HStack(spacing: 12) {
                    Button("Skip for now") {
                        withAnimation { currentStep += 1 }
                    }
                    .foregroundColor(.secondary)
                    .buttonStyle(.plain)
                    .font(.caption)

                    Button("Next") {
                        withAnimation { currentStep += 1 }
                    }
                    .buttonStyle(.borderedProminent)
                }

            case 5:
                // Summary — next to final
                Button("Next") {
                    withAnimation { currentStep += 1 }
                }
                .buttonStyle(.borderedProminent)

            case 6:
                // Ready — final step with actions
                EmptyView()

            default:
                EmptyView()
            }
        }
    }

    // MARK: - Step Content

    @ViewBuilder
    private func stepContent(for step: Int) -> some View {
        switch step {
        case 0: welcomeStep
        case 1: howItWorksStep
        case 2: permissionsStep
        case 3: vlmSetupStep
        case 4: chromeExtensionStep
        case 5: summaryStep
        case 6: readyStep
        default: EmptyView()
        }
    }

    // MARK: - Step 0: Welcome

    private var welcomeStep: some View {
        VStack(spacing: 20) {
            Image(systemName: "binoculars.fill")
                .font(.system(size: 56))
                .foregroundStyle(
                    LinearGradient(
                        colors: [.orange, .yellow],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .padding(.bottom, 4)

            Text("AgentHandover")
                .font(.largeTitle)
                .fontWeight(.bold)

            Text("Your work, turned into agent instructions")
                .font(.title3)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)

            VStack(alignment: .leading, spacing: 14) {
                valueBullet(
                    icon: "magnifyingglass",
                    color: .blue,
                    text: "Silently watches your screen as you work"
                )
                valueBullet(
                    icon: "brain.head.profile",
                    color: .purple,
                    text: "Learns your repeatable workflows automatically"
                )
                valueBullet(
                    icon: "list.clipboard",
                    color: .green,
                    text: "Produces step-by-step procedures agents can follow"
                )
            }
            .padding(.vertical, 8)

            HStack(spacing: 6) {
                Image(systemName: "lock.shield.fill")
                    .foregroundColor(.green)
                    .font(.caption)
                Text("Everything runs locally on your Mac. Nothing leaves your machine.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.green.opacity(0.08))
            )
        }
    }

    private func valueBullet(icon: String, color: Color, text: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundColor(color)
                .frame(width: 28)
            Text(text)
                .font(.body)
        }
    }

    // MARK: - Step 1: How It Works

    private var howItWorksStep: some View {
        VStack(spacing: 20) {
            Text("Two ways to teach")
                .font(.title2)
                .fontWeight(.semibold)

            HStack(spacing: 16) {
                // Focus Recording card
                modeCard(
                    icon: "record.circle",
                    iconColor: .red,
                    title: "Focus Recording",
                    badge: "Start here",
                    badgeColor: .orange,
                    bullets: [
                        "Record a specific task",
                        "Click Record \u{2192} do the workflow \u{2192} Stop",
                        "Get a procedure in ~60 seconds",
                        "Best for: your most important 5-10 tasks",
                    ]
                )

                // Passive Discovery card
                modeCard(
                    icon: "eye",
                    iconColor: .blue,
                    title: "Passive Discovery",
                    badge: nil,
                    badgeColor: .clear,
                    bullets: [
                        "Learns automatically in the background",
                        "Detects patterns when you repeat workflows",
                        "Gets smarter over time with more observations",
                        "Best for: discovering workflows you didn't think to record",
                    ]
                )
            }

            Text("We recommend starting with Focus Recording to see results immediately.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 460)
        }
    }

    private func modeCard(
        icon: String,
        iconColor: Color,
        title: String,
        badge: String?,
        badgeColor: Color,
        bullets: [String]
    ) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: icon)
                    .font(.title2)
                    .foregroundColor(iconColor)
                Spacer()
                if let badge = badge {
                    Text(badge)
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            RoundedRectangle(cornerRadius: 4)
                                .fill(badgeColor.opacity(0.15))
                        )
                        .foregroundColor(badgeColor)
                }
            }

            Text(title)
                .font(.headline)
                .fontWeight(.semibold)

            VStack(alignment: .leading, spacing: 6) {
                ForEach(bullets, id: \.self) { bullet in
                    HStack(alignment: .top, spacing: 6) {
                        Text("\u{2022}")
                            .foregroundColor(.secondary)
                            .font(.caption)
                        Text(bullet)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.secondary.opacity(0.12), lineWidth: 1)
        )
    }

    // MARK: - Step 2: Permissions (Combined)

    private var permissionsStep: some View {
        VStack(spacing: 20) {
            Text("Two permissions needed")
                .font(.title2)
                .fontWeight(.semibold)

            HStack(spacing: 16) {
                // Accessibility card
                permissionCard(
                    icon: "hand.raised.circle.fill",
                    title: "Accessibility",
                    granted: appState.accessibilityGranted,
                    action: {
                        PermissionChecker.requestAccessibility()
                    },
                    actionLabel: "Grant Access"
                )

                // Screen Recording card
                permissionCard(
                    icon: "rectangle.dashed.badge.record",
                    title: "Screen Recording",
                    granted: appState.screenRecordingGranted,
                    action: {
                        PermissionChecker.openScreenRecordingSettings()
                    },
                    actionLabel: "Open Settings"
                )
            }

            Text("These let AgentHandover see what's on your screen. It reads window titles and takes screenshots -- never types or clicks anything.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 460)
        }
    }

    private func permissionCard(
        icon: String,
        title: String,
        granted: Bool,
        action: @escaping () -> Void,
        actionLabel: String
    ) -> some View {
        VStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 32))
                .foregroundColor(granted ? .green : .accentColor)

            Text(title)
                .font(.headline)

            if granted {
                HStack(spacing: 4) {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundColor(.green)
                    Text("Granted")
                        .foregroundColor(.green)
                }
                .font(.caption)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color.green.opacity(0.1))
                )
            } else {
                Button(actionLabel) {
                    action()
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.06))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(granted ? Color.green.opacity(0.3) : Color.secondary.opacity(0.12), lineWidth: 1)
        )
    }

    // MARK: - Step 3: VLM Setup (Required)

    private var vlmSetupStep: some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 48))
                .foregroundColor(.orange)

            Text("Set up the AI brain")
                .font(.title2)
                .fontWeight(.semibold)

            Text("AgentHandover uses a local AI model to understand what's on your screen. This runs entirely on your Mac.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)

            if appState.vlmAvailable {
                PermissionStatusBadge(
                    granted: true,
                    grantedLabel: "AI Model Ready",
                    deniedLabel: ""
                )
            } else {
                // Local / Cloud toggle
                Picker("Mode", selection: $vlmMode) {
                    ForEach(VLMMode.allCases, id: \.self) { mode in
                        Text(mode.rawValue).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 240)

                if vlmMode == .cloud {
                    cloudVLMContent
                } else {
                    localVLMContent
                }
            }
        }
    }

    // MARK: - Local VLM Content

    private var localVLMContent: some View {
        VStack(spacing: 8) {
            let ollamaInstalled = isOllamaInstalled()

            if ollamaInstalled {
                PermissionStatusBadge(
                    granted: true,
                    grantedLabel: "Ollama Installed",
                    deniedLabel: ""
                )

                if vlmPullInProgress {
                    VStack(spacing: 4) {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .controlSize(.small)
                        Text("Pulling models...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        if !vlmPullOutput.isEmpty {
                            Text(vlmPullOutput)
                                .font(.caption2)
                                .fontDesign(.monospaced)
                                .foregroundColor(.secondary)
                                .lineLimit(2)
                        }
                    }
                } else {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Recommended models (~6 GB total):")
                            .font(.caption)
                            .fontWeight(.semibold)

                        VStack(alignment: .leading, spacing: 4) {
                            modelRow("qwen3.5:2b", "2.7 GB", "Screen annotation -- reads your screen and describes what you're doing")
                            modelRow("qwen3.5:4b", "3.4 GB", "SOP generation -- writes step-by-step procedures from observations")
                            modelRow("all-minilm:l6-v2", "45 MB", "Task matching -- groups similar work together")
                        }

                        Button("Pull All Recommended Models") {
                            pullOllamaModel()
                        }
                        .buttonStyle(.borderedProminent)

                        Text("Or use any Ollama-compatible model -- edit annotation_model and sop_model in config.toml after setup.")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .frame(maxWidth: 380)
                    }
                    .frame(maxWidth: 440)
                }
            } else {
                Text("Ollama not installed")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Button("Download Ollama") {
                    if let url = URL(string: "https://ollama.com/download/mac") {
                        NSWorkspace.shared.open(url)
                    }
                }
                .buttonStyle(.bordered)

                Text("Or install via: brew install ollama")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Cloud VLM Content

    private var cloudVLMContent: some View {
        VStack(spacing: 12) {
            // Privacy consent
            if !remoteConsentGiven {
                VStack(spacing: 8) {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("Privacy Notice")
                            .font(.caption)
                            .fontWeight(.semibold)
                    }
                    Text("Cloud VLM sends screenshots of your desktop to a third-party API for analysis. Only enable this if you accept this trade-off.")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 350)

                    Button("I Understand & Consent") {
                        remoteConsentGiven = true
                    }
                    .buttonStyle(.bordered)
                }
            } else {
                // Provider picker
                Picker("Provider", selection: $selectedProvider) {
                    ForEach(RemoteProvider.allCases) { provider in
                        Text(provider.displayName).tag(provider)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: 280)
                .onChange(of: selectedProvider) { _ in
                    customModelName = ""
                }

                // Model selection
                HStack(spacing: 8) {
                    Text("Model:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    TextField("Model name", text: $customModelName)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 220)
                }

                Text("Default: \(selectedProvider.defaultModel)")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                // API Key input
                SecureField("API Key", text: $apiKeyInput)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 300)

                Text("Stored securely in macOS Keychain")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                // Save & Test button
                HStack(spacing: 8) {
                    Button("Save Configuration") {
                        saveCloudVLMConfig()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(apiKeyInput.count < 10)

                    if apiKeyValidating {
                        ProgressView()
                            .controlSize(.small)
                    }
                }

                if let valid = apiKeyValid {
                    PermissionStatusBadge(
                        granted: valid,
                        grantedLabel: "Configuration Saved",
                        deniedLabel: "Failed to save"
                    )
                }
            }
        }
    }

    // MARK: - Cloud VLM Config Save

    private func saveCloudVLMConfig() {
        apiKeyValidating = true

        DispatchQueue.global(qos: .userInitiated).async {
            // Store key in Keychain
            let stored = KeychainHelper.store(
                key: "agenthandover-\(selectedProvider.rawValue)-key",
                value: apiKeyInput
            )

            // Write config.toml update
            if stored {
                writeRemoteVLMConfig(
                    provider: selectedProvider.rawValue,
                    model: customModelName.isEmpty ? selectedProvider.defaultModel : customModelName,
                    apiKeyEnv: selectedProvider.envVar
                )
            }

            DispatchQueue.main.async {
                apiKeyValidating = false
                apiKeyValid = stored
                if stored && !apiKeyInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    appState.vlmAvailable = true
                }
            }
        }
    }

    private func writeRemoteVLMConfig(provider: String, model: String, apiKeyEnv: String) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let configDir = home
            .appendingPathComponent("Library/Application Support/agenthandover")
        let configPath = configDir.appendingPathComponent("config.toml")

        try? FileManager.default.createDirectory(at: configDir, withIntermediateDirectories: true)

        // Read existing config or start fresh
        var content = (try? String(contentsOf: configPath, encoding: .utf8)) ?? ""

        // Strip any existing remote-mode keys from [vlm] section to avoid
        // duplicates on repeated saves.  Then insert the new values.
        let remoteKeys = ["mode", "provider", "model", "api_key_env"]
        for key in remoteKeys {
            // Match lines like: mode = "remote"  or  provider = "openai"
            // (with optional leading whitespace and any quoted value)
            let pattern = "(?m)^[ \\t]*\(key)[ \\t]*=[ \\t]*\"[^\"]*\"[ \\t]*\\n?"
            if let regex = try? NSRegularExpression(pattern: pattern) {
                let range = NSRange(content.startIndex..., in: content)
                content = regex.stringByReplacingMatches(
                    in: content, range: range, withTemplate: ""
                )
            }
        }

        let newFields = "mode = \"remote\"\nprovider = \"\(provider)\"\nmodel = \"\(model)\"\napi_key_env = \"\(apiKeyEnv)\"\n"

        if content.contains("[vlm]") {
            if let vlmRange = content.range(of: "[vlm]") {
                let afterVlm = content[vlmRange.upperBound...]
                if let nextSection = afterVlm.range(of: "\n[") {
                    content.insert(contentsOf: "\n" + newFields, at: nextSection.lowerBound)
                } else {
                    content += "\n" + newFields
                }
            }
        } else {
            content += "\n[vlm]\n" + newFields
        }

        try? content.write(to: configPath, atomically: true, encoding: .utf8)
    }

    // MARK: - Step 4: Chrome Extension (Optional)

    private var chromeExtensionStep: some View {
        VStack(spacing: 16) {
            HStack(spacing: 6) {
                Image(systemName: "globe.badge.chevron.backward")
                    .font(.system(size: 48))
                    .foregroundColor(.accentColor)
            }

            Text("Supercharge with browser context")
                .font(.title2)
                .fontWeight(.semibold)

            HStack(spacing: 6) {
                Text("Optional")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.secondary.opacity(0.15))
                    )
                    .foregroundColor(.secondary)
            }

            Text("The Chrome extension adds CSS selectors, form fields, and page structure to your procedures -- making them more precise for browser automation.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)

            // Connection status
            if appState.extensionConnected {
                PermissionStatusBadge(
                    granted: true,
                    grantedLabel: "Extension Connected",
                    deniedLabel: ""
                )
            } else {
                // Show extension path
                if !extensionPath.isEmpty {
                    VStack(spacing: 8) {
                        Text("Extension location:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(extensionPath)
                            .font(.caption)
                            .fontDesign(.monospaced)
                            .textSelection(.enabled)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(
                                RoundedRectangle(cornerRadius: 4)
                                    .fill(Color.secondary.opacity(0.1))
                            )
                    }

                    Button("Copy Path & Open Chrome") {
                        copyPathAndOpenChrome()
                    }
                    .buttonStyle(.bordered)

                    if let error = chromeOpenError {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        instructionRow(number: "1", text: "Enable Developer Mode (top-right toggle)")
                        instructionRow(number: "2", text: "Click \"Load Unpacked\"")
                        instructionRow(number: "3", text: "Paste path (Cmd+V) and click Select")
                    }
                    .font(.caption)
                    .foregroundColor(.secondary)
                } else {
                    Text("Extension files not found. Install via:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text("brew install --HEAD agenthandover")
                        .font(.caption)
                        .fontDesign(.monospaced)
                        .textSelection(.enabled)
                }
            }
        }
    }

    // MARK: - Step 5: Summary

    private var summaryStep: some View {
        VStack(spacing: 20) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 48))
                .foregroundStyle(
                    LinearGradient(
                        colors: [.green, .mint],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )

            Text("Setup complete")
                .font(.title2)
                .fontWeight(.semibold)

            VStack(spacing: 8) {
                summaryRow(label: "Accessibility", ok: appState.accessibilityGranted)
                summaryRow(label: "Screen Recording", ok: appState.screenRecordingGranted)
                summaryRow(label: "AI Model", ok: appState.vlmAvailable)
                summaryRow(label: "Chrome Extension", ok: appState.extensionConnected, optional: true)
            }
            .padding(.horizontal, 20)
        }
    }

    // MARK: - Step 6: Ready — First Recording

    private var readyStep: some View {
        VStack(spacing: 20) {
            Text("You're all set!")
                .font(.title2)
                .fontWeight(.bold)

            // Primary: Record first workflow
            VStack(spacing: 14) {
                Image(systemName: "record.circle")
                    .font(.system(size: 28))
                    .foregroundColor(.red)

                Text("Record your first workflow")
                    .font(.headline)

                Text("What's something you do regularly? Type a name and we'll record you doing it.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 340)

                TextField("e.g. \"Process expense report\"", text: $firstRecordingTitle)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 320)

                Button {
                    startServicesAndRecord()
                } label: {
                    HStack(spacing: 6) {
                        Image(systemName: "record.circle")
                        Text("Start Recording")
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(.red)
                .controlSize(.large)
                .disabled(
                    firstRecordingTitle.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || !appState.accessibilityGranted
                    || !appState.vlmAvailable
                )
            }
            .padding(20)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color.secondary.opacity(0.06))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(Color.secondary.opacity(0.12), lineWidth: 1)
            )

            // Secondary: Just start observing
            VStack(spacing: 6) {
                Text("or")
                    .font(.caption)
                    .foregroundColor(.secondary)

                Button("Just start observing") {
                    startServicesOnly()
                }
                .foregroundColor(.accentColor)
                .buttonStyle(.plain)
                .font(.callout)
            }

            if serviceStartFailed {
                Text("Services may not have started. Check agenthandover status in Terminal.")
                    .font(.caption2)
                    .foregroundColor(.red)
            } else if !appState.accessibilityGranted {
                Text("Accessibility permission is required (go back to step 3)")
                    .font(.caption2)
                    .foregroundColor(.orange)
            } else if !appState.vlmAvailable {
                Text("An AI model must be configured (go back to step 4)")
                    .font(.caption2)
                    .foregroundColor(.orange)
            }

            HStack(spacing: 4) {
                Text("AgentHandover will run quietly in your menu bar")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Image(systemName: "arrow.up.right")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: - Actions

    private func startServicesOnly() {
        let ok = ServiceController.startAll()
        if ok {
            onComplete?()
            NSApplication.shared.keyWindow?.close()
        } else {
            serviceStartFailed = true
        }
    }

    private func startServicesAndRecord() {
        let ok = ServiceController.startAll()
        if ok {
            // Write focus-session.json to trigger a recording
            let sessionId = UUID().uuidString
            let signal: [String: Any] = [
                "session_id": sessionId,
                "title": firstRecordingTitle.trimmingCharacters(in: .whitespacesAndNewlines),
                "started_at": ISO8601DateFormatter().string(from: Date()),
                "status": "recording",
            ]
            writeFocusSignalFile(signal)
            onComplete?()
            NSApplication.shared.keyWindow?.close()
        } else {
            serviceStartFailed = true
        }
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
            // Silently fail — the recording won't start but services are already running.
            // The user can start a recording from the menu bar.
        }
    }

    // MARK: - Helpers

    private func resolveExtensionPath() {
        // 1. Check installed paths (pkg + Homebrew opt symlinks)
        let installedPaths: [String] = [
            "/usr/local/lib/agenthandover/extension",
            "/usr/local/opt/agenthandover/libexec/extension",
            "/opt/homebrew/opt/agenthandover/libexec/extension",
        ]

        for candidate in installedPaths {
            let manifestPath = (candidate as NSString).appendingPathComponent("manifest.json")
            if FileManager.default.fileExists(atPath: manifestPath) {
                extensionPath = candidate
                return
            }
        }

        // 2. Resolve Homebrew Cellar path by following the CLI binary symlink.
        //    This covers cases where the opt symlink is broken or not yet created.
        //    Pattern: /usr/local/bin/agenthandover -> .../Cellar/agenthandover/HEAD-xxx/bin/agenthandover
        //             -> .../Cellar/agenthandover/HEAD-xxx/libexec/extension/
        let cliBinaryPaths = ["/usr/local/bin/agenthandover", "/opt/homebrew/bin/agenthandover"]
        for binaryPath in cliBinaryPaths {
            let url = URL(fileURLWithPath: binaryPath)
            let resolved = url.resolvingSymlinksInPath().path
                .components(separatedBy: "/")
            if !resolved.isEmpty {
                if let binIdx = resolved.lastIndex(of: "bin") {
                    let prefix = resolved[..<binIdx].joined(separator: "/")
                    let cellarExt = prefix + "/libexec/extension"
                    if FileManager.default.fileExists(atPath: cellarExt + "/manifest.json") {
                        extensionPath = cellarExt
                        return
                    }
                }
            }
        }

        // 3. Check for dev/source build by looking for the agenthandover CLI binary
        //    and walking ancestors to find extension/dist relative to the repo root.
        if let cliPath = findCLIBinary() {
            var dir = URL(fileURLWithPath: cliPath).deletingLastPathComponent()
            for _ in 0..<6 {
                let candidate = dir.appendingPathComponent("extension/dist").path
                if FileManager.default.fileExists(atPath: candidate) {
                    extensionPath = candidate
                    return
                }
                dir = dir.deletingLastPathComponent()
            }
        }
    }

    /// Find the agenthandover CLI binary on the system.
    private func findCLIBinary() -> String? {
        let knownPaths = [
            "/usr/local/bin/agenthandover",
            "/opt/homebrew/bin/agenthandover",
        ]
        for path in knownPaths {
            if FileManager.default.fileExists(atPath: path) {
                // Resolve symlinks to find the real location
                return URL(fileURLWithPath: path).resolvingSymlinksInPath().path
            }
        }
        return nil
    }

    private func copyPathAndOpenChrome() {
        chromeOpenError = nil

        // Copy extension path to clipboard
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        let copied = pasteboard.setString(extensionPath, forType: .string)
        if !copied {
            chromeOpenError = "Failed to copy path to clipboard."
            return
        }

        // Open chrome://extensions via /usr/bin/open -a "Google Chrome"
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = ["-a", "Google Chrome", "chrome://extensions"]
        do {
            try process.run()
            process.waitUntilExit()
            if process.terminationStatus != 0 {
                chromeOpenError = "Could not open Chrome. Is Google Chrome installed?"
            }
        } catch {
            chromeOpenError = "Could not open Chrome: \(error.localizedDescription)"
        }
    }

    private func isOllamaInstalled() -> Bool {
        let paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
        ]
        return paths.contains { FileManager.default.fileExists(atPath: $0) }
    }

    private func findOllamaPath() -> String? {
        let paths = [
            "/usr/local/bin/ollama",
            "/opt/homebrew/bin/ollama",
        ]
        return paths.first { FileManager.default.fileExists(atPath: $0) }
    }

    private func modelRow(_ name: String, _ size: String, _ description: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("\u{2022}")
                .foregroundColor(.orange)
                .font(.caption)
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text(name)
                        .font(.caption)
                        .fontWeight(.medium)
                        .fontDesign(.monospaced)
                    Text(size)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                Text(description)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
        }
    }

    private func pullOllamaModel() {
        guard let ollamaPath = findOllamaPath() else { return }

        vlmPullInProgress = true
        vlmPullOutput = "Starting download..."

        let models = [
            ("qwen3.5:2b", "scene annotation"),
            ("qwen3.5:4b", "SOP generation"),
            ("all-minilm:l6-v2", "embeddings"),
        ]

        DispatchQueue.global(qos: .userInitiated).async {
            for (index, (model, purpose)) in models.enumerated() {
                DispatchQueue.main.async {
                    vlmPullOutput = "[\(index + 1)/\(models.count)] Pulling \(model) (\(purpose))..."
                }

                let process = Process()
                process.executableURL = URL(fileURLWithPath: ollamaPath)
                process.arguments = ["pull", model]

                let pipe = Pipe()
                process.standardOutput = pipe
                process.standardError = pipe

                do {
                    try process.run()

                    // Read output asynchronously
                    pipe.fileHandleForReading.readabilityHandler = { handle in
                        let data = handle.availableData
                        if !data.isEmpty, let output = String(data: data, encoding: .utf8) {
                            let lastLine = output.components(separatedBy: "\n")
                                .filter { !$0.isEmpty }
                                .last ?? ""
                            DispatchQueue.main.async {
                                vlmPullOutput = "[\(index + 1)/\(models.count)] \(model): \(String(lastLine.prefix(60)))"
                            }
                        }
                    }

                    process.waitUntilExit()
                    pipe.fileHandleForReading.readabilityHandler = nil

                    if process.terminationStatus != 0 {
                        DispatchQueue.main.async {
                            vlmPullInProgress = false
                            vlmPullOutput = "Failed to pull \(model). Make sure Ollama is running."
                        }
                        return
                    }
                } catch {
                    DispatchQueue.main.async {
                        vlmPullInProgress = false
                        vlmPullOutput = "Failed to run ollama: \(error.localizedDescription)"
                    }
                    return
                }
            }

            DispatchQueue.main.async {
                vlmPullInProgress = false
                vlmPullOutput = "All models downloaded successfully!"
                appState.vlmAvailable = true
            }
        }
    }

    private func instructionRow(number: String, text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text(number + ".")
                .fontWeight(.semibold)
            Text(text)
        }
    }

    private func summaryRow(label: String, ok: Bool, optional: Bool = false) -> some View {
        HStack(spacing: 6) {
            Image(systemName: ok ? "checkmark.circle.fill" : (optional ? "minus.circle" : "xmark.circle.fill"))
                .foregroundColor(ok ? .green : (optional ? .secondary : .orange))
            Text(label)
                .font(.caption)
            Spacer()
            if ok {
                Text("Ready")
                    .font(.caption2)
                    .foregroundColor(.green)
            } else if optional {
                Text("Skipped")
                    .font(.caption2)
                    .foregroundColor(.secondary)
            } else {
                Text("Not Set Up")
                    .font(.caption2)
                    .foregroundColor(.orange)
            }
        }
        .padding(.horizontal, 40)
    }
}

// MARK: - Models

enum OnboardingAction {
    case none
    case accessibility
    case screenRecording
    case chromeExtension
    case vlmSetup
}

// MARK: - Subviews

/// Simple macOS Keychain wrapper for storing/retrieving API keys.
struct KeychainHelper {
    static func store(key: String, value: String) -> Bool {
        guard let data = value.data(using: .utf8) else { return false }

        // Delete existing item first
        let deleteQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.agenthandover.app",
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(deleteQuery as CFDictionary)

        // Add new item
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.agenthandover.app",
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
        ]
        let status = SecItemAdd(addQuery as CFDictionary, nil)
        return status == errSecSuccess
    }

    static func retrieve(key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.agenthandover.app",
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        guard status == errSecSuccess,
              let data = result as? Data,
              let string = String(data: data, encoding: .utf8) else {
            return nil
        }
        return string
    }
}

struct PermissionStatusBadge: View {
    let granted: Bool
    let grantedLabel: String
    let deniedLabel: String

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: granted ? "checkmark.circle.fill" : "xmark.circle.fill")
                .foregroundColor(granted ? .green : .orange)
            Text(granted ? grantedLabel : deniedLabel)
                .font(.caption)
                .foregroundColor(granted ? .green : .orange)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 6)
                .fill((granted ? Color.green : Color.orange).opacity(0.1))
        )
    }
}
