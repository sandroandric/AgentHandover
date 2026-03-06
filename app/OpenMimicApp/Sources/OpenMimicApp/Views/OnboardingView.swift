import SwiftUI

/// Step-by-step onboarding for first-run setup.
///
/// 6 steps: Welcome → Accessibility → Screen Recording → Chrome Extension →
/// VLM Setup (optional) → Ready to Go!
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
            case .openai: return "gpt-4o-mini"
            case .anthropic: return "claude-sonnet-4-20250514"
            case .google: return "gemini-2.0-flash"
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
    @State private var apiKeyValidating = false
    @State private var apiKeyValid: Bool? = nil
    @State private var remoteConsentGiven = false

    /// Called when onboarding completes (sets hasCompletedOnboarding).
    var onComplete: (() -> Void)?

    private let totalSteps = 6

    var body: some View {
        VStack(spacing: 0) {
            // Progress indicators
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
            HStack {
                if currentStep > 0 {
                    Button("Back") {
                        withAnimation { currentStep -= 1 }
                    }
                }

                Spacer()

                if currentStep < totalSteps - 1 {
                    Button("Next") {
                        withAnimation { currentStep += 1 }
                    }
                    .buttonStyle(.borderedProminent)
                } else {
                    VStack(spacing: 4) {
                        Button("Start Observing") {
                            let ok = ServiceController.startAll()
                            if ok {
                                onComplete?()
                                NSApplication.shared.keyWindow?.close()
                            } else {
                                serviceStartFailed = true
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!appState.extensionConnected)

                        if serviceStartFailed {
                            Text("Services may not have started. Check openmimic status in Terminal.")
                                .font(.caption2)
                                .foregroundColor(.red)
                        } else if !appState.extensionConnected {
                            Text("Connect the Chrome extension first to enable observation")
                                .font(.caption2)
                                .foregroundColor(.orange)
                        }
                    }
                }
            }
            .padding(.horizontal, 40)
            .padding(.bottom, 24)
        }
        .onAppear {
            resolveExtensionPath()
        }
    }

    // MARK: - Step Content

    @ViewBuilder
    private func stepContent(for step: Int) -> some View {
        switch step {
        case 0: welcomeStep
        case 1: accessibilityStep
        case 2: screenRecordingStep
        case 3: chromeExtensionStep
        case 4: vlmSetupStep
        case 5: readyStep
        default: EmptyView()
        }
    }

    // MARK: - Step 0: Welcome

    private var welcomeStep: some View {
        VStack(spacing: 16) {
            Image(systemName: "eye.circle.fill")
                .font(.system(size: 48))
                .foregroundColor(.accentColor)

            Text("Welcome to OpenMimic")
                .font(.title2)
                .fontWeight(.semibold)

            Text("OpenMimic silently observes your workflows and generates semantic SOPs that AI agents can execute.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)
        }
    }

    // MARK: - Step 1: Accessibility

    private var accessibilityStep: some View {
        VStack(spacing: 16) {
            Image(systemName: "hand.raised.circle.fill")
                .font(.system(size: 48))
                .foregroundColor(.accentColor)

            Text("Accessibility Permission")
                .font(.title2)
                .fontWeight(.semibold)

            Text("OpenMimic needs Accessibility access to observe window titles and UI elements. This is read-only — it never takes actions.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)

            PermissionStatusBadge(
                granted: appState.accessibilityGranted,
                grantedLabel: "Accessibility Granted",
                deniedLabel: "Accessibility Not Granted"
            )

            if !appState.accessibilityGranted {
                Button("Grant Accessibility Access") {
                    PermissionChecker.requestAccessibility()
                }
                .buttonStyle(.bordered)
            }
        }
    }

    // MARK: - Step 2: Screen Recording

    private var screenRecordingStep: some View {
        VStack(spacing: 16) {
            Image(systemName: "rectangle.dashed.badge.record")
                .font(.system(size: 48))
                .foregroundColor(.accentColor)

            Text("Screen Recording Permission")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Screen Recording access allows OpenMimic to capture screenshots for visual context. Images are stored locally and encrypted.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)

            PermissionStatusBadge(
                granted: appState.screenRecordingGranted,
                grantedLabel: "Screen Recording Granted",
                deniedLabel: "Screen Recording Not Granted"
            )

            if !appState.screenRecordingGranted {
                Button("Open Screen Recording Settings") {
                    PermissionChecker.openScreenRecordingSettings()
                }
                .buttonStyle(.bordered)
            }
        }
    }

    // MARK: - Step 3: Chrome Extension (Enhanced)

    private var chromeExtensionStep: some View {
        VStack(spacing: 16) {
            Image(systemName: "globe.badge.chevron.backward")
                .font(.system(size: 48))
                .foregroundColor(.accentColor)

            Text("Chrome Extension")
                .font(.title2)
                .fontWeight(.semibold)

            Text("Install the OpenMimic Chrome extension for rich browser observation.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)

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

                    // Copy path + open Chrome button
                    Button("Copy Path & Open Chrome") {
                        copyPathAndOpenChrome()
                    }
                    .buttonStyle(.bordered)

                    if let error = chromeOpenError {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                    }

                    // Instructions
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
                    Text("brew install --HEAD openmimic")
                        .font(.caption)
                        .fontDesign(.monospaced)
                        .textSelection(.enabled)
                }
            }
        }
    }

    // MARK: - Step 4: VLM Setup (Optional)

    private var vlmSetupStep: some View {
        VStack(spacing: 16) {
            Image(systemName: "eye.trianglebadge.exclamationmark")
                .font(.system(size: 48))
                .foregroundColor(.accentColor)

            Text("VLM Setup")
                .font(.title2)
                .fontWeight(.semibold)

            HStack(spacing: 4) {
                Text("Optional")
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.blue.opacity(0.1))
                    )
                    .foregroundColor(.blue)
                Text("— enables visual understanding of native app UI")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            if appState.vlmAvailable {
                PermissionStatusBadge(
                    granted: true,
                    grantedLabel: "VLM Available",
                    deniedLabel: ""
                )
            } else {
                // Local / Cloud toggle
                Picker("VLM Mode", selection: $vlmMode) {
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
                        Text("Pulling required models...")
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
                    Button("Pull Recommended Models (qwen3.5:2b + 4b)") {
                        pullOllamaModel()
                    }
                    .buttonStyle(.bordered)
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

                Text("Default model: \(selectedProvider.defaultModel)")
                    .font(.caption2)
                    .foregroundColor(.secondary)

                // API Key input
                SecureField("API Key", text: $apiKeyInput)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 300)

                Text("Key will be stored in env var: \(selectedProvider.envVar)")
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
                key: "openmimic-\(selectedProvider.rawValue)-key",
                value: apiKeyInput
            )

            // Write config.toml update
            if stored {
                writeRemoteVLMConfig(
                    provider: selectedProvider.rawValue,
                    model: selectedProvider.defaultModel,
                    apiKeyEnv: selectedProvider.envVar
                )
            }

            DispatchQueue.main.async {
                apiKeyValidating = false
                apiKeyValid = stored
            }
        }
    }

    private func writeRemoteVLMConfig(provider: String, model: String, apiKeyEnv: String) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let configDir = home
            .appendingPathComponent("Library/Application Support/oc-apprentice")
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

    // MARK: - Step 5: Ready

    private var readyStep: some View {
        VStack(spacing: 16) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 48))
                .foregroundColor(.green)

            Text("Ready to Go!")
                .font(.title2)
                .fontWeight(.semibold)

            Text("OpenMimic will now observe your workflows in the background. Check the menu bar icon for status. SOPs appear once enough patterns are detected.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)

            // Summary badges
            VStack(spacing: 6) {
                summaryRow(label: "Accessibility", ok: appState.accessibilityGranted)
                summaryRow(label: "Screen Recording", ok: appState.screenRecordingGranted)
                summaryRow(label: "Chrome Extension", ok: appState.extensionConnected)
                summaryRow(label: "VLM", ok: appState.vlmAvailable, optional: true)
            }
        }
    }

    // MARK: - Helpers

    private func resolveExtensionPath() {
        // 1. Check installed paths (pkg + Homebrew opt symlinks)
        let installedPaths: [String] = [
            "/usr/local/lib/openmimic/extension",
            "/usr/local/opt/openmimic/libexec/extension",
            "/opt/homebrew/opt/openmimic/libexec/extension",
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
        //    Pattern: /usr/local/bin/openmimic → .../Cellar/openmimic/HEAD-xxx/bin/openmimic
        //             → .../Cellar/openmimic/HEAD-xxx/libexec/extension/
        let cliBinaryPaths = ["/usr/local/bin/openmimic", "/opt/homebrew/bin/openmimic"]
        for binaryPath in cliBinaryPaths {
            let url = URL(fileURLWithPath: binaryPath)
            let resolved = url.resolvingSymlinksInPath().path
                .components(separatedBy: "/")
            if !resolved.isEmpty {
                // resolved: ["", "usr", "local", "Cellar", "openmimic", "HEAD-xxx", "bin", "openmimic"]
                // We want: .../Cellar/openmimic/HEAD-xxx/libexec/extension
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

        // 3. Check for dev/source build by looking for the openmimic CLI binary
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

    /// Find the openmimic CLI binary on the system.
    private func findCLIBinary() -> String? {
        let knownPaths = [
            "/usr/local/bin/openmimic",
            "/opt/homebrew/bin/openmimic",
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
            kSecAttrService as String: "com.openmimic.apprentice",
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(deleteQuery as CFDictionary)

        // Add new item
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.openmimic.apprentice",
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
        ]
        let status = SecItemAdd(addQuery as CFDictionary, nil)
        return status == errSecSuccess
    }

    static func retrieve(key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.openmimic.apprentice",
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
