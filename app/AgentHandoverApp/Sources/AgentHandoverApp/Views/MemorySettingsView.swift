import AppKit
import SwiftUI

/// Memory Settings: what AgentHandover does (Skills + Memory or Memory
/// only, `SkillsMode`), Memory on or off, how long thumbnails and moments are kept,
/// whether agents may search, and forgetting everything.  Each change is
/// written to config.toml `[memory]` at once; the worker and the daemon
/// restart (they read it at start) a moment after the last change, except
/// for the agents switch, which the MCP server reads on every search.
struct MemorySettingsView: View {
    @ObservedObject private var center = MemoryCenter.shared
    @EnvironmentObject private var appState: AppState
    @State private var settings = MemorySettings.defaults
    @State private var saved = MemorySettings.defaults
    @State private var loaded = false
    @State private var pendingSave: Task<Void, Never>?
    @State private var saving = false
    @State private var mode = SkillsMode.skillsAndMemory
    @State private var savedMode = SkillsMode.skillsAndMemory

    /// The retention choices; a value set by hand in config.toml is listed too.
    private static let retentionChoices = [30, 90, 180, 365, 730, 1825, 3650]

    var body: some View {
        Form {
            Section("What AgentHandover does") {
                Picker("Mode", selection: $mode) {
                    Text(SkillsMode.skillsAndMemory.title).tag(SkillsMode.skillsAndMemory)
                    Text(SkillsMode.memoryOnly.title).tag(SkillsMode.memoryOnly)
                }
                .pickerStyle(.radioGroup)
                .labelsHidden()
                .disabled(saving)
                Text(mode.isMemoryOnly
                     ? mode.summary + " Your Skills stay available to your agents."
                     : mode.summary + " If its models are not on this Mac yet, the menu offers to download them.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Section {
                Toggle(isOn: $settings.enabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Memory")
                        Text(mode.isMemoryOnly
                             ? "Memory only is Memory: to turn Memory off, choose Skills + Memory first."
                             : "A private, searchable record of what you see: a short description and a search vector of each screen, kept only on this Mac.")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .disabled(mode.isMemoryOnly)
            }

            Section("Keep") {
                Picker("Keep Memory for", selection: $settings.retentionDays) {
                    ForEach(retentionChoices, id: \.self) { days in
                        Text(Self.retentionLabel(days)).tag(days)
                    }
                }
                Stepper(value: $settings.thumbnailDays, in: MemorySettings.thumbnailRange) {
                    HStack {
                        Text("Keep thumbnails for")
                        Spacer()
                        Text(settings.thumbnailDays == 0 ? "None"
                             : settings.thumbnailDays == 1 ? "1 day" : "\(settings.thumbnailDays) days")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                }
                Text("Thumbnails are small, encrypted with a key in your Keychain, and shown only in this app. After they expire, the description and search vector remain.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .disabled(!settings.enabled)

            if let upgrade = appState.workerStatus?.embedding_upgrade {
                Section("Skills index") {
                    Text(upgrade.line)
                        .font(.system(size: 12))
                        .foregroundStyle(upgrade.state == "failed" ? .red : .secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    if upgrade.state == "running", upgrade.total > 0 {
                        ProgressView(value: Double(upgrade.done), total: Double(upgrade.total))
                    }
                }
            }

            Section("Agents") {
                Toggle(isOn: $settings.agentsAccess) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Let agents search my Memory")
                        Text("Connected agents see descriptions, apps, window titles and times, never screenshots.")
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }

            Section("Privacy") {
                Text("Password managers, macOS password prompts, Keychain Access, and banking, payroll and sign-in pages are never kept. To exclude more apps or sites, add them to [privacy.zones] in config.toml (blocked, blocked_urls); AgentHandover reads it when it restarts.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Button("Open config.toml") { NSWorkspace.shared.open(MemorySettings.configURL) }
            }

            Section {
                HStack {
                    Button("Forget Everything\u{2026}", role: .destructive) {
                        Task { await center.forget(.everything) }
                    }
                    .disabled(center.status == nil)
                    Spacer()
                    if saving {
                        ProgressView().controlSize(.small)
                        Text("Restarting AgentHandover\u{2026}").font(.system(size: 11)).foregroundStyle(.secondary)
                    } else if let status = center.status {
                        Text(MemoryFormat().statusLine(status, now: Date()))
                            .font(.system(size: 11))
                            .foregroundStyle(.secondary)
                    }
                }
                if let notice = center.notice {
                    Text(notice).font(.system(size: 11)).foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 520)
        .frame(minHeight: 560)
        .task {
            let (current, currentMode) = await Task.detached { (MemorySettings.load(), SkillsMode.load()) }.value
            settings = current
            saved = current
            mode = currentMode
            savedMode = currentMode
            loaded = true
            await center.refresh()
        }
        .onChange(of: mode) { _, new in
            guard loaded, new != savedMode else { return }
            switchMode(to: new)
        }
        .onChange(of: settings) { _, new in
            guard loaded else { return }
            scheduleSave(new)
        }
        // The Commonplace window's agents line writes the same switch.
        .onChange(of: center.status?.agentsAccess) { _, access in
            guard loaded, let access, access != saved.agentsAccess else { return }
            saved.agentsAccess = access
            settings.agentsAccess = access
        }
    }

    private var retentionChoices: [Int] {
        Array(Set(Self.retentionChoices + [settings.retentionDays])).sorted()
    }

    private static let retentionNames = [
        30: "1 month", 90: "3 months", 180: "6 months", 365: "1 year",
        730: "2 years", 1825: "5 years", 3650: "10 years",
    ]

    static func retentionLabel(_ days: Int) -> String {
        retentionNames[days] ?? "\(days) days"
    }

    /// Write the mode (Memory only also turns Memory on) and restart the
    /// worker and the daemon, which read it at start; then show Memory's
    /// settings as the file now has them.
    private func switchMode(to new: SkillsMode) {
        pendingSave?.cancel()
        saving = true
        savedMode = new
        Task {
            let current = await Task.detached { () -> MemorySettings in
                if new.write() { MemorySettings.restartServices() }
                return MemorySettings.load()
            }.value
            saved = current
            settings = current
            saving = false
            await center.refresh()
        }
    }

    /// Write at once when only the agents switch changed, else shortly
    /// after the last change (one restart for several clicks).
    private func scheduleSave(_ new: MemorySettings) {
        let keys = MemorySettings.changes(from: saved, to: new)
        guard !keys.isEmpty else { return }
        pendingSave?.cancel()
        let delay: UInt64 = keys.contains(where: \.needsRestart) ? 1_200_000_000 : 0
        pendingSave = Task {
            try? await Task.sleep(nanoseconds: delay)
            guard !Task.isCancelled else { return }
            saving = keys.contains(where: \.needsRestart)
            saved = new
            await center.save(new, keys: keys)
            saving = false
        }
    }
}
