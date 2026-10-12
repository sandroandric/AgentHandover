import AppKit
import SwiftUI

/// The offer an updated install sees once (`MemoryDecisions.offer`):
/// "Memory only", "Upgrade everything" or "Not now", recorded in
/// memory-offer.json.  With an Ollama too old for Memory's model it
/// explains the update and offers "Remind me later" instead.
struct MemoryOfferView: View {
    let visualSearch: Bool
    /// The running Ollama's version when it cannot run Memory's model.
    let ollamaTooOld: String?
    let minimumOllama: String
    let memoryModel: String
    /// "710 MB", when the table knows the model's size.
    let memoryModelSize: String?
    let close: () -> Void

    private enum Phase: Equatable {
        case choosing
        case working
        case download(MemoryOfferState.Choice)
        case done(String)
    }

    @State private var phase: Phase = .choosing

    var body: some View {
        switch phase {
        case .download(let choice):
            ModelDownloadView(
                question: "Downloading \(memoryModel)\(memoryModelSize.map { " (\($0))" } ?? "") for Memory.",
                detail: "Memory starts once the download finishes.",
                missing: [memoryModel],
                afterDownload: {
                    MemorySettings.restartServices()
                    return .success(Self.doneMessage(choice))
                },
                autostart: true,
                close: close
            )
        default:
            offer
        }
    }

    private var offer: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                Image(systemName: "clock.arrow.circlepath")
                    .font(.system(size: 28, weight: .light))
                    .foregroundStyle(.secondary)
                VStack(alignment: .leading, spacing: 2) {
                    Text("New in AgentHandover 0.6")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                    Text("Memory")
                        .font(.system(size: 20, weight: .semibold))
                }
            }

            Text("Find anything you saw on your Mac by describing it. AgentHandover keeps a short description and a search vector of each screen it keeps, and a small encrypted thumbnail for 7 days. Everything stays on this Mac; password managers, password prompts, banking and sign-in pages, and the apps and sites you exclude are never kept.")
                .font(.system(size: 13))
                .fixedSize(horizontal: false, vertical: true)

            if visualSearch {
                Text("You had Visual search on; Memory replaces it.")
                    .font(.system(size: 13, weight: .medium))
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text("Agents cannot search your Memory unless you allow it in Memory Settings.")
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if let version = ollamaTooOld {
                tooOld(version)
            } else {
                choices
            }
        }
        .padding(24)
        .frame(width: 480)
    }

    private var choices: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 8) {
                choiceLine("Memory only", "Turns on Memory. Skill search keeps the model it uses now.")
                choiceLine("Upgrade everything",
                           "Also moves Skill search and task grouping to EmbeddingGemma 2. AgentHandover will re-index your Skills in the background and switch when it is done.")
                Text("Either one downloads \(memoryModel)\(memoryModelSize.map { " (\($0))" } ?? "") if it is not on this Mac yet.")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(12)
            .background(RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Color.primary.opacity(0.04)))

            HStack {
                // Once a choice is made, "Not Now" would overwrite it.
                if case .done = phase {
                } else {
                    Button("Not Now") { choose(.declined) }
                        .keyboardShortcut(.cancelAction)
                }
                Spacer()
                if phase == .working {
                    ProgressView().controlSize(.small)
                }
                if case .done(let message) = phase {
                    Text(message).font(.system(size: 12)).foregroundStyle(.secondary)
                    Button("Done") { close() }.keyboardShortcut(.defaultAction)
                } else {
                    Button("Upgrade Everything") { choose(.everything) }
                    Button("Memory Only") { choose(.memoryOnly) }
                        .keyboardShortcut(.defaultAction)
                }
            }
            .disabled(phase == .working)
        }
    }

    private func choiceLine(_ title: String, _ text: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.system(size: 12, weight: .semibold))
            Text(text)
                .font(.system(size: 12))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func tooOld(_ version: String) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(MemoryDecisions.ollamaTooOldText(version: version, minimum: minimumOllama))
                .font(.system(size: 12))
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Button("Not Now") { choose(.declined) }
                Spacer()
                Button("Get Ollama") {
                    if let url = URL(string: "https://ollama.com/download/mac") { NSWorkspace.shared.open(url) }
                }
                Button("Remind Me Later") { close() }
                    .keyboardShortcut(.defaultAction)
            }
        }
    }

    static func doneMessage(_ choice: MemoryOfferState.Choice) -> String {
        choice == .everything
            ? "Memory is on. Your Skills are being re-indexed in the background."
            : "Memory is on. Search it from the menu bar: Search Memory."
    }

    /// Record the choice; for Memory, turn it on and pull its model when
    /// this Mac does not have it.
    private func choose(_ choice: MemoryOfferState.Choice) {
        MemoryOfferState.record(choice)
        guard choice != .declined else {
            close()
            return
        }
        phase = .working
        let model = memoryModel
        DispatchQueue.global(qos: .userInitiated).async {
            var settings = MemorySettings.load()
            settings.enabled = true
            MemorySettings.write(settings, keys: [.enabled])
            let missing = ModelUpgrade.installedModels().map { OllamaModels.missing([model], installed: $0) } ?? []
            if missing.isEmpty {
                MemorySettings.restartServices()
            }
            DispatchQueue.main.async {
                phase = missing.isEmpty ? .done(Self.doneMessage(choice)) : .download(choice)
            }
        }
    }
}
