import Foundation

/// What AgentHandover does on this Mac: config.toml `[skills] enabled`.
///
/// `.skillsAndMemory` (the default) learns Skills from how you work and
/// keeps Memory. `.memoryOnly` keeps capturing and filling Memory with
/// Memory's embedding model alone: no screen descriptions, Skill drafts,
/// questions or daily summaries, and no other model is loaded (the
/// worker's `SkillsConfig`). The worker and the daemon read it at start.
enum SkillsMode: Equatable {
    case skillsAndMemory
    case memoryOnly

    /// The mode `config` names; an absent or unreadable key is Skills and
    /// Memory, as the worker reads it.
    static func parse(_ config: String) -> SkillsMode {
        ConfigTOML.value(section: "skills", key: "enabled", in: config) == "false" ? .memoryOnly : .skillsAndMemory
    }

    static func load() -> SkillsMode { parse(MemorySettings.readConfig()) }

    var isMemoryOnly: Bool { self == .memoryOnly }

    /// `config` in this mode: `[skills] enabled`, and for Memory only also
    /// `[memory] enabled = true` (the mode is Memory); every other line kept.
    func apply(to config: String) -> String {
        let block = "[skills]\nenabled = \(isMemoryOnly ? "false" : "true")\n"
        let withMode = ModelTierTable.applyVLMBlock(
            block, to: config, managedKeys: ["enabled"], sectionHeader: "[skills]"
        )
        guard isMemoryOnly else { return withMode }
        var memory = MemorySettings.parse(withMode)
        memory.enabled = true
        return memory.apply([.enabled], to: withMode)
    }

    /// Write this mode into config.toml when it changes the file; true when
    /// it did (the caller restarts the services, `MemorySettings.restartServices`).
    @discardableResult
    func write() -> Bool {
        let url = MemorySettings.configURL
        let current = MemorySettings.readConfig()
        let updated = apply(to: current)
        guard updated != current else { return false }
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        return (try? updated.write(to: url, atomically: true, encoding: .utf8)) != nil
    }

    // MARK: - Words

    /// The setup and Settings choice, one line each.
    var title: String {
        switch self {
        case .skillsAndMemory: return "Skills + Memory"
        case .memoryOnly: return "Memory only"
        }
    }

    var summary: String {
        switch self {
        case .skillsAndMemory:
            return "Learns Skills from how you work, asks about what it could not see, summarises your days, and remembers what you saw."
        case .memoryOnly:
            return "Remembers what you saw so you can search it, and nothing else. One small model; no Skills, questions or daily summaries."
        }
    }

    /// The Commonplace line for a scope this mode does not fill.
    static let memoryOnlyNote = "Memory only: no new Skills or daily summaries are made. The ones you have stay searchable."
}
