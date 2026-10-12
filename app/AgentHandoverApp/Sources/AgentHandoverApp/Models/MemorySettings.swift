import Foundation

/// Memory's settings in config.toml `[memory]`: the keys the app reads and
/// writes. The worker and the daemon read them at start; the MCP server
/// re-reads `agents_access` on every search, so that one needs no restart.
struct MemorySettings: Equatable {
    var enabled: Bool
    var retentionDays: Int
    var thumbnailDays: Int
    var agentsAccess: Bool

    /// The worker's defaults (`config.MemoryConfig`).
    static let defaults = MemorySettings(enabled: false, retentionDays: 365, thumbnailDays: 7, agentsAccess: false)
    static let retentionRange = 30...3650
    static let thumbnailRange = 0...30

    enum Key: String, CaseIterable {
        case enabled
        case retentionDays = "retention_days"
        case thumbnailDays = "thumbnail_days"
        case agentsAccess = "agents_access"

        /// Whether a change needs the worker and the daemon restarted.
        var needsRestart: Bool { self != .agentsAccess }
    }

    static var configURL: URL { MemoryAPI.dataDirectory.appendingPathComponent("config.toml") }

    /// The `[memory]` settings of `config`; a key that is absent or not
    /// readable keeps its default.
    static func parse(_ config: String) -> MemorySettings {
        var settings = defaults
        let value = { (key: Key) in ConfigTOML.value(section: "memory", key: key.rawValue, in: config) }
        if let raw = value(.enabled), let on = Bool(raw) { settings.enabled = on }
        if let raw = value(.retentionDays), let days = Int(raw) { settings.retentionDays = days }
        if let raw = value(.thumbnailDays), let days = Int(raw) { settings.thumbnailDays = days }
        if let raw = value(.agentsAccess), let on = Bool(raw) { settings.agentsAccess = on }
        return settings
    }

    /// `[memory] enabled` as written: nil when the key (or the section) is
    /// absent, as in every config written before 0.6.
    static func enabledValue(in config: String) -> Bool? {
        ConfigTOML.value(section: "memory", key: Key.enabled.rawValue, in: config).flatMap(Bool.init)
    }

    /// The changed keys between `old` and `new`, in `Key` order.
    static func changes(from old: MemorySettings, to new: MemorySettings) -> [Key] {
        Key.allCases.filter { old.literal($0) != new.literal($0) }
    }

    /// The TOML value of `key`, clamped to its range.
    func literal(_ key: Key) -> String {
        switch key {
        case .enabled: return String(enabled)
        case .agentsAccess: return String(agentsAccess)
        case .retentionDays: return String(retentionDays.clamped(to: Self.retentionRange))
        case .thumbnailDays: return String(thumbnailDays.clamped(to: Self.thumbnailRange))
        }
    }

    /// `config` with `keys` of `[memory]` set to this value, by the same
    /// section updater as `[vlm]` and `[embedding]`: every other line of
    /// the file is kept byte for byte.
    func apply(_ keys: [Key], to config: String) -> String {
        guard !keys.isEmpty else { return config }
        let block = "[memory]\n" + keys.map { "\($0.rawValue) = \(literal($0))\n" }.joined()
        return ModelTierTable.applyVLMBlock(
            block, to: config, managedKeys: keys.map(\.rawValue), sectionHeader: "[memory]"
        )
    }

    // MARK: - The file

    static func readConfig() -> String {
        (try? String(contentsOf: configURL, encoding: .utf8)) ?? ""
    }

    static func load() -> MemorySettings { parse(readConfig()) }

    /// Write `keys` of `settings` into config.toml when they change it;
    /// true when the file changed.
    @discardableResult
    static func write(_ settings: MemorySettings, keys: [Key]) -> Bool {
        let current = readConfig()
        let updated = settings.apply(keys, to: current)
        guard updated != current else { return false }
        try? FileManager.default.createDirectory(
            at: configURL.deletingLastPathComponent(), withIntermediateDirectories: true
        )
        return (try? updated.write(to: configURL, atomically: true, encoding: .utf8)) != nil
    }

    /// The worker and the daemon read `[memory]` at start: restart the
    /// worker, and the daemon only when it runs (a paused Mac stays
    /// paused).  Blocking.
    static func restartServices() {
        ServiceController.restartWorker()
        if ServiceController.isDaemonRunning() {
            ServiceController.restartDaemon()
        }
    }
}

/// Reading one value of config.toml (simple `key = value` lines).
enum ConfigTOML {
    /// The value of `key` in `[section]`, with a trailing comment and
    /// surrounding quotes removed; nil when absent.
    static func value(section: String, key: String, in config: String) -> String? {
        var inSection = false
        for raw in config.components(separatedBy: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("[") {
                inSection = line.components(separatedBy: "#")[0]
                    .trimmingCharacters(in: .whitespaces) == "[\(section)]"
                continue
            }
            guard inSection, !line.hasPrefix("#") else { continue }
            let parts = line.split(separator: "=", maxSplits: 1).map {
                $0.trimmingCharacters(in: .whitespaces)
            }
            guard parts.count == 2, parts[0] == key else { continue }
            var value = parts[1]
            if value.hasPrefix("\"") {
                let rest = value.dropFirst()
                return rest.firstIndex(of: "\"").map { String(rest[..<$0]) }
            }
            value = value.components(separatedBy: "#")[0].trimmingCharacters(in: .whitespaces)
            return value
        }
        return nil
    }
}

extension Comparable {
    func clamped(to range: ClosedRange<Self>) -> Self {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
