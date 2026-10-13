import XCTest
@testable import AgentHandoverApp

/// WP-E8: the app's writes of `[memory]` (and setup's `[embedding] model`)
/// change only those keys; every other line stays byte for byte.
final class MemorySettingsTests: XCTestCase {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()

    /// The config a 2.0 install starts from.
    private func example() throws -> String {
        try String(contentsOf: Self.repoRoot.appendingPathComponent("config.example.toml"), encoding: .utf8)
    }

    /// `text` without the lines of `[section]` that set `keys`.
    private func removingKeys(_ keys: [String], of section: String, from text: String) -> [String] {
        var inSection = false
        return text.components(separatedBy: "\n").filter { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("[") { inSection = trimmed == "[\(section)]" }
            let key = trimmed.split(separator: "=", maxSplits: 1).first.map {
                $0.trimmingCharacters(in: .whitespaces)
            }
            return !(inSection && trimmed.contains("=") && keys.contains(key ?? ""))
        }
    }

    func testTheExampleConfigReadsAsTheWorkersDefaults() throws {
        XCTAssertEqual(MemorySettings.parse(try example()), MemorySettings.defaults)
        XCTAssertEqual(MemorySettings.enabledValue(in: try example()), false)
    }

    func testAbsentSectionOrKeyKeepsTheDefaultAndReadsAsAbsent() {
        XCTAssertEqual(MemorySettings.parse(""), .defaults)
        XCTAssertNil(MemorySettings.enabledValue(in: "[vlm]\nenabled = true\n"))
        XCTAssertNil(MemorySettings.enabledValue(in: "[memory]\nretention_days = 90\n"))
        let config = "[memory]\nenabled = true   # setup\nretention_days = 90\nthumbnail_days = 0\nagents_access = true\n"
        XCTAssertEqual(MemorySettings.parse(config),
                       MemorySettings(enabled: true, retentionDays: 90, thumbnailDays: 0, agentsAccess: true))
    }

    func testOneChangedKeyRewritesOnlyItsLine() throws {
        let config = try example()
        var settings = MemorySettings.parse(config)
        settings.agentsAccess = true
        let keys = MemorySettings.changes(from: .parse(config), to: settings)
        XCTAssertEqual(keys, [.agentsAccess])
        let updated = settings.apply(keys, to: config)
        XCTAssertEqual(MemorySettings.parse(updated), settings)
        XCTAssertEqual(removingKeys(["agents_access"], of: "memory", from: updated),
                       removingKeys(["agents_access"], of: "memory", from: config))
        // Applying again changes nothing.
        XCTAssertEqual(settings.apply(keys, to: updated), updated)
    }

    func testEveryKeyAtOnceKeepsEverythingElse() throws {
        let config = try example()
        let settings = MemorySettings(enabled: true, retentionDays: 90, thumbnailDays: 14, agentsAccess: true)
        let keys = MemorySettings.Key.allCases
        let updated = settings.apply(keys, to: config)
        XCTAssertEqual(MemorySettings.parse(updated), settings)
        let names = keys.map(\.rawValue)
        XCTAssertEqual(removingKeys(names, of: "memory", from: updated),
                       removingKeys(names, of: "memory", from: config))
        // [memory]'s other keys (model, dimensions, ...) are kept.
        XCTAssertEqual(ConfigTOML.value(section: "memory", key: "model", in: updated), "embeddinggemma-2:440m")
        XCTAssertEqual(ConfigTOML.value(section: "memory", key: "max_frames_per_hour", in: updated), "600")
    }

    func testAConfigWithoutMemoryGetsTheSectionAppended() {
        let config = "[vlm]\ntier = \"recommended\"\n"
        var settings = MemorySettings.defaults
        settings.enabled = true
        XCTAssertEqual(settings.apply([.enabled], to: config),
                       "[vlm]\ntier = \"recommended\"\n\n[memory]\nenabled = true\n")
    }

    func testValuesAreClampedToTheirRanges() {
        let settings = MemorySettings(enabled: true, retentionDays: 5, thumbnailDays: 90, agentsAccess: false)
        XCTAssertEqual(settings.literal(.retentionDays), "30")
        XCTAssertEqual(settings.literal(.thumbnailDays), "30")
        XCTAssertEqual(MemorySettings(enabled: true, retentionDays: 9000, thumbnailDays: -1, agentsAccess: false)
            .literal(.retentionDays), "3650")
        XCTAssertEqual(MemorySettings(enabled: true, retentionDays: 9000, thumbnailDays: -1, agentsAccess: false)
            .literal(.thumbnailDays), "0")
    }

    func testOnlyTheAgentsSwitchSkipsTheRestart() {
        XCTAssertEqual(MemorySettings.Key.allCases.filter { !$0.needsRestart }, [.agentsAccess])
    }

    func testSetupsEmbeddingWriteChangesOnlyThatKey() throws {
        let config = try example()
        let updated = ModelTierTable.applyEmbeddingModel("embeddinggemma-2:440m", to: config)
        XCTAssertEqual(ConfigTOML.value(section: "embedding", key: "model", in: updated), "embeddinggemma-2:440m")
        XCTAssertEqual(removingKeys(["model"], of: "embedding", from: updated),
                       removingKeys(["model"], of: "embedding", from: config))
    }

    func testConfigValuesIgnoreCommentsAndOtherSections() {
        let config = """
            [privacy]
            enabled = true
            [memory]
            # enabled = true
            enabled = false # off
            [embedding]
            model = "nomic-embed-text"   # text
            """
        XCTAssertEqual(ConfigTOML.value(section: "memory", key: "enabled", in: config), "false")
        XCTAssertEqual(ConfigTOML.value(section: "embedding", key: "model", in: config), "nomic-embed-text")
        XCTAssertNil(ConfigTOML.value(section: "embedding", key: "image_embeddings", in: config))
    }
}
