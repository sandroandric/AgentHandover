import XCTest
@testable import AgentHandoverApp

/// WP-E8: who sees the Memory upgrade offer, what setup writes for Memory,
/// and what the offer records.
final class MemoryOfferTests: XCTestCase {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent()

    /// The table the installer writes (resources/tiers.json).
    private var table: ModelTierTable {
        try! ModelTierTable.load(from: Self.repoRoot.appendingPathComponent("resources/tiers.json"))
    }

    private func offer(
        setupAnswered: Bool = false, memoryEnabled: Bool? = nil, choice: String? = nil,
        wantedVisualSearch: Bool = false, imageEmbeddings: Bool = false, ollama: String? = "0.40.3"
    ) -> MemoryOfferDecision {
        MemoryDecisions.offer(
            setupAnswered: setupAnswered, memoryEnabled: memoryEnabled,
            offer: MemoryOfferState(choice: choice, wantedVisualSearch: wantedVisualSearch),
            imageEmbeddingsInConfig: imageEmbeddings, ollamaVersion: ollama, table: table
        )
    }

    // MARK: - The upgrade offer

    func testAFreshInstallNeverSeesTheOffer() {
        // Setup asked about Memory, whatever the answer was.
        XCTAssertEqual(offer(setupAnswered: true), .none)
        XCTAssertEqual(offer(setupAnswered: true, memoryEnabled: false), .none)
        XCTAssertEqual(offer(setupAnswered: true, memoryEnabled: false, ollama: "0.34.2"), .none)
    }

    func testAnUpgraderWithoutAChoiceIsOffered() {
        XCTAssertEqual(offer(), .offer(visualSearch: false, ollamaTooOld: nil))
        XCTAssertEqual(offer(memoryEnabled: false), .offer(visualSearch: false, ollamaTooOld: nil))
    }

    func testAnUpgraderWithAChoiceOrMemoryOnIsNotOffered() {
        for choice in ["memory_only", "everything", "declined"] {
            XCTAssertEqual(offer(choice: choice), .none, choice)
        }
        XCTAssertEqual(offer(memoryEnabled: true), .none)
    }

    func testAnOldOllamaIsNamedSoTheOfferExplainsTheUpdate() {
        XCTAssertEqual(offer(ollama: "0.34.2"), .offer(visualSearch: false, ollamaTooOld: "0.34.2"))
        XCTAssertEqual(offer(ollama: "0.40.0"), .offer(visualSearch: false, ollamaTooOld: nil))
        // No Ollama answering: nothing to say about its version.
        XCTAssertEqual(offer(ollama: nil), .offer(visualSearch: false, ollamaTooOld: nil))
    }

    func testVisualSearchIsMentionedFromTheFileOrALeftoverConfigKey() {
        XCTAssertEqual(offer(wantedVisualSearch: true), .offer(visualSearch: true, ollamaTooOld: nil))
        XCTAssertEqual(offer(imageEmbeddings: true), .offer(visualSearch: true, ollamaTooOld: nil))
    }

    func testATableWithoutMemoryOffersNothing() throws {
        let old = try ModelTierTable.decode(Data("""
            {"schema_version": 1, "minimum_ollama_version_for_gemma4": "0.20.0", "embedding_model": "nomic-embed-text",
             "model_sizes_gb": {}, "tiers": [{"id": "standard", "label": "S", "subtitle": "", "min_ram_gb": 0,
             "annotation_model": "a", "sop_model": "b", "ab_sop_model": ""}]}
            """.utf8))
        XCTAssertEqual(MemoryDecisions.offer(setupAnswered: false, memoryEnabled: nil,
                                             offer: MemoryOfferState(choice: nil, wantedVisualSearch: false),
                                             imageEmbeddingsInConfig: false, ollamaVersion: "0.40.3", table: old), .none)
        XCTAssertEqual(MemoryDecisions.offer(setupAnswered: false, memoryEnabled: nil,
                                             offer: MemoryOfferState(choice: nil, wantedVisualSearch: false),
                                             imageEmbeddingsInConfig: false, ollamaVersion: "0.40.3", table: nil), .none)
    }

    // MARK: - Setup

    func testANewInstallRunsEverythingOnEmbeddingGemmaWithMemoryOn() {
        let plan = MemoryDecisions.setupPlan(wantsMemory: true, ollamaVersion: "0.40.3", table: table)
        XCTAssertEqual(plan, MemorySetupPlan(memoryEnabled: true, embeddingModel: "embeddinggemma-2:440m", deferred: false))
    }

    func testMemoryTurnedOffInSetupStillRunsTextOnEmbeddingGemma() {
        let plan = MemoryDecisions.setupPlan(wantsMemory: false, ollamaVersion: "0.41.0", table: table)
        XCTAssertEqual(plan, MemorySetupPlan(memoryEnabled: false, embeddingModel: "embeddinggemma-2:440m", deferred: false))
    }

    func testAnOldOllamaKeepsNomicAndMemoryOffUntilItIsUpdated() {
        let plan = MemoryDecisions.setupPlan(wantsMemory: true, ollamaVersion: "0.34.2", table: table)
        XCTAssertEqual(plan, MemorySetupPlan(memoryEnabled: false, embeddingModel: "nomic-embed-text", deferred: true))
        let unknown = MemoryDecisions.setupPlan(wantsMemory: true, ollamaVersion: nil, table: table)
        XCTAssertEqual(unknown, MemorySetupPlan(memoryEnabled: false, embeddingModel: "nomic-embed-text", deferred: true))
        let declined = MemoryDecisions.setupPlan(wantsMemory: false, ollamaVersion: "0.34.2", table: table)
        XCTAssertFalse(declined.deferred)
    }

    func testTheSetupPullListFollowsThePlan() {
        let tier = table.recommended(forRAMGB: 16)
        let eg2 = MemoryDecisions.setupPlan(wantsMemory: true, ollamaVersion: "0.40.3", table: table)
        XCTAssertEqual(table.models(for: tier, embeddingModel: eg2.embeddingModel),
                       [tier.annotationModel, "embeddinggemma-2:440m"].uniqued)
        let old = MemoryDecisions.setupPlan(wantsMemory: true, ollamaVersion: "0.34.2", table: table)
        XCTAssertTrue(table.models(for: tier, embeddingModel: old.embeddingModel).contains("nomic-embed-text"))
        XCTAssertFalse(table.models(for: tier, embeddingModel: old.embeddingModel).contains("embeddinggemma-2:440m"))
    }

    func testTheDeferredMemoryTurnsOnOnceOllamaIsUpdated() {
        let installed: Set<String> = ["qwen3.5:4b", "nomic-embed-text"]
        func action(_ deferred: Bool = true, enabled: Bool? = false, ollama: String? = "0.40.3",
                    installed: Set<String>? = installed) -> MemoryDeferredAction {
            MemoryDecisions.deferredAction(deferred: deferred, memoryEnabled: enabled, ollamaVersion: ollama,
                                           installed: installed, table: table)
        }
        XCTAssertEqual(action(), .download("embeddinggemma-2:440m"))
        XCTAssertEqual(action(installed: installed.union(["embeddinggemma-2:440m"])), .enable)
        XCTAssertEqual(action(ollama: "0.34.2"), .none)
        XCTAssertEqual(action(ollama: nil), .none)
        XCTAssertEqual(action(installed: nil), .none)
        XCTAssertEqual(action(false), .none)
        XCTAssertEqual(action(enabled: true), .none)
    }

    func testSetupDefaultsAreMemoryOnAndAgentsOff() {
        let suite = "memory-setup-\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suite)!
        defer { defaults.removePersistentDomain(forName: suite) }
        XCTAssertTrue(MemorySetupChoice.wantsMemory(defaults))
        XCTAssertFalse(MemorySetupChoice.agentsAccess(defaults))
        XCTAssertFalse(defaults.bool(forKey: MemoryDecisions.setupAnsweredKey))
        MemorySetupChoice.save(memoryOn: false, agentsAccess: true, defaults: defaults)
        XCTAssertFalse(MemorySetupChoice.wantsMemory(defaults))
        XCTAssertTrue(MemorySetupChoice.agentsAccess(defaults))
        // A Mac whose setup asked never sees the upgrade offer.
        XCTAssertTrue(defaults.bool(forKey: MemoryDecisions.setupAnsweredKey))
    }

    func testSetupWritesOnlyMemoryAndTheEmbeddingModel() throws {
        let example = try String(contentsOf: Self.repoRoot.appendingPathComponent("config.example.toml"), encoding: .utf8)
        let plan = MemoryDecisions.setupPlan(wantsMemory: true, ollamaVersion: "0.40.3", table: table)
        let updated = MemorySetupChoice.apply(plan, agentsAccess: false, to: example)
        XCTAssertEqual(ConfigTOML.value(section: "memory", key: "enabled", in: updated), "true")
        XCTAssertEqual(ConfigTOML.value(section: "memory", key: "agents_access", in: updated), "false")
        XCTAssertEqual(ConfigTOML.value(section: "embedding", key: "model", in: updated), "embeddinggemma-2:440m")
        let changed = updated.components(separatedBy: "\n").filter { !example.components(separatedBy: "\n").contains($0) }
        XCTAssertEqual(Set(changed), ["enabled = true", "agents_access = false", "model = \"embeddinggemma-2:440m\""])
        // An old Ollama: Memory off, text model unchanged.
        let old = MemoryDecisions.setupPlan(wantsMemory: true, ollamaVersion: "0.34.2", table: table)
        let kept = MemorySetupChoice.apply(old, agentsAccess: false, to: example)
        XCTAssertEqual(ConfigTOML.value(section: "memory", key: "enabled", in: kept), "false")
        XCTAssertEqual(ConfigTOML.value(section: "embedding", key: "model", in: kept), "nomic-embed-text")
    }

    // MARK: - The offer file

    func testRecordingAChoiceKeepsTheOtherKeys() throws {
        let existing = Data(#"{"wanted_visual_search": true, "extra": 1}"#.utf8)
        let at = Date(timeIntervalSince1970: 1_791_819_600)
        let data = MemoryOfferState.recording(.everything, at: at, in: existing)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(object["choice"] as? String, "everything")
        XCTAssertEqual(object["offered_at"] as? String, "2026-10-12T15:40:00Z")
        XCTAssertEqual(object["wanted_visual_search"] as? Bool, true)
        XCTAssertEqual(object["extra"] as? Int, 1)
        XCTAssertEqual(MemoryOfferState.parse(data), MemoryOfferState(choice: "everything", wantedVisualSearch: true))
    }

    func testRecordingWithoutAFileOrWithGarbageStartsAnObject() throws {
        for existing in [nil, Data("[1, 2]".utf8), Data("not json".utf8)] {
            let data = MemoryOfferState.recording(.declined, at: Date(timeIntervalSince1970: 0), in: existing)
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(Set(object.keys), ["choice", "offered_at"])
        }
    }

    func testTheChoicePersistsOnDiskAndThenTheOfferIsNotShown() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("memory-offer.json")
        XCTAssertEqual(MemoryOfferState.load(from: url), MemoryOfferState(choice: nil, wantedVisualSearch: false))
        MemoryOfferState.record(.memoryOnly, url: url)
        let state = MemoryOfferState.load(from: url)
        XCTAssertEqual(state.choice, "memory_only")
        let mode = try FileManager.default.attributesOfItem(atPath: url.path)[.posixPermissions] as? Int
        XCTAssertEqual(mode, 0o600)
        XCTAssertEqual(MemoryDecisions.offer(setupAnswered: false, memoryEnabled: nil, offer: state,
                                             imageEmbeddingsInConfig: false, ollamaVersion: "0.40.3", table: table), .none)
    }

    // MARK: - Ollama versions

    func testOllamaVersionComparison() {
        XCTAssertTrue(OllamaModels.isVersion("0.40.0", atLeast: "0.40.0"))
        XCTAssertTrue(OllamaModels.isVersion("0.40.3", atLeast: "0.40.0"))
        XCTAssertTrue(OllamaModels.isVersion("1.0", atLeast: "0.40.0"))
        XCTAssertTrue(OllamaModels.isVersion("0.41.0-rc1", atLeast: "0.40.0"))
        XCTAssertFalse(OllamaModels.isVersion("0.34.2", atLeast: "0.40.0"))
        XCTAssertFalse(OllamaModels.isVersion("0.4.0", atLeast: "0.40.0"))
        XCTAssertFalse(OllamaModels.isVersion("unknown", atLeast: "0.40.0"))
    }

    func testClosingTheOfferWithoutAChoiceIsNotNowButRemindLaterAsksAgain() {
        XCTAssertEqual(MemoryOfferState.choiceOnClose(remindLater: false, recorded: nil), .declined)
        XCTAssertNil(MemoryOfferState.choiceOnClose(remindLater: true, recorded: nil))
        XCTAssertNil(MemoryOfferState.choiceOnClose(remindLater: false, recorded: "memory_only"), "a choice is kept")
    }
}

private extension Array where Element: Hashable {
    var uniqued: [Element] {
        var seen = Set<Element>()
        return filter { seen.insert($0).inserted }
    }
}
