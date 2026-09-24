import XCTest
@testable import AgentHandoverApp

/// The tier table onboarding decodes is the worker's: `resources/tiers.json`
/// is pinned to `model_profiles.TIERS` by the worker's test_model_tiers.py
/// and is the file the installer writes; these tests decode that same file.
final class ModelTiersTests: XCTestCase {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // AgentHandoverAppTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // AgentHandoverApp
        .deletingLastPathComponent()  // app
        .deletingLastPathComponent()  // repo root

    private static let tiersURL = repoRoot.appendingPathComponent("resources/tiers.json")

    private func fixture(_ name: String) throws -> String {
        try String(
            contentsOf: Self.repoRoot.appendingPathComponent("tests/fixtures/vlm_block/\(name)"),
            encoding: .utf8
        )
    }

    func testDecodesEveryFieldOfTheWorkersTable() throws {
        let data = try Data(contentsOf: Self.tiersURL)
        let table = try ModelTierTable.decode(data)
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(table.schemaVersion, raw["schema_version"] as? Int)
        XCTAssertEqual(table.minimumOllamaVersionForGemma4, raw["minimum_ollama_version_for_gemma4"] as? String)
        XCTAssertEqual(table.embeddingModel, raw["embedding_model"] as? String)
        XCTAssertEqual(table.modelSizesGB, raw["model_sizes_gb"] as? [String: Double])
        let rawTiers = try XCTUnwrap(raw["tiers"] as? [[String: Any]])
        XCTAssertEqual(table.tiers.count, rawTiers.count)
        for (tier, rawTier) in zip(table.tiers, rawTiers) {
            // Every key of the JSON row is decoded, none invented.
            XCTAssertEqual(Set(rawTier.keys), Set(ModelTier.CodingKeys.allCases.map(\.rawValue)))
            XCTAssertEqual(tier.id, rawTier["id"] as? String)
            XCTAssertEqual(tier.label, rawTier["label"] as? String)
            XCTAssertEqual(tier.subtitle, rawTier["subtitle"] as? String)
            XCTAssertEqual(tier.minRAMGB, rawTier["min_ram_gb"] as? Int)
            XCTAssertEqual(tier.annotationModel, rawTier["annotation_model"] as? String)
            XCTAssertEqual(tier.sopModel, rawTier["sop_model"] as? String)
            XCTAssertEqual(tier.abSOPModel, rawTier["ab_sop_model"] as? String)
            let rawPairs = try XCTUnwrap(rawTier["previous_defaults"] as? [[String: String]])
            XCTAssertEqual(tier.previousDefaults.map { [$0.annotationModel, $0.sopModel] },
                           rawPairs.map { [$0["annotation_model"] ?? "", $0["sop_model"] ?? ""] })
        }
        // WP-F6 moved the 16 GB tier off gemma4:12b-it-qat.
        XCTAssertEqual(table.tier(id: "recommended")?.previousDefaults,
                       [ModelPair(annotationModel: "qwen3.5:4b", sopModel: "gemma4:12b-it-qat")])
        // WP-F8 moved the 8 GB tier's annotator off qwen3.5:4b.
        XCTAssertEqual(table.tier(id: "standard")?.previousDefaults,
                       [ModelPair(annotationModel: "qwen3.5:4b", sopModel: "qwen3.5:4b")])
    }

    /// A tiers.json from an install before `previous_defaults` still decodes.
    func testDecodesATableWithoutPreviousDefaults() throws {
        let json = """
            {"schema_version": 1, "minimum_ollama_version_for_gemma4": "0.34.0",
             "embedding_model": "nomic-embed-text", "model_sizes_gb": {},
             "tiers": [{"id": "standard", "label": "Standard", "subtitle": "s",
                        "min_ram_gb": 0, "annotation_model": "qwen3.5:4b",
                        "sop_model": "qwen3.5:4b", "ab_sop_model": ""}]}
            """
        let table = try ModelTierTable.decode(Data(json.utf8))
        XCTAssertEqual(table.tiers[0].previousDefaults, [])
    }

    func testRecommendsByRAM() throws {
        let table = try ModelTierTable.load(from: Self.tiersURL)
        let picks = [8, 15, 16, 24, 36, 48, 128].map { table.recommended(forRAMGB: $0).id }
        XCTAssertEqual(picks, ["standard", "standard", "recommended", "performance",
                               "performance", "max_quality", "max_quality"])
        XCTAssertEqual(table.available(forRAMGB: 16).map(\.id), ["standard", "recommended"])
    }

    func testTheMeasuredSplit() throws {
        let table = try ModelTierTable.load(from: Self.tiersURL)
        // WP-F8: the 8 GB tier annotates with the 2B, every other with the 4B.
        XCTAssertEqual(table.tiers.map(\.annotationModel),
                       ["qwen3.5:2b-q4_K_M", "qwen3.5:4b", "qwen3.5:4b", "qwen3.5:4b"])
        // WP-F6: 16 GB runs qwen3.5:4b for both roles; Gemma 4 writes Skills from 24 GB.
        XCTAssertEqual(table.tiers.map(\.sopModel),
                       ["qwen3.5:4b", "qwen3.5:4b", "gemma4:12b-it-qat", "gemma4:31b-it-qat"])
        let recommended = try XCTUnwrap(table.tier(id: "recommended"))
        XCTAssertEqual(table.models(for: recommended), ["qwen3.5:4b", "nomic-embed-text"])
        XCTAssertFalse(table.needsNewOllama(table.models(for: recommended)))
        XCTAssertEqual(table.downloadLabel(for: recommended), "~4 GB")
        let performance = try XCTUnwrap(table.tier(id: "performance"))
        XCTAssertEqual(table.models(for: performance), ["qwen3.5:4b", "gemma4:12b-it-qat", "nomic-embed-text"])
        XCTAssertTrue(table.needsNewOllama(table.models(for: performance)))
        XCTAssertFalse(table.needsNewOllama(table.models(for: table.tiers[0])))
        XCTAssertEqual(table.sizeLabel("qwen3.5:4b"), "3.4 GB")
        XCTAssertEqual(table.sizeLabel("nomic-embed-text"), "270 MB")
        XCTAssertEqual(table.downloadLabel(for: performance), "~11 GB")
        let standard = try XCTUnwrap(table.tier(id: "standard"))
        XCTAssertEqual(table.models(for: standard), ["qwen3.5:2b-q4_K_M", "qwen3.5:4b", "nomic-embed-text"])
        XCTAssertFalse(table.needsNewOllama(table.models(for: standard)))
        XCTAssertEqual(table.sizeLabel("qwen3.5:2b-q4_K_M"), "1.9 GB")
        // 1.9 + 3.4 + 0.27 GB.
        XCTAssertEqual(table.downloadLabel(for: standard), "~6 GB")
    }

    func testVLMBlockMatchesTheSharedFixture() throws {
        let table = try ModelTierTable.load(from: Self.tiersURL)
        let tier = try XCTUnwrap(table.tier(id: "performance"))
        let block = ModelTierTable.vlmBlock(
            tierID: tier.id, annotationModel: tier.annotationModel, sopModel: tier.sopModel)
        let once = ModelTierTable.applyVLMBlock(block, to: try fixture("input.toml"))
        XCTAssertEqual(once, try fixture("performance.toml"))
        XCTAssertEqual(ModelTierTable.applyVLMBlock(block, to: once), once)
    }

    /// The worker's tier migration makes onboarding's edit: a config on the
    /// 16 GB tier's pre-WP-F6 models becomes recommended-migrated.toml.
    func testVLMBlockMatchesTheMigrationFixture() throws {
        let table = try ModelTierTable.load(from: Self.tiersURL)
        let tier = try XCTUnwrap(table.tier(id: "recommended"))
        let block = ModelTierTable.vlmBlock(
            tierID: tier.id, annotationModel: tier.annotationModel, sopModel: tier.sopModel)
        XCTAssertEqual(ModelTierTable.applyVLMBlock(block, to: try fixture("recommended-previous.toml")),
                       try fixture("recommended-migrated.toml"))
    }

    /// WP-F8: a config on the 8 GB tier's pre-WP-F8 models becomes
    /// standard-migrated.toml, as the worker's tier migration makes it.
    func testVLMBlockMatchesTheStandardMigrationFixture() throws {
        let table = try ModelTierTable.load(from: Self.tiersURL)
        let tier = try XCTUnwrap(table.tier(id: "standard"))
        let block = ModelTierTable.vlmBlock(
            tierID: tier.id, annotationModel: tier.annotationModel, sopModel: tier.sopModel)
        let once = ModelTierTable.applyVLMBlock(block, to: try fixture("standard-previous.toml"))
        XCTAssertEqual(once, try fixture("standard-migrated.toml"))
        XCTAssertEqual(ModelTierTable.applyVLMBlock(block, to: once), once)
    }

    func testVLMBlockIsAppendedWithoutAVLMSection() {
        let block = ModelTierTable.vlmBlock(tierID: "standard", annotationModel: "qwen3.5:4b", sopModel: "qwen3.5:4b")
        XCTAssertEqual(
            ModelTierTable.applyVLMBlock(block, to: "[embedding]\nimage_embeddings = false"),
            "[embedding]\nimage_embeddings = false\n\n" + block
        )
        XCTAssertEqual(ModelTierTable.applyVLMBlock(block, to: ""), "\n" + block)
    }

    /// Only `[vlm]` is edited: a `mode =` or `model =` in another section
    /// belongs to that section (an earlier writer deleted the embedding
    /// model this way).
    func testTierBlockEditsOnlyTheVLMSection() {
        let config = """
        [vlm]
        mode = "local"
        keep_alive = "30m"

        [embedding]
        model = "nomic-embed-text"

        [privacy]
        mode = "strict"

        """
        let block = ModelTierTable.vlmBlock(tierID: "standard", annotationModel: "a", sopModel: "s")
        let updated = ModelTierTable.applyVLMBlock(block, to: config)
        XCTAssertEqual(updated, """
        [vlm]
        mode = "local"
        tier = "standard"
        annotation_model = "a"
        sop_model = "s"
        keep_alive = "30m"

        [embedding]
        model = "nomic-embed-text"

        [privacy]
        mode = "strict"

        """)
        // Saving twice leaves one copy of each key.
        XCTAssertEqual(ModelTierTable.applyVLMBlock(block, to: updated), updated)
    }

    func testAnEmptyTableIsAnError() {
        let json = #"{"schema_version": 1, "minimum_ollama_version_for_gemma4": "0.34.0", "embedding_model": "e", "model_sizes_gb": {}, "tiers": []}"#
        XCTAssertThrowsError(try ModelTierTable.decode(Data(json.utf8)))
    }
}
