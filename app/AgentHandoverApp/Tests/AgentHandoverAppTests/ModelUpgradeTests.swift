import XCTest
@testable import AgentHandoverApp

/// WP-F16: the app asks, one click.  The launch decision, the menu row,
/// the plan the worker prints and the Ollama helpers the window uses.
final class ModelUpgradeTests: XCTestCase {

    /// `python -m agenthandover_worker.model_profiles upgrade-plan` on the
    /// config v0.4.0 left on a 24 GB Mac (tests/fixtures/vlm_block/
    /// v0.4.0-onboarding.toml; the worker's test_plan_for_each_v040_pair).
    static let planOutput = """
        {"plan": {"tier": "performance", "label": "Performance", \
        "current": {"annotation_model": "gemma4:12b-it-qat", "sop_model": "gemma4:12b-it-qat"}, \
        "target": {"annotation_model": "qwen3.5:4b", "sop_model": "gemma4:12b-it-qat"}, \
        "models": ["qwen3.5:4b", "gemma4:12b-it-qat"], \
        "sizes_gb": {"qwen3.5:4b": 3.4, "gemma4:12b-it-qat": 7.2}}, \
        "reason": "tier \\"performance\\": annotation_model gemma4:12b-it-qat -> qwen3.5:4b"}
        """

    private var plan: ModelUpgradePlan { ModelUpgrade.decodePlan(Self.planOutput)! }

    func testThePlanDecodes() {
        XCTAssertEqual(plan.tier, "performance")
        XCTAssertEqual(plan.label, "Performance")
        XCTAssertEqual(plan.current.annotationModel, "gemma4:12b-it-qat")
        XCTAssertEqual(plan.target.annotationModel, "qwen3.5:4b")
        XCTAssertEqual(plan.target.sopModel, "gemma4:12b-it-qat")
        XCTAssertEqual(plan.models, ["qwen3.5:4b", "gemma4:12b-it-qat"])
        XCTAssertEqual(plan.sizesGB["qwen3.5:4b"], 3.4)
    }

    func testNoPlanDecodesAsNil() {
        XCTAssertNil(ModelUpgrade.decodePlan(#"{"plan": null, "reason": "already the current models of tier \"standard\""}"#))
        XCTAssertNil(ModelUpgrade.decodePlan(""))
        XCTAssertNil(ModelUpgrade.decodePlan("Traceback (most recent call last):"))
    }

    // MARK: - The launch decision

    func testNothingToDoWithoutAPlanOrAnOllama() {
        XCTAssertEqual(ModelUpgrade.decide(plan: nil, installed: [], declinedVersion: nil, appVersion: "0.5.0"), .none)
        XCTAssertEqual(ModelUpgrade.decide(plan: plan, installed: nil, declinedVersion: nil, appVersion: "0.5.0"), .none)
    }

    func testEveryModelPulledSwitchesWithoutAWindow() {
        let installed: Set<String> = ["qwen3.5:4b", "gemma4:12b-it-qat", "nomic-embed-text"]
        XCTAssertEqual(
            ModelUpgrade.decide(plan: plan, installed: installed, declinedVersion: nil, appVersion: "0.5.0"),
            .applySilently)
        // Even after "Not now": nothing to download, nothing to ask.
        XCTAssertEqual(
            ModelUpgrade.decide(plan: plan, installed: installed, declinedVersion: "0.5.0", appVersion: "0.5.0"),
            .applySilently)
    }

    func testAMissingModelAsksUnlessDeclinedForThisVersion() {
        let installed: Set<String> = ["gemma4:12b-it-qat", "nomic-embed-text"]
        XCTAssertEqual(
            ModelUpgrade.decide(plan: plan, installed: installed, declinedVersion: nil, appVersion: "0.5.0"),
            .ask(missing: ["qwen3.5:4b"]))
        XCTAssertEqual(
            ModelUpgrade.decide(plan: plan, installed: installed, declinedVersion: "0.5.0", appVersion: "0.5.0"),
            .none)
        // The next upgrade asks again.
        XCTAssertEqual(
            ModelUpgrade.decide(plan: plan, installed: installed, declinedVersion: "0.5.0", appVersion: "0.5.1"),
            .ask(missing: ["qwen3.5:4b"]))
        XCTAssertEqual(
            ModelUpgrade.decide(plan: plan, installed: [], declinedVersion: nil, appVersion: "0.5.0"),
            .ask(missing: ["qwen3.5:4b", "gemma4:12b-it-qat"]))
    }

    func testTheQuestionNamesTheModelsAndTheDownload() {
        XCTAssertEqual(
            ModelUpgrade.question(plan: plan, missing: ["qwen3.5:4b"]),
            "0.5 uses a faster model (qwen3.5:4b, ~3.4 GB). Download now?")
        XCTAssertEqual(
            ModelUpgrade.question(plan: plan, missing: ["qwen3.5:4b", "gemma4:12b-it-qat"]),
            "0.5 uses faster models (qwen3.5:4b and gemma4:12b-it-qat, ~10.6 GB). Download now?")
        XCTAssertEqual(
            ModelUpgrade.question(plan: nil, missing: ["qwen3.5:4b"]),
            "AgentHandover needs qwen3.5:4b to process your recordings. Download now?")
    }

    // MARK: - The menu row

    func testTheMenuRowShowsOnlyWhenAModelIsMissing() {
        XCTAssertEqual(
            ModelUpgrade.missingModelRow(vlmAvailable: false, missingModels: ["qwen3.5:4b", "gemma4:12b-it-qat"]),
            "AI model missing: qwen3.5:4b. Download\u{2026}")
        XCTAssertNil(ModelUpgrade.missingModelRow(vlmAvailable: true, missingModels: ["qwen3.5:4b"]))
        // Ollama did not answer (or the worker is still starting): no key.
        XCTAssertNil(ModelUpgrade.missingModelRow(vlmAvailable: false, missingModels: nil))
        XCTAssertNil(ModelUpgrade.missingModelRow(vlmAvailable: false, missingModels: []))
    }

    func testWorkerStatusDecodesWithAndWithoutMissingModels() throws {
        let base = #""pid": 1, "version": "0.5.0", "started_at": "t", "heartbeat": "t", "events_processed_today": 0, "sops_generated": 0, "last_pipeline_duration_ms": null, "consecutive_errors": 0, "vlm_available": false"#
        let with = try JSONDecoder().decode(
            WorkerStatusFile.self, from: Data("{\(base), \"missing_models\": [\"qwen3.5:4b\"]}".utf8))
        XCTAssertEqual(with.missing_models, ["qwen3.5:4b"])
        let without = try JSONDecoder().decode(WorkerStatusFile.self, from: Data("{\(base)}".utf8))
        XCTAssertNil(without.missing_models)
    }

    // MARK: - OllamaModels

    func testInstalledModelsFromOllamaList() {
        let output = """
            NAME                  ID              SIZE      MODIFIED
            gemma4:12b-it-qat     1c6b1a0e3a4f    7.2 GB    5 months ago
            nomic-embed-text:latest  0a109f422b47  274 MB   5 months ago

            """
        let installed = OllamaModels.parseList(output)
        XCTAssertEqual(installed, ["gemma4:12b-it-qat", "nomic-embed-text"])
        XCTAssertEqual(OllamaModels.missing(["qwen3.5:4b", "gemma4:12b-it-qat", "nomic-embed-text:latest"],
                                            installed: installed), ["qwen3.5:4b"])
        // A tag is not a prefix match (qwen3.5:2b is not qwen3.5:2b-q4_K_M).
        XCTAssertEqual(OllamaModels.missing(["qwen3.5:2b-q4_K_M"], installed: ["qwen3.5:2b"]),
                       ["qwen3.5:2b-q4_K_M"])
    }

    func testPullProgress() {
        XCTAssertEqual(OllamaModels.fraction(in: "pulling 8eeb52dfb3bb:  45% ▕████      ▏ 1.5 GB/3.4 GB"), 0.45)
        XCTAssertEqual(OllamaModels.fraction(in: "pulling 8eeb52dfb3bb: 100% ▕██████████▏ 3.4 GB"), 1.0)
        XCTAssertNil(OllamaModels.fraction(in: "pulling manifest"))
        XCTAssertNil(OllamaModels.fraction(in: "verifying sha256 digest"))
        XCTAssertEqual(OllamaModels.lastLine("pulling a:  1%\rpulling a:  2%\r\n"), "pulling a:  2%")
        XCTAssertNil(OllamaModels.lastLine("\r\n  \n"))
    }
}
