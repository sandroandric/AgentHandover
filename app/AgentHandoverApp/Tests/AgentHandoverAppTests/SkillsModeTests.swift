import XCTest
@testable import AgentHandoverApp

/// WP-E14: Memory only. The app's write of `[skills] enabled` changes only
/// that key (and, for Memory only, `[memory] enabled`); the worker status
/// says which mode the running worker uses.
final class SkillsModeTests: XCTestCase {
    private let config = """
        [vlm]
        tier = "recommended"

        [memory]
        enabled = false
        agents_access = true

        """

    func testAnAbsentKeyIsSkillsAndMemoryAsTheWorkerReadsIt() {
        XCTAssertEqual(SkillsMode.parse(""), .skillsAndMemory)
        XCTAssertEqual(SkillsMode.parse(config), .skillsAndMemory)
        XCTAssertEqual(SkillsMode.parse("[skills]\nenabled = false\n"), .memoryOnly)
        XCTAssertEqual(SkillsMode.parse("[skills]\nenabled = true\n"), .skillsAndMemory)
    }

    func testMemoryOnlyTurnsMemoryOnAndKeepsEveryOtherLine() {
        let written = SkillsMode.memoryOnly.apply(to: config)
        XCTAssertEqual(SkillsMode.parse(written), .memoryOnly)
        XCTAssertEqual(MemorySettings.enabledValue(in: written), true)
        XCTAssertEqual(MemorySettings.parse(written).agentsAccess, true)
        XCTAssertTrue(written.contains("[vlm]\ntier = \"recommended\"\n"), written)
        // Writing it again changes nothing.
        XCTAssertEqual(SkillsMode.memoryOnly.apply(to: written), written)
    }

    func testSkillsAndMemoryLeavesMemoryAsItIs() {
        let memoryOnly = SkillsMode.memoryOnly.apply(to: config)
        let back = SkillsMode.skillsAndMemory.apply(to: memoryOnly)
        XCTAssertEqual(SkillsMode.parse(back), .skillsAndMemory)
        XCTAssertEqual(MemorySettings.enabledValue(in: back), true, "Memory stays as the person left it")
        let fresh = SkillsMode.skillsAndMemory.apply(to: config)
        XCTAssertEqual(MemorySettings.enabledValue(in: fresh), false)
    }

    func testTheWorkerStatusSaysWhetherItRunsMemoryOnly() throws {
        let base = #""pid": 1, "version": "2.0.0", "started_at": "t", "heartbeat": "t", "events_processed_today": 0, "sops_generated": 0, "last_pipeline_duration_ms": null, "consecutive_errors": 0, "vlm_available": false"#
        let decode = { (extra: String) in
            try JSONDecoder().decode(WorkerStatusFile.self, from: Data("{\(base)\(extra)}".utf8))
        }
        XCTAssertTrue(try decode(#", "skills_enabled": false, "missing_models": []"#).memoryOnly)
        XCTAssertFalse(try decode(#", "skills_enabled": true"#).memoryOnly)
        XCTAssertFalse(try decode("").memoryOnly, "a worker before 2.0 learns Skills")
    }

    func testMemoryOnlyDoesNotAskToDownloadSkillModels() {
        // The worker in Memory only reports no model missing.
        XCTAssertNil(ModelUpgrade.missingModelRow(vlmAvailable: false, missingModels: []))
    }
}
