import XCTest
import Foundation
@testable import AgentHandoverApp

final class SkillRunSummaryTests: XCTestCase {

    private func procedure(_ json: String) throws -> [String: Any] {
        let obj = try JSONSerialization.jsonObject(with: Data(json.utf8))
        return try XCTUnwrap(obj as? [String: Any])
    }

    func testNoRunsNoSummary() throws {
        let proc = try procedure(#"{"id": "x", "steps": [{"step_id": "step_1", "action": "Open"}]}"#)
        XCTAssertNil(SkillRunSummary(procedure: proc))
    }

    func testRunsDeviationsAndOpenQuestions() throws {
        // The shape skill_improver.py writes (execution_stats from the runs table).
        let proc = try procedure(#"""
        {
          "id": "export-weekly-report",
          "steps": [
            {"step_id": "step_1", "action": "Open the dashboard"},
            {"step_id": "step_2", "action": "Export the table as CSV"},
            {"action": "Upload the CSV"}
          ],
          "execution_stats": {"total": 3, "completed": 1, "deviated": 2, "failed": 0,
                              "aborted": 0, "deviations": 3, "success_rate": 0.333,
                              "last_run": "2026-09-22T10:00:00+00:00", "last_failure": null},
          "observed_alternatives": [
            {"step_id": "step_3", "observed_action": "Emailed the CSV", "execution_id": "e1"},
            {"step_id": "step_2", "observed_action": "Copied the table by hand", "execution_id": "e1"},
            {"step_id": "step_2", "observed_action": "copied the table by hand ", "execution_id": "e2"},
            {"step_id": "step_9", "observed_action": "Something else", "execution_id": "e2"}
          ]
        }
        """#)
        let summary = try XCTUnwrap(SkillRunSummary(procedure: proc))
        XCTAssertEqual(summary.runs, 3)
        XCTAssertEqual(summary.completedRuns, 1)
        XCTAssertEqual(summary.deviatedRuns, 2)
        XCTAssertEqual(summary.deviations, 3)
        XCTAssertEqual(summary.lastRun, "2026-09-22T10:00:00+00:00")
        XCTAssertEqual(summary.headline, "Ran 3 times · 3 deviations")

        XCTAssertEqual(summary.openQuestions.map(\.stepId), ["step_2", "step_3", "step_9"])
        let top = summary.openQuestions[0]
        XCTAssertEqual(top.count, 2)
        XCTAssertEqual(top.stepLabel, "Step 2: Export the table as CSV")
        XCTAssertEqual(top.prompt, "An agent did: Copied the table by hand (2 runs). Should the Skill change to match?")
        // A step without step_id is `step_<n>`, as the packet prints it.
        XCTAssertEqual(summary.openQuestions[1].stepLabel, "Step 3: Upload the CSV")
        // An alternative for a step the Skill no longer has keeps its id.
        XCTAssertEqual(summary.openQuestions[2].stepLabel, "step_9")
    }

    func testSingleRunWording() throws {
        let proc = try procedure(#"{"execution_stats": {"total": 1, "completed": 1, "deviations": 1}}"#)
        XCTAssertEqual(SkillRunSummary(procedure: proc)?.headline, "Ran once · 1 deviation")
    }
}
