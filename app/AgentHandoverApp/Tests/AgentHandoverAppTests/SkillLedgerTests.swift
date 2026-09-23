import XCTest
import Foundation
@testable import AgentHandoverApp

final class SkillLedgerTests: XCTestCase {

    private func procedure(_ json: String) throws -> [String: Any] {
        let obj = try JSONSerialization.jsonObject(with: Data(json.utf8))
        return try XCTUnwrap(obj as? [String: Any])
    }

    func testNoLedgerNoSection() throws {
        let proc = try procedure(#"{"id": "x", "steps": [{"action": "Open"}]}"#)
        XCTAssertNil(SkillLedger(procedure: proc))
    }

    func testOneLinePerSlotInTheWorkersOrder() throws {
        // The shape ledger.py writes on every save.
        let proc = try procedure(#"""
        {
          "id": "daily-digest",
          "knowledge": {
            "trigger": {"status": "unknown", "value_ref": null, "evidence": [], "source": null,
                        "updated_at": "2026-09-15T09:00:00+00:00"},
            "steps": {"status": "known", "value_ref": "steps", "source": "observation",
                      "evidence": [{"kind": "instance", "id": "s1"}, {"kind": "instance", "id": "s2"},
                                   {"kind": "frame", "id": "f1"}],
                      "updated_at": "2026-09-15T09:00:00+00:00"},
            "goal": {"status": "hypothesis", "value_ref": "goal", "source": "observation",
                     "evidence": [{"kind": "instance", "id": "s1"}],
                     "updated_at": "2026-09-15T09:00:00+00:00"},
            "done_criteria": {"status": "known", "value_ref": "agent_clarifications.0",
                              "source": "user",
                              "evidence": [{"kind": "answer", "id": "agent_clarifications.0"}],
                              "updated_at": "2026-09-15T09:00:00+00:00"},
            "context": {"status": "known", "value_ref": "environment.required_apps",
                        "source": "observation", "evidence": [{"kind": "frame", "id": "f1"}],
                        "updated_at": "2026-09-15T09:00:00+00:00"}
          }
        }
        """#)
        let ledger = try XCTUnwrap(SkillLedger(procedure: proc))
        XCTAssertEqual(ledger.slots.map(\.key), ["goal", "done_criteria", "trigger", "steps", "context"])
        XCTAssertEqual(ledger.slots.map(\.line), [
            "Seen in 1 run, not confirmed",
            "You told me",
            "Still unknown",
            "Known from 2 runs",
            "Known from the recording",
        ])
        XCTAssertEqual(ledger.slots[1].label, "Done criteria")
        // The optional context slot does not count toward the headline.
        XCTAssertEqual(ledger.headline, "2 of 4 known")
    }
}
