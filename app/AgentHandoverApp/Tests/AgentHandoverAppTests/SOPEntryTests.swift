import XCTest
import Foundation
@testable import AgentHandoverApp

final class SOPEntryTests: XCTestCase {

    // MARK: - JSON Decoding

    func testDecodesMinimalJSON() throws {
        let json = """
        {
            "sop_id": "test-1",
            "slug": "check-email",
            "title": "Check Email",
            "source": "passive",
            "status": "approved",
            "confidence": 0.85,
            "created_at": "2026-03-10T10:00:00Z"
        }
        """.data(using: .utf8)!

        let entry = try JSONDecoder().decode(SOPEntry.self, from: json)
        XCTAssertEqual(entry.sop_id, "test-1")
        XCTAssertEqual(entry.slug, "check-email")
        XCTAssertEqual(entry.title, "Check Email")
        XCTAssertEqual(entry.confidence, 0.85)
        // Default lifecycle state when missing from JSON
        XCTAssertEqual(entry.lifecycleState, "observed")
    }

    func testDecodesWithLifecycleState() throws {
        let json = """
        {
            "sop_id": "test-2",
            "slug": "deploy-api",
            "title": "Deploy API",
            "source": "focus",
            "status": "approved",
            "confidence": 0.92,
            "created_at": "2026-03-10T10:00:00Z",
            "lifecycle_state": "agent_ready"
        }
        """.data(using: .utf8)!

        let entry = try JSONDecoder().decode(SOPEntry.self, from: json)
        XCTAssertEqual(entry.lifecycleState, "agent_ready")
    }

    func testDecodesOptionalFields() throws {
        let json = """
        {
            "sop_id": "test-3",
            "slug": "full-entry",
            "title": "Full Entry",
            "short_title": "Full",
            "tags": ["development", "deployment"],
            "source": "focus",
            "status": "draft",
            "confidence": 0.75,
            "created_at": "2026-03-10T10:00:00Z",
            "reviewed_at": "2026-03-11T10:00:00Z",
            "lifecycle_state": "reviewed"
        }
        """.data(using: .utf8)!

        let entry = try JSONDecoder().decode(SOPEntry.self, from: json)
        XCTAssertEqual(entry.short_title, "Full")
        XCTAssertEqual(entry.tags, ["development", "deployment"])
        XCTAssertEqual(entry.reviewed_at, "2026-03-11T10:00:00Z")
        XCTAssertEqual(entry.lifecycleState, "reviewed")
    }

    // MARK: - Display Title

    func testDisplayTitleUsesShortTitle() throws {
        let json = """
        {
            "sop_id": "t1", "slug": "s", "title": "Very Long Title Here",
            "short_title": "Short",
            "source": "focus", "status": "ok", "confidence": 0.5,
            "created_at": "2026-03-10T10:00:00Z"
        }
        """.data(using: .utf8)!

        let entry = try JSONDecoder().decode(SOPEntry.self, from: json)
        XCTAssertEqual(entry.displayTitle, "Short")
    }

    func testDisplayTitleStripsNoisePrefix() throws {
        let json = """
        {
            "sop_id": "t2", "slug": "s",
            "title": "The user is filing an expense report in Chrome",
            "source": "passive", "status": "ok", "confidence": 0.5,
            "created_at": "2026-03-10T10:00:00Z"
        }
        """.data(using: .utf8)!

        let entry = try JSONDecoder().decode(SOPEntry.self, from: json)
        // Should strip "The user is " prefix and capitalize
        XCTAssertTrue(entry.displayTitle.hasPrefix("Filing"))
        XCTAssertFalse(entry.displayTitle.contains("The user is"))
    }

    // MARK: - Lifecycle Labels

    func testLifecycleLabelMapping() throws {
        let cases: [(String, String)] = [
            ("observed", "Observed"),
            ("draft", "Draft"),
            ("reviewed", "Reviewed"),
            ("verified", "Verified"),
            ("agent_ready", "Agent Ready"),
            ("stale", "Stale"),
            ("archived", "Archived"),
        ]

        for (state, expected) in cases {
            let json = """
            {
                "sop_id": "t", "slug": "s", "title": "T",
                "source": "focus", "status": "ok", "confidence": 0.5,
                "created_at": "2026-03-10T10:00:00Z",
                "lifecycle_state": "\(state)"
            }
            """.data(using: .utf8)!

            let entry = try JSONDecoder().decode(SOPEntry.self, from: json)
            XCTAssertEqual(entry.lifecycleLabel, expected,
                           "Expected '\(expected)' for state '\(state)'")
        }
    }

    func testCanPromote() throws {
        let promotable = ["observed", "draft", "reviewed", "verified"]
        let notPromotable = ["agent_ready", "stale", "archived"]

        for state in promotable {
            let json = """
            {"sop_id":"t","slug":"s","title":"T","source":"f","status":"ok",
             "confidence":0.5,"created_at":"2026-03-10T10:00:00Z",
             "lifecycle_state":"\(state)"}
            """.data(using: .utf8)!
            let entry = try JSONDecoder().decode(SOPEntry.self, from: json)
            XCTAssertTrue(entry.canPromote, "\(state) should be promotable")
        }

        for state in notPromotable {
            let json = """
            {"sop_id":"t","slug":"s","title":"T","source":"f","status":"ok",
             "confidence":0.5,"created_at":"2026-03-10T10:00:00Z",
             "lifecycle_state":"\(state)"}
            """.data(using: .utf8)!
            let entry = try JSONDecoder().decode(SOPEntry.self, from: json)
            XCTAssertFalse(entry.canPromote, "\(state) should not be promotable")
        }
    }

    // MARK: - Source Labels

    func testSourceLabels() throws {
        let cases: [(String, String)] = [
            ("focus", "Focus Recording"),
            ("passive", "Auto-discovered"),
            ("unknown", "Imported"),
        ]

        for (source, expected) in cases {
            let json = """
            {"sop_id":"t","slug":"s","title":"T","source":"\(source)",
             "status":"ok","confidence":0.5,"created_at":"2026-03-10T10:00:00Z"}
            """.data(using: .utf8)!
            let entry = try JSONDecoder().decode(SOPEntry.self, from: json)
            XCTAssertEqual(entry.sourceLabel, expected)
        }
    }
}
