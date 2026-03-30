import XCTest
import Foundation
@testable import AgentHandoverApp

final class AppStateTests: XCTestCase {

    // MARK: - ServiceHealth

    func testServiceHealthLabels() {
        XCTAssertEqual(ServiceHealth.healthy.label, "Healthy")
        XCTAssertEqual(ServiceHealth.warning.label, "Warning")
        XCTAssertEqual(ServiceHealth.down.label, "Down")
        XCTAssertEqual(ServiceHealth.stopped.label, "Stopped")
    }

    func testServiceHealthRawValues() {
        XCTAssertEqual(ServiceHealth(rawValue: "healthy"), .healthy)
        XCTAssertEqual(ServiceHealth(rawValue: "warning"), .warning)
        XCTAssertEqual(ServiceHealth(rawValue: "down"), .down)
        XCTAssertEqual(ServiceHealth(rawValue: "stopped"), .stopped)
        XCTAssertNil(ServiceHealth(rawValue: "invalid"))
    }

    // MARK: - DaemonStatusFile Decoding

    func testDaemonStatusDecoding() throws {
        let json = """
        {
            "pid": 12345,
            "version": "0.2.0",
            "started_at": "2026-03-10T10:00:00Z",
            "heartbeat": "2026-03-10T10:05:00Z",
            "events_today": 150,
            "permissions_ok": true,
            "accessibility_permitted": true,
            "screen_recording_permitted": true,
            "db_path": "/tmp/events.db",
            "uptime_seconds": 300
        }
        """.data(using: .utf8)!

        let status = try JSONDecoder().decode(DaemonStatusFile.self, from: json)
        XCTAssertEqual(status.pid, 12345)
        XCTAssertEqual(status.version, "0.2.0")
        XCTAssertEqual(status.events_today, 150)
        XCTAssertTrue(status.permissions_ok)
        XCTAssertTrue(status.accessibility_permitted)
        XCTAssertEqual(status.uptime_seconds, 300)
        XCTAssertNil(status.last_extension_message)
    }

    // MARK: - FocusSessionSignalFile Decoding

    func testFocusSessionDecoding() throws {
        let json = """
        {
            "session_id": "sess-001",
            "title": "File expense report",
            "started_at": "2026-03-10T10:00:00Z",
            "status": "recording"
        }
        """.data(using: .utf8)!

        let session = try JSONDecoder().decode(FocusSessionSignalFile.self, from: json)
        XCTAssertEqual(session.session_id, "sess-001")
        XCTAssertEqual(session.title, "File expense report")
        XCTAssertEqual(session.status, "recording")
    }
}
