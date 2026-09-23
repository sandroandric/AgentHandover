import XCTest
import Foundation
@testable import AgentHandoverApp

/// The popover's focus-recording state comes only from focus-session.json,
/// so a session started or stopped with `agenthandover focus start/stop`
/// shows the same way in the app, and a stop from the app keeps the
/// session id the CLI (and the daemon's event tags) use.
final class FocusSignalTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("focus-signal-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    /// Exactly what `agenthandover focus start "Send weekly report"` writes
    /// (serde_json pretty print of FocusSessionSignal, chrono to_rfc3339).
    private func writeCLISignal(status: String) throws {
        let json = """
        {
          "session_id": "550e8400-e29b-41d4-a716-446655440000",
          "title": "Send weekly report",
          "started_at": "2026-09-23T10:00:00.123456789+00:00",
          "status": "\(status)"
        }
        """
        try json.write(
            to: dir.appendingPathComponent(FocusSignal.fileName),
            atomically: true,
            encoding: .utf8
        )
    }

    func testNoSignalIsIdle() {
        XCTAssertEqual(FocusRecordState(signal: FocusSignal.read(in: dir)), .idle)
    }

    func testCLIStartedSessionShowsAsRecordingWithItsTitleAndStart() throws {
        try writeCLISignal(status: "recording")
        let state = FocusRecordState(signal: FocusSignal.read(in: dir))
        guard case .recording(let id, let title, let startedAt) = state else {
            return XCTFail("expected recording, got \(state)")
        }
        XCTAssertEqual(id, "550e8400-e29b-41d4-a716-446655440000")
        XCTAssertEqual(title, "Send weekly report")
        let expected = ISO8601DateFormatter().date(from: "2026-09-23T10:00:00Z")!
        XCTAssertEqual(startedAt!.timeIntervalSince1970, expected.timeIntervalSince1970, accuracy: 1)
    }

    func testCLIStoppedSessionShowsAsProcessing() throws {
        try writeCLISignal(status: "stopped")
        XCTAssertEqual(
            FocusRecordState(signal: FocusSignal.read(in: dir)),
            .processing(title: "Send weekly report")
        )
    }

    func testStopFromAppKeepsTheCLISessionIdTitleAndStart() throws {
        try writeCLISignal(status: "recording")
        let stopped = try XCTUnwrap(FocusSignal.stop(in: dir))
        XCTAssertEqual(stopped.status, "stopped")

        // Read back as raw JSON: the fields the Rust side parses must be
        // byte-for-byte what the CLI wrote (the daemon tagged events with
        // this lowercase id).
        let data = try Data(contentsOf: dir.appendingPathComponent(FocusSignal.fileName))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: String])
        XCTAssertEqual(json, [
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Send weekly report",
            "started_at": "2026-09-23T10:00:00.123456789+00:00",
            "status": "stopped",
        ])
    }

    func testStopWithNothingRecordingChangesNothing() throws {
        XCTAssertNil(try FocusSignal.stop(in: dir))
        try writeCLISignal(status: "stopped")
        XCTAssertNil(try FocusSignal.stop(in: dir))
        XCTAssertEqual(FocusSignal.read(in: dir)?.status, "stopped")
    }

    func testAppStartWritesWhatTheCLIStopReads() throws {
        let now = ISO8601DateFormatter().date(from: "2026-09-23T09:30:00Z")!
        let started = try FocusSignal.start(title: "File expense report", in: dir, now: now)

        let data = try Data(contentsOf: dir.appendingPathComponent(FocusSignal.fileName))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: String])
        XCTAssertEqual(Set(json.keys), ["session_id", "title", "started_at", "status"])
        XCTAssertEqual(json["status"], "recording")
        XCTAssertEqual(json["started_at"], "2026-09-23T09:30:00Z")
        XCTAssertNotNil(UUID(uuidString: started.session_id))
        XCTAssertEqual(started.session_id, started.session_id.lowercased())
        XCTAssertEqual(
            FocusRecordState(signal: FocusSignal.read(in: dir)),
            .recording(sessionId: started.session_id, title: "File expense report", startedAt: now)
        )
    }

    func testUnknownStatusIsIdle() {
        let signal = FocusSessionSignalFile(
            session_id: "s", title: "t", started_at: "2026-09-23T10:00:00Z", status: "weird"
        )
        XCTAssertEqual(FocusRecordState(signal: signal), .idle)
    }
}
