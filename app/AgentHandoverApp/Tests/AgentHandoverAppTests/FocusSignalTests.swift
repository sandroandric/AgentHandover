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
        let now = ISO8601DateFormatter().date(from: "2026-09-23T10:05:00Z")!
        let stopped = try XCTUnwrap(FocusSignal.stop(in: dir, now: now))
        XCTAssertEqual(stopped.status, "stopped")

        // Read back as raw JSON: the fields the Rust side parses must be
        // byte-for-byte what the CLI wrote (the daemon tagged events with
        // this lowercase id), plus the stop time the worker reads.
        let data = try Data(contentsOf: dir.appendingPathComponent(FocusSignal.fileName))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: String])
        XCTAssertEqual(json, [
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Send weekly report",
            "started_at": "2026-09-23T10:00:00.123456789+00:00",
            "status": "stopped",
            "stopped_at": "2026-09-23T10:05:00Z",
        ])
    }

    /// `agenthandover focus stop` writes `stopped_at` with chrono's
    /// `to_rfc3339` (fractional seconds, offset): the app still reads the
    /// signal, and a signal without it (older writers) reads as nil.
    func testCLIStoppedAtDecodesAndIsOptional() throws {
        let json = """
        {
          "session_id": "550e8400-e29b-41d4-a716-446655440000",
          "title": "Send weekly report",
          "started_at": "2026-09-23T10:00:00.123456789+00:00",
          "status": "stopped",
          "stopped_at": "2026-09-23T10:05:00.987654321+00:00"
        }
        """
        try json.write(
            to: dir.appendingPathComponent(FocusSignal.fileName), atomically: true, encoding: .utf8
        )
        let signal = try XCTUnwrap(FocusSignal.read(in: dir))
        XCTAssertEqual(signal.stopped_at, "2026-09-23T10:05:00.987654321+00:00")
        XCTAssertNotNil(FocusSignal.parseTimestamp(signal.stopped_at!))

        try writeCLISignal(status: "recording")
        XCTAssertNil(try XCTUnwrap(FocusSignal.read(in: dir)).stopped_at)
    }

    /// A second stop does not move the first stop time.
    func testStopOfAStoppedSessionKeepsItsStoppedAt() throws {
        try FocusSignal.start(title: "Send weekly report", in: dir)
        let first = ISO8601DateFormatter().date(from: "2026-09-23T10:05:00Z")!
        try FocusSignal.stop(in: dir, now: first)
        XCTAssertNil(try FocusSignal.stop(in: dir, now: first.addingTimeInterval(60)))
        XCTAssertEqual(FocusSignal.read(in: dir)?.stopped_at, "2026-09-23T10:05:00Z")
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

    /// Start from the menu when the daemon would not start: the recording
    /// is taken back, and only that one.
    func testDiscardRemovesOnlyThatRecordingSession() throws {
        let started = try FocusSignal.start(title: "Send weekly report", in: dir)
        FocusSignal.discard(sessionID: "another-session", in: dir)
        XCTAssertEqual(FocusSignal.read(in: dir), started)

        FocusSignal.discard(sessionID: started.session_id, in: dir)
        XCTAssertNil(FocusSignal.read(in: dir))
    }

    func testDiscardLeavesAStoppedSessionForTheWorker() throws {
        let started = try FocusSignal.start(title: "Send weekly report", in: dir)
        try FocusSignal.stop(in: dir)
        FocusSignal.discard(sessionID: started.session_id, in: dir)
        XCTAssertEqual(FocusSignal.read(in: dir)?.status, "stopped")
    }
}

/// The worker's `focus-result.json` for a stopped session with no events,
/// and what the menu shows for it.
final class FocusResultTests: XCTestCase {
    private var dir: URL!

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("focus-result-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: dir)
    }

    private let sessionID = "550e8400-e29b-41d4-a716-446655440000"

    /// Exactly what the worker's `_close_empty_focus_session` writes
    /// (json.dumps with indent 2, Python `isoformat()` timestamps).
    private func writeWorkerResult(sessionID: String, stoppedAt: String = "\"2026-09-23T10:05:00+00:00\"",
                                   status: String = "nothing_recorded") throws {
        let json = """
        {
          "session_id": "\(sessionID)",
          "title": "Send weekly report",
          "status": "\(status)",
          "message": "Nothing was recorded: observation was off during this session",
          "stopped_at": \(stoppedAt),
          "closed_at": "2026-09-23T10:07:00.412345+00:00"
        }
        """
        try json.write(
            to: dir.appendingPathComponent(FocusResult.fileName), atomically: true, encoding: .utf8
        )
    }

    private func signal(_ id: String, _ status: String) -> FocusSessionSignalFile {
        FocusSessionSignalFile(
            session_id: id, title: "Next session", started_at: "2026-09-23T11:00:00Z", status: status
        )
    }

    // MARK: Decoding

    func testDecodesTheWorkersFile() throws {
        try writeWorkerResult(sessionID: sessionID)
        let result = try XCTUnwrap(FocusResult.read(in: dir))
        XCTAssertEqual(result.session_id, sessionID)
        XCTAssertEqual(result.title, "Send weekly report")
        XCTAssertEqual(result.status, "nothing_recorded")
        XCTAssertEqual(result.stopped_at, "2026-09-23T10:05:00+00:00")
        XCTAssertEqual(result.closed_at, "2026-09-23T10:07:00.412345+00:00")
    }

    func testDecodesANullStoppedAt() throws {
        try writeWorkerResult(sessionID: sessionID, stoppedAt: "null")
        XCTAssertNil(try XCTUnwrap(FocusResult.read(in: dir)).stopped_at)
    }

    func testMissingOrBrokenFileReadsAsNil() throws {
        XCTAssertNil(FocusResult.read(in: dir))
        try "{not json".write(
            to: dir.appendingPathComponent(FocusResult.fileName), atomically: true, encoding: .utf8
        )
        XCTAssertNil(FocusResult.read(in: dir))
    }

    // MARK: Menu state

    func testResultWithNoSignalShowsTheMessageInsteadOfAnalyzing() throws {
        try writeWorkerResult(sessionID: sessionID)
        let state = FocusRecordState(signal: nil, result: FocusResult.read(in: dir))
        XCTAssertEqual(state, .nothingRecorded(sessionId: sessionID, title: "Send weekly report"))
        XCTAssertEqual(
            state.nothingRecordedMessage,
            "Nothing was recorded for 'Send weekly report': observation was off during this session"
        )
    }

    /// The worker writes the result before it removes the signal.
    func testResultForTheStoppedSessionReplacesAnalyzing() throws {
        try writeWorkerResult(sessionID: sessionID)
        let state = FocusRecordState(
            signal: signal(sessionID, "stopped"), result: FocusResult.read(in: dir)
        )
        XCTAssertEqual(state, .nothingRecorded(sessionId: sessionID, title: "Send weekly report"))
    }

    func testResultIsIgnoredWhileANewSessionRecords() throws {
        try writeWorkerResult(sessionID: sessionID)
        let state = FocusRecordState(
            signal: signal("new-session", "recording"), result: FocusResult.read(in: dir)
        )
        guard case .recording(let id, let title, _) = state else {
            return XCTFail("expected recording, got \(state)")
        }
        XCTAssertEqual(id, "new-session")
        XCTAssertEqual(title, "Next session")
        XCTAssertNil(state.nothingRecordedMessage)
    }

    func testAnOlderResultDoesNotHideANewStoppedSession() throws {
        try writeWorkerResult(sessionID: sessionID)
        let state = FocusRecordState(
            signal: signal("new-session", "stopped"), result: FocusResult.read(in: dir)
        )
        XCTAssertEqual(state, .processing(title: "Next session"))
    }

    func testUnknownResultStatusShowsNothing() throws {
        try writeWorkerResult(sessionID: sessionID, status: "something_else")
        XCTAssertEqual(FocusRecordState(signal: nil, result: FocusResult.read(in: dir)), .idle)
    }

    func testOtherStatesHaveNoMessage() {
        XCTAssertNil(FocusRecordState.idle.nothingRecordedMessage)
        XCTAssertNil(FocusRecordState.processing(title: "x").nothingRecordedMessage)
    }

    // MARK: Dismiss

    func testDismissDeletesTheFileForThatSessionOnly() throws {
        try writeWorkerResult(sessionID: sessionID)
        FocusResult.dismiss(sessionID: "another-session", in: dir)
        XCTAssertNotNil(FocusResult.read(in: dir))

        FocusResult.dismiss(sessionID: sessionID, in: dir)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: dir.appendingPathComponent(FocusResult.fileName).path
        ))
        XCTAssertEqual(FocusRecordState(signal: nil, result: FocusResult.read(in: dir)), .idle)
    }

    /// The 2 s quick check notices the file through its FileStamp.
    func testFileStampNoticesTheResultAppearingAndGoing() throws {
        let url = dir.appendingPathComponent(FocusResult.fileName)
        var stamp = FileStamp()
        XCTAssertTrue(stamp.changed(url))
        XCTAssertFalse(stamp.changed(url))
        try writeWorkerResult(sessionID: sessionID)
        XCTAssertTrue(stamp.changed(url))
        FocusResult.dismiss(sessionID: sessionID, in: dir)
        XCTAssertTrue(stamp.changed(url))
    }
}
