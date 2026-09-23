import Foundation

/// What the popover shows for focus recording, derived only from
/// `focus-session.json`: the one signal the CLI (`agenthandover focus
/// start/stop`), the app, the daemon and the worker all share. The popover
/// keeps no recording state of its own, so a session started or stopped
/// from either side shows the same way in both.
enum FocusRecordState: Equatable {
    case idle
    case recording(sessionId: String, title: String, startedAt: Date?)
    /// Stopped, waiting for the worker to turn it into a draft.
    case processing(title: String)

    init(signal: FocusSessionSignalFile?) {
        guard let signal else {
            self = .idle
            return
        }
        switch signal.status {
        case "recording":
            self = .recording(
                sessionId: signal.session_id,
                title: signal.title,
                startedAt: FocusSignal.parseTimestamp(signal.started_at)
            )
        case "stopped":
            self = .processing(title: signal.title)
        default:
            self = .idle
        }
    }
}

/// Reads and writes `focus-session.json` in the same shape and with the same
/// atomic replace as `agenthandover_common::focus_session`.
enum FocusSignal {
    static let fileName = "focus-session.json"

    static var defaultDirectory: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover")
    }

    static func read(in dir: URL) -> FocusSessionSignalFile? {
        guard let data = try? Data(contentsOf: dir.appendingPathComponent(fileName)) else {
            return nil
        }
        return try? JSONDecoder().decode(FocusSessionSignalFile.self, from: data)
    }

    static func write(_ signal: FocusSessionSignalFile, in dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        // `.atomic` writes a temporary file and renames it over the target.
        try encoder.encode(signal).write(to: dir.appendingPathComponent(fileName), options: .atomic)
    }

    /// Starts a recording: a new session id and `status: "recording"`.
    @discardableResult
    static func start(title: String, in dir: URL, now: Date = Date()) throws -> FocusSessionSignalFile {
        let signal = FocusSessionSignalFile(
            session_id: UUID().uuidString.lowercased(),
            title: title,
            started_at: ISO8601DateFormatter().string(from: now),
            status: "recording"
        )
        try write(signal, in: dir)
        return signal
    }

    /// Stops the session that is recording, whoever started it: the session
    /// id, title and start time stay exactly as written (the daemon tagged
    /// the session's events with that id), only the status changes.
    /// Returns nil when nothing is recording.
    @discardableResult
    static func stop(in dir: URL) throws -> FocusSessionSignalFile? {
        guard var signal = read(in: dir), signal.status == "recording" else {
            return nil
        }
        signal.status = "stopped"
        try write(signal, in: dir)
        return signal
    }

    /// Takes back a recording that never captured anything (the daemon
    /// did not start): removes the signal if it is still that session
    /// recording, so no empty session is stopped and processed.
    static func discard(sessionID: String, in dir: URL) {
        guard let signal = read(in: dir),
              signal.session_id == sessionID,
              signal.status == "recording" else { return }
        try? FileManager.default.removeItem(at: dir.appendingPathComponent(fileName))
    }

    /// What the menu says when Start could not start observation.
    static let daemonDidNotStart =
        "Observation could not start, so nothing was recorded. Check the menu's status, or run: agenthandover logs daemon"

    /// Parses the signal's `started_at`: the CLI writes RFC 3339 with
    /// fractional seconds and an offset (chrono's `to_rfc3339`), the app
    /// writes whole seconds with `Z`.
    static func parseTimestamp(_ text: String) -> Date? {
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: text) {
            return date
        }
        return ISO8601DateFormatter().date(from: text)
    }
}
