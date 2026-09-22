import Foundation

/// The worker's trigger spool: how the app asks the worker to do something.
///
/// Each request is its own file, `<state>/triggers/<kind>-<uuid>.json`,
/// written to a dot-prefixed tmp file and renamed into place.  The worker
/// drains the directory once per tick, oldest first, so two clicks before
/// the worker's next tick are both honoured (the old single-slot
/// `<kind>-trigger.json` files kept only the last one).
enum TriggerSpool {
    static let directoryName = "triggers"

    /// The worker's state directory.
    static var defaultStateDir: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover")
    }

    /// Queue one request of `kind`; returns the spool file, or nil on failure.
    @discardableResult
    static func write(kind: String, payload: [String: Any], stateDir: URL = defaultStateDir) -> URL? {
        let dir = stateDir.appendingPathComponent(directoryName)
        guard
            (try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)) != nil,
            let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys])
        else { return nil }

        let name = "\(kind)-\(UUID().uuidString).json"
        let target = dir.appendingPathComponent(name)
        let tmp = dir.appendingPathComponent(".\(name).tmp")
        do {
            try data.write(to: tmp)
            try FileManager.default.moveItem(at: tmp, to: target)
            return target
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            return nil
        }
    }
}
