import AppKit
import Foundation

/// Memory's state for the menu, the Memory window and Memory Settings, and
/// the actions they share: pause, resume, forget (after a confirmation),
/// and turning Memory on or off.  Everything goes through the worker's
/// query API (`MemoryAPI`); the switches live in config.toml.
@MainActor
final class MemoryCenter: ObservableObject {
    static let shared = MemoryCenter()

    @Published private(set) var status: MemoryStatus?
    /// Why the last status request failed; nil after a success.
    @Published private(set) var statusError: MemoryAPIError?
    /// The last action's outcome, in one line ("Forgot 3 moments.").
    @Published var notice: String?
    /// Bumped after a forget, so open result lists search again.
    @Published private(set) var forgetCount = 0

    let api: MemoryAPI

    init(api: MemoryAPI = MemoryAPI()) {
        self.api = api
    }

    var menuState: MemoryMenuState { MemoryMenuState(status: status) }

    func refresh() async {
        do {
            status = try await api.status()
            statusError = nil
        } catch {
            status = nil
            statusError = error as? MemoryAPIError ?? .workerNotRunning
        }
    }

    func pause(_ option: MemoryPauseOption) async {
        await apply { try await self.api.pause(minutes: option.minutes(now: Date())) }
    }

    func resume() async {
        await apply { try await self.api.resume() }
    }

    private func apply(_ call: @escaping () async throws -> MemoryStatus) async {
        do {
            status = try await call()
            statusError = nil
            notice = nil
        } catch {
            notice = Self.describe(error)
        }
    }

    /// Confirm, then forget.  True when the worker forgot.
    @discardableResult
    func forget(_ request: MemoryForgetRequest) async -> Bool {
        let now = Date()
        let confirmation = request.confirmation(now: now)
        NSApp.activate(ignoringOtherApps: true)
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = confirmation.title
        alert.informativeText = confirmation.message
        alert.addButton(withTitle: confirmation.button)
        alert.addButton(withTitle: "Cancel")
        alert.buttons.first?.hasDestructiveAction = true
        guard alert.runModal() == .alertFirstButtonReturn else { return false }
        do {
            let counts = try await api.forget(request.selector(now: now))
            notice = MemoryForgetRequest.summary(counts)
            forgetCount += 1
            await refresh()
            return true
        } catch {
            notice = Self.describe(error)
            return false
        }
    }

    /// Write changed `[memory]` settings; the worker and the daemon are
    /// restarted when a key they read at start changed.
    func save(_ settings: MemorySettings, keys: [MemorySettings.Key]) async {
        guard !keys.isEmpty else { return }
        let restart = keys.contains { $0.needsRestart }
        await Task.detached(priority: .userInitiated) {
            if MemorySettings.write(settings, keys: keys), restart {
                MemorySettings.restartServices()
            }
        }.value
        await refresh()
    }

    /// Turn Memory on (the window's and the menu's "Turn On").
    func turnOn() async {
        var settings = await Task.detached { MemorySettings.load() }.value
        settings.enabled = true
        await save(settings, keys: [.enabled])
    }

    static func describe(_ error: Error) -> String {
        switch error as? MemoryAPIError {
        case .workerNotRunning?, .noToken?:
            return "The worker is not running."
        case .memoryOff?:
            return "Memory is off."
        case .modelUnavailable(let detail)?:
            return detail.isEmpty ? "Memory's model is not available." : detail
        case .badRequest(let reason)?:
            return reason
        case .unexpected(let status, _)?:
            return "The worker answered \(status)."
        case nil:
            return error.localizedDescription
        }
    }
}
