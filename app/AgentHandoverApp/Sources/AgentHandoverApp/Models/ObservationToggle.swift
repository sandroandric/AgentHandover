import Foundation

/// Starts and stops the services off the main thread.
///
/// `ServiceController.startAll()` / `stopCapture()` spawn processes, talk to
/// launchd and wait for the daemon, which takes up to a few seconds. Run
/// on the main thread (as the menu bar buttons used to) that beachballs
/// the menu. This runs them on a background queue, exposes the in-flight
/// phase so the button can say "Starting…"/"Stopping…" and disable
/// itself, and calls back on the main queue when done.
@MainActor
final class ObservationToggle: ObservableObject {

    enum Phase: Equatable {
        case starting
        case stopping

        var label: String {
            switch self {
            case .starting: return "Starting…"
            case .stopping: return "Stopping…"
            }
        }
    }

    @Published private(set) var phase: Phase?

    var isTransitioning: Bool { phase != nil }

    private let startServices: () -> Bool
    private let stopServices: () -> Void

    init(
        startServices: @escaping () -> Bool = { ServiceController.startAll() },
        stopServices: @escaping () -> Void = { ServiceController.stopCapture() }
    ) {
        self.startServices = startServices
        self.stopServices = stopServices
    }

    /// Start the services in the background. Ignored while a start or stop
    /// is already in flight. `completion` runs on the main queue.
    func start(completion: @escaping @MainActor (Bool) -> Void = { _ in }) {
        guard phase == nil else { return }
        phase = .starting
        let work = startServices
        DispatchQueue.global(qos: .userInitiated).async {
            let ok = work()
            DispatchQueue.main.async {
                self.phase = nil
                completion(ok)
            }
        }
    }

    /// Stop the services in the background. Ignored while a start or stop
    /// is already in flight. `completion` runs on the main queue.
    func stop(completion: @escaping @MainActor () -> Void = {}) {
        guard phase == nil else { return }
        phase = .stopping
        let work = stopServices
        DispatchQueue.global(qos: .userInitiated).async {
            work()
            DispatchQueue.main.async {
                self.phase = nil
                completion()
            }
        }
    }
}
