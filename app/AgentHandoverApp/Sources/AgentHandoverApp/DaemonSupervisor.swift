import Foundation

/// Restarts the app-spawned daemon when it exits abnormally.
///
/// The daemon is a child of this app (TCC attributes its capture to the
/// app), so launchd cannot supervise it. Without this, a daemon crash left
/// capture silently off until the user noticed and pressed Start again.
///
/// Policy, mirroring launchd's `KeepAlive { SuccessfulExit = false }`:
/// - a stop the app asked for (`stopRequested`) is never undone;
/// - exit status 0, SIGTERM and SIGINT are deliberate stops (the CLI's
///   `agenthandover stop` sends SIGTERM) and are not restarted;
/// - anything else is a crash: restart after 1, 2, 4, ... 60 s, with the
///   count reset once a daemon has stayed up for a minute.
final class DaemonSupervisor {

    static let shared = DaemonSupervisor(
        restart: { ServiceController.startDaemon() },
        schedule: { delay, work in
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay, execute: work)
        }
    )

    /// Posted on the main queue whenever `statusLine` changes.
    static let statusChanged = Notification.Name("AgentHandoverDaemonSupervisorStatusChanged")

    static let stableRunSeconds: TimeInterval = 60
    static let maxBackoffSeconds: TimeInterval = 60

    /// Delay before the `restartNumber`-th consecutive restart (1-based).
    static func backoffSeconds(forRestart restartNumber: Int) -> TimeInterval {
        let exponent = Double(max(restartNumber, 1) - 1)
        return min(maxBackoffSeconds, pow(2, exponent))
    }

    /// Whether an exit is a crash (restart) rather than a deliberate stop.
    static func isAbnormalExit(reason: Process.TerminationReason, status: Int32) -> Bool {
        switch reason {
        case .exit:
            return status != 0
        case .uncaughtSignal:
            return status != SIGTERM && status != SIGINT
        @unknown default:
            return true
        }
    }

    private let restart: () -> Bool
    private let schedule: (TimeInterval, @escaping () -> Void) -> Void
    private let now: () -> Date
    private let lock = NSLock()
    private var stopWasRequested = false
    private var lastStart: Date?
    private var consecutiveRestarts = 0
    /// Bumped by every stop so a restart scheduled before it is dropped.
    private var generation = 0
    private var status: String?

    init(
        restart: @escaping () -> Bool,
        schedule: @escaping (TimeInterval, @escaping () -> Void) -> Void,
        now: @escaping () -> Date = Date.init
    ) {
        self.restart = restart
        self.schedule = schedule
        self.now = now
    }

    /// One line for the menu: why the daemon was restarted, or nil.
    var statusLine: String? {
        lock.lock()
        defer { lock.unlock() }
        return status
    }

    /// Called right before the app spawns a daemon.
    func daemonWillStart() {
        lock.lock()
        stopWasRequested = false
        lastStart = now()
        lock.unlock()
    }

    /// Called when the app stops the daemon on purpose.
    func stopRequested() {
        lock.lock()
        stopWasRequested = true
        generation += 1
        consecutiveRestarts = 0
        lock.unlock()
        setStatus(nil)
    }

    /// Termination handler of a daemon process the app spawned.
    func daemonExited(reason: Process.TerminationReason, status exitStatus: Int32) {
        lock.lock()
        guard !stopWasRequested,
              Self.isAbnormalExit(reason: reason, status: exitStatus) else {
            lock.unlock()
            return
        }
        if let started = lastStart, now().timeIntervalSince(started) >= Self.stableRunSeconds {
            consecutiveRestarts = 0
        }
        consecutiveRestarts += 1
        let restartNumber = consecutiveRestarts
        let scheduledGeneration = generation
        lock.unlock()

        let delay = Self.backoffSeconds(forRestart: restartNumber)
        let how = reason == .uncaughtSignal ? "signal \(exitStatus)" : "exit code \(exitStatus)"
        setStatus("Capture daemon stopped unexpectedly (\(how)); restarting in \(Int(delay)) s")

        schedule(delay) { [weak self] in
            guard let self else { return }
            self.lock.lock()
            let stillWanted = !self.stopWasRequested && self.generation == scheduledGeneration
            self.lock.unlock()
            guard stillWanted else { return }

            if self.restart() {
                let time = DateFormatter.localizedString(from: self.now(), dateStyle: .none, timeStyle: .short)
                self.setStatus("Capture daemon restarted at \(time) after an unexpected exit (restart \(restartNumber))")
            } else {
                // The spawn itself failed, so no termination handler will
                // fire: go round again with the next backoff step.
                self.daemonExited(reason: .exit, status: -1)
            }
        }
    }

    private func setStatus(_ line: String?) {
        lock.lock()
        let changed = status != line
        status = line
        lock.unlock()
        guard changed else { return }
        DispatchQueue.main.async {
            NotificationCenter.default.post(name: Self.statusChanged, object: nil)
        }
    }
}
