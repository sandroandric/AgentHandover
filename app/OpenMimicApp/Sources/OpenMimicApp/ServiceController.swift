import Foundation

/// Controls OpenMimic services via launchctl.
final class ServiceController {

    static let daemonLabel = "com.openmimic.daemon"
    static let workerLabel = "com.openmimic.worker"

    private static var launchAgentsDir: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/LaunchAgents")
    }

    // MARK: - Start

    /// Start daemon and return whether the job is actually running afterward.
    @discardableResult
    static func startDaemon() -> Bool {
        launchctl(["load", "-w", plistPath(daemonLabel)])
        Thread.sleep(forTimeInterval: 0.5)
        return isJobRunning(label: daemonLabel)
    }

    /// Start worker and return whether the job is actually running afterward.
    @discardableResult
    static func startWorker() -> Bool {
        launchctl(["load", "-w", plistPath(workerLabel)])
        Thread.sleep(forTimeInterval: 0.5)
        return isJobRunning(label: workerLabel)
    }

    /// Start all services and return true only if both are verified running.
    @discardableResult
    static func startAll() -> Bool {
        let d = startDaemon()
        let w = startWorker()
        return d && w
    }

    // MARK: - Stop

    static func stopDaemon() {
        launchctl(["unload", plistPath(daemonLabel)])
    }

    static func stopWorker() {
        launchctl(["unload", plistPath(workerLabel)])
    }

    static func stopAll() {
        stopDaemon()
        stopWorker()
    }

    // MARK: - Restart

    /// Restart daemon: unload, wait for process to exit, then reload.
    static func restartDaemon() {
        stopDaemon()
        waitForPidExit(label: daemonLabel) {
            startDaemon()
        }
    }

    /// Restart worker: unload, wait for process to exit, then reload.
    static func restartWorker() {
        stopWorker()
        waitForPidExit(label: workerLabel) {
            startWorker()
        }
    }

    /// Restart all: stop both, wait for exit, then start both.
    static func restartAll() {
        stopAll()
        waitForPidExit(label: daemonLabel) {
            self.waitForPidExit(label: self.workerLabel) {
                startAll()
            }
        }
    }

    /// Poll until the launchd job is no longer running, then call completion.
    /// Falls back to a 2-second timeout if the job doesn't exit cleanly.
    private static func waitForPidExit(label: String, attempt: Int = 0, completion: @escaping () -> Void) {
        let result = launchctl(["list", label])
        if result.exitCode != 0 || attempt >= 10 {
            // Job is gone or we've waited long enough (10 × 0.2s = 2s max)
            DispatchQueue.main.async { completion() }
        } else {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                waitForPidExit(label: label, attempt: attempt + 1, completion: completion)
            }
        }
    }

    // MARK: - Helpers

    /// Check whether a launchd job is actually running via `launchctl list <label>`.
    static func isJobRunning(label: String) -> Bool {
        let result = launchctl(["list", label])
        return result.exitCode == 0
    }

    private static func plistPath(_ label: String) -> String {
        launchAgentsDir.appendingPathComponent("\(label).plist").path
    }

    @discardableResult
    private static func launchctl(_ args: [String]) -> (exitCode: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = args

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            process.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8) ?? ""
            return (process.terminationStatus, output)
        } catch {
            return (-1, "Failed to run launchctl: \(error)")
        }
    }
}
