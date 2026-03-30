import Foundation
import AppKit

/// Controls AgentHandover services.
///
/// **Daemon**: Launched as a plain helper executable inside the main app
/// bundle. The main app owns all TCC-sensitive permissions, so the daemon
/// should not exist as a second macOS app principal at all.
///
/// **Worker**: Managed via launchd (no TCC requirements).
final class ServiceController {

    static let workerLabel = "com.agenthandover.worker"

    /// Path to the daemon executable. Lives OUTSIDE the app bundle so
    /// codesign --deep doesn't register it as a second TCC principal.
    static var daemonExecutableURL: URL {
        URL(fileURLWithPath: "/usr/local/lib/agenthandover/ah-observer")
    }

    private static var daemonExecPath: String {
        daemonExecutableURL.path
    }

    private static var uid: uid_t { getuid() }
    private static var guiDomain: String { "gui/\(uid)" }

    private static var launchAgentsDir: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/LaunchAgents")
    }

    private static var nativeMessagingManifestTargets: [(supportRoot: URL, manifestURL: URL)] {
        let appSupport = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support")
        let relativeRoots = [
            "Google/Chrome",
            "Chromium",
            "BraveSoftware/Brave-Browser",
            "Microsoft Edge",
            "Arc/User Data",
        ]

        return relativeRoots.map { relativeRoot in
            let supportRoot = appSupport.appendingPathComponent(relativeRoot)
            let manifestURL = supportRoot
                .appendingPathComponent("NativeMessagingHosts")
                .appendingPathComponent("com.agenthandover.host.json")
            return (supportRoot, manifestURL)
        }
    }

    // MARK: - Daemon (app-launched)

    /// Start daemon as a detached helper executable. Returns true if process is running.
    @discardableResult
    static func startDaemon() -> Bool {

        // Check if already running
        if isDaemonRunning() { return true }

        let process = Process()
        process.executableURL = daemonExecutableURL
        process.arguments = []
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
            Thread.sleep(forTimeInterval: 0.5)
            return isDaemonRunning()
        } catch {
            NSLog("ServiceController: failed to launch daemon executable %@ (%@)", daemonExecPath, error.localizedDescription)
            return false
        }
    }

    /// Stop daemon by sending SIGTERM to its process.
    static func stopDaemon() {
        guard let pid = daemonPid() else { return }
        kill(pid, SIGTERM)
    }

    /// Check if the daemon process is running.
    ///
    /// Validates the PID file by checking that the process at that PID is
    /// actually `ah-observer`, not a recycled PID from an unrelated process.
    static func isDaemonRunning() -> Bool {
        if let pid = daemonPid(), kill(pid, 0) == 0 {
            // Verify the PID is actually ah-observer, not a recycled PID
            let result = shell("/bin/ps", args: ["-p", "\(pid)", "-o", "comm="])
            if result.contains("ah-observer") {
                return true
            }
            // Stale PID file — remove it
            let home = FileManager.default.homeDirectoryForCurrentUser
            let pidPath = home.appendingPathComponent(
                "Library/Application Support/agenthandover/daemon.pid")
            try? FileManager.default.removeItem(at: pidPath)
            return false
        }
        // Fallback: check by process name
        let result = shell("/bin/ps", args: ["-ax", "-o", "pid,comm"])
        return result.contains("ah-observer")
    }

    /// Read the daemon's PID from its PID file.
    private static func daemonPid() -> Int32? {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let pidPath = home.appendingPathComponent(
            "Library/Application Support/agenthandover/daemon.pid")
        guard let content = try? String(contentsOf: pidPath, encoding: .utf8),
              let pid = Int32(content.trimmingCharacters(in: .whitespacesAndNewlines)),
              pid > 0 else { return nil }
        return pid
    }

    /// Block until the daemon process exits (up to timeout).
    static func waitForDaemonExit(timeoutSeconds: Int = 5) {
        let iterations = timeoutSeconds * 5
        for _ in 0..<iterations {
            if !isDaemonRunning() { return }
            Thread.sleep(forTimeInterval: 0.2)
        }
    }

    // MARK: - Worker (launchd-managed)

    @discardableResult
    static func startWorker() -> Bool {
        bootstrapIfNeeded(label: workerLabel)
        kickstart(label: workerLabel)
        Thread.sleep(forTimeInterval: 0.5)
        return isJobRunning(label: workerLabel)
    }

    static func stopWorker() {
        bootout(label: workerLabel)
    }

    // MARK: - Combined

    @discardableResult
    static func startAll() -> Bool {
        let d = startDaemon()
        let w = startWorker()
        return d && w
    }

    static func stopAll() {
        stopDaemon()
        stopWorker()
    }

    static func restartDaemon() {
        stopDaemon()
        waitForDaemonExit(timeoutSeconds: 3)
        startDaemon()
    }

    static func restartWorker() {
        stopWorker()
        Thread.sleep(forTimeInterval: 0.5)
        startWorker()
    }

    /// Quiesce any daemon process/job before app-owned permission requests.
    ///
    /// Screen Recording and Accessibility must be attributed to AgentHandover
    /// itself. If a legacy daemon process is still around from an older build
    /// or a stale launchd registration, stop it before asking TCC.
    static func prepareForAppOwnedPermissionRequest() {
        stopDaemon()
        waitForDaemonExit(timeoutSeconds: 2)
        bootout(label: "com.agenthandover.daemon")
        disable(label: "com.agenthandover.daemon")
        _ = shell("/usr/bin/killall", args: ["ah-observer"])
    }

    /// Remove native host registration while onboarding/permissions are active.
    /// This prevents an already-installed browser extension from respawning the
    /// daemon and surfacing it as a second TCC principal during setup.
    static func removeNativeMessagingHostManifest() {
        for (_, manifestURL) in nativeMessagingManifestTargets {
            try? FileManager.default.removeItem(at: manifestURL)
        }
    }

    /// Reinstall the native messaging host manifest once onboarding has
    /// progressed to extension setup or has fully completed.
    @discardableResult
    static func installNativeMessagingHostManifest() -> Bool {
        let manifest: [String: Any] = [
            "name": "com.agenthandover.host",
            "description": "AgentHandover native messaging host",
            "path": daemonExecPath,
            "type": "stdio",
            "args": ["--native-messaging"],
            "allowed_origins": [
                "chrome-extension://knldjmfmopnpolahpmmgbagdohdnhkik/",
            ],
        ]

        let fileManager = FileManager.default
        let targets = nativeMessagingManifestTargets.filter { supportRoot, _ in
            fileManager.fileExists(atPath: supportRoot.path)
        }

        guard !targets.isEmpty else {
            NSLog("ServiceController: no supported Chromium browser support roots found for native host install")
            return false
        }

        do {
            let data = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted])
            var installedAny = false

            for (_, manifestURL) in targets {
                let manifestDir = manifestURL.deletingLastPathComponent()
                try fileManager.createDirectory(
                    at: manifestDir,
                    withIntermediateDirectories: true
                )
                try data.write(to: manifestURL, options: .atomic)
                installedAny = true
            }

            return installedAny
        } catch {
            NSLog("ServiceController: failed to write native host manifests (%@)", error.localizedDescription)
            return false
        }
    }

    // MARK: - Launchd (worker only)

    private static func bootstrapIfNeeded(label: String) {
        launchctl(["bootstrap", guiDomain, plistPath(label)])
    }

    @discardableResult
    private static func kickstart(label: String) -> Bool {
        let result = launchctl(["kickstart", "\(guiDomain)/\(label)"])
        return result.exitCode == 0
    }

    private static func bootout(label: String) {
        launchctl(["bootout", "\(guiDomain)/\(label)"])
    }

    private static func disable(label: String) {
        launchctl(["disable", "\(guiDomain)/\(label)"])
    }

    static func isJobRunning(label: String) -> Bool {
        let result = launchctl(["print", "\(guiDomain)/\(label)"])
        if result.exitCode != 0 { return false }
        let lines = result.output.components(separatedBy: "\n")
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("pid =") || trimmed.hasPrefix("pid=") {
                let parts = trimmed.components(separatedBy: "=")
                if let pidStr = parts.last?.trimmingCharacters(in: .whitespaces),
                   let pid = Int(pidStr), pid > 0 {
                    return true
                }
            }
        }
        return false
    }

    static func isServiceHealthy(label: String) -> Bool {
        guard isJobRunning(label: label) else { return false }

        let statusFileName: String
        switch label {
        case workerLabel:
            statusFileName = "worker-status.json"
        default:
            return true
        }

        let statusDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover")
        let statusFile = statusDir.appendingPathComponent(statusFileName)

        guard let data = try? Data(contentsOf: statusFile),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let heartbeatString = json["heartbeat"] as? String else {
            return true
        }

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        var heartbeatDate = formatter.date(from: heartbeatString)
        if heartbeatDate == nil {
            formatter.formatOptions = [.withInternetDateTime]
            heartbeatDate = formatter.date(from: heartbeatString)
        }

        guard let date = heartbeatDate else { return true }
        return Date().timeIntervalSince(date) <= 30
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

    private static func shell(_ path: String, args: [String]) -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = args
        let pipe = Pipe()
        process.standardOutput = pipe
        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8) ?? ""
        } catch {
            return ""
        }
    }
}
