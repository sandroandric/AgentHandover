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
            "Comet",
        ]

        return relativeRoots.map { relativeRoot in
            let supportRoot = appSupport.appendingPathComponent(relativeRoot)
            let manifestURL = supportRoot
                .appendingPathComponent("NativeMessagingHosts")
                .appendingPathComponent("com.agenthandover.host.json")
            return (supportRoot, manifestURL)
        }
    }

    // MARK: - Bounded polling

    /// Poll `condition` every `interval` until it holds or `timeout` passes.
    /// Returns whether it held. Replaces fixed sleeps: the common fast case
    /// returns after the first check or two instead of a full sleep.
    @discardableResult
    static func waitUntil(
        timeout: TimeInterval = 5,
        interval: TimeInterval = 0.05,
        _ condition: () -> Bool
    ) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while true {
            if condition() { return true }
            if Date() >= deadline { return false }
            Thread.sleep(forTimeInterval: interval)
        }
    }

    // MARK: - Daemon (app-launched)

    /// Start daemon as a detached helper executable. Returns true if process is running.
    ///
    /// A daemon this app spawns is supervised: if it exits abnormally while
    /// the user wants observation on, it is restarted with backoff (see
    /// `DaemonSupervisor`).
    @discardableResult
    static func startDaemon() -> Bool {

        // Check if already running
        if isDaemonRunning() {
            let pidFile = daemonPid().map { String($0) } ?? "absent"
            launchLog.notice("daemon already running (pid file \(pidFile, privacy: .public)), not starting another")
            return true
        }
        launchLog.notice("starting the daemon")

        let process = Process()
        process.executableURL = daemonExecutableURL
        process.arguments = []
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice

        // Redirect stderr to a log file instead of /dev/null.  Previously
        // piped to nullDevice, which swallowed Rust panic backtraces and
        // ObjC NSException messages — so when the daemon died abnormally
        // (hikoae's v0.2.7 OCR-timeout abort), there was no evidence
        // anywhere of WHY it crashed.  Capturing stderr to a rotating log
        // file means future mystery exits leave a diagnostic trail.
        let logsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover/logs")
        try? FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
        let stderrLogURL = logsDir.appendingPathComponent("daemon.stderr.log")
        if !FileManager.default.fileExists(atPath: stderrLogURL.path) {
            FileManager.default.createFile(atPath: stderrLogURL.path, contents: nil)
        }
        if let stderrHandle = try? FileHandle(forWritingTo: stderrLogURL) {
            stderrHandle.seekToEndOfFile()
            process.standardError = stderrHandle
        } else {
            process.standardError = FileHandle.nullDevice
        }

        process.terminationHandler = { finished in
            DaemonSupervisor.shared.daemonExited(
                reason: finished.terminationReason,
                status: finished.terminationStatus
            )
        }

        do {
            DaemonSupervisor.shared.daemonWillStart()
            try process.run()
            return waitUntil { isDaemonRunning() }
        } catch {
            NSLog("ServiceController: failed to launch daemon executable %@ (%@)", daemonExecPath, error.localizedDescription)
            return false
        }
    }

    /// Stop daemon by sending SIGTERM to its process.
    ///
    /// Also clears `daemon-status.json` so `agenthandover status` and the menu
    /// bar app don't keep reading a stale heartbeat pointing at the dead PID.
    /// Without this, v0.2.6 and earlier left the file behind on SIGTERM — the
    /// CLI would then check `kill(stale_pid, 0)`, see a dead process, and
    /// report the daemon as "not responding" even when the user had
    /// deliberately toggled off observation (hikoae's 0.2.6 report).
    ///
    /// `daemon.pid` is left alone — `isDaemonRunning()` self-cleans stale PID
    /// files (lines 97-101) and other code still uses it as the source of
    /// truth for which process to signal.
    static func stopDaemon() {
        DaemonSupervisor.shared.stopRequested()
        guard let pid = daemonPid() else { return }
        kill(pid, SIGTERM)
        waitUntil { !isDaemonRunning() }

        // Remove stale status file so status tooling reports a clean "stopped"
        // state instead of "not responding". Best-effort — file may not exist.
        let statusPath = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/agenthandover/daemon-status.json")
        try? FileManager.default.removeItem(at: statusPath)
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
        // Fallback (no PID file): a daemon process, not a browser's
        // native-messaging bridge, which is the same binary.
        return daemonListed(inProcessList: shell("/bin/ps", args: ["-ax", "-o", "args="]))
    }

    /// Whether a `ps -ax -o args=` listing holds a daemon: `ah-observer`
    /// started without arguments (the app and the CLI start it bare).  A
    /// browser starts the same binary as its native-messaging bridge with
    /// the extension's origin (`chrome-extension://…/`), and counting that
    /// as the daemon kept capture off whenever the browser was open.
    static func daemonListed(inProcessList listing: String) -> Bool {
        listing.split(separator: "\n").contains { line in
            let words = line.split(separator: " ", omittingEmptySubsequences: true)
            guard let executable = words.first else { return false }
            return (executable == "ah-observer" || executable.hasSuffix("/ah-observer"))
                && words.count == 1
        }
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
        waitUntil(timeout: TimeInterval(timeoutSeconds)) { !isDaemonRunning() }
    }

    // MARK: - Worker (launchd-managed)

    @discardableResult
    static func startWorker() -> Bool {
        bootstrapIfNeeded(label: workerLabel)
        kickstart(label: workerLabel)
        return waitUntil { isJobRunning(label: workerLabel) }
    }

    static func stopWorker() {
        bootout(label: workerLabel)
    }

    // MARK: - Combined

    @discardableResult
    static func startAll() -> Bool {
        let d = startDaemon()
        let w = startWorker()
        // Re-install native messaging host manifest on resume. It was removed
        // by stopCapture() on pause to prevent Chrome from spawning transient
        // ah-observer bridge instances while observation was supposed to be
        // off. Without this, toggling "Observe Me" back on would leave the
        // extension permanently disconnected until the app relaunched.
        installNativeMessagingHostManifest()
        return d && w
    }

    /// Pause (and app quit): stop capture only. The worker stays loaded:
    /// approvals, search and MCP reporting all go through it, paused or not.
    static func stopCapture() {
        stopDaemon()
        // Remove the native messaging host manifest so Chrome can't spawn
        // transient ah-observer bridge instances while observation is paused.
        // Previously the extension would keep writing extension-heartbeat.json
        // via those bridges even after the main daemon was stopped, showing
        // as "connected" while the daemon showed as "not responding" — exactly
        // the confusing state hikoae reported on v0.2.6.
        removeNativeMessagingHostManifest()
    }

    static func restartDaemon() {
        stopDaemon()
        waitForDaemonExit(timeoutSeconds: 3)
        startDaemon()
    }

    static func restartWorker() {
        stopWorker()
        waitUntil { !isJobRunning(label: workerLabel) }
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
            "allowed_origins": [
                "chrome-extension://jpemkdcihaijkolbkankcldmiimmmnfo/",
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
            // Read before waiting: a full pipe blocks the child (see `shell`).
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            let output = String(data: data, encoding: .utf8) ?? ""
            return (process.terminationStatus, output)
        } catch {
            return (-1, "Failed to run launchctl: \(error)")
        }
    }

    /// *path*'s standard output. Read to the end before waiting for the
    /// exit: a child writing more than the pipe holds (64 KB) blocks until
    /// it is read, so waiting first deadlocked. Seen on the test Mac, where
    /// `ps -ax -o args=` printed 64,320 bytes and the app never started
    /// the daemon.
    static func shell(_ path: String, args: [String]) -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = args
        let pipe = Pipe()
        process.standardOutput = pipe
        do {
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            return String(data: data, encoding: .utf8) ?? ""
        } catch {
            return ""
        }
    }
}
