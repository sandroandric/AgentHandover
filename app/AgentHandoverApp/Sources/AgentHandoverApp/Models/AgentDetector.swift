import Foundation
import SwiftUI

struct DetectedAgent: Identifiable {
    let id: String
    let name: String
    let icon: String
    /// The file pairing writes (for Claude Code: the CLI, which owns its config).
    let configPath: URL
    var isConnected: Bool
    /// Why the last Connect did not end connected; shown in the menu.
    var failure: String? = nil
    /// A Connect or a Claude Code probe is running.
    var isBusy: Bool = false
}

/// Claude Code keeps its MCP servers in its own config; the supported way
/// to change and read it is the `claude` CLI (`claude mcp add` /
/// `claude mcp list`).
struct ClaudeCodeCLI {
    enum ServerState: Equatable {
        case absent
        /// Registered, but Claude Code's health check could not start it.
        case failing
        case connected
        /// The CLI could not be run or failed; the message says why.
        case unknown(String)
    }

    static let serverName = "agenthandover"

    let binary: String
    var timeout: TimeInterval = 120

    /// `$AH_CLAUDE_BIN`, else the usual install locations (a GUI app has no
    /// shell `PATH`).  Nil when Claude Code's CLI is not installed.
    static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> String? {
        if let override = environment["AH_CLAUDE_BIN"], !override.isEmpty {
            return override
        }
        let candidates = [
            home.appendingPathComponent(".claude/local/claude").path,
            home.appendingPathComponent(".local/bin/claude").path,
            "/opt/homebrew/bin/claude",
            "/usr/local/bin/claude",
            home.appendingPathComponent(".npm-global/bin/claude").path,
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    /// Our server's state in `claude mcp list` output.  A line looks like
    /// `agenthandover: /usr/local/bin/agenthandover-mcp  - ✓ Connected`.
    static func parseList(_ output: String) -> ServerState {
        for line in output.split(separator: "\n", omittingEmptySubsequences: true) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard trimmed.hasPrefix("\(serverName):") else { continue }
            let status = trimmed.components(separatedBy: " - ").last ?? ""
            return status.contains("Connected") && !status.contains("Failed") ? .connected : .failing
        }
        return .absent
    }

    func state() -> ServerState {
        let result = run(["mcp", "list"])
        guard result.status == 0 else { return .unknown(result.output) }
        return Self.parseList(result.output)
    }

    /// Registers the server at user scope; nil on success, else the error.
    func add(mcpCommand: String) -> String? {
        let result = run([
            "mcp", "add", "--scope", "user", "--transport", "stdio",
            Self.serverName, "--", mcpCommand,
        ])
        return result.status == 0 ? nil : result.output
    }

    func remove() {
        _ = run(["mcp", "remove", "--scope", "user", Self.serverName])
    }

    /// Runs the CLI and returns its exit status and combined output.
    func run(_ arguments: [String]) -> (status: Int32, output: String) {
        runAgentCLI(binary, arguments, timeout: timeout)
    }
}

/// Runs an agent's command-line tool and returns its exit status and its
/// output (stdout, plus stderr unless `discardStderr`, for output that is
/// parsed as JSON).
func runAgentCLI(
    _ binary: String,
    _ arguments: [String],
    timeout: TimeInterval,
    discardStderr: Bool = false
) -> (status: Int32, output: String) {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: binary)
    process.arguments = arguments
    // npm installs are `#!/usr/bin/env node` scripts: give them the PATH a
    // login shell would have.
    var environment = ProcessInfo.processInfo.environment
    let binDir = URL(fileURLWithPath: binary).deletingLastPathComponent().path
    environment["PATH"] = ([binDir, "/opt/homebrew/bin", "/usr/local/bin"]
        + (environment["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin").split(separator: ":").map(String.init))
        .joined(separator: ":")
    process.environment = environment
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = discardStderr ? FileHandle.nullDevice : pipe
    process.standardInput = FileHandle.nullDevice
    do {
        try process.run()
    } catch {
        return (-1, "could not run \(binary): \(error.localizedDescription)")
    }
    let deadline = DispatchWorkItem { if process.isRunning { process.terminate() } }
    DispatchQueue.global().asyncAfter(deadline: .now() + timeout, execute: deadline)
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    process.waitUntilExit()
    deadline.cancel()
    let output = String(data: data, encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    return (process.terminationStatus, output)
}

/// Codex reads MCP servers from `~/.codex/config.toml`
/// (`[mcp_servers.<name>]`); the supported way to change and read that is
/// its CLI (`codex mcp add` / `codex mcp list --json`), which keeps the rest
/// of the file as it is. (Pairing used to write `~/.codex/mcp.json`, which
/// Codex never reads.)
struct CodexCLI {
    static let serverName = ClaudeCodeCLI.serverName

    let binary: String
    var timeout: TimeInterval = 60

    /// `$AH_CODEX_BIN`, else the usual install locations (a GUI app has no
    /// shell `PATH`).  Nil when Codex's CLI is not installed.
    static func resolve(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        home: URL = FileManager.default.homeDirectoryForCurrentUser
    ) -> String? {
        if let override = environment["AH_CODEX_BIN"], !override.isEmpty {
            return override
        }
        let candidates = [
            "/opt/homebrew/bin/codex",
            "/usr/local/bin/codex",
            home.appendingPathComponent(".local/bin/codex").path,
            home.appendingPathComponent(".npm-global/bin/codex").path,
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    /// Whether `codex mcp list --json` output lists our server, enabled.
    static func listsServer(_ json: String) -> Bool {
        guard let data = json.data(using: .utf8),
              let servers = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return false }
        return servers.contains {
            $0["name"] as? String == serverName && ($0["enabled"] as? Bool ?? true)
        }
    }

    /// Whether Codex lists our server; nil with the reason when the CLI failed.
    func isRegistered() -> (listed: Bool, error: String?) {
        let result = runAgentCLI(binary, ["mcp", "list", "--json"], timeout: timeout, discardStderr: true)
        guard result.status == 0 else {
            return (false, "codex mcp list failed (exit \(result.status))")
        }
        return (Self.listsServer(result.output), nil)
    }

    /// Registers (or replaces) the server; nil on success, else the error.
    func add(mcpCommand: String) -> String? {
        let result = runAgentCLI(binary, ["mcp", "add", Self.serverName, "--", mcpCommand], timeout: timeout)
        return result.status == 0 ? nil : result.output
    }
}

class AgentDetector: ObservableObject {
    @Published var agents: [DetectedAgent] = []

    private let home: URL
    private let claude: ClaudeCodeCLI?
    private let codex: CodexCLI?
    private let mcpCommand: String
    private let work = DispatchQueue(label: "com.agenthandover.agent-detector")

    init(
        home: URL = FileManager.default.homeDirectoryForCurrentUser,
        claudeBinary: String? = ClaudeCodeCLI.resolve(),
        codexBinary: String? = CodexCLI.resolve(),
        mcpCommand: String = AgentDetector.defaultMCPCommand()
    ) {
        self.home = home
        self.claude = claudeBinary.map { ClaudeCodeCLI(binary: $0) }
        self.codex = codexBinary.map { CodexCLI(binary: $0) }
        self.mcpCommand = mcpCommand
    }

    /// The full path of the MCP server binary, so agents find it even when
    /// the venv bin dir is not in their PATH.
    static func defaultMCPCommand() -> String {
        let candidates = [
            "/usr/local/bin/agenthandover-mcp",
            "/usr/local/lib/agenthandover/venv/bin/agenthandover-mcp",
        ]
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
            ?? "agenthandover-mcp" // bare name as last resort
    }

    /// Lists installed agents.  File-based states are read now; Claude
    /// Code's comes from `claude mcp list`, which health-checks every
    /// server, so it runs off the main thread and `completion` is called
    /// on the main thread once it is known.
    func detect(completion: (() -> Void)? = nil) {
        var detected: [DetectedAgent] = []
        let fm = FileManager.default

        // Claude Code — installed if its CLI or ~/.claude/ exists.
        let claudeDir = home.appendingPathComponent(".claude")
        if claude != nil || fm.fileExists(atPath: claudeDir.path) {
            let previous = agents.first { $0.id == "claude-code" }
            detected.append(DetectedAgent(
                id: "claude-code",
                name: "Claude Code",
                icon: "terminal",
                configPath: URL(fileURLWithPath: claude?.binary ?? claudeDir.path),
                isConnected: previous?.isConnected ?? false,
                failure: previous?.failure,
                isBusy: claude != nil
            ))
        }

        // Cursor — installed if ~/.cursor/ exists
        let cursorDir = home.appendingPathComponent(".cursor")
        let cursorConfig = cursorDir.appendingPathComponent("mcp.json")
        if fm.fileExists(atPath: cursorDir.path) {
            detected.append(DetectedAgent(
                id: "cursor",
                name: "Cursor",
                icon: "cursorarrow.click",
                configPath: cursorConfig,
                isConnected: Self.hasServer(in: cursorConfig, keyPath: ["mcpServers"])
            ))
        }

        // Windsurf — installed if ~/.windsurf/ exists
        let windsurfDir = home.appendingPathComponent(".windsurf")
        let windsurfConfig = windsurfDir.appendingPathComponent("mcp.json")
        if fm.fileExists(atPath: windsurfDir.path) {
            detected.append(DetectedAgent(
                id: "windsurf",
                name: "Windsurf",
                icon: "wind",
                configPath: windsurfConfig,
                isConnected: Self.hasServer(in: windsurfConfig, keyPath: ["mcpServers"])
            ))
        }

        // Codex — installed if its CLI exists; connected only when
        // `codex mcp list` lists the server (read off the main thread).
        if codex != nil {
            let previous = agents.first { $0.id == "codex" }
            detected.append(DetectedAgent(
                id: "codex",
                name: "Codex",
                icon: "chevron.left.forwardslash.chevron.right",
                configPath: home.appendingPathComponent(".codex/config.toml"),
                isConnected: previous?.isConnected ?? false,
                failure: previous?.failure,
                isBusy: true
            ))
        }

        // OpenClaw — installed if ~/.openclaw/ exists.  Its config file is
        // ~/.openclaw/openclaw.json, with MCP servers under mcp.servers.
        let openclawDir = home.appendingPathComponent(".openclaw")
        if fm.fileExists(atPath: openclawDir.path) {
            let openclawConfig = openclawDir.appendingPathComponent("openclaw.json")
            detected.append(DetectedAgent(
                id: "openclaw",
                name: "OpenClaw",
                icon: "pawprint.fill",
                configPath: openclawConfig,
                isConnected: Self.hasServer(in: openclawConfig, keyPath: ["mcp", "servers"])
            ))
        }

        agents = detected
        let pending = DispatchGroup()
        pending.enter()
        refreshClaudeCode { pending.leave() }
        pending.enter()
        refreshCodex { pending.leave() }
        pending.notify(queue: .main) { completion?() }
    }

    func connect(_ agent: DetectedAgent, completion: (() -> Void)? = nil) {
        guard agents.contains(where: { $0.id == agent.id }) else { completion?(); return }

        switch agent.id {
        case "claude-code":
            guard let claude else {
                update("claude-code") {
                    $0.failure = "Claude Code's command-line tool was not found. Install it, then click Connect again."
                }
                completion?()
                return
            }
            update("claude-code") { $0.isBusy = true; $0.failure = nil }
            let command = mcpCommand
            work.async { [weak self] in
                if case .failing = claude.state() { claude.remove() }
                let addError = claude.add(mcpCommand: command)
                let state = claude.state()
                DispatchQueue.main.async {
                    self?.update("claude-code") {
                        $0.isBusy = false
                        $0.isConnected = state == .connected
                        if let addError {
                            $0.failure = "claude mcp add failed: \(addError)"
                        } else if state != .connected {
                            $0.failure = Self.describe(state, command: command)
                        } else {
                            $0.failure = nil
                        }
                    }
                    completion?()
                }
            }
        case "codex":
            guard let codex else { completion?(); return }
            update("codex") { $0.isBusy = true; $0.failure = nil }
            let command = mcpCommand
            work.async { [weak self] in
                let addError = codex.add(mcpCommand: command)
                let registered = codex.isRegistered()
                DispatchQueue.main.async {
                    self?.update("codex") {
                        $0.isBusy = false
                        $0.isConnected = registered.listed
                        if let addError {
                            $0.failure = "codex mcp add failed: \(addError)"
                        } else if let error = registered.error {
                            $0.failure = error
                        } else if !registered.listed {
                            $0.failure = "Codex does not list the AgentHandover MCP server."
                        } else {
                            $0.failure = nil
                        }
                    }
                    completion?()
                }
            }
        case "openclaw":
            write(agent, keyPath: ["mcp", "servers"])
            completion?()
        default:
            write(agent, keyPath: ["mcpServers"])
            completion?()
        }
    }

    // MARK: - Claude Code

    private func refreshClaudeCode(completion: (() -> Void)?) {
        guard let claude, agents.contains(where: { $0.id == "claude-code" }) else {
            completion?()
            return
        }
        let command = mcpCommand
        work.async { [weak self] in
            let state = claude.state()
            DispatchQueue.main.async {
                self?.update("claude-code") {
                    $0.isBusy = false
                    $0.isConnected = state == .connected
                    if state == .failing { $0.failure = Self.describe(state, command: command) }
                }
                completion?()
            }
        }
    }

    // MARK: - Codex

    private func refreshCodex(completion: @escaping () -> Void) {
        guard let codex, agents.contains(where: { $0.id == "codex" }) else {
            completion()
            return
        }
        work.async { [weak self] in
            let registered = codex.isRegistered()
            DispatchQueue.main.async {
                self?.update("codex") {
                    $0.isBusy = false
                    $0.isConnected = registered.listed
                    if let error = registered.error { $0.failure = error }
                }
                completion()
            }
        }
    }

    private static func describe(_ state: ClaudeCodeCLI.ServerState, command: String) -> String {
        switch state {
        case .connected: return ""
        case .absent: return "Claude Code does not list the AgentHandover MCP server."
        case .failing: return "Claude Code could not start \(command)."
        case .unknown(let message): return "claude mcp list failed: \(message)"
        }
    }

    // MARK: - JSON config files

    /// Adds our server under `keyPath` in the agent's JSON config.  A file
    /// that exists but is not plain JSON is never rewritten.
    private func write(_ agent: DetectedAgent, keyPath: [String]) {
        do {
            var config: [String: Any] = [:]
            if FileManager.default.fileExists(atPath: agent.configPath.path) {
                let data = try Data(contentsOf: agent.configPath)
                guard let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                    throw PairingError.notAnObject(agent.configPath.path)
                }
                config = parsed
            }
            config = Self.setting(config, keyPath: keyPath, value: ["command": mcpCommand])
            try FileManager.default.createDirectory(
                at: agent.configPath.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            let data = try JSONSerialization.data(
                withJSONObject: config,
                options: [.prettyPrinted, .sortedKeys]
            )
            try data.write(to: agent.configPath, options: .atomic)
            update(agent.id) { $0.isConnected = true; $0.failure = nil }
        } catch {
            let message = "\(agent.configPath.path) was not changed: it is not plain JSON "
                + "(\(error.localizedDescription)). Add the server \"\(ClaudeCodeCLI.serverName)\" "
                + "with command \(mcpCommand) by hand."
            update(agent.id) { $0.failure = message }
        }
    }

    private enum PairingError: LocalizedError {
        case notAnObject(String)
        var errorDescription: String? {
            switch self {
            case .notAnObject(let path): return "\(path) is not a JSON object"
            }
        }
    }

    private static func setting(_ object: [String: Any], keyPath: [String], value: Any) -> [String: Any] {
        guard let key = keyPath.first else { return object }
        var result = object
        if keyPath.count == 1 {
            var servers = result[key] as? [String: Any] ?? [:]
            servers[ClaudeCodeCLI.serverName] = value
            result[key] = servers
        } else {
            let child = result[key] as? [String: Any] ?? [:]
            result[key] = setting(child, keyPath: Array(keyPath.dropFirst()), value: value)
        }
        return result
    }

    /// Whether the config at `path` lists our server under `keyPath`.
    /// JSON5 (comments, trailing commas) is accepted for reading.
    static func hasServer(in path: URL, keyPath: [String]) -> Bool {
        guard let data = try? Data(contentsOf: path),
              var node = try? JSONSerialization.jsonObject(with: data, options: [.json5Allowed]) as? [String: Any]
        else { return false }
        for key in keyPath {
            guard let child = node[key] as? [String: Any] else { return false }
            node = child
        }
        return node[ClaudeCodeCLI.serverName] != nil
    }

    private func update(_ id: String, _ change: (inout DetectedAgent) -> Void) {
        guard let idx = agents.firstIndex(where: { $0.id == id }) else { return }
        change(&agents[idx])
    }
}
