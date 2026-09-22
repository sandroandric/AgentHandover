import XCTest
import Foundation
@testable import AgentHandoverApp

/// Pairing writes where each agent reads: Claude Code through its CLI
/// (`claude mcp add` / `claude mcp list`), OpenClaw into openclaw.json.
/// The Claude Code CLI is a stub that records its argv; the real one would
/// change this machine's Claude Code configuration.
final class AgentDetectorTests: XCTestCase {
    private var root: URL!
    private var home: URL!

    private let connected = """
        Checking MCP server health...

        other: npx other-server - ✓ Connected
        agenthandover: /usr/local/bin/agenthandover-mcp  - ✓ Connected
        """
    private let failing = """
        Checking MCP server health...

        agenthandover: /old/agenthandover-mcp  - ✗ Failed to connect
        """
    private let absent = "Checking MCP server health...\n\nother: npx other-server - ✓ Connected"

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ah-agents-\(UUID().uuidString)")
        home = root.appendingPathComponent("home")
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - Stub CLI

    private var argvLog: URL { root.appendingPathComponent("claude-argv.log") }

    /// A `claude` stand-in: appends its argv to `argvLog`; `mcp list` prints
    /// `list` until an `mcp add` succeeded, then `listAfterAdd`.
    private func stubClaude(list: String, listAfterAdd: String? = nil, addStatus: Int32 = 0) throws -> String {
        let added = root.appendingPathComponent("claude-added").path
        let before = root.appendingPathComponent("list-before.txt")
        let after = root.appendingPathComponent("list-after.txt")
        try list.write(to: before, atomically: true, encoding: .utf8)
        try (listAfterAdd ?? list).write(to: after, atomically: true, encoding: .utf8)
        let script = root.appendingPathComponent("claude")
        try """
            #!/bin/sh
            echo "$*" >> '\(argvLog.path)'
            if [ "$1 $2" = "mcp list" ]; then
                if [ -f '\(added)' ]; then cat '\(after.path)'; else cat '\(before.path)'; fi
            elif [ "$1 $2" = "mcp add" ]; then
                if [ \(addStatus) -eq 0 ]; then touch '\(added)'; else echo "agenthandover already exists" >&2; fi
                exit \(addStatus)
            fi
            """.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return script.path
    }

    private func calls() -> [String] {
        ((try? String(contentsOf: argvLog, encoding: .utf8)) ?? "")
            .split(separator: "\n").map(String.init)
    }

    @MainActor
    private func detected(_ detector: AgentDetector) -> [DetectedAgent] {
        let done = expectation(description: "detect")
        detector.detect { done.fulfill() }
        wait(for: [done], timeout: 10)
        return detector.agents
    }

    @MainActor
    private func connect(_ detector: AgentDetector, _ id: String) {
        let done = expectation(description: "connect")
        detector.connect(detector.agents.first { $0.id == id }!) { done.fulfill() }
        wait(for: [done], timeout: 10)
    }

    private func claudeCode(_ agents: [DetectedAgent]) -> DetectedAgent? {
        agents.first { $0.id == "claude-code" }
    }

    // MARK: - Connected probe

    func testParseList() {
        XCTAssertEqual(ClaudeCodeCLI.parseList(connected), .connected)
        XCTAssertEqual(ClaudeCodeCLI.parseList(failing), .failing)
        XCTAssertEqual(ClaudeCodeCLI.parseList(absent), .absent)
        XCTAssertEqual(ClaudeCodeCLI.parseList(""), .absent)
        XCTAssertEqual(ClaudeCodeCLI.parseList("agenthandover-dev: x - ✓ Connected"), .absent)
    }

    @MainActor
    func testConnectedStateComesFromClaudeMcpList() throws {
        let detector = AgentDetector(home: home, claudeBinary: try stubClaude(list: connected), mcpCommand: "/x/agenthandover-mcp")
        let agent = try XCTUnwrap(claudeCode(detected(detector)))
        XCTAssertTrue(agent.isConnected)
        XCTAssertFalse(agent.isBusy)
        XCTAssertNil(agent.failure)
        XCTAssertEqual(calls(), ["mcp list"])
    }

    @MainActor
    func testASettingsJsonEntryIsNotConnected() throws {
        // The old pairing wrote ~/.claude/settings.json, which Claude Code
        // does not read MCP servers from.
        let claudeDir = home.appendingPathComponent(".claude")
        try FileManager.default.createDirectory(at: claudeDir, withIntermediateDirectories: true)
        try #"{"mcpServers": {"agenthandover": {"command": "agenthandover-mcp"}}}"#
            .write(to: claudeDir.appendingPathComponent("settings.json"), atomically: true, encoding: .utf8)
        let detector = AgentDetector(home: home, claudeBinary: try stubClaude(list: absent), mcpCommand: "/x/agenthandover-mcp")
        XCTAssertEqual(claudeCode(detected(detector))?.isConnected, false)
    }

    @MainActor
    func testAFailingRegistrationIsNotConnected() throws {
        let detector = AgentDetector(home: home, claudeBinary: try stubClaude(list: failing), mcpCommand: "/x/agenthandover-mcp")
        let agent = try XCTUnwrap(claudeCode(detected(detector)))
        XCTAssertFalse(agent.isConnected)
        XCTAssertNotNil(agent.failure)
    }

    // MARK: - Pairing Claude Code

    @MainActor
    func testConnectRunsClaudeMcpAddAtUserScope() throws {
        let detector = AgentDetector(
            home: home,
            claudeBinary: try stubClaude(list: absent, listAfterAdd: connected),
            mcpCommand: "/usr/local/bin/agenthandover-mcp"
        )
        XCTAssertEqual(claudeCode(detected(detector))?.isConnected, false)
        connect(detector, "claude-code")

        XCTAssertEqual(calls(), [
            "mcp list",
            "mcp list",
            "mcp add --scope user --transport stdio agenthandover -- /usr/local/bin/agenthandover-mcp",
            "mcp list",
        ])
        let agent = try XCTUnwrap(claudeCode(detector.agents))
        XCTAssertTrue(agent.isConnected)
        XCTAssertNil(agent.failure)
        // Nothing is written into ~/.claude by the app itself.
        XCTAssertFalse(FileManager.default.fileExists(atPath: home.appendingPathComponent(".claude/settings.json").path))
    }

    @MainActor
    func testConnectReplacesAFailingRegistration() throws {
        let detector = AgentDetector(
            home: home,
            claudeBinary: try stubClaude(list: failing, listAfterAdd: connected),
            mcpCommand: "/x/agenthandover-mcp"
        )
        _ = detected(detector)
        connect(detector, "claude-code")
        XCTAssertEqual(Array(calls().dropFirst()), [
            "mcp list",
            "mcp remove --scope user agenthandover",
            "mcp add --scope user --transport stdio agenthandover -- /x/agenthandover-mcp",
            "mcp list",
        ])
        XCTAssertEqual(claudeCode(detector.agents)?.isConnected, true)
    }

    @MainActor
    func testAFailedAddIsShown() throws {
        let detector = AgentDetector(
            home: home, claudeBinary: try stubClaude(list: absent, addStatus: 1), mcpCommand: "/x/agenthandover-mcp"
        )
        _ = detected(detector)
        connect(detector, "claude-code")
        let agent = try XCTUnwrap(claudeCode(detector.agents))
        XCTAssertFalse(agent.isConnected)
        XCTAssertTrue(agent.failure?.contains("already exists") ?? false, agent.failure ?? "")
    }

    @MainActor
    func testWithoutTheCLIConnectExplains() throws {
        try FileManager.default.createDirectory(
            at: home.appendingPathComponent(".claude"), withIntermediateDirectories: true
        )
        let detector = AgentDetector(home: home, claudeBinary: nil, mcpCommand: "/x/agenthandover-mcp")
        XCTAssertNotNil(claudeCode(detected(detector)))
        connect(detector, "claude-code")
        let agent = try XCTUnwrap(claudeCode(detector.agents))
        XCTAssertFalse(agent.isConnected)
        XCTAssertTrue(agent.failure?.contains("not found") ?? false)
    }

    // MARK: - OpenClaw

    @MainActor
    func testOpenClawPairingWritesTheConfigFileNotTheWorkspace() throws {
        let openclaw = home.appendingPathComponent(".openclaw")
        let workspace = openclaw.appendingPathComponent("workspace")
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        let config = openclaw.appendingPathComponent("openclaw.json")
        try #"{"agents": {"defaults": {"model": "m"}}, "mcp": {"servers": {"docs": {"command": "d"}}}}"#
            .write(to: config, atomically: true, encoding: .utf8)

        let detector = AgentDetector(home: home, claudeBinary: nil, mcpCommand: "/x/agenthandover-mcp")
        XCTAssertEqual(detected(detector).first { $0.id == "openclaw" }?.isConnected, false)
        connect(detector, "openclaw")

        var isDir: ObjCBool = false
        XCTAssertTrue(FileManager.default.fileExists(atPath: workspace.path, isDirectory: &isDir) && isDir.boolValue)
        let written = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: config)) as? [String: Any]
        )
        let servers = try XCTUnwrap((written["mcp"] as? [String: Any])?["servers"] as? [String: Any])
        XCTAssertEqual((servers["agenthandover"] as? [String: String])?["command"], "/x/agenthandover-mcp")
        XCTAssertNotNil(servers["docs"])
        XCTAssertNotNil(written["agents"])
        XCTAssertEqual(detector.agents.first { $0.id == "openclaw" }?.isConnected, true)
        XCTAssertTrue(AgentDetector.hasServer(in: config, keyPath: ["mcp", "servers"]))
    }

    @MainActor
    func testAJson5ConfigIsReadButNeverRewritten() throws {
        let openclaw = home.appendingPathComponent(".openclaw")
        try FileManager.default.createDirectory(at: openclaw, withIntermediateDirectories: true)
        let config = openclaw.appendingPathComponent("openclaw.json")
        let json5 = "// my settings\n{\n  agents: { defaults: { model: \"m\" } },\n}\n"
        try json5.write(to: config, atomically: true, encoding: .utf8)

        let detector = AgentDetector(home: home, claudeBinary: nil, mcpCommand: "/x/agenthandover-mcp")
        _ = detected(detector)
        connect(detector, "openclaw")

        XCTAssertEqual(try String(contentsOf: config, encoding: .utf8), json5)
        let agent = try XCTUnwrap(detector.agents.first { $0.id == "openclaw" })
        XCTAssertFalse(agent.isConnected)
        XCTAssertTrue(agent.failure?.contains("not changed") ?? false)

        let paired = "// mine\n{ mcp: { servers: { agenthandover: { command: \"x\" } } }, }\n"
        try paired.write(to: config, atomically: true, encoding: .utf8)
        XCTAssertTrue(AgentDetector.hasServer(in: config, keyPath: ["mcp", "servers"]))
    }

    @MainActor
    func testAnUnparseableCursorConfigIsNotOverwritten() throws {
        let cursor = home.appendingPathComponent(".cursor")
        try FileManager.default.createDirectory(at: cursor, withIntermediateDirectories: true)
        let config = cursor.appendingPathComponent("mcp.json")
        try "not json".write(to: config, atomically: true, encoding: .utf8)
        let detector = AgentDetector(home: home, claudeBinary: nil, mcpCommand: "/x/agenthandover-mcp")
        _ = detected(detector)
        connect(detector, "cursor")
        XCTAssertEqual(try String(contentsOf: config, encoding: .utf8), "not json")
        XCTAssertEqual(detector.agents.first { $0.id == "cursor" }?.isConnected, false)
    }
}
