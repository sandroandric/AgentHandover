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
        let detector = AgentDetector(home: home, claudeBinary: try stubClaude(list: connected), codexBinary: nil, mcpCommand: "/x/agenthandover-mcp")
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
        let detector = AgentDetector(home: home, claudeBinary: try stubClaude(list: absent), codexBinary: nil, mcpCommand: "/x/agenthandover-mcp")
        XCTAssertEqual(claudeCode(detected(detector))?.isConnected, false)
    }

    @MainActor
    func testAFailingRegistrationIsNotConnected() throws {
        let detector = AgentDetector(home: home, claudeBinary: try stubClaude(list: failing), codexBinary: nil, mcpCommand: "/x/agenthandover-mcp")
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
            codexBinary: nil, mcpCommand: "/usr/local/bin/agenthandover-mcp"
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
            codexBinary: nil, mcpCommand: "/x/agenthandover-mcp"
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
            home: home, claudeBinary: try stubClaude(list: absent, addStatus: 1), codexBinary: nil, mcpCommand: "/x/agenthandover-mcp"
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
        let detector = AgentDetector(home: home, claudeBinary: nil, codexBinary: nil, mcpCommand: "/x/agenthandover-mcp")
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

        let detector = AgentDetector(home: home, claudeBinary: nil, codexBinary: nil, mcpCommand: "/x/agenthandover-mcp")
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

        let detector = AgentDetector(home: home, claudeBinary: nil, codexBinary: nil, mcpCommand: "/x/agenthandover-mcp")
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
        let detector = AgentDetector(home: home, claudeBinary: nil, codexBinary: nil, mcpCommand: "/x/agenthandover-mcp")
        _ = detected(detector)
        connect(detector, "cursor")
        XCTAssertEqual(try String(contentsOf: config, encoding: .utf8), "not json")
        XCTAssertEqual(detector.agents.first { $0.id == "cursor" }?.isConnected, false)
    }

    // MARK: - Codex
    //
    // Real-life finding (test Mac, 2026-09-23): pairing wrote
    // ~/.codex/mcp.json, which Codex does not read. Codex reads
    // ~/.codex/config.toml; its CLI (`codex mcp add`, `codex mcp list
    // --json`) edits and reads that file. The stub below records its argv
    // and prints what codex-cli 0.153 prints.

    private var codexLog: URL { root.appendingPathComponent("codex-argv.log") }

    private let codexListed = """
        [{"name": "agenthandover", "enabled": true, "disabled_reason": null,
          "transport": {"type": "stdio", "command": "/x/agenthandover-mcp", "args": []}},
         {"name": "other", "enabled": true, "transport": {"type": "stdio", "command": "x"}}]
        """

    private func stubCodex(listAfterAdd: String, addStatus: Int32 = 0) throws -> String {
        let added = root.appendingPathComponent("codex-added").path
        let after = root.appendingPathComponent("codex-list-after.json")
        try listAfterAdd.write(to: after, atomically: true, encoding: .utf8)
        let script = root.appendingPathComponent("codex")
        try """
            #!/bin/sh
            echo "$*" >> '\(codexLog.path)'
            if [ "$1 $2 $3" = "mcp list --json" ]; then
                echo "WARNING: proceeding, even though we could not update PATH" >&2
                if [ -f '\(added)' ]; then cat '\(after.path)'; else echo '[]'; fi
            elif [ "$1 $2" = "mcp add" ]; then
                if [ \(addStatus) -eq 0 ]; then touch '\(added)'; echo "Added global MCP server '$3'."; else echo "Error: invalid config.toml" >&2; fi
                exit \(addStatus)
            fi
            """.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return script.path
    }

    private func codexCalls() -> [String] {
        ((try? String(contentsOf: codexLog, encoding: .utf8)) ?? "")
            .split(separator: "\n").map(String.init)
    }

    func testCodexListParsing() {
        XCTAssertTrue(CodexCLI.listsServer(codexListed))
        XCTAssertFalse(CodexCLI.listsServer("[]"))
        XCTAssertFalse(CodexCLI.listsServer(#"[{"name": "agenthandover", "enabled": false}]"#))
        XCTAssertFalse(CodexCLI.listsServer(#"[{"name": "agenthandover-dev", "enabled": true}]"#))
        XCTAssertFalse(CodexCLI.listsServer("Error: no config"))
    }

    @MainActor
    func testCodexConnectRunsCodexMcpAddAndIsConnectedOnlyWhenListed() throws {
        let detector = AgentDetector(
            home: home, claudeBinary: nil,
            codexBinary: try stubCodex(listAfterAdd: codexListed),
            mcpCommand: "/x/agenthandover-mcp"
        )
        let before = try XCTUnwrap(detected(detector).first { $0.id == "codex" })
        XCTAssertFalse(before.isConnected)
        XCTAssertFalse(before.isBusy)
        XCTAssertEqual(before.configPath.path, home.appendingPathComponent(".codex/config.toml").path)

        connect(detector, "codex")

        XCTAssertEqual(codexCalls(), [
            "mcp list --json",
            "mcp add agenthandover -- /x/agenthandover-mcp",
            "mcp list --json",
        ])
        let agent = try XCTUnwrap(detector.agents.first { $0.id == "codex" })
        XCTAssertTrue(agent.isConnected)
        XCTAssertNil(agent.failure)
        // The app itself writes nothing under ~/.codex.
        XCTAssertFalse(FileManager.default.fileExists(atPath: home.appendingPathComponent(".codex").path))
    }

    @MainActor
    func testCodexAddThatDoesNotShowUpInTheListIsNotConnected() throws {
        let detector = AgentDetector(
            home: home, claudeBinary: nil,
            codexBinary: try stubCodex(listAfterAdd: "[]"),
            mcpCommand: "/x/agenthandover-mcp"
        )
        _ = detected(detector)
        connect(detector, "codex")
        let agent = try XCTUnwrap(detector.agents.first { $0.id == "codex" })
        XCTAssertFalse(agent.isConnected)
        XCTAssertEqual(agent.failure, "Codex does not list the AgentHandover MCP server.")
    }

    @MainActor
    func testCodexFailedAddIsShown() throws {
        let detector = AgentDetector(
            home: home, claudeBinary: nil,
            codexBinary: try stubCodex(listAfterAdd: codexListed, addStatus: 1),
            mcpCommand: "/x/agenthandover-mcp"
        )
        _ = detected(detector)
        connect(detector, "codex")
        let agent = try XCTUnwrap(detector.agents.first { $0.id == "codex" })
        XCTAssertFalse(agent.isConnected)
        XCTAssertTrue(agent.failure?.contains("invalid config.toml") ?? false, agent.failure ?? "")
    }

    @MainActor
    func testTheOldCodexMcpJsonEntryIsNotConnected() throws {
        let codexDir = home.appendingPathComponent(".codex")
        try FileManager.default.createDirectory(at: codexDir, withIntermediateDirectories: true)
        try #"{"mcpServers": {"agenthandover": {"command": "agenthandover-mcp"}}}"#
            .write(to: codexDir.appendingPathComponent("mcp.json"), atomically: true, encoding: .utf8)
        let detector = AgentDetector(
            home: home, claudeBinary: nil,
            codexBinary: try stubCodex(listAfterAdd: codexListed),
            mcpCommand: "/x/agenthandover-mcp"
        )
        XCTAssertEqual(detected(detector).first { $0.id == "codex" }?.isConnected, false)
    }

    @MainActor
    func testWithoutTheCodexCLICodexIsNotListed() {
        let detector = AgentDetector(home: home, claudeBinary: nil, codexBinary: nil, mcpCommand: "/x/agenthandover-mcp")
        XCTAssertNil(detected(detector).first { $0.id == "codex" })
    }
}
