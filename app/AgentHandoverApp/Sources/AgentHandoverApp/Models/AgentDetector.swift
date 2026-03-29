import Foundation
import SwiftUI

struct DetectedAgent: Identifiable {
    let id: String
    let name: String
    let icon: String
    let configPath: URL
    var isConnected: Bool
}

class AgentDetector: ObservableObject {
    @Published var agents: [DetectedAgent] = []

    private var mcpEntry: [String: Any] {
        // Use the full path to the MCP server binary so agents can find it
        // even if the venv bin dir is not in PATH.
        let candidates = [
            "/usr/local/bin/agenthandover-mcp",
            "/usr/local/lib/agenthandover/venv/bin/agenthandover-mcp",
        ]
        let resolved = candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
            ?? "agenthandover-mcp" // bare name as last resort
        return ["command": resolved]
    }

    func detect() {
        var detected: [DetectedAgent] = []
        let home = FileManager.default.homeDirectoryForCurrentUser
        let fm = FileManager.default

        // Claude Code — installed if ~/.claude/ exists
        let claudeDir = home.appendingPathComponent(".claude")
        let claudeConfig = claudeDir.appendingPathComponent("settings.json")
        if fm.fileExists(atPath: claudeDir.path) {
            detected.append(DetectedAgent(
                id: "claude-code",
                name: "Claude Code",
                icon: "terminal",
                configPath: claudeConfig,
                isConnected: isAgentConnected(at: claudeConfig)
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
                isConnected: isAgentConnected(at: cursorConfig)
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
                isConnected: isAgentConnected(at: windsurfConfig)
            ))
        }

        // Codex — installed if codex CLI exists
        if fm.fileExists(atPath: "/usr/local/bin/codex") || fm.fileExists(atPath: "/opt/homebrew/bin/codex") {
            let codexConfig = home.appendingPathComponent(".codex/mcp.json")
            detected.append(DetectedAgent(
                id: "codex",
                name: "Codex",
                icon: "chevron.left.forwardslash.chevron.right",
                configPath: codexConfig,
                isConnected: isAgentConnected(at: codexConfig)
            ))
        }

        // OpenClaw — installed if ~/.openclaw/ exists
        let openclawDir = home.appendingPathComponent(".openclaw")
        if fm.fileExists(atPath: openclawDir.path) {
            let openclawConfig = openclawDir.appendingPathComponent("workspace")
            detected.append(DetectedAgent(
                id: "openclaw",
                name: "OpenClaw",
                icon: "pawprint.fill",
                configPath: openclawConfig,
                isConnected: fm.fileExists(atPath: openclawConfig.path)
            ))
        }

        agents = detected
    }

    func connect(_ agent: DetectedAgent) {
        guard let idx = agents.firstIndex(where: { $0.id == agent.id }) else { return }

        do {
            var config = readConfig(at: agent.configPath) ?? [:]
            var servers = config["mcpServers"] as? [String: Any] ?? [:]
            servers["agenthandover"] = mcpEntry
            config["mcpServers"] = servers
            try writeConfig(config, to: agent.configPath)
            agents[idx].isConnected = true
        } catch {
            print("Failed to connect \(agent.name): \(error)")
        }
    }

    func disconnect(_ agent: DetectedAgent) {
        guard let idx = agents.firstIndex(where: { $0.id == agent.id }) else { return }

        do {
            var config = readConfig(at: agent.configPath) ?? [:]
            var servers = config["mcpServers"] as? [String: Any] ?? [:]
            servers.removeValue(forKey: "agenthandover")
            config["mcpServers"] = servers
            try writeConfig(config, to: agent.configPath)
            agents[idx].isConnected = false
        } catch {
            print("Failed to disconnect \(agent.name): \(error)")
        }
    }

    private func isAgentConnected(at path: URL) -> Bool {
        guard let config = readConfig(at: path),
              let servers = config["mcpServers"] as? [String: Any] else {
            return false
        }
        return servers["agenthandover"] != nil
    }

    private func readConfig(at path: URL) -> [String: Any]? {
        guard FileManager.default.fileExists(atPath: path.path),
              let data = try? Data(contentsOf: path),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json
    }

    private func writeConfig(_ config: [String: Any], to path: URL) throws {
        try FileManager.default.createDirectory(
            at: path.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try JSONSerialization.data(
            withJSONObject: config,
            options: [.prettyPrinted, .sortedKeys]
        )
        try data.write(to: path, options: .atomic)
    }
}
