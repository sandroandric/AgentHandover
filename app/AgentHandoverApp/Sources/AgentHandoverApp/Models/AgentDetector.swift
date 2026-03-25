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

    private let mcpEntry: [String: Any] = [
        "command": "agenthandover-mcp"
    ]

    func detect() {
        var detected: [DetectedAgent] = []
        let home = FileManager.default.homeDirectoryForCurrentUser

        // Claude Code
        let claudeConfig = home.appendingPathComponent(".claude/settings.json")
        detected.append(DetectedAgent(
            id: "claude-code",
            name: "Claude Code",
            icon: "terminal",
            configPath: claudeConfig,
            isConnected: isAgentConnected(at: claudeConfig)
        ))

        // Cursor
        let cursorConfig = home.appendingPathComponent(".cursor/mcp.json")
        detected.append(DetectedAgent(
            id: "cursor",
            name: "Cursor",
            icon: "cursorarrow.click",
            configPath: cursorConfig,
            isConnected: isAgentConnected(at: cursorConfig)
        ))

        // Windsurf
        let windsurfConfig = home.appendingPathComponent(".windsurf/mcp.json")
        detected.append(DetectedAgent(
            id: "windsurf",
            name: "Windsurf",
            icon: "wind",
            configPath: windsurfConfig,
            isConnected: isAgentConnected(at: windsurfConfig)
        ))

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
