import SwiftUI

struct AgentConnectView: View {
    @StateObject private var detector = AgentDetector()

    private let darkNavy = Color(red: 0.09, green: 0.10, blue: 0.12)
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)
    private let brightGreen = Color(red: 0.18, green: 0.80, blue: 0.34)
    private let lightGray = Color(red: 0.96, green: 0.96, blue: 0.96)

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Connect Agents")
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .foregroundColor(darkNavy)

            Text("One click to connect AgentHandover's MCP server to your AI agent.")
                .font(.system(size: 13))
                .foregroundColor(darkNavy.opacity(0.5))

            ForEach(detector.agents) { agent in
                agentCard(agent)
            }

            Spacer()
        }
        .padding(24)
        .frame(width: 400, height: 360)
        .onAppear { detector.detect() }
    }

    private func agentCard(_ agent: DetectedAgent) -> some View {
        HStack(spacing: 12) {
            Image(systemName: agent.icon)
                .font(.system(size: 18, weight: .medium))
                .foregroundColor(darkNavy)
                .frame(width: 36, height: 36)
                .background(lightGray)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(darkNavy.opacity(0.1), lineWidth: 1.5)
                )

            VStack(alignment: .leading, spacing: 2) {
                Text(agent.name)
                    .font(.system(size: 14, weight: .semibold, design: .rounded))
                    .foregroundColor(darkNavy)

                if agent.isConnected {
                    Text("Connected")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(brightGreen)
                } else {
                    Text("Not connected")
                        .font(.system(size: 11))
                        .foregroundColor(darkNavy.opacity(0.4))
                }
            }

            Spacer()

            if agent.isConnected {
                Button("Disconnect") {
                    detector.disconnect(agent)
                }
                .font(.system(size: 12, weight: .medium))
                .foregroundColor(darkNavy.opacity(0.5))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(darkNavy.opacity(0.15), lineWidth: 1.5)
                )
                .buttonStyle(.plain)
            } else {
                Button("Connect") {
                    detector.connect(agent)
                }
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(
                    RoundedRectangle(cornerRadius: 8)
                        .fill(darkNavy)
                )
                .buttonStyle(.plain)
            }
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(agent.isConnected ? brightGreen.opacity(0.05) : Color.white)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(agent.isConnected ? brightGreen.opacity(0.3) : darkNavy.opacity(0.08), lineWidth: 1.5)
        )
    }
}
