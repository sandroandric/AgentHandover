import SwiftUI

struct FAQView: View {
    @State private var expandedId: String?

    private let darkNavy = Color.primary
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                // Header
                VStack(alignment: .leading, spacing: 6) {
                    Text("FAQ")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                    Text("Everything you need to know about AgentHandover")
                        .font(.system(size: 14))
                        .foregroundColor(.secondary)
                }
                .padding(.bottom, 20)

                ForEach(sections) { section in
                    sectionHeader(section.title, icon: section.icon)

                    ForEach(section.items) { item in
                        faqItem(item)
                    }

                    if section.id != sections.last?.id {
                        Divider()
                            .padding(.vertical, 12)
                    }
                }
            }
            .padding(24)
        }
        .frame(minWidth: 520, minHeight: 500)
    }

    // MARK: - Section Header

    private func sectionHeader(_ title: String, icon: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 13, weight: .semibold))
                .foregroundColor(warmOrange)
            Text(title)
                .font(.system(size: 16, weight: .bold, design: .rounded))
        }
        .padding(.bottom, 8)
        .padding(.top, 4)
    }

    // MARK: - FAQ Item

    private func faqItem(_ item: FAQItem) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: {
                withAnimation(.easeInOut(duration: 0.2)) {
                    expandedId = expandedId == item.id ? nil : item.id
                }
            }) {
                HStack(alignment: .top, spacing: 10) {
                    Text(item.question)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(darkNavy)
                        .multilineTextAlignment(.leading)
                    Spacer()
                    Image(systemName: expandedId == item.id ? "chevron.up" : "chevron.down")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.secondary)
                        .padding(.top, 3)
                }
                .padding(.vertical, 10)
                .padding(.horizontal, 12)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            if expandedId == item.id {
                Text(item.answer)
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
                    .lineSpacing(4)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 12)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(expandedId == item.id
                      ? Color(nsColor: .controlBackgroundColor)
                      : Color.clear)
        )
    }

    // MARK: - Data

    private var sections: [FAQSection] {
        [
            FAQSection(
                id: "basics",
                title: "Getting Started",
                icon: "sparkles",
                items: [
                    FAQItem(
                        id: "what-is",
                        question: "What is AgentHandover?",
                        answer: "AgentHandover is a local apprentice that silently watches how you work on your Mac, learns your workflows, and turns them into step-by-step Skills that AI agents (like Claude Code, Cursor, or Codex) can execute for you.\n\nEverything runs locally on your machine. Your data never leaves your computer."
                    ),
                    FAQItem(
                        id: "focus-vs-observe",
                        question: "What's the difference between Focus Recording and Observe Me?",
                        answer: "Focus Recording: You click \"Record a Focus Session\", name it (e.g. \"Send weekly report\"), perform the task, then stop. AgentHandover creates a precise, high-quality Skill from that single recording. Best for teaching specific workflows.\n\nObserve Me: Passive background observation. AgentHandover watches everything you do and detects repeated patterns over time. It takes longer but discovers workflows you might not think to record. You control when it's on."
                    ),
                    FAQItem(
                        id: "privacy",
                        question: "Is my data private?",
                        answer: "Yes. Everything stays on your machine:\n\n- Screenshots are processed locally by Qwen (a small AI model running on your Mac)\n- No data is sent to any cloud service\n- All analysis happens on-device\n- You can delete all data at any time\n\nAgentHandover needs Accessibility and Screen Recording permissions to observe your screen, but the data never leaves your computer."
                    ),
                ]
            ),
            FAQSection(
                id: "skills",
                title: "Skills & Workflows",
                icon: "lightbulb",
                items: [
                    FAQItem(
                        id: "how-long-focus",
                        question: "How long does it take to create a Skill from a Focus Recording?",
                        answer: "After you stop recording, the worker processes your session in 2-5 minutes depending on the number of steps and your Mac's speed. You'll see a Q&A popup asking a few clarifying questions (like credentials or preferences), then the Skill appears in your Workflows."
                    ),
                    FAQItem(
                        id: "how-long-observe",
                        question: "How long does Observe Me take to create Skills?",
                        answer: "Passive observation needs to see you repeat a workflow pattern multiple times before it generates a Skill. This typically takes days or weeks of regular use. Focus Recording is much faster for workflows you already know."
                    ),
                    FAQItem(
                        id: "qa-questions",
                        question: "Why does it ask me questions after recording?",
                        answer: "The Q&A step fills in information that can't be seen on screen: login credentials to use, which option to pick when there are choices, whether to send or save a draft, etc. Your answers are stored in the Skill so the agent knows exactly what to do.\n\nYou can skip questions — the agent will ask you at execution time instead."
                    ),
                    FAQItem(
                        id: "approve",
                        question: "Do I need to approve Skills before agents can use them?",
                        answer: "Focus Recording Skills are auto-approved and immediately available to agents. Passive observation Skills start as drafts that you can review in the Workflows screen before approving."
                    ),
                ]
            ),
            FAQSection(
                id: "agents",
                title: "Connecting Agents",
                icon: "terminal",
                items: [
                    FAQItem(
                        id: "how-connect",
                        question: "How do I connect an AI agent?",
                        answer: "Click the agent name in the menu bar and hit \"Connect\". This adds AgentHandover's MCP server to the agent's configuration.\n\nSupported agents: Claude Code, Cursor, Windsurf, Codex, and any MCP-compatible tool.\n\nYou can also connect manually:\n  agenthandover connect claude-code\n  agenthandover connect mcp"
                    ),
                    FAQItem(
                        id: "how-use-skill",
                        question: "How do I tell the agent to use a Skill?",
                        answer: "Two ways:\n\n1. Slash command: Type /ah-skill-name in Claude Code (e.g. /ah-send-weekly-report). The agent gets the full procedure with steps, strategy, and your preferences.\n\n2. Just ask: If the MCP server is connected, the agent can search your Skills. Say \"check my AgentHandover skills\" or \"do you know how to send the weekly report?\" and it will find the matching Skill."
                    ),
                    FAQItem(
                        id: "mcp-what",
                        question: "What is the MCP server?",
                        answer: "MCP (Model Context Protocol) is a standard that lets AI agents use external tools. AgentHandover's MCP server gives agents access to:\n\n- list_ready_skills: See all your executable Skills\n- get_skill: Get the full procedure for a specific Skill\n- search_skills: Find Skills by description\n- report_execution_*: Report progress during execution\n\nWhen you click \"Connect\", this server is automatically configured."
                    ),
                ]
            ),
            FAQSection(
                id: "improvement",
                title: "Self-Improving Skills",
                icon: "arrow.triangle.2.circlepath",
                items: [
                    FAQItem(
                        id: "how-improve",
                        question: "How does automatic Skill improvement work?",
                        answer: "When an agent executes a Skill, it reports back what happened at each step:\n\n1. Before starting: \"I'm about to execute this Skill\"\n2. After each step: \"Completed\" or \"I had to deviate because...\"\n3. When done: \"Finished successfully\" or \"Failed at step 3\"\n\nAgentHandover analyzes these reports and updates the Skill. If the agent consistently deviates at a step, that step gets rewritten. If it fails, the Skill gets flagged for your review.\n\nOver time, Skills get more accurate and reliable without you doing anything."
                    ),
                    FAQItem(
                        id: "knowledge-base",
                        question: "What is the Knowledge Base?",
                        answer: "The Knowledge Base is where AgentHandover stores everything it learns:\n\n- Skills (step-by-step procedures)\n- Your profile (tools you use, working hours, preferences)\n- Decision rules (when you choose X over Y)\n- Constraints and guardrails\n\nIt lives at ~/.agenthandover/knowledge/ and is readable by any tool. The MCP server reads from it to serve Skills to agents."
                    ),
                ]
            ),
            FAQSection(
                id: "troubleshooting",
                title: "Troubleshooting",
                icon: "wrench",
                items: [
                    FAQItem(
                        id: "permissions",
                        question: "The app says permissions are missing",
                        answer: "AgentHandover needs two macOS permissions:\n\n1. Accessibility: Lets it see window titles and app names\n2. Screen Recording: Lets it capture screenshots for analysis\n\nGo to System Settings > Privacy & Security and toggle AgentHandover ON for both. If AgentHandover doesn't appear in the list, try removing and re-adding it, or reinstall the app."
                    ),
                    FAQItem(
                        id: "no-skills",
                        question: "I recorded a Focus Session but no Skill appeared",
                        answer: "Check these:\n\n1. Is the worker running? (Look for the green dot next to \"Worker\" in the menu bar footer)\n2. Wait 2-5 minutes — processing takes time, especially on 8GB Macs\n3. Check for Q&A: A popup may be waiting for your answers\n4. Run agenthandover status in Terminal to see what's happening\n\nIf the worker shows errors, try: agenthandover sops failed"
                    ),
                    FAQItem(
                        id: "vlm",
                        question: "What is VLM / Qwen and do I need it?",
                        answer: "VLM (Vision Language Model) is the local AI that analyzes your screenshots. AgentHandover uses Qwen, a small model that runs on your Mac via Ollama.\n\nYes, you need it. Without VLM, AgentHandover can't understand what's on your screen. The app will prompt you to set up Ollama during onboarding.\n\nRequirements: 8GB+ RAM (16GB recommended), ~4GB disk for the model."
                    ),
                    FAQItem(
                        id: "agent-zero",
                        question: "Agent says 0 procedures available",
                        answer: "Make sure you've:\n\n1. Recorded and completed at least one Focus Session\n2. Answered the Q&A questions when prompted\n3. Run: agenthandover connect claude-code (to refresh)\n\nThe Skill needs to reach \"agent_ready\" status. Check with: agenthandover skills list"
                    ),
                ]
            ),
        ]
    }
}

// MARK: - Models

private struct FAQSection: Identifiable {
    let id: String
    let title: String
    let icon: String
    let items: [FAQItem]
}

private struct FAQItem: Identifiable {
    let id: String
    let question: String
    let answer: String
}

#Preview {
    FAQView()
        .frame(width: 560, height: 700)
}
