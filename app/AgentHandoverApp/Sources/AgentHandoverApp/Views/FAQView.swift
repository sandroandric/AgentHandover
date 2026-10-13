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

    private var sections: [FAQSection] { FAQContent.sections() }
}

// MARK: - Models

struct FAQSection: Identifiable {
    let id: String
    let title: String
    let icon: String
    let items: [FAQItem]
}

struct FAQItem: Identifiable {
    let id: String
    let question: String
    let answer: String
}

// MARK: - Content

/// The FAQ's questions and answers.  Every model runs on the Mac through
/// Ollama (a configured `ollama_host` is still the user's own Ollama).
enum FAQContent {
    /// Where the models run.
    static let onDevice = "Screen annotation, Skill writing and questions all run on your Mac through Ollama, and nothing you do leaves your computer."

    static func sections() -> [FAQSection] {
        let privacyAnswer = "Yes. \(onDevice)\n\n"
            + "- Screenshots are read by the local model and deleted once annotated; frames never annotated are deleted after 7 days.\n"
            + "- API keys, tokens and passwords are redacted before anything is stored or sent to the model.\n"
            + "- Password managers, macOS password and permission prompts, and banking or login pages are never captured.\n"
            + "- The only request the app makes on its own is its daily update check.\n"
            + "- To remove everything: agenthandover uninstall --purge-data\n\n"
            + "AgentHandover needs Accessibility and Screen Recording permissions to observe your screen."
        return [
            FAQSection(
                id: "basics",
                title: "Getting Started",
                icon: "sparkles",
                items: [
                    FAQItem(
                        id: "what-is",
                        question: "What is AgentHandover?",
                        answer: "AgentHandover watches how you work on your Mac, learns your workflows, and turns them into step-by-step Skills that AI agents (like Claude Code, Cursor, or Codex) can execute for you.\n\n\(onDevice)"
                    ),
                    FAQItem(
                        id: "focus-vs-observe",
                        question: "What's the difference between Focus Recording and Observe Me?",
                        answer: "Focus Recording: You click \"Record a Focus Session\", name it (e.g. \"Send weekly report\"), perform the task, then stop. AgentHandover creates a Skill from that single recording. Best for teaching specific workflows.\n\nObserve Me: Passive background observation. AgentHandover notices when you repeat the same task and, once it has seen enough, drafts a Skill for it (at most three new drafts a week). It takes longer but discovers workflows you might not think to record. You control when it's on."
                    ),
                    FAQItem(
                        id: "privacy",
                        question: "Is my data private?",
                        answer: privacyAnswer
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
                        answer: "A few minutes. On a 16 GB test Mac, questions were ready 2.5 to 5 minutes after Stop. A window then asks up to three questions about what the recording could not show, and the Skill is in your Workflows as a draft."
                    ),
                    FAQItem(
                        id: "how-long-observe",
                        question: "How long does Observe Me take to create Skills?",
                        answer: "Passive observation needs to see you repeat a workflow several times before it drafts a Skill. This typically takes days or weeks of regular use. Focus Recording is much faster for workflows you already know."
                    ),
                    FAQItem(
                        id: "qa-questions",
                        question: "Why does it ask me questions after recording?",
                        answer: "The questions fill in what the screen can't show: what starts the task, when it counts as done, why you chose one option over another, or a step the recording did not settle (for example whether you pasted or typed a line). They come from what AgentHandover still doesn't know about the Skill: at most three after a recording, and five a week in all.\n\nYou can skip any question. The Skill keeps it as not yet known, and agents are told to ask you instead."
                    ),
                    FAQItem(
                        id: "approve",
                        question: "Do I need to approve Skills before agents can use them?",
                        answer: "Yes. Every Skill, focus-recorded or passively discovered, starts as a draft. Open it in Workflows, review the steps, and click Approve for Agents. Until then agents cannot see it.\n\nAfter approval, agents see it only while it is fresh and its goal, steps and done criteria are known. The Skill page shows what AgentHandover knows and what it still doesn't; agents are told to ask about the rest."
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
                        answer: "Click the agent name in the menu bar and hit \"Connect\". For Claude Code this runs `claude mcp add --scope user`, for Codex `codex mcp add`; for Cursor, Windsurf and OpenClaw it adds AgentHandover's MCP server to the agent's MCP config file.\n\nYou can also connect from Terminal:\n  agenthandover connect claude-code\n  agenthandover connect codex\n  agenthandover connect mcp   (prints the MCP config for any other agent)"
                    ),
                    FAQItem(
                        id: "how-use-skill",
                        question: "How do I tell the agent to use a Skill?",
                        answer: "Two ways:\n\n1. Skill: Type /skill-name in Claude Code (e.g. /send-weekly-report). The agent gets the full procedure with steps, strategy, and your preferences.\n\n2. Just ask: If the MCP server is connected, the agent can search your Skills. Say \"check my AgentHandover skills\" or \"do you know how to send the weekly report?\" and it will find the matching Skill."
                    ),
                    FAQItem(
                        id: "mcp-what",
                        question: "What is the MCP server?",
                        answer: "MCP (Model Context Protocol) is a standard that lets AI agents use external tools. AgentHandover's MCP server gives agents:\n\n- search_commonplace: one search over Commonplace, everything AgentHandover knows about your work (Skills, daily summaries, your profile, and Memory if you let agents search it), grouped by kind, text only\n- list_ready_skills: the Skills an agent may run, and the approved ones held back and why\n- get_skill: the full Skill\n- search_skills: find Skills by meaning\n- search_memory: search Memory, only if you let agents search it\n- list_all_skills: every Skill, drafts included\n- get_user_profile: your tools, working hours and writing style\n- report_execution_start, report_step_result, report_execution_complete: report progress while running a Skill\n\nWhen you click \"Connect\", this server is configured for the agent."
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
                        question: "How do agent runs improve a Skill?",
                        answer: "When an agent runs a Skill through the MCP server, it reports each step (completed, or what it did instead) and the outcome. AgentHandover keeps every run and:\n\n- tracks how agent runs go, apart from how well the Skill is grounded in what you did\n- keeps what the agent did at a deviated step as an observed alternative\n- after three runs that follow the steps as written, counts the steps as confirmed\n- when a run deviates where a decision was only a guess, marks that decision unknown and asks you about it\n\nIt never rewrites your steps on its own: the Skill changes through your answers and your review."
                    ),
                    FAQItem(
                        id: "knowledge-base",
                        question: "What is the Knowledge Base?",
                        answer: "The Knowledge Base is where AgentHandover keeps what it learns:\n\n- Skills (steps, strategy, decisions, guardrails, and what is still unknown)\n- Your profile (tools you use, working hours, writing style)\n- Voice profiles per workflow\n\nIt is stored in AgentHandover's database and exported as files to ~/.agenthandover/knowledge/, readable by any tool. The MCP server serves it to agents."
                    ),
                    FAQItem(
                        id: "voice-profile",
                        question: "How does AgentHandover learn my voice and style?",
                        answer: "From the text you type and copy while you work, AgentHandover builds a Voice Profile per workflow: formality, sentence length, vocabulary and emoji use.\n\nWhen an agent runs a Skill that involves writing (emails, messages, docs), the Skill carries your Voice Profile so the output sounds like you.\n\nThe profile strengthens as AgentHandover sees more of your writing. Agents can read your overall style with the MCP server's get_user_profile tool."
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
                        answer: "AgentHandover needs two macOS permissions:\n\n1. Accessibility: Lets it see window titles and app names\n2. Screen Recording: Lets it capture screenshots for analysis\n\nGo to System Settings > Privacy & Security and toggle AgentHandover ON for both (macOS asks you to quit and reopen the app after Screen Recording). If AgentHandover doesn't appear in the list, try removing and re-adding it, or reinstall the app."
                    ),
                    FAQItem(
                        id: "no-skills",
                        question: "I recorded a Focus Session but no Skill appeared",
                        answer: "Check these:\n\n1. Is AgentHandover observing? A recording made while observation was off captures nothing, and the menu says so.\n2. Wait a few minutes: processing takes time.\n3. Check for questions: a window may be waiting for your answers.\n4. Did you record a task you already have as a Skill? The recording then counts as a run of that Skill, and no new Skill appears.\n5. Run agenthandover status in Terminal to see what's happening.\n\nIf the worker shows errors, try: agenthandover skills failed"
                    ),
                    FAQItem(
                        id: "vlm",
                        question: "What is VLM / Qwen and do I need it?",
                        answer: "VLM (Vision Language Model) is the local AI that reads your screen. AgentHandover runs small models on your Mac via Ollama: Qwen 3.5 2B annotates on 8 GB Macs and Qwen 3.5 4B elsewhere; Qwen 3.5 4B writes Skills up to 16 GB, and a Gemma 4 model from 24 GB.\n\nYes, you need it. Without it, AgentHandover can't understand what's on your screen. The app sets up Ollama during onboarding.\n\nRequirements: 8 GB+ RAM (16 GB recommended); the models take ~4 GB of disk on 16 GB Macs and ~6 GB on 8 GB Macs."
                    ),
                    FAQItem(
                        id: "agent-zero",
                        question: "Agent says 0 procedures available",
                        answer: "Make sure you've:\n\n1. Opened the Skill in Workflows and clicked Approve for Agents\n2. Answered its open questions: a Skill whose goal, steps or done criteria are unknown is held back\n3. Run: agenthandover connect claude-code (to refresh)\n\nThe MCP server's list_ready_skills names every approved Skill it holds back and why. Check your Skills with: agenthandover skills list"
                    ),
                ]
            ),
        ]
    }
}

#Preview {
    FAQView()
        .environmentObject(AppState())
        .frame(width: 560, height: 700)
}
