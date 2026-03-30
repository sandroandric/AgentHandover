import SwiftUI

/// Window for answering focus recording questions.
///
/// When the Python worker generates clarifying questions after a focus
/// recording, this view displays them and writes answers back to
/// `focus-questions.json` for the worker to consume.
struct FocusQAView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var questionsFile: FocusQuestionsFile?
    @State private var answers: [Int: String] = [:]
    @State private var writeError: String?
    @State private var didSubmit = false

    // Contra design tokens (system-adaptive)
    private let darkNavy = Color.primary
    private let warmOrange = Color(red: 0.92, green: 0.57, blue: 0.20)
    private let goldenYellow = Color(red: 1.0, green: 0.74, blue: 0.07)
    private let warmCream = Color(nsColor: .windowBackgroundColor)
    private let lightGray = Color(nsColor: .controlBackgroundColor)
    private let brightGreen = Color(red: 0.18, green: 0.80, blue: 0.34)
    private let cardRadius: CGFloat = 14
    private let contraBorder: CGFloat = 1.5

    private var statusDir: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent("Library/Application Support/agenthandover")
    }

    var body: some View {
        Group {
            if let file = questionsFile {
                questionsContent(file)
            } else {
                emptyState
            }
        }
        .onAppear {
            loadQuestions()
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 14) {
            ZStack {
                Circle()
                    .fill(brightGreen.opacity(0.1))
                    .frame(width: 56, height: 56)
                    .overlay(
                        Circle()
                            .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                    )
                Image(systemName: "checkmark.circle")
                    .font(.system(size: 28))
                    .foregroundColor(brightGreen)
            }
            Text("No questions pending")
                .font(.system(size: 16, weight: .bold, design: .rounded))
                .foregroundColor(darkNavy)
            Text("Questions will appear here after a focus recording is processed.")
                .font(.system(size: 13))
                .foregroundColor(darkNavy.opacity(0.5))
                .multilineTextAlignment(.center)
                .lineSpacing(3)
        }
        .padding(40)
        .frame(minWidth: 400, minHeight: 200)
    }

    // MARK: - Questions Content

    private func questionsContent(_ file: FocusQuestionsFile) -> some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 8) {
                ZStack {
                    Circle()
                        .fill(warmOrange.opacity(0.12))
                        .frame(width: 48, height: 48)
                        .overlay(
                            Circle()
                                .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                        )
                    Image(systemName: "questionmark.bubble.fill")
                        .font(.system(size: 22))
                        .foregroundColor(warmOrange)
                }

                Text(displayTitle(for: file.slug))
                    .font(.system(size: 18, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)

                Text("Answer these questions so your agent can execute this workflow reliably.")
                    .font(.system(size: 13))
                    .foregroundColor(darkNavy.opacity(0.5))
                    .multilineTextAlignment(.center)
                    .lineSpacing(3)
                    .frame(maxWidth: 380)
            }
            .padding(.top, 28)
            .padding(.bottom, 20)

            Rectangle()
                .fill(darkNavy.opacity(0.08))
                .frame(height: 1)
                .padding(.horizontal, 24)

            // Questions list
            ScrollView {
                VStack(spacing: 14) {
                    ForEach(file.questions) { question in
                        questionCard(question)
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 18)
            }

            Rectangle()
                .fill(darkNavy.opacity(0.08))
                .frame(height: 1)
                .padding(.horizontal, 24)

            // Footer
            footer

            if let error = writeError {
                Text(error)
                    .font(.caption2)
                    .foregroundColor(.red)
                    .padding(.horizontal, 24)
                    .padding(.bottom, 8)
            }
        }
        .frame(minWidth: 520, maxWidth: 600, minHeight: 550)
    }

    // MARK: - Question Card

    private func questionCard(_ question: FocusQuestion) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            // Category badge + question
            HStack(alignment: .top, spacing: 8) {
                categoryBadge(for: question)

                Text(question.question)
                    .font(.system(size: 13, weight: .bold))
                    .foregroundColor(darkNavy)
                    .fixedSize(horizontal: false, vertical: true)
                    .lineSpacing(3)
            }

            if !question.context.isEmpty {
                Text(question.context)
                    .font(.system(size: 12))
                    .foregroundColor(darkNavy.opacity(0.5))
                    .italic()
                    .fixedSize(horizontal: false, vertical: true)
                    .lineSpacing(2)
                    .padding(.leading, 4)
            }

            TextField("Your answer", text: answerBinding(for: question.index, default: question.default))
                .textFieldStyle(.plain)
                .font(.system(size: 13))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(nsColor: .controlBackgroundColor))
                .cornerRadius(cardRadius)
                .overlay(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .stroke(darkNavy.opacity(0.15), lineWidth: contraBorder)
                )
        }
        .padding(14)
        .background(
            RoundedRectangle(cornerRadius: cardRadius)
                .fill(lightGray)
        )
        .overlay(
            RoundedRectangle(cornerRadius: cardRadius)
                .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
        )
    }

    /// Color-coded category badge based on the question context.
    private func categoryBadge(for question: FocusQuestion) -> some View {
        let (icon, color) = categorize(question)
        return ZStack {
            RoundedRectangle(cornerRadius: 6)
                .fill(color.opacity(0.12))
                .frame(width: 26, height: 26)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
                )
            Image(systemName: icon)
                .font(.system(size: 12))
                .foregroundColor(color)
        }
    }

    /// Categorize a question by analyzing its text content.
    private func categorize(_ question: FocusQuestion) -> (icon: String, color: Color) {
        let text = (question.question + " " + question.context).lowercased()
        if text.contains("credential") || text.contains("login") || text.contains("password") || text.contains("account") {
            return ("key.fill", .red)
        }
        if text.contains("decide") || text.contains("choose") || text.contains("which") || text.contains("option") {
            return ("arrow.triangle.branch", .purple)
        }
        if text.contains("verify") || text.contains("confirm") || text.contains("check") || text.contains("success") {
            return ("checkmark.circle", brightGreen)
        }
        if text.contains("how often") || text.contains("frequen") || text.contains("schedule") || text.contains("when") {
            return ("clock", .blue)
        }
        return ("questionmark.circle", warmOrange)
    }

    // MARK: - Footer

    private var footer: some View {
        HStack {
            Button("Skip All") {
                writeSkipped()
            }
            .buttonStyle(.plain)
            .font(.system(size: 13, weight: .medium))
            .foregroundColor(darkNavy.opacity(0.5))
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .overlay(
                RoundedRectangle(cornerRadius: cardRadius)
                    .stroke(darkNavy.opacity(0.12), lineWidth: contraBorder)
            )

            Spacer()

            Button {
                writeAnswered()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "paperplane.fill")
                        .font(.system(size: 11))
                    Text("Submit Answers")
                        .font(.system(size: 13, weight: .bold))
                }
                .padding(.horizontal, 18)
                .padding(.vertical, 9)
                .background(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .fill(Color.accentColor)
                )
                .foregroundColor(.white)
                .overlay(
                    RoundedRectangle(cornerRadius: cardRadius)
                        .stroke(Color.accentColor, lineWidth: contraBorder)
                )
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
    }

    // MARK: - Data Loading

    private func loadQuestions() {
        let path = statusDir.appendingPathComponent("focus-questions.json")
        guard let data = try? Data(contentsOf: path),
              let file = try? JSONDecoder().decode(FocusQuestionsFile.self, from: data),
              file.status == "pending" else {
            questionsFile = nil
            return
        }

        questionsFile = file

        // Pre-fill answers with defaults
        for question in file.questions {
            if answers[question.index] == nil {
                answers[question.index] = question.default
            }
        }
    }

    // MARK: - Answer Binding

    private func answerBinding(for index: Int, default defaultValue: String) -> Binding<String> {
        Binding(
            get: { answers[index] ?? defaultValue },
            set: { answers[index] = $0 }
        )
    }

    // MARK: - Write Back

    private func writeAnswered() {
        guard var file = questionsFile else { return }
        writeError = nil

        file.status = "answered"
        var answersDict: [String: String] = [:]
        for question in file.questions {
            let answer = answers[question.index] ?? question.default
            answersDict[String(question.index)] = answer
        }
        file.answers = answersDict

        if writeFile(file) {
            didSubmit = true
            appState.focusQuestionsAvailable = false
            appState.focusQuestionsSlug = ""
            dismiss()
        }
    }

    private func writeSkipped() {
        guard var file = questionsFile else { return }
        writeError = nil

        file.status = "skipped"

        if writeFile(file) {
            didSubmit = true
            appState.focusQuestionsAvailable = false
            appState.focusQuestionsSlug = ""
            dismiss()
        }
    }

    /// Atomic write: temp file + rename.
    private func writeFile(_ file: FocusQuestionsFile) -> Bool {
        let target = statusDir.appendingPathComponent("focus-questions.json")
        let tmp = statusDir.appendingPathComponent(".focus-questions.json.tmp")

        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(file)
            try data.write(to: tmp, options: .atomic)

            // Atomic rename
            if FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.removeItem(at: target)
            }
            try FileManager.default.moveItem(at: tmp, to: target)
            return true
        } catch {
            writeError = "Failed to save: \(error.localizedDescription)"
            // Clean up temp file if it exists
            try? FileManager.default.removeItem(at: tmp)
            return false
        }
    }

    // MARK: - Helpers

    /// Convert a slug like "check-gmail-inbox" to "Check Gmail Inbox".
    private func displayTitle(for slug: String) -> String {
        slug.split(separator: "-")
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
            .joined(separator: " ")
    }
}
