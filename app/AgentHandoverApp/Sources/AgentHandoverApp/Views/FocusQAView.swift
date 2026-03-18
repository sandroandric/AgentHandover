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
        VStack(spacing: 12) {
            Image(systemName: "checkmark.circle")
                .font(.system(size: 36))
                .foregroundColor(.secondary)
            Text("No questions pending")
                .font(.headline)
                .foregroundColor(.secondary)
            Text("Questions will appear here after a focus recording is processed.")
                .font(.caption)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(minWidth: 400, minHeight: 200)
    }

    // MARK: - Questions Content

    private func questionsContent(_ file: FocusQuestionsFile) -> some View {
        VStack(spacing: 0) {
            // Header
            VStack(spacing: 6) {
                Image(systemName: "questionmark.bubble")
                    .font(.system(size: 28))
                    .foregroundColor(.orange)

                Text(displayTitle(for: file.slug))
                    .font(.title3)
                    .fontWeight(.semibold)

                Text("Answer these questions so your agent can execute this workflow reliably.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 380)
            }
            .padding(.top, 24)
            .padding(.bottom, 16)

            Divider()
                .padding(.horizontal, 24)

            // Questions list
            ScrollView {
                VStack(spacing: 16) {
                    ForEach(file.questions) { question in
                        questionCard(question)
                    }
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 16)
            }

            Divider()
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
        .frame(minWidth: 480, maxWidth: 480, minHeight: 300)
    }

    // MARK: - Question Card

    private func questionCard(_ question: FocusQuestion) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(question.question)
                .font(.callout)
                .fontWeight(.semibold)
                .fixedSize(horizontal: false, vertical: true)

            if !question.context.isEmpty {
                Text(question.context)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .italic()
                    .fixedSize(horizontal: false, vertical: true)
            }

            TextField("Your answer", text: answerBinding(for: question.index, default: question.default))
                .textFieldStyle(.roundedBorder)
                .font(.callout)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color.secondary.opacity(0.05))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.secondary.opacity(0.1), lineWidth: 1)
        )
    }

    // MARK: - Footer

    private var footer: some View {
        HStack {
            Button("Skip All") {
                writeSkipped()
            }
            .buttonStyle(.plain)
            .font(.callout)
            .foregroundColor(.secondary)

            Spacer()

            Button("Submit Answers") {
                writeAnswered()
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
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
