import SwiftUI

/// Window for answering focus recording questions.
///
/// When the Python worker generates clarifying questions after a focus
/// recording, this view displays them and writes answers back to
/// `focus-questions.json` for the worker to consume.
///
/// Each answer field starts with AgentHandover's guess as editable text.
/// Submitted unchanged it is an accepted guess (the worker keeps it a
/// hypothesis); edited, or picked from the question's choices, it is the
/// user's answer; cleared, the question stays open.
struct FocusQAView: View {
    @EnvironmentObject var appState: AppState
    /// Closes the window (the app delegate hosts it).
    var onDone: () -> Void = {}

    @State private var questionsFile: FocusQuestionsFile?
    @State private var answers: [Int: String] = [:]
    /// Question index -> the choice the user picked (the user's answer
    /// even when it equals the guess).
    @State private var picked: [Int: String] = [:]
    @State private var writeError: String?
    @State private var didSubmit = false
    @State private var offerNotifications = false

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
            FocusQuestionsNotifier.shared.canAsk { offerNotifications = $0 }
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

                Text(Self.displayTitle(for: file))
                    .font(.system(size: 18, weight: .bold, design: .rounded))
                    .foregroundColor(darkNavy)

                Text("Each answer starts with AgentHandover's guess. Keep it, pick a choice, change it, or clear it to leave the question open.")
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

            if !Self.choices(for: question).isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(Self.choices(for: question), id: \.self) { choice in
                        Button {
                            answers[question.index] = choice
                            picked[question.index] = choice
                        } label: {
                            Text(choice)
                                .font(.system(size: 12))
                                .multilineTextAlignment(.leading)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 5)
                                .background(
                                    RoundedRectangle(cornerRadius: 8)
                                        .fill(picked[question.index] == choice
                                              ? warmOrange.opacity(0.2) : Color.clear)
                                )
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(warmOrange.opacity(0.4), lineWidth: 1)
                                )
                        }
                        .buttonStyle(.plain)
                    }
                }
            }

            TextField("Your answer", text: answerBinding(for: question.index), axis: .vertical)
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

            if let note = Self.answerNote(for: question, answers: answers, picked: picked) {
                Text(note)
                    .font(.system(size: 11))
                    .foregroundColor(darkNavy.opacity(0.5))
                    .padding(.leading, 4)
            }
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
        VStack(spacing: 10) {
            if offerNotifications {
                notificationOffer
            }
            footerButtons
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 16)
    }

    /// The one place the app asks for notification permission: here, where
    /// the user sees what they are for.
    private var notificationOffer: some View {
        HStack(spacing: 8) {
            Image(systemName: "bell")
                .font(.system(size: 12))
                .foregroundColor(darkNavy.opacity(0.5))
            Text("Get a notification when the next recording's questions are ready.")
                .font(.system(size: 12))
                .foregroundColor(darkNavy.opacity(0.6))
                .fixedSize(horizontal: false, vertical: true)
            Spacer()
            Button("Notify me") {
                FocusQuestionsNotifier.shared.requestAuthorization { _ in
                    offerNotifications = false
                }
            }
            .font(.system(size: 12, weight: .medium))
        }
    }

    private var footerButtons: some View {
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
        answers = Self.initialAnswers(for: file.questions)
    }

    // MARK: - Answers

    static let typed = "typed"
    static let acceptedGuess = "accepted_guess"

    private static func guess(of question: FocusQuestion) -> String {
        question.default.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// What the answer fields start with: each question's guess, as text
    /// the user can keep, edit or clear.
    static func initialAnswers(for questions: [FocusQuestion]) -> [Int: String] {
        var result: [Int: String] = [:]
        for question in questions where !guess(of: question).isEmpty {
            result[question.index] = guess(of: question)
        }
        return result
    }

    /// The text an answer field shows.
    static func answerText(for index: Int, in answers: [Int: String]) -> String {
        answers[index] ?? ""
    }

    /// The answers the question offers (the worker's choices, drawn from
    /// what the recorded runs showed), in its order.
    static func choices(for question: FocusQuestion) -> [String] {
        (question.choices ?? []).filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    }

    /// The line under a field: what submitting it will record.
    static func answerNote(
        for question: FocusQuestion, answers: [Int: String], picked: [Int: String] = [:]
    ) -> String? {
        let text = answerText(for: question.index, in: answers)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty {
            return "Left open: your agent will ask."
        }
        if text == guess(of: question) && picked[question.index] != text {
            return "AgentHandover's guess: submitted as is, it stays a guess."
        }
        return nil
    }

    /// The answers written back to the worker, keyed by question index, and
    /// where each came from: the guess submitted unchanged is
    /// `accepted_guess`, anything else non-blank (a choice the user picked
    /// included, even the guess) is `typed`.  A cleared field is left out,
    /// and the worker records its question unanswered.
    static func submission(
        for questions: [FocusQuestion],
        answers: [Int: String],
        picked: [Int: String] = [:]
    ) -> (answers: [String: String], sources: [String: String]) {
        var texts: [String: String] = [:]
        var sources: [String: String] = [:]
        for question in questions {
            let text = answerText(for: question.index, in: answers)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let key = String(question.index)
            texts[key] = text
            let pickedIt = picked[question.index] == text
            sources[key] = text == guess(of: question) && !pickedIt ? acceptedGuess : typed
        }
        return (texts, sources)
    }

    /// The file Submit writes: *file* answered with *answers*.
    static func answeredFile(
        _ file: FocusQuestionsFile, answers: [Int: String], picked: [Int: String] = [:]
    ) -> FocusQuestionsFile {
        var answered = file
        let submitted = submission(for: file.questions, answers: answers, picked: picked)
        answered.status = "answered"
        answered.answers = submitted.answers
        answered.answer_sources = submitted.sources
        return answered
    }

    private func answerBinding(for index: Int) -> Binding<String> {
        Binding(
            get: { Self.answerText(for: index, in: answers) },
            set: {
                answers[index] = $0
                // Typing over a picked choice makes it the user's own text.
                if picked[index] != $0 { picked[index] = nil }
            }
        )
    }

    // MARK: - Write Back

    private func writeAnswered() {
        guard let file = questionsFile else { return }
        writeError = nil

        if writeFile(Self.answeredFile(file, answers: answers, picked: picked)) {
            didSubmit = true
            appState.focusQuestionsAvailable = false
            appState.focusQuestionsSlug = ""
            onDone()
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
            onDone()
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

    /// The session title as the user typed it ("Look up the IANA
    /// reserved domains page"); for a file from an older worker, the slug
    /// with spaces and only its first letter capitalised (capitalising every
    /// word turned "IANA" into "Iana").
    static func displayTitle(for file: FocusQuestionsFile) -> String {
        let title = (file.title ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !title.isEmpty { return title }
        let words = file.slug.replacingOccurrences(of: "-", with: " ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }
}
