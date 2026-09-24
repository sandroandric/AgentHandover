import Foundation

/// One review-card question the worker asks (`questions.py`): a slot of a
/// Skill it does not know yet, or whether a workflow it discovered is a
/// routine the user would hand off (a labelling card, with a name field).
struct MicroQuestion: Codable, Identifiable, Equatable, Sendable {
    let id: String
    /// `slot` or `label`.
    let kind: String
    /// The Skill's slug, or the workflow's key.
    let subject: String
    /// What the subject is called.
    let title: String
    let question: String
    let context: String
    /// The answers the recorded runs suggest, in order.
    let choices: [String]
    var slot: String? = nil
    /// Labelling cards: the worker's name for the workflow, the name
    /// field's starting text.
    var `default`: String = ""
    var asks_name: Bool = false

    var isLabel: Bool { kind == "label" }
}

/// `micro-questions.json` (written by the worker's `questions.write_cards`):
/// the open review cards.  The app answers them through the trigger spool.
struct MicroQuestionsFile: Codable {
    let questions: [MicroQuestion]
}

enum MicroQuestions {
    static let fileName = "micro-questions.json"
    /// The trigger the worker's `questions.apply_answer` handles.
    static let triggerKind = "micro-answer"
    /// Posted when the file changed (AppState's quick check), so an open
    /// Review Queue reloads.
    static let changed = Notification.Name("AgentHandoverMicroQuestionsChanged")

    /// The open cards in *statusDir*, oldest first; none when the file is
    /// missing or unreadable.
    static func load(from statusDir: URL) -> [MicroQuestion] {
        let path = statusDir.appendingPathComponent(fileName)
        guard let data = try? Data(contentsOf: path),
              let file = try? JSONDecoder().decode(MicroQuestionsFile.self, from: data)
        else { return [] }
        return file.questions
    }

    /// The trigger payload for an answer: the choice (or typed text) and,
    /// for a labelling card, the name.  A blank answer skips the question
    /// (its slot stays as it was).
    static func answerPayload(
        for question: MicroQuestion, answer: String, name: String = "", at date: Date = Date()
    ) -> [String: String] {
        var payload = [
            "question_id": question.id,
            "answer": answer.trimmingCharacters(in: .whitespacesAndNewlines),
            "requested_at": ISO8601DateFormatter().string(from: date),
        ]
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if question.isLabel && !trimmedName.isEmpty {
            payload["name"] = trimmedName
        }
        return payload
    }
}

/// When the app tells the user about new review-card questions: once per
/// new set of question ids.  Pure, so the tests pin it.
enum MicroQuestionsNotice {
    static let lastNotifiedKey = "microQuestionsLastNotified"

    /// The ids as the defaults store them.
    static func key(for questions: [MicroQuestion]) -> String {
        questions.map(\.id).sorted().joined(separator: "\n")
    }

    /// True when *questions* hold an id the last notification did not.
    static func isNew(_ questions: [MicroQuestion], lastNotified: String?) -> Bool {
        let seen = Set((lastNotified ?? "").split(separator: "\n").map(String.init))
        return questions.contains { !seen.contains($0.id) }
    }

    static func title(for questions: [MicroQuestion]) -> String {
        questions.count == 1
            ? "A quick question about '\(questions[0].title)'"
            : "\(questions.count) quick questions about your workflows"
    }

    static let body = "Answer them in the Review Queue so your agent does not have to guess."
}
