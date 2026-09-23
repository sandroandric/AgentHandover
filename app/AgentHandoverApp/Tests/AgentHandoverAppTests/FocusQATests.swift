import XCTest
import Foundation
@testable import AgentHandoverApp

final class FocusQATests: XCTestCase {

    // MARK: - FocusQuestion Decoding

    func testDecodesSingleQuestion() throws {
        let json = """
        {
            "index": 0,
            "question": "Does this workflow require logging into Gmail?",
            "category": "access",
            "context": "Browser URLs detected at mail.google.com",
            "default": "No login required"
        }
        """.data(using: .utf8)!

        let question = try JSONDecoder().decode(FocusQuestion.self, from: json)
        XCTAssertEqual(question.index, 0)
        XCTAssertEqual(question.question, "Does this workflow require logging into Gmail?")
        XCTAssertEqual(question.category, "access")
        XCTAssertEqual(question.context, "Browser URLs detected at mail.google.com")
        XCTAssertEqual(question.default, "No login required")
        XCTAssertEqual(question.id, 0)
    }

    // MARK: - FocusQuestionsFile Decoding

    func testDecodesPendingQuestionsFile() throws {
        let json = """
        {
            "session_id": "abc123",
            "slug": "check-gmail-inbox",
            "questions": [
                {
                    "index": 0,
                    "question": "Does this require login?",
                    "category": "access",
                    "context": "Browser at mail.google.com",
                    "default": "No"
                },
                {
                    "index": 1,
                    "question": "Which emails do you open?",
                    "category": "decision_logic",
                    "context": "Some emails were skipped",
                    "default": "Open all unread"
                }
            ],
            "status": "pending"
        }
        """.data(using: .utf8)!

        let file = try JSONDecoder().decode(FocusQuestionsFile.self, from: json)
        XCTAssertEqual(file.session_id, "abc123")
        XCTAssertEqual(file.slug, "check-gmail-inbox")
        XCTAssertEqual(file.questions.count, 2)
        XCTAssertEqual(file.status, "pending")
        XCTAssertNil(file.answers)

        XCTAssertEqual(file.questions[0].index, 0)
        XCTAssertEqual(file.questions[0].category, "access")
        XCTAssertEqual(file.questions[1].index, 1)
        XCTAssertEqual(file.questions[1].category, "decision_logic")
    }

    func testDecodesAnsweredQuestionsFile() throws {
        let json = """
        {
            "session_id": "abc123",
            "slug": "check-gmail-inbox",
            "questions": [
                {
                    "index": 0,
                    "question": "Does this require login?",
                    "category": "access",
                    "context": "Browser at mail.google.com",
                    "default": "No"
                }
            ],
            "status": "answered",
            "answers": {
                "0": "Yes, SSO via Google workspace"
            }
        }
        """.data(using: .utf8)!

        let file = try JSONDecoder().decode(FocusQuestionsFile.self, from: json)
        XCTAssertEqual(file.status, "answered")
        XCTAssertNotNil(file.answers)
        XCTAssertEqual(file.answers?["0"], "Yes, SSO via Google workspace")
    }

    func testDecodesSkippedQuestionsFile() throws {
        let json = """
        {
            "session_id": "abc123",
            "slug": "check-gmail-inbox",
            "questions": [],
            "status": "skipped"
        }
        """.data(using: .utf8)!

        let file = try JSONDecoder().decode(FocusQuestionsFile.self, from: json)
        XCTAssertEqual(file.status, "skipped")
        XCTAssertEqual(file.questions.count, 0)
        XCTAssertNil(file.answers)
    }

    // MARK: - Answer Serialization

    func testAnswerSerializationRoundTrip() throws {
        let questionsJSON = """
        {
            "session_id": "sess-42",
            "slug": "deploy-to-prod",
            "questions": [
                {
                    "index": 0,
                    "question": "Which environment?",
                    "category": "scope",
                    "context": "Multiple deploy targets detected",
                    "default": "staging"
                },
                {
                    "index": 1,
                    "question": "Run tests first?",
                    "category": "verification",
                    "context": "No test step was observed",
                    "default": "Yes"
                }
            ],
            "status": "pending"
        }
        """.data(using: .utf8)!

        var file = try JSONDecoder().decode(FocusQuestionsFile.self, from: questionsJSON)

        // Simulate answering
        file.status = "answered"
        file.answers = [
            "0": "production",
            "1": "Always run full test suite"
        ]

        // Encode
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let encoded = try encoder.encode(file)

        // Decode back
        let decoded = try JSONDecoder().decode(FocusQuestionsFile.self, from: encoded)
        XCTAssertEqual(decoded.status, "answered")
        XCTAssertEqual(decoded.session_id, "sess-42")
        XCTAssertEqual(decoded.slug, "deploy-to-prod")
        XCTAssertEqual(decoded.questions.count, 2)
        XCTAssertEqual(decoded.answers?["0"], "production")
        XCTAssertEqual(decoded.answers?["1"], "Always run full test suite")
    }

    func testSkipSerializationRoundTrip() throws {
        let questionsJSON = """
        {
            "session_id": "sess-99",
            "slug": "file-expense",
            "questions": [
                {
                    "index": 0,
                    "question": "Which category?",
                    "category": "decision_logic",
                    "context": "Category selection was ambiguous",
                    "default": "Travel"
                }
            ],
            "status": "pending"
        }
        """.data(using: .utf8)!

        var file = try JSONDecoder().decode(FocusQuestionsFile.self, from: questionsJSON)

        // Simulate skip
        file.status = "skipped"

        // Encode
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let encoded = try encoder.encode(file)

        // Decode back
        let decoded = try JSONDecoder().decode(FocusQuestionsFile.self, from: encoded)
        XCTAssertEqual(decoded.status, "skipped")
        XCTAssertNil(decoded.answers)
        XCTAssertEqual(decoded.questions.count, 1)
    }

    // MARK: - Atomic File Write

    func testAtomicWriteToTempDir() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("focusqa-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let target = tmpDir.appendingPathComponent("focus-questions.json")
        let tmp = tmpDir.appendingPathComponent(".focus-questions.json.tmp")

        let file = FocusQuestionsFile(
            session_id: "test-session",
            slug: "test-workflow",
            questions: [
                FocusQuestion(index: 0, question: "Test?", category: "scope",
                              context: "Test context", default: "default answer")
            ],
            status: "answered",
            answers: ["0": "real answer"]
        )

        // Write via atomic pattern (temp + rename)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(file)
        try data.write(to: tmp, options: .atomic)
        try FileManager.default.moveItem(at: tmp, to: target)

        // Verify temp file is gone
        XCTAssertFalse(FileManager.default.fileExists(atPath: tmp.path))

        // Verify target file exists and is valid
        XCTAssertTrue(FileManager.default.fileExists(atPath: target.path))
        let readBack = try Data(contentsOf: target)
        let decoded = try JSONDecoder().decode(FocusQuestionsFile.self, from: readBack)
        XCTAssertEqual(decoded.status, "answered")
        XCTAssertEqual(decoded.answers?["0"], "real answer")
        XCTAssertEqual(decoded.slug, "test-workflow")
    }

    // MARK: - Status Detection

    func testPendingStatusDetected() throws {
        let json = """
        { "session_id": "s1", "slug": "wf", "questions": [], "status": "pending" }
        """.data(using: .utf8)!

        let file = try JSONDecoder().decode(FocusQuestionsFile.self, from: json)
        XCTAssertEqual(file.status, "pending")
    }

    func testAnsweredStatusNotPending() throws {
        let json = """
        { "session_id": "s1", "slug": "wf", "questions": [], "status": "answered", "answers": {} }
        """.data(using: .utf8)!

        let file = try JSONDecoder().decode(FocusQuestionsFile.self, from: json)
        XCTAssertNotEqual(file.status, "pending")
    }

    func testSkippedStatusNotPending() throws {
        let json = """
        { "session_id": "s1", "slug": "wf", "questions": [], "status": "skipped" }
        """.data(using: .utf8)!

        let file = try JSONDecoder().decode(FocusQuestionsFile.self, from: json)
        XCTAssertNotEqual(file.status, "pending")
    }

    // MARK: - Answers start with the guess; submitted as is, it is an accepted guess

    private static let questions = [
        FocusQuestion(index: 0, question: "Does the IANA page need a login?", category: "access",
                      context: "No login step was recorded", default: "Assume the page is public",
                      step_indexes: []),
        FocusQuestion(index: 1, question: "How often do you do this?", category: "scope",
                      context: "One run was recorded", default: "Every day", step_indexes: [0]),
        FocusQuestion(index: 2, question: "How do you know the note is right?",
                      category: "verification", context: "No check was recorded",
                      default: "The note shows the domains", step_indexes: [2]),
    ]

    private static let pending = FocusQuestionsFile(
        session_id: "83f40b9b",
        slug: "look-up-the-iana-reserved-domains-page-and-note-it",
        questions: questions,
        status: "pending",
        answers: nil,
        title: "Look up the IANA reserved domains page and note it"
    )

    func testFieldsStartWithTheGuessAsEditableText() {
        let answers = FocusQAView.initialAnswers(for: Self.questions)
        XCTAssertEqual(FocusQAView.answerText(for: 0, in: answers), "Assume the page is public")
        XCTAssertEqual(
            FocusQAView.answerNote(for: Self.questions[0], answers: answers),
            "AgentHandover's guess: submitted as is, it stays a guess."
        )
        let noGuess = FocusQuestion(index: 3, question: "Q?", category: "scope", context: "", default: "  ")
        XCTAssertNil(FocusQAView.initialAnswers(for: [noGuess])[3])
        XCTAssertEqual(FocusQAView.answerNote(for: noGuess, answers: [:]), "Left open: your agent will ask.")
        XCTAssertNil(FocusQAView.answerNote(for: Self.questions[1], answers: [1: "Weekly"]))
    }

    func testSubmitRecordsAcceptedGuessesTypedAnswersAndClearedFields() {
        var answers = FocusQAView.initialAnswers(for: Self.questions)
        answers[1] = "  Once a week, on Mondays "  // typed over the guess
        answers[2] = "   "                           // cleared
        let submitted = FocusQAView.submission(for: Self.questions, answers: answers)
        XCTAssertEqual(submitted.answers, ["0": "Assume the page is public", "1": "Once a week, on Mondays"])
        XCTAssertEqual(submitted.sources, ["0": "accepted_guess", "1": "typed"])
    }

    func testUntouchedSubmitIsNotUnanswered() {
        // The live run: Submit without touching the pre-filled fields.
        let submitted = FocusQAView.submission(
            for: Self.questions, answers: FocusQAView.initialAnswers(for: Self.questions)
        )
        XCTAssertEqual(submitted.answers.count, 3)
        XCTAssertEqual(Set(submitted.sources.values), ["accepted_guess"])
    }

    /// The worker's reader parses the same file
    /// (`worker/tests/test_skill_grounding.py`, `ANSWERED_FILE`).
    func testSubmitWritesTheSharedFixture() throws {
        var answers = FocusQAView.initialAnswers(for: Self.questions)
        answers[1] = "Once a week, on Mondays"
        answers[2] = ""
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let written = try encoder.encode(FocusQAView.answeredFile(Self.pending, answers: answers))

        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // AgentHandoverAppTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // AgentHandoverApp
            .deletingLastPathComponent()  // app
            .deletingLastPathComponent()  // repo root
            .appendingPathComponent("worker/tests/fixtures/focus_questions_answered.json")
        let expected = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture)) as? NSDictionary
        let actual = try JSONSerialization.jsonObject(with: written) as? NSDictionary
        XCTAssertEqual(actual, expected)

        // The worker's fields (title, step indexes) survive the app's rewrite.
        let decoded = try JSONDecoder().decode(FocusQuestionsFile.self, from: Data(contentsOf: fixture))
        XCTAssertEqual(decoded.title, Self.pending.title)
        XCTAssertEqual(decoded.questions.map { $0.step_indexes ?? [] }, [[], [0], [2]])
        let rewritten = try JSONSerialization.jsonObject(with: encoder.encode(decoded)) as? NSDictionary
        XCTAssertEqual(rewritten, expected)
    }

    // MARK: - Title as typed

    func testTitleIsShownAsTheUserTypedIt() {
        XCTAssertEqual(FocusQAView.displayTitle(for: Self.pending),
                       "Look up the IANA reserved domains page and note it")
        var older = Self.pending
        older.title = nil  // a file from a worker before WP-Q1
        XCTAssertEqual(FocusQAView.displayTitle(for: older),
                       "Look up the iana reserved domains page and note it")
    }

    // MARK: - Notification

    func testNotificationNamesTheCountAndTheTitle() {
        XCTAssertEqual(FocusQuestionsNotice.title(for: Self.pending),
                       "3 questions about 'Look up the IANA reserved domains page and note it'")
        let one = FocusQuestionsFile(session_id: "s1", slug: "x", questions: [Self.questions[0]],
                                     status: "pending", answers: nil, title: "Check the inbox")
        XCTAssertTrue(FocusQuestionsNotice.title(for: one).hasPrefix("1 question about"))
    }

    func testOneNotificationPerSession() {
        XCTAssertTrue(FocusQuestionsNotice.isNew(Self.pending, lastNotified: nil))
        XCTAssertTrue(FocusQuestionsNotice.isNew(Self.pending, lastNotified: "older-session"))
        XCTAssertFalse(FocusQuestionsNotice.isNew(Self.pending, lastNotified: "83f40b9b"))
        let empty = FocusQuestionsFile(session_id: "s2", slug: "x", questions: [],
                                       status: "pending", answers: nil)
        XCTAssertFalse(FocusQuestionsNotice.isNew(empty, lastNotified: nil))
    }

    // MARK: - Edge Cases

    func testEmptyQuestionsArray() throws {
        let json = """
        { "session_id": "s1", "slug": "empty-workflow", "questions": [], "status": "pending" }
        """.data(using: .utf8)!

        let file = try JSONDecoder().decode(FocusQuestionsFile.self, from: json)
        XCTAssertEqual(file.questions.count, 0)
        XCTAssertEqual(file.slug, "empty-workflow")
    }

    func testQuestionIdentifiableConformance() throws {
        let q1 = FocusQuestion(index: 0, question: "Q1", category: "a", context: "", default: "")
        let q2 = FocusQuestion(index: 1, question: "Q2", category: "b", context: "", default: "")
        let q3 = FocusQuestion(index: 0, question: "Q3", category: "c", context: "", default: "")

        XCTAssertEqual(q1.id, 0)
        XCTAssertEqual(q2.id, 1)
        XCTAssertEqual(q1.id, q3.id) // Same index = same id
        XCTAssertNotEqual(q1.id, q2.id)
    }
}
