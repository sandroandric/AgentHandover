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
