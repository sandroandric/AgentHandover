import XCTest
import Foundation
@testable import AgentHandoverApp

/// WP-L4: the worker's micro-questions as review cards (choices, name
/// field, the micro-answer payload) and as choices in the focus Q&A window.
final class MicroQuestionsTests: XCTestCase {

    /// A card as the worker's `questions.write_cards` writes it (extra keys
    /// included: the app ignores them).
    private static let labelCardJSON = """
    {"questions": [
      {"id": "label:4f2a", "kind": "label", "subject": "4f2a", "title": "Stripe payouts",
       "question": "Was this a routine you would hand off?",
       "context": "Seen 15 times, about 5 hours a month, in Safari, dashboard.stripe.com.",
       "choices": ["Yes", "No", "Not sure"], "slot": null, "default": "Stripe payouts",
       "asks_name": true, "workflow_keys": ["4f2a"], "obs_ids": ["o1"], "value": 4000000.5,
       "asked_at": "2026-09-21T09:00:00.000Z"},
      {"id": "slot:weekly-report:decisions", "kind": "slot", "subject": "weekly-report",
       "title": "Weekly report", "question": "Why do you skip the entries you don't pick?",
       "context": "The runs show you pick some entries of a list and skip others.",
       "choices": ["Too old (the ones you pick are at most 2 hours old)",
                   "Off-topic (the ones you pick mention \\"rust\\")"],
       "slot": "decisions", "default": "", "asks_name": false, "workflow_keys": [],
       "obs_ids": [], "value": 2990000.0, "asked_at": "2026-09-21T09:00:00.000Z"}
    ]}
    """

    private func questions() throws -> [MicroQuestion] {
        try JSONDecoder().decode(
            MicroQuestionsFile.self, from: Data(Self.labelCardJSON.utf8)
        ).questions
    }

    func testDecodesTheWorkersCards() throws {
        let decoded = try questions()
        XCTAssertEqual(decoded.map(\.id), ["label:4f2a", "slot:weekly-report:decisions"])
        XCTAssertTrue(decoded[0].isLabel && decoded[0].asks_name)
        XCTAssertNil(decoded[0].slot)
        XCTAssertEqual(decoded[1].slot, "decisions")
    }

    func testLoadsFromTheStatusDirAndIsEmptyWithoutAFile() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("micro-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertEqual(MicroQuestions.load(from: dir), [])
        try Data(Self.labelCardJSON.utf8).write(to: dir.appendingPathComponent(MicroQuestions.fileName))
        XCTAssertEqual(MicroQuestions.load(from: dir).count, 2)
    }

    // MARK: - Cards render their choices

    func testQuestionCardsRenderTheChoicesInOrder() throws {
        let cards = MicroReviewViewModel.questionCards(try questions())
        XCTAssertEqual(cards.map(\.type), [.question, .question])
        XCTAssertEqual(cards.map(\.typeLabel), ["Question", "Question"])
        XCTAssertEqual(ReviewCard.answerButtons(for: cards[0]), ["Yes", "No", "Not sure"])
        XCTAssertEqual(ReviewCard.answerButtons(for: cards[1]), [
            "Too old (the ones you pick are at most 2 hours old)",
            "Off-topic (the ones you pick mention \"rust\")",
        ])
        // A labelling card asks for a name, pre-filled with the worker's;
        // a slot card lets the user type their own answer instead.
        XCTAssertTrue(cards[0].asksName)
        XCTAssertEqual(cards[0].defaultName, "Stripe payouts")
        XCTAssertEqual(ReviewCard.fieldPrompt(for: cards[0]), "What do you call it? (optional)")
        XCTAssertFalse(cards[1].asksName)
        XCTAssertEqual(cards[1].defaultName, "")
        XCTAssertEqual(ReviewCard.fieldPrompt(for: cards[1]), "Or type your answer")
        XCTAssertEqual(cards[0].evidenceText, try questions()[0].context)
        XCTAssertEqual(cards[1].slug, "weekly-report")
    }

    // MARK: - What an answer sends

    func testAChoiceSendsTheAnswerAndALabelsName() throws {
        let cards = MicroReviewViewModel.questionCards(try questions())
        let label = MicroReviewViewModel.questionPayload(
            card: cards[0], action: .answer("No", name: "  Payouts ")
        )
        XCTAssertEqual(label?["question_id"], "label:4f2a")
        XCTAssertEqual(label?["answer"], "No")
        XCTAssertEqual(label?["name"], "Payouts")
        let slot = MicroReviewViewModel.questionPayload(
            card: cards[1], action: .answer(ReviewCard.answerButtons(for: cards[1])[0], name: "ignored")
        )
        XCTAssertEqual(slot?["answer"], "Too old (the ones you pick are at most 2 hours old)")
        XCTAssertNil(slot?["name"])
    }

    func testSkipSendsABlankAnswerAndNothingElseIsSent() throws {
        let cards = MicroReviewViewModel.questionCards(try questions())
        XCTAssertEqual(MicroReviewViewModel.questionPayload(card: cards[1], action: .dismiss)?["answer"], "")
        XCTAssertNil(MicroReviewViewModel.questionPayload(card: cards[1], action: .answer("  ", name: "")))
        XCTAssertNil(MicroReviewViewModel.questionPayload(card: cards[1], action: .approve))
        let draft = ReviewCardData(
            id: "d", type: .draftProcedure, title: "T", recurrence: "-", duration: "-",
            observations: 1, confidence: 0, variables: [], outcome: nil, evidenceText: "",
            stepsPreview: [], slug: "t"
        )
        XCTAssertNil(MicroReviewViewModel.questionPayload(card: draft, action: .answer("Yes", name: "")))
        XCTAssertEqual(MicroQuestions.triggerKind, "micro-answer")
    }

    // MARK: - Notification

    func testOneNotificationPerNewQuestion() throws {
        let all = try questions()
        XCTAssertTrue(MicroQuestionsNotice.isNew(all, lastNotified: nil))
        let key = MicroQuestionsNotice.key(for: all)
        XCTAssertFalse(MicroQuestionsNotice.isNew(all, lastNotified: key))
        // One answered: nothing new.
        XCTAssertFalse(MicroQuestionsNotice.isNew([all[1]], lastNotified: key))
        XCTAssertFalse(MicroQuestionsNotice.isNew([], lastNotified: nil))
        XCTAssertEqual(MicroQuestionsNotice.title(for: [all[0]]), "A quick question about 'Stripe payouts'")
        XCTAssertEqual(MicroQuestionsNotice.title(for: all), "2 quick questions about your workflows")
    }

    // MARK: - The focus Q&A window's choices

    private static let focusQuestion = FocusQuestion(
        index: 0, question: "What starts this task?", category: "trigger", context: "",
        default: "Most Mondays around 09:00", questionID: "slot:weekly-report:trigger",
        choices: ["Most Mondays around 09:00", "Nothing fixed: I start it when I decide to", " "]
    )

    func testFocusQuestionsKeepTheirIdAndChoicesThroughTheApp() throws {
        let json = """
        {"index": 0, "id": "slot:a:trigger", "question": "What starts this task?",
         "category": "trigger", "context": "", "default": "", "step_indexes": [],
         "choices": ["Mondays"]}
        """
        let decoded = try JSONDecoder().decode(FocusQuestion.self, from: Data(json.utf8))
        XCTAssertEqual(decoded.questionID, "slot:a:trigger")
        XCTAssertEqual(decoded.choices, ["Mondays"])
        XCTAssertEqual(decoded.id, 0)
        let rewritten = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(decoded)
        ) as? [String: Any]
        XCTAssertEqual(rewritten?["id"] as? String, "slot:a:trigger")
        XCTAssertEqual(rewritten?["choices"] as? [String], ["Mondays"])
    }

    func testThePickedChoiceIsTheUsersAnswerEvenWhenItIsTheGuess() {
        let q = Self.focusQuestion
        XCTAssertEqual(FocusQAView.choices(for: q).count, 2)  // the blank one is dropped
        let answers = [0: q.default]
        // Left as pre-filled: an accepted guess.
        XCTAssertEqual(FocusQAView.submission(for: [q], answers: answers).sources, ["0": "accepted_guess"])
        // Picked: typed, and the note does not call it a guess.
        let picked = [0: q.default]
        XCTAssertEqual(
            FocusQAView.submission(for: [q], answers: answers, picked: picked).sources, ["0": "typed"]
        )
        XCTAssertNil(FocusQAView.answerNote(for: q, answers: answers, picked: picked))
        let other = FocusQAView.submission(
            for: [q], answers: [0: "Nothing fixed: I start it when I decide to"],
            picked: [0: "Nothing fixed: I start it when I decide to"]
        )
        XCTAssertEqual(other.answers, ["0": "Nothing fixed: I start it when I decide to"])
        XCTAssertEqual(other.sources, ["0": "typed"])
    }

    // MARK: - WP-F10: step questions (from step grounding)

    /// A step question as the worker writes it into the focus Q&A file.
    private static let stepQuestionJSON = """
    {"index": 2, "id": "step:note:clipboard:0a1b2c3d4e5f",
     "question": "Step 5 says \\"Paste the copied text\\", but nothing was copied before it in the recording. What did you do?",
     "category": "steps", "context": "The Skill has it as: \\"Type the text\\".", "default": "",
     "step_indexes": [4], "choices": ["Typed it", "Pasted it from somewhere else"]}
    """

    func testAStepQuestionRendersItsChoicesInTheQAWindow() throws {
        let q = try JSONDecoder().decode(FocusQuestion.self, from: Data(Self.stepQuestionJSON.utf8))
        XCTAssertEqual(FocusQAView.choices(for: q), ["Typed it", "Pasted it from somewhere else"])
        // No guess: the field starts empty and says the question is left open.
        XCTAssertEqual(FocusQAView.initialAnswers(for: [q]), [:])
        XCTAssertEqual(FocusQAView.answerNote(for: q, answers: [:]), "Left open: your agent will ask.")
        // A picked choice is sent as typed, word for word: the worker matches it.
        let picked = FocusQAView.submission(
            for: [q], answers: [2: "Typed it"], picked: [2: "Typed it"]
        )
        XCTAssertEqual(picked.answers, ["2": "Typed it"])
        XCTAssertEqual(picked.sources, ["2": "typed"])
        // The step it is about survives the round trip.
        let rewritten = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(q)
        ) as? [String: Any]
        XCTAssertEqual(rewritten?["step_indexes"] as? [Int], [4])
        XCTAssertEqual(rewritten?["id"] as? String, "step:note:clipboard:0a1b2c3d4e5f")
    }

    func testAStepQuestionCardRendersItsChoicesAndTheOtherField() throws {
        let json = """
        {"questions": [
          {"id": "step:look:cut:9f8e7d6c5b4a", "kind": "step", "subject": "look", "title": "Look",
           "question": "The recording did not show a scroll at step 1 (\\"Open Safari\\"). Did you scroll?",
           "context": "The model wrote: \\"Open Safari and scroll down\\".",
           "choices": ["No, I didn't", "Yes, I did"], "slot": "steps", "default": "",
           "asks_name": false, "workflow_keys": [], "obs_ids": [], "value": 2500000.0,
           "origin": {}, "step_indexes": [0], "asked_at": "2026-09-24T09:00:00.000Z"}
        ]}
        """
        let decoded = try JSONDecoder().decode(MicroQuestionsFile.self, from: Data(json.utf8)).questions
        let cards = MicroReviewViewModel.questionCards(decoded)
        XCTAssertEqual(cards.map(\.type), [.question])
        XCTAssertFalse(cards[0].asksName)
        XCTAssertEqual(ReviewCard.answerButtons(for: cards[0]), ["No, I didn't", "Yes, I did"])
        XCTAssertEqual(ReviewCard.fieldPrompt(for: cards[0]), "Or type your answer")
        XCTAssertEqual(cards[0].evidenceText, "The model wrote: \"Open Safari and scroll down\".")
        let payload = MicroReviewViewModel.questionPayload(
            card: cards[0], action: .answer(ReviewCard.answerButtons(for: cards[0])[1], name: "")
        )
        XCTAssertEqual(payload?["question_id"], "step:look:cut:9f8e7d6c5b4a")
        XCTAssertEqual(payload?["answer"], "Yes, I did")
    }
}
