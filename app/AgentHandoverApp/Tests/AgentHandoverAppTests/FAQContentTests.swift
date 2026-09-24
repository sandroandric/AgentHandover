import XCTest
@testable import AgentHandoverApp

/// The in-app FAQ is what a non-technical user reads: it must say where
/// their data goes (WP-S3, #59).  Cloud mode is gone (WP-F13): every model
/// runs through Ollama, so the FAQ has one answer.
final class FAQContentTests: XCTestCase {
    private func answers() -> [String: String] {
        let items = FAQContent.sections().flatMap(\.items)
        return Dictionary(uniqueKeysWithValues: items.map { ($0.id, $0.answer) })
    }

    func testTheFAQSaysOnDeviceAndNamesNoCloudProvider() {
        let faq = answers()
        for id in ["privacy", "what-is"] {
            let answer = faq[id] ?? ""
            XCTAssertTrue(answer.contains(FAQContent.onDevice), id)
        }
        for (id, answer) in faq {
            for word in ["Cloud mode", "cloud provider", "OpenAI", "Anthropic", "Gemini", "Keychain"] {
                XCTAssertFalse(answer.contains(word), "\(id): \(word)")
            }
        }
    }

    func testApprovalIsRequiredAndNeverClaimedAutomatic() {
        let approve = answers()["approve"] ?? ""
        XCTAssertTrue(approve.contains("starts as a draft"))
        XCTAssertTrue(approve.contains("Approve for Agents"))
        XCTAssertTrue(approve.contains("Until then agents cannot see it"))
        // [sop] auto_approve never reaches agents (the hand-off gate needs
        // a human's Approve for Agents), so the FAQ must not offer it.
        XCTAssertFalse(approve.contains("auto_approve"))
    }
}
