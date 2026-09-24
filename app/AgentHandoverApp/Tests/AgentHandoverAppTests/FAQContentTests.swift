import XCTest
@testable import AgentHandoverApp

/// The in-app FAQ is what a non-technical user reads: it must say where
/// their data goes for the model mode they configured (WP-S3, #59).
final class FAQContentTests: XCTestCase {
    private func answers(_ mode: String, provider: String? = nil, model: String? = nil) -> [String: String] {
        let items = FAQContent.sections(vlmMode: mode, provider: provider, model: model)
            .flatMap(\.items)
        return Dictionary(uniqueKeysWithValues: items.map { ($0.id, $0.answer) })
    }

    func testRemoteModeNamesTheProviderModelAndKeychainAndNeverSaysOnDevice() {
        let faq = answers("remote", provider: "anthropic", model: "claude-sonnet-4-6-20260320")
        for id in ["privacy", "what-is"] {
            let answer = faq[id] ?? ""
            XCTAssertFalse(answer.contains(FAQContent.onDevice), id)
            XCTAssertFalse(answer.contains("nothing you do leaves"), id)
            XCTAssertTrue(answer.contains("Anthropic"), id)
            XCTAssertTrue(answer.contains("claude-sonnet-4-6-20260320"), id)
            XCTAssertTrue(answer.contains("Keychain (service com.agenthandover.app)"), id)
        }
    }

    func testLocalModeSaysOnDeviceAndNamesNoProvider() {
        let faq = answers("local")
        for id in ["privacy", "what-is"] {
            let answer = faq[id] ?? ""
            XCTAssertTrue(answer.contains(FAQContent.onDevice), id)
            XCTAssertFalse(answer.contains("Cloud mode is on"), id)
            XCTAssertFalse(answer.contains("Keychain"), id)
        }
    }

    func testApprovalIsRequiredAndNeverClaimedAutomatic() {
        let approve = answers("local")["approve"] ?? ""
        XCTAssertTrue(approve.contains("starts as a draft"))
        XCTAssertTrue(approve.contains("Approve for Agents"))
        XCTAssertTrue(approve.contains("Until then agents cannot see it"))
        // [sop] auto_approve never reaches agents (the hand-off gate needs
        // a human's Approve for Agents), so the FAQ must not offer it.
        XCTAssertFalse(approve.contains("auto_approve"))
    }

    func testTheVLMSettingsReaderMatchesWholeKeys() {
        // "model" starts with "mode": it must not overwrite the mode.
        let remote = AppState.vlmSettings(inConfig: """
        [vlm]
        mode = "remote"
        provider = "openai"
        model = "gpt-4.1-mini"  # chosen in setup
        api_key_env = "OPENAI_API_KEY"

        [embedding]
        model = "nomic-embed-text"
        """)
        XCTAssertEqual(remote.mode, "remote")
        XCTAssertEqual(remote.provider, "openai")
        XCTAssertEqual(remote.model, "gpt-4.1-mini")

        let local = AppState.vlmSettings(inConfig: "[vlm]\ntier = \"recommended\"\n# mode = \"remote\"\n")
        XCTAssertEqual(local.mode, "local")
        XCTAssertNil(local.provider)
        XCTAssertNil(local.model)
    }
}
