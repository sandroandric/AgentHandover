import XCTest
import Security
@testable import AgentHandoverApp

/// Cloud mode was removed in 0.5.0 (WP-F13): onboarding offers only local
/// models, and the API keys earlier versions stored are deleted once.
final class CloudModeRemovedTests: XCTestCase {
    private var suite: String!
    private var defaults: UserDefaults!

    override func setUpWithError() throws {
        suite = "ah-legacy-keys-\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    }

    override func tearDownWithError() throws {
        defaults.removePersistentDomain(forName: suite)
    }

    private var sourcesDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/AgentHandoverApp")
    }

    /// No source file keeps a cloud choice, a provider, an API-key field or
    /// a remote `[vlm]` writer.
    func testNoSourceOffersCloudMode() throws {
        let files = try XCTUnwrap(FileManager.default.enumerator(at: sourcesDir, includingPropertiesForKeys: nil))
            .compactMap { $0 as? URL }
            .filter { $0.pathExtension == "swift" && $0.lastPathComponent != "LegacyAPIKeys.swift" }
        XCTAssertTrue(files.contains { $0.lastPathComponent == "OnboardingView.swift" })
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for needle in [
                "case cloud", "RemoteProvider", "remoteVLMBlock", "writeRemoteVLMConfig",
                "KeychainHelper", "SecureField(\"API Key\"", "mode = \\\"remote\\\"",
                "Cloud AI", "vlmProvider", "vlm_provider",
            ] {
                XCTAssertFalse(text.contains(needle), "\(file.lastPathComponent) still has \(needle)")
            }
        }
    }

    func testTheOldKeysAreDeletedOnceAndLogged() {
        var asked: [String] = []
        let removed = LegacyAPIKeys.removeOnce(defaults: defaults) { account in
            asked.append(account)
            return account == "agenthandover-openai-key" ? errSecSuccess : errSecItemNotFound
        }
        XCTAssertEqual(asked, [
            "agenthandover-openai-key", "agenthandover-anthropic-key", "agenthandover-google-key",
        ])
        XCTAssertEqual(removed, ["agenthandover-openai-key"])

        // Later launches leave the Keychain alone.
        let again = LegacyAPIKeys.removeOnce(defaults: defaults) { _ in
            XCTFail("the Keychain is touched once")
            return errSecSuccess
        }
        XCTAssertEqual(again, [])
    }

    func testAFailedDeleteIsTriedAgainNextLaunch() {
        LegacyAPIKeys.removeOnce(defaults: defaults) { _ in errSecInteractionNotAllowed }
        XCTAssertFalse(defaults.bool(forKey: LegacyAPIKeys.doneKey))
        let removed = LegacyAPIKeys.removeOnce(defaults: defaults) { _ in errSecSuccess }
        XCTAssertEqual(removed.count, 3)
        XCTAssertTrue(defaults.bool(forKey: LegacyAPIKeys.doneKey))
    }
}
