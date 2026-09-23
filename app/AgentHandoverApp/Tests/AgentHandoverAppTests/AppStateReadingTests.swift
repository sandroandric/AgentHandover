import CoreServices
import XCTest
@testable import AgentHandoverApp

/// `[privacy] read_app_state`: the onboarding toggle's config edit, the
/// consent codes, and the app list shared with the daemon.
final class AppStateReadingTests: XCTestCase {
    private static let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // AgentHandoverAppTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // AgentHandoverApp
        .deletingLastPathComponent()  // app
        .deletingLastPathComponent()  // repo root

    func testOffUnlessThePrivacySectionTurnsItOn() {
        XCTAssertFalse(AppStateReading.isEnabled(in: ""))
        XCTAssertFalse(AppStateReading.isEnabled(in: "[privacy]\nread_app_state = false\n"))
        XCTAssertTrue(AppStateReading.isEnabled(in: "[privacy]\nread_app_state = true  # opt-in\n"))
        // The key in another section is not the setting.
        XCTAssertFalse(AppStateReading.isEnabled(in: "[vlm]\nread_app_state = true\n[privacy]\n"))
    }

    func testTheToggleEditsOnlyItsKey() {
        let config = """
        [privacy]
        secure_field_drop = true
        read_app_state = false

        [vlm]
        mode = "local"

        """
        let on = AppStateReading.apply(enabled: true, to: config)
        XCTAssertTrue(AppStateReading.isEnabled(in: on))
        XCTAssertTrue(on.contains("secure_field_drop = true"))
        XCTAssertTrue(on.contains("[vlm]\nmode = \"local\""))
        XCTAssertEqual(on.components(separatedBy: "read_app_state").count, 2, on)
        // Idempotent, and back off again.
        XCTAssertEqual(AppStateReading.apply(enabled: true, to: on), on)
        XCTAssertFalse(AppStateReading.isEnabled(in: AppStateReading.apply(enabled: false, to: on)))
        // No [privacy] section yet: one is appended.
        let appended = AppStateReading.apply(enabled: true, to: "[vlm]\nmode = \"local\"\n")
        XCTAssertEqual(appended, "[vlm]\nmode = \"local\"\n\n[privacy]\nread_app_state = true\n")
    }

    func testConsentCodes() {
        XCTAssertEqual(AppStateReading.Consent(status: OSStatus(noErr)), .granted)
        XCTAssertEqual(AppStateReading.Consent(status: OSStatus(errAEEventNotPermitted)), .denied)
        XCTAssertEqual(AppStateReading.Consent(status: OSStatus(errAEEventWouldRequireUserConsent)), .notAsked)
        XCTAssertEqual(AppStateReading.Consent(status: OSStatus(procNotFound)), .notRunning)
        XCTAssertEqual(AppStateReading.Consent(status: -50), .unknown)
    }

    /// Asking without a prompt answers at once (an app that does not exist
    /// is never granted).
    func testTheSilentCheckDoesNotPrompt() {
        XCTAssertNotEqual(
            AppStateReading.consent(for: "com.agenthandover.no-such-app", ask: false), .granted
        )
    }

    /// The app asks for exactly the apps the daemon reads.
    func testSupportedAppsAreTheDaemons() throws {
        let bridge = try String(
            contentsOf: Self.repoRoot.appendingPathComponent(
                "crates/daemon/src/platform/applescript_bridge.rs"),
            encoding: .utf8
        )
        let list = try XCTUnwrap(bridge.range(of: "const KNOWN_APPS")).upperBound
        let end = try XCTUnwrap(bridge.range(of: "];", range: list..<bridge.endIndex)).lowerBound
        let pairs = try NSRegularExpression(pattern: #"\("([^"]+)",\s*"([^"]+)"\)"#)
        let text = String(bridge[list..<end])
        var daemon: [String: String] = [:]
        for m in pairs.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            let name = String(text[Range(m.range(at: 1), in: text)!])
            let id = String(text[Range(m.range(at: 2), in: text)!])
            daemon[id] = name
        }
        XCTAssertFalse(daemon.isEmpty)
        XCTAssertEqual(AppStateReading.supportedApps, daemon)
    }
}
