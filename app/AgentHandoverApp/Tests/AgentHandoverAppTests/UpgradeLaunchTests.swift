import XCTest
import Foundation
@testable import AgentHandoverApp

/// An upgrade keeps what the user had. The installer resets the grants
/// when the app's code signature changes; setup then opens only for the
/// missing grant, closes itself once the grants are back, and observation
/// resumes when the user was observing. A first install still starts
/// paused until the user chooses.
final class UpgradeLaunchTests: XCTestCase {
    private var suite: String!
    private var defaults: UserDefaults!

    override func setUpWithError() throws {
        suite = "ah-upgrade-\(UUID().uuidString)"
        defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    }

    override func tearDownWithError() throws {
        defaults.removePersistentDomain(forName: suite)
    }

    private func launch(accessibility: Bool, screenRecording: Bool) -> OnboardingProgress.LaunchAction {
        OnboardingProgress.launchAction(
            defaults: defaults, accessibilityGranted: accessibility, screenRecordingGranted: screenRecording
        )
    }

    /// An install that finished setup before, observing or paused.
    private func existingInstall(paused: Bool) {
        defaults.set(true, forKey: OnboardingProgress.completedKey)
        defaults.set(paused, forKey: OnboardingProgress.pausedKey)
    }

    // MARK: - Launch

    func testFirstInstallOpensSetupAtWelcome() {
        XCTAssertEqual(launch(accessibility: false, screenRecording: false), .setup(step: 0))
    }

    func testFirstInstallRelaunchedAfterGrantingAccessibilityGoesOnToScreenRecording() {
        OnboardingProgress.save(step: OnboardingProgress.accessibilityStep, in: defaults)
        XCTAssertEqual(
            launch(accessibility: true, screenRecording: false),
            .setup(step: OnboardingProgress.screenRecordingStep)
        )
        XCTAssertEqual(
            launch(accessibility: true, screenRecording: true),
            .setup(step: OnboardingProgress.screenRecordingStep)
        )
    }

    func testUpgradeThatLostBothGrantsOpensAtAccessibility() {
        existingInstall(paused: false)
        XCTAssertEqual(
            launch(accessibility: false, screenRecording: false),
            .setup(step: OnboardingProgress.accessibilityStep)
        )
    }

    func testUpgradeOpensAtTheFirstMissingGrantNotAtASavedGrantedStep() {
        // The live case: setup was on the Accessibility step, the user
        // granted it and Quit & Reopen relaunched the app.
        existingInstall(paused: false)
        OnboardingProgress.save(step: OnboardingProgress.accessibilityStep, in: defaults)
        XCTAssertEqual(
            launch(accessibility: true, screenRecording: false),
            .setup(step: OnboardingProgress.screenRecordingStep)
        )
    }

    func testUpgradeWithBothGrantsBackResumesObservingWithoutSetup() {
        existingInstall(paused: false)
        OnboardingProgress.save(step: OnboardingProgress.accessibilityStep, in: defaults)
        XCTAssertEqual(launch(accessibility: true, screenRecording: true), .observe)
    }

    func testUpgradeOfAPausedInstallStaysPaused() {
        existingInstall(paused: true)
        XCTAssertEqual(launch(accessibility: true, screenRecording: true), .stayPaused)
    }

    func testAnInstallFromBeforeThePauseSettingObserves() {
        // No observingPaused written: the user never paused.
        defaults.set(true, forKey: OnboardingProgress.completedKey)
        XCTAssertEqual(launch(accessibility: true, screenRecording: true), .observe)
    }

    // MARK: - Finishing setup

    func testFinishingAFirstInstallStartsPaused() {
        XCTAssertTrue(OnboardingProgress.finish(in: defaults))
        XCTAssertTrue(defaults.bool(forKey: OnboardingProgress.pausedKey))
        XCTAssertTrue(defaults.bool(forKey: OnboardingProgress.completedKey))
    }

    func testFinishingAnUpgradeKeepsObserving() {
        existingInstall(paused: false)
        OnboardingProgress.save(step: OnboardingProgress.screenRecordingStep, in: defaults)
        XCTAssertFalse(OnboardingProgress.finish(in: defaults))
        XCTAssertFalse(defaults.bool(forKey: OnboardingProgress.pausedKey))
        XCTAssertNil(OnboardingProgress.savedStep(defaults))
    }

    func testFinishingAnUpgradeKeepsPaused() {
        existingInstall(paused: true)
        XCTAssertTrue(OnboardingProgress.finish(in: defaults))
    }

    // MARK: - Grants coming back while the app runs

    private func regrant(setupOpen: Bool, existing: Bool, observing: Bool = false) -> OnboardingProgress.RegrantAction {
        OnboardingProgress.regrantAction(
            defaults: defaults, setupOpen: setupOpen, setupForExistingInstall: existing, observing: observing
        )
    }

    func testGrantsBackDuringAnUpgradesSetupFinishIt() {
        existingInstall(paused: false)
        XCTAssertEqual(regrant(setupOpen: true, existing: true), .finishSetup)
    }

    func testGrantsBackDuringAFirstInstallsSetupLeaveItOpen() {
        XCTAssertEqual(regrant(setupOpen: true, existing: false), .nothing)
    }

    func testGrantsBackWithoutSetupResumeObservingOnlyForAnObservingUser() {
        existingInstall(paused: false)
        XCTAssertEqual(regrant(setupOpen: false, existing: false), .observe)
        XCTAssertEqual(regrant(setupOpen: false, existing: false, observing: true), .nothing)
        existingInstall(paused: true)
        XCTAssertEqual(regrant(setupOpen: false, existing: false), .nothing)
    }

    func testGrantsBackBeforeSetupWasEverFinishedStartNothing() {
        XCTAssertEqual(regrant(setupOpen: false, existing: false), .nothing)
    }

    func testAReturnIsATransitionFromMissingToGranted() {
        XCTAssertFalse(AppState.grantsReturned(before: nil, now: true))
        XCTAssertFalse(AppState.grantsReturned(before: true, now: true))
        XCTAssertFalse(AppState.grantsReturned(before: false, now: false))
        XCTAssertTrue(AppState.grantsReturned(before: false, now: true))
    }

    /// The 2 s check reads the grants only while one is missing: a re-grant
    /// still starts capture within seconds, and an idle app with both
    /// grants stops re-verifying its signature every 2 s.
    func testTheQuickCheckReadsGrantsOnlyWhileOneIsMissing() {
        XCTAssertTrue(AppState.quickCheckReadsGrants(grantedAtLastCheck: nil))
        XCTAssertTrue(AppState.quickCheckReadsGrants(grantedAtLastCheck: false))
        XCTAssertFalse(AppState.quickCheckReadsGrants(grantedAtLastCheck: true))
    }
}

/// focus-questions.json is watched with a stat every couple of seconds, so
/// the Q&A shows within seconds of the worker writing it.
final class FileStampTests: XCTestCase {
    func testChangesAreSeenOnceEach() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("file-stamp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("focus-questions.json")

        var stamp = FileStamp()
        XCTAssertTrue(stamp.changed(file), "the first look reads the file")
        XCTAssertFalse(stamp.changed(file), "still missing")

        try Data("{}".utf8).write(to: file)
        XCTAssertTrue(stamp.changed(file), "created")
        XCTAssertFalse(stamp.changed(file))

        let later = Date().addingTimeInterval(5)
        try FileManager.default.setAttributes([.modificationDate: later], ofItemAtPath: file.path)
        XCTAssertTrue(stamp.changed(file), "rewritten")

        try FileManager.default.removeItem(at: file)
        XCTAssertTrue(stamp.changed(file), "removed")
        XCTAssertFalse(stamp.changed(file))
    }

    func testTheQuickCheckIsSeconds() {
        XCTAssertLessThanOrEqual(AppState.quickCheckInterval, 3)
    }
}

/// Setup's browser-extension step shows what `agenthandover doctor
/// --extension-json` finds, the check doctor's "Browser extension" line
/// uses.
final class ExtensionReportTests: XCTestCase {
    private func report(_ json: String) throws -> ExtensionReport {
        try ExtensionReport.decode(Data(json.utf8))
    }

    func testDecodesWhatTheCLIPrints() throws {
        // serde_json::to_string_pretty of browser_extension::Report.
        let parsed = try report("""
        {
          "extension_id": "jpemkdcihaijkolbkankcldmiimmmnfo",
          "build_path": "/usr/local/lib/agenthandover/extension",
          "installed_in": [
            {"browser": "Google Chrome", "profile": "Default", "enabled": true},
            {"browser": "Google Chrome", "profile": "Profile 1", "enabled": true},
            {"browser": "Brave", "profile": "Default", "enabled": false}
          ]
        }
        """)
        XCTAssertEqual(parsed.extensionId, "jpemkdcihaijkolbkankcldmiimmmnfo")
        XCTAssertEqual(parsed.status, .installed(browsers: ["Google Chrome"]))
    }

    func testTurnedOffNotLoadedAndNotBuilt() throws {
        XCTAssertEqual(try report("""
        {"extension_id": "x", "build_path": "/p", "installed_in": [{"browser": "Brave", "profile": "Default", "enabled": false}]}
        """).status, .turnedOff(browser: "Brave"))
        XCTAssertEqual(try report("""
        {"extension_id": "x", "build_path": "/usr/local/lib/agenthandover/extension", "installed_in": []}
        """).status, .notLoaded(path: "/usr/local/lib/agenthandover/extension"))
        XCTAssertEqual(try report("""
        {"extension_id": "x", "build_path": null, "installed_in": []}
        """).status, .notBuilt)
    }
}
