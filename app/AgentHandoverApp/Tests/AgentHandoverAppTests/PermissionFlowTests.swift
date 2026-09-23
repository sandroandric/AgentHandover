import XCTest
import Foundation
@testable import AgentHandoverApp

/// Findings from a real upgrade on the test Mac (2026-09-23): "Open
/// Settings" opened an Accessibility list without AgentHandover in it, a
/// recheck with the switch already ON was a dead end, setup did not come
/// back after "Quit & Reopen", and the menu showed a red Offline while the
/// services were starting.
final class PermissionFlowTests: XCTestCase {

    // MARK: - Open Settings registers the app first

    private final class Calls {
        var log: [String] = []
    }

    private func registration(trusted: Bool, calls: Calls) -> PermissionChecker.SettingsRegistration {
        var system = PermissionChecker.SettingsRegistration()
        system.isAccessibilityTrusted = { calls.log.append("check"); return trusted }
        system.registerAccessibility = { calls.log.append("register-ax") }
        system.openAccessibilityPane = { calls.log.append("open-ax") }
        system.registerScreenRecording = { calls.log.append("register-sr") }
        system.openScreenRecordingPane = { calls.log.append("open-sr") }
        return system
    }

    @MainActor
    func testAccessibilityOpenSettingsRegistersTheAppBeforeOpeningThePane() async {
        let calls = Calls()
        let granted = await PermissionChecker.requestAccessibilityAndOpenSettingsIfNeeded(
            registration(trusted: false, calls: calls)
        )
        XCTAssertFalse(granted)
        XCTAssertEqual(calls.log, ["check", "register-ax", "open-ax"])
    }

    @MainActor
    func testAccessibilityAlreadyGrantedOpensNothing() async {
        let calls = Calls()
        let granted = await PermissionChecker.requestAccessibilityAndOpenSettingsIfNeeded(
            registration(trusted: true, calls: calls)
        )
        XCTAssertTrue(granted)
        XCTAssertEqual(calls.log, ["check"])
    }

    @MainActor
    func testScreenRecordingOpenSettingsProbesBeforeOpeningThePane() async {
        let calls = Calls()
        await PermissionChecker.registerAndOpenScreenRecordingSettings(registration(trusted: false, calls: calls))
        XCTAssertEqual(calls.log, ["register-sr", "open-sr"])
    }

    // MARK: - Recheck failure guidance

    func testAFailedRecheckExplainsRemovingAndReAddingTheEntry() throws {
        let guidance = try XCTUnwrap(PermissionChecker.recheckGuidance(granted: false))
        XCTAssertEqual(
            guidance,
            "If AgentHandover is already switched on, select it, click \u{2212} to remove it, then add it again with +."
        )
        XCTAssertNil(PermissionChecker.recheckGuidance(granted: true))
    }

    // MARK: - Where setup opens

    func testFirstRunOpensAtWelcome() {
        XCTAssertEqual(OnboardingProgress.startStep(
            savedStep: nil, completed: false, accessibilityGranted: false, screenRecordingGranted: false
        ), 0)
    }

    func testQuitAndReopenResumesAtTheSavedStep() {
        // First install: the Screen Recording grant relaunched the app at step 6 of 9.
        XCTAssertEqual(OnboardingProgress.startStep(
            savedStep: 5, completed: false, accessibilityGranted: true, screenRecordingGranted: true
        ), 5)
        // Upgrade: hasCompletedOnboarding is true from the old install, and
        // the relaunch comes before the user got past the permission steps.
        XCTAssertEqual(OnboardingProgress.startStep(
            savedStep: 5, completed: true, accessibilityGranted: true, screenRecordingGranted: false
        ), 5)
    }

    func testAnUpgradeThatLostAGrantOpensAtThatPermission() {
        XCTAssertEqual(OnboardingProgress.startStep(
            savedStep: nil, completed: true, accessibilityGranted: false, screenRecordingGranted: false
        ), OnboardingProgress.accessibilityStep)
        XCTAssertEqual(OnboardingProgress.startStep(
            savedStep: nil, completed: true, accessibilityGranted: true, screenRecordingGranted: false
        ), OnboardingProgress.screenRecordingStep)
    }

    func testNothingToSetUpOpensNothing() {
        XCTAssertNil(OnboardingProgress.startStep(
            savedStep: nil, completed: true, accessibilityGranted: true, screenRecordingGranted: true
        ))
        // A step saved by a resumed upgrade setup the user then closed.
        XCTAssertNil(OnboardingProgress.startStep(
            savedStep: 5, completed: true, accessibilityGranted: true, screenRecordingGranted: true
        ))
    }

    func testTheStepIsPersistedAndFinishingClearsIt() throws {
        let suite = "ah-onboarding-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }

        XCTAssertEqual(OnboardingProgress.startStep(
            defaults: defaults, accessibilityGranted: false, screenRecordingGranted: false
        ), 0)
        OnboardingProgress.save(step: 5, in: defaults)
        XCTAssertEqual(OnboardingProgress.startStep(
            defaults: defaults, accessibilityGranted: true, screenRecordingGranted: false
        ), 5)

        OnboardingProgress.finish(in: defaults)
        XCTAssertTrue(defaults.bool(forKey: OnboardingProgress.completedKey))
        XCTAssertNil(OnboardingProgress.savedStep(defaults))
        XCTAssertNil(OnboardingProgress.startStep(
            defaults: defaults, accessibilityGranted: true, screenRecordingGranted: true
        ))
    }

    // MARK: - Menu health while starting

    private func health(
        startedSecondsAgo: TimeInterval?, daemon: Bool, worker: Bool, stopped: Bool = false
    ) -> ServiceHealth {
        let now = Date()
        return ServiceHealth.derive(
            userStopped: stopped,
            startedAt: startedSecondsAgo.map { now.addingTimeInterval(-$0) },
            now: now,
            daemonRunning: daemon,
            workerRunning: worker,
            hasIssues: false
        )
    }

    func testJustAfterObserveMeTheMenuSaysStartingNotOffline() {
        XCTAssertEqual(health(startedSecondsAgo: 1, daemon: false, worker: false), .starting)
        XCTAssertEqual(health(startedSecondsAgo: 1, daemon: true, worker: false), .starting)
        XCTAssertEqual(health(startedSecondsAgo: 1, daemon: false, worker: true), .starting)
        XCTAssertNotEqual(ServiceHealth.starting.color, ServiceHealth.down.color)
    }

    func testStartingEndsWhenBothReportOrTheGracePeriodPasses() {
        XCTAssertEqual(health(startedSecondsAgo: 1, daemon: true, worker: true), .healthy)
        XCTAssertEqual(
            health(startedSecondsAgo: ServiceHealth.startGracePeriod + 1, daemon: false, worker: false), .down
        )
        XCTAssertEqual(
            health(startedSecondsAgo: ServiceHealth.startGracePeriod + 1, daemon: true, worker: false), .warning
        )
        XCTAssertEqual(health(startedSecondsAgo: nil, daemon: false, worker: false), .down)
        XCTAssertEqual(health(startedSecondsAgo: 1, daemon: false, worker: false, stopped: true), .stopped)
    }
}
