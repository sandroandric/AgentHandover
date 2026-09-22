import XCTest
import Foundation
@testable import AgentHandoverApp

/// Start/stop off the main thread, daemon restart policy, bounded polling,
/// and the Screen Recording registration path.
final class ServiceSupervisionTests: XCTestCase {

    // MARK: - ObservationToggle

    @MainActor
    func testStartReturnsToMainActorQuicklyAndClearsTransition() {
        let toggle = ObservationToggle(
            startServices: { Thread.sleep(forTimeInterval: 0.5); return true },
            stopServices: {}
        )
        let done = expectation(description: "start completed")

        let began = Date()
        toggle.start { ok in
            XCTAssertTrue(ok)
            XCTAssertTrue(Thread.isMainThread)
            done.fulfill()
        }
        let elapsed = Date().timeIntervalSince(began)

        XCTAssertLessThan(elapsed, 0.05, "start() must not block the main thread")
        XCTAssertEqual(toggle.phase, .starting)
        XCTAssertTrue(toggle.isTransitioning)

        wait(for: [done], timeout: 5)
        XCTAssertNil(toggle.phase)
        XCTAssertFalse(toggle.isTransitioning)
    }

    @MainActor
    func testStopReturnsQuicklyAndClearsTransition() {
        let toggle = ObservationToggle(
            startServices: { true },
            stopServices: { Thread.sleep(forTimeInterval: 0.5) }
        )
        let done = expectation(description: "stop completed")
        let began = Date()
        toggle.stop { done.fulfill() }
        XCTAssertLessThan(Date().timeIntervalSince(began), 0.05)
        XCTAssertEqual(toggle.phase, .stopping)
        XCTAssertEqual(toggle.phase?.label, "Stopping…")
        wait(for: [done], timeout: 5)
        XCTAssertNil(toggle.phase)
    }

    @MainActor
    func testSecondTapWhileInFlightIsIgnored() {
        let calls = LockedCounter()
        let toggle = ObservationToggle(
            startServices: { calls.increment(); Thread.sleep(forTimeInterval: 0.2); return true },
            stopServices: { calls.increment() }
        )
        let done = expectation(description: "first start completed")
        toggle.start { _ in done.fulfill() }
        toggle.start()
        toggle.stop()
        wait(for: [done], timeout: 5)
        XCTAssertEqual(calls.value, 1)
    }

    // MARK: - ServiceController.waitUntil

    func testWaitUntilReturnsImmediatelyWhenConditionHolds() {
        let began = Date()
        XCTAssertTrue(ServiceController.waitUntil { true })
        XCTAssertLessThan(Date().timeIntervalSince(began), 0.05)
    }

    func testWaitUntilGivesUpAtTimeout() {
        let began = Date()
        XCTAssertFalse(ServiceController.waitUntil(timeout: 0.2) { false })
        let elapsed = Date().timeIntervalSince(began)
        XCTAssertGreaterThanOrEqual(elapsed, 0.2)
        XCTAssertLessThan(elapsed, 1.0)
    }

    // MARK: - DaemonSupervisor

    func testExitClassification() {
        XCTAssertFalse(DaemonSupervisor.isAbnormalExit(reason: .exit, status: 0))
        XCTAssertTrue(DaemonSupervisor.isAbnormalExit(reason: .exit, status: 101))
        XCTAssertFalse(DaemonSupervisor.isAbnormalExit(reason: .uncaughtSignal, status: SIGTERM))
        XCTAssertFalse(DaemonSupervisor.isAbnormalExit(reason: .uncaughtSignal, status: SIGINT))
        XCTAssertTrue(DaemonSupervisor.isAbnormalExit(reason: .uncaughtSignal, status: SIGABRT))
        XCTAssertTrue(DaemonSupervisor.isAbnormalExit(reason: .uncaughtSignal, status: SIGKILL))
    }

    func testBackoffDoublesAndCaps() {
        let delays = (1...8).map { DaemonSupervisor.backoffSeconds(forRestart: $0) }
        XCTAssertEqual(delays, [1, 2, 4, 8, 16, 32, 60, 60])
    }

    func testCrashIsRestartedWithBackoffAndStatusLine() {
        let harness = SupervisorHarness(restartSucceeds: true)
        harness.supervisor.daemonWillStart()
        harness.supervisor.daemonExited(reason: .uncaughtSignal, status: SIGABRT)
        XCTAssertEqual(harness.scheduledDelays, [1])
        XCTAssertEqual(harness.restarts, 0, "restart waits for the backoff")
        XCTAssertTrue(harness.supervisor.statusLine?.contains("signal \(SIGABRT)") ?? false)

        harness.runScheduled()
        XCTAssertEqual(harness.restarts, 1)
        XCTAssertTrue(harness.supervisor.statusLine?.contains("restarted") ?? false)

        // A second crash soon after backs off further.
        harness.supervisor.daemonExited(reason: .exit, status: 1)
        XCTAssertEqual(harness.scheduledDelays, [1, 2])
    }

    func testCleanExitAndSigtermAreNotRestarted() {
        let harness = SupervisorHarness(restartSucceeds: true)
        harness.supervisor.daemonWillStart()
        harness.supervisor.daemonExited(reason: .exit, status: 0)
        harness.supervisor.daemonExited(reason: .uncaughtSignal, status: SIGTERM)
        XCTAssertTrue(harness.scheduledDelays.isEmpty)
        XCTAssertNil(harness.supervisor.statusLine)
    }

    func testRequestedStopIsNeverUndone() {
        let harness = SupervisorHarness(restartSucceeds: true)
        harness.supervisor.daemonWillStart()
        harness.supervisor.stopRequested()
        harness.supervisor.daemonExited(reason: .uncaughtSignal, status: SIGKILL)
        XCTAssertTrue(harness.scheduledDelays.isEmpty)
    }

    func testStopAfterCrashCancelsPendingRestart() {
        let harness = SupervisorHarness(restartSucceeds: true)
        harness.supervisor.daemonWillStart()
        harness.supervisor.daemonExited(reason: .exit, status: 1)
        harness.supervisor.stopRequested()
        harness.runScheduled()
        XCTAssertEqual(harness.restarts, 0)
        XCTAssertNil(harness.supervisor.statusLine)
    }

    func testFailedSpawnRetriesWithNextBackoff() {
        let harness = SupervisorHarness(restartSucceeds: false)
        harness.supervisor.daemonWillStart()
        harness.supervisor.daemonExited(reason: .exit, status: 1)
        harness.runScheduled()
        XCTAssertEqual(harness.restarts, 1)
        XCTAssertEqual(harness.scheduledDelays, [1, 2])
    }

    func testCountResetsAfterAStableRun() {
        let harness = SupervisorHarness(restartSucceeds: true)
        harness.supervisor.daemonWillStart()
        harness.supervisor.daemonExited(reason: .exit, status: 1)
        harness.runScheduled()
        harness.supervisor.daemonWillStart()
        harness.clock = harness.clock.addingTimeInterval(DaemonSupervisor.stableRunSeconds + 1)
        harness.supervisor.daemonExited(reason: .exit, status: 1)
        XCTAssertEqual(harness.scheduledDelays, [1, 1])
    }

    // MARK: - Screen Recording registration

    func testPermissionFlowUsesTheRegistrationProbe() async {
        let probeCalls = LockedCounter()
        let granted = LockedFlag()
        let service = ScreenCaptureService(
            preflight: { granted.value },
            registrationProbe: {
                probeCalls.increment()
                granted.set(true)
            },
            pollIntervalNanoseconds: 1_000_000
        )
        let result = await service.registerAndAwaitGrant()
        XCTAssertTrue(result)
        XCTAssertEqual(probeCalls.value, 1)
    }

    func testPermissionFlowReportsDenialAfterBoundedPolling() async {
        let probeCalls = LockedCounter()
        let service = ScreenCaptureService(
            preflight: { false },
            registrationProbe: { probeCalls.increment() },
            pollIntervalNanoseconds: 1_000_000
        )
        let result = await service.registerAndAwaitGrant()
        XCTAssertFalse(result)
        XCTAssertEqual(probeCalls.value, 1)
    }
}

// MARK: - Helpers

private final class LockedCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func increment() { lock.lock(); count += 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
}

private final class LockedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    func set(_ v: Bool) { lock.lock(); flag = v; lock.unlock() }
    var value: Bool { lock.lock(); defer { lock.unlock() }; return flag }
}

/// A supervisor whose schedule and clock the test drives by hand.
private final class SupervisorHarness {
    var scheduledDelays: [TimeInterval] = []
    var pending: [() -> Void] = []
    var restarts = 0
    var clock = Date(timeIntervalSince1970: 1_800_000_000)
    lazy var supervisor = DaemonSupervisor(
        restart: { [unowned self] in
            self.restarts += 1
            return self.restartSucceeds
        },
        schedule: { [unowned self] delay, work in
            self.scheduledDelays.append(delay)
            self.pending.append(work)
        },
        now: { [unowned self] in self.clock }
    )
    private let restartSucceeds: Bool

    init(restartSucceeds: Bool) {
        self.restartSucceeds = restartSucceeds
    }

    /// Run the work scheduled so far (not work those runs schedule).
    func runScheduled() {
        let work = pending
        pending = []
        work.forEach { $0() }
    }
}
