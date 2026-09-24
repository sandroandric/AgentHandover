import XCTest
@testable import AgentHandoverApp

/// Seen on the test Mac after the 0.5.0 upgrade: with Comet open, its
/// native-messaging bridge (`ah-observer chrome-extension://…/`) counted as
/// the daemon, so the relaunched app never started capture.
final class DaemonDetectionTests: XCTestCase {
    func testABrowsersBridgeIsNotTheDaemon() {
        let listing = """
        /Applications/Comet.app/Contents/MacOS/Comet
        /usr/local/lib/agenthandover/ah-observer chrome-extension://jpemkdcihaijkolbkankcldmiimmmnfo/
        /usr/local/lib/agenthandover/ah-observer chrome-extension://jpemkdcihaijkolbkankcldmiimmmnfo/ --parent-window=0
        """
        XCTAssertFalse(ServiceController.daemonListed(inProcessList: listing))
    }

    func testTheDaemonStartedBareIsFound() {
        let listing = """
        /usr/local/lib/agenthandover/ah-observer chrome-extension://jpemkdcihaijkolbkankcldmiimmmnfo/
        /usr/local/lib/agenthandover/ah-observer
        """
        XCTAssertTrue(ServiceController.daemonListed(inProcessList: listing))
    }

    func testOtherProcessesNamingItAreNotTheDaemon() {
        let listing = """
        /usr/bin/pgrep -l ah-observer
        /bin/zsh -c tail -f /tmp/ah-observer
        """
        XCTAssertFalse(ServiceController.daemonListed(inProcessList: listing))
    }

    /// A child printing more than a pipe holds (64 KB) must not deadlock the
    /// app: `ps -ax -o args=` printed 64,320 bytes on the test Mac.
    func testALargeOutputIsReadWithoutDeadlock() {
        let done = expectation(description: "shell returned")
        var output = ""
        DispatchQueue.global().async {
            output = ServiceController.shell("/bin/sh", args: ["-c", "head -c 300000 /dev/zero | tr '\\0' 'x'"])
            done.fulfill()
        }
        wait(for: [done], timeout: 10)
        XCTAssertEqual(output.count, 300_000)
    }
}

/// Measured on the test Mac: re-checking the daemon's code signature on
/// every connection was most of the idle app's CPU.
final class ValidatedPeersTests: XCTestCase {
    private func token(pid: UInt32, version: UInt32) -> audit_token_t {
        var t = audit_token_t()
        t.val.5 = pid
        t.val.7 = version
        return t
    }

    func testAPeerIsRememberedByPidAndVersion() {
        let peers = PeerPolicy.ValidatedPeers()
        let requirement = "anchor apple generic"
        XCTAssertFalse(peers.contains(token(pid: 100, version: 1), requirement: requirement))
        peers.insert(token(pid: 100, version: 1), requirement: requirement)
        XCTAssertTrue(peers.contains(token(pid: 100, version: 1), requirement: requirement))
        // The same PID re-executed or reused: checked again.
        XCTAssertFalse(peers.contains(token(pid: 100, version: 2), requirement: requirement))
        // Another requirement: checked again.
        XCTAssertFalse(peers.contains(token(pid: 100, version: 1), requirement: "anchor apple"))
    }

    func testTheCacheIsBounded() {
        let peers = PeerPolicy.ValidatedPeers()
        for pid in 1...40 { peers.insert(token(pid: UInt32(pid), version: 1), requirement: "r") }
        XCTAssertFalse(peers.contains(token(pid: 1, version: 1), requirement: "r"))
        XCTAssertTrue(peers.contains(token(pid: 40, version: 1), requirement: "r"))
    }
}
