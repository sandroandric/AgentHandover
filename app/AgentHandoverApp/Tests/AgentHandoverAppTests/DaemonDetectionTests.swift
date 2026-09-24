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
}
