import CoreGraphics
import XCTest
@testable import AgentHandoverApp

/// Which window a frame is of: the one you work in, not a toolbar or
/// another app's bar on top (2.0.2 captured Comet's toolbar strips as
/// black bands).
final class FocusedWindowTests: XCTestCase {
    private func window(_ number: Int, pid: Int, owner: String, title: String = "", layer: Int = 0,
                        _ x: Double, _ y: Double, _ w: Double, _ h: Double) -> [String: Any] {
        [
            kCGWindowNumber as String: number,
            kCGWindowOwnerPID as String: pid,
            kCGWindowOwnerName as String: owner,
            kCGWindowName as String: title,
            kCGWindowLayer as String: layer,
            kCGWindowBounds as String: ["X": x, "Y": y, "Width": w, "Height": h],
        ]
    }

    private func number(_ dict: [String: Any]?) -> Int? {
        dict?[kCGWindowNumber as String] as? Int
    }

    /// The MacBook Pro's list (2026-10-14): Comet's three untitled toolbar
    /// windows above its titled page window, then VS Code.
    private lazy var list: [[String: Any]] = [
        window(1, pid: 1, owner: "Window Server", layer: 25, 0, 0, 1512, 33),
        window(40486, pid: 81218, owner: "Comet", 0, 33, 1512, 43),
        window(40487, pid: 81218, owner: "Comet", 0, 76, 1512, 42),
        window(40488, pid: 81218, owner: "Comet", 0, 33, 1512, 121),
        window(40100, pid: 81218, owner: "Comet", title: "Bending Spoons (BSP) - Yahoo Finance", 0, 118, 1512, 864),
        window(39000, pid: 500, owner: "Code", title: "AgentHandover", 0, 33, 1400, 880),
    ]

    func testToolbarsAboveThePageAreNotTheWindowYouWorkIn() {
        XCTAssertEqual(number(ObservationService.focusedWindow(in: list, frontmostPID: 81218)), 40100)
    }

    func testTheFrontmostAppsWindowWinsOverAnotherAppsOnTop() {
        // Using VS Code while Comet's windows sit above it in the list.
        XCTAssertEqual(number(ObservationService.focusedWindow(in: list, frontmostPID: 500)), 39000)
    }

    func testATitledPopUpInFrontIsTaken() {
        // Comet's "Restore pages?" bubble (320 x 148, titled) above its page.
        let popUp = window(86, pid: 81218, owner: "Comet", title: "Restore pages?", 1035, 78, 320, 148)
        XCTAssertEqual(number(ObservationService.focusedWindow(in: [popUp] + list, frontmostPID: 81218)), 86)
    }

    func testWithoutATitleTheLargestWindowIsTaken() {
        let untitled = [
            window(5, pid: 7, owner: "Editor", 0, 0, 800, 120),
            window(6, pid: 7, owner: "Editor", 0, 120, 800, 700),
        ]
        XCTAssertEqual(number(ObservationService.focusedWindow(in: untitled, frontmostPID: 7)), 6)
    }

    func testWithoutAFrontmostAppWindowTheFrontmostTitledOneIsTaken() {
        XCTAssertEqual(number(ObservationService.focusedWindow(in: list, frontmostPID: nil)), 40100)
        // A frontmost app with no window on screen (a menu-bar app).
        XCTAssertEqual(number(ObservationService.focusedWindow(in: list, frontmostPID: 4242)), 40100)
    }

    func testOnlyBarsAndOtherLayersIsNoWindow() {
        let bars = [list[0], list[1], list[2], window(7, pid: 9, owner: "", 0, 0, 800, 600)]
        XCTAssertNil(ObservationService.focusedWindow(in: bars, frontmostPID: 81218))
        // A small but real window (100 x 100) counts.
        let small = [window(8, pid: 9, owner: "Calculator", title: "Calculator", 0, 0, 100, 100)]
        XCTAssertEqual(number(ObservationService.focusedWindow(in: small, frontmostPID: 9)), 8)
    }
}
