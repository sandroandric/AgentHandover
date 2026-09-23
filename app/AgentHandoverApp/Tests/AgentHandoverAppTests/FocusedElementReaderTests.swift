import XCTest
@testable import AgentHandoverApp

/// A fake Accessibility tree: every read is recorded so the tests can pin
/// what the reader is allowed to touch.
private final class FakeNode: AXNode {
    var strings: [String: String]
    var children: [FakeNode] = []
    var related: [String: FakeNode] = [:]
    var lists: [String: [FakeNode]] = [:]
    var ranges: [String: Int] = [:]
    weak var parent: FakeNode?
    private(set) var reads: [String] = []

    init(_ strings: [String: String]) { self.strings = strings }

    @discardableResult
    func add(_ child: FakeNode) -> FakeNode {
        children.append(child)
        child.parent = self
        return child
    }

    func string(_ attribute: String) -> String? {
        reads.append(attribute)
        return strings[attribute]
    }

    func element(_ attribute: String) -> FakeNode? {
        reads.append(attribute)
        return attribute == "AXParent" ? parent : related[attribute]
    }

    func elements(_ attribute: String) -> [FakeNode] {
        reads.append(attribute)
        return lists[attribute] ?? []
    }

    func rangeLocation(_ attribute: String) -> Int? { ranges[attribute] }

    func childIndex(of child: FakeNode) -> Int? {
        children.firstIndex { $0 === child }
    }
}

final class FocusedElementReaderTests: XCTestCase {

    /// A spreadsheet: window > table (with column headers) > row > cell.
    private func spreadsheet() -> (window: FakeNode, cell: FakeNode) {
        let window = FakeNode(["AXRole": "AXWindow", "AXDocument": "file:///Users/me/Q3.xlsx"])
        let table = window.add(FakeNode(["AXRole": "AXTable", "AXTitle": "Sheet1"]))
        table.lists["AXColumnHeaderUIElements"] = [
            FakeNode(["AXRole": "AXStaticText", "AXValue": "Customer"]),
            FakeNode(["AXRole": "AXStaticText", "AXTitle": "Amount"]),
        ]
        table.add(FakeNode(["AXRole": "AXRow"]))
        let row = table.add(FakeNode(["AXRole": "AXRow", "AXDescription": "Row 2"]))
        row.add(FakeNode(["AXRole": "AXCell", "AXValue": "Acme"]))
        let cell = row.add(FakeNode(["AXRole": "AXCell", "AXValue": "1,250.00"]))
        cell.related["AXWindow"] = window
        return (window, cell)
    }

    func testCellCarriesColumnHeaderValueAncestorsAndDocument() {
        let (_, cell) = spreadsheet()
        let element = FocusedElementReader.read(focused: cell, captureValues: true)

        XCTAssertEqual(element?.role, "AXCell")
        XCTAssertEqual(element?.value, "1,250.00")
        XCTAssertEqual(element?.columnHeader, "Amount", "header at the cell's index in its row")
        XCTAssertEqual(element?.document, "file:///Users/me/Q3.xlsx")
        XCTAssertEqual(element?.ancestors, [
            ObservationAXAncestor(role: "AXRow", title: nil, description: "Row 2", index: 1),
            ObservationAXAncestor(role: "AXTable", title: "Sheet1", description: nil, index: 1),
        ], "the walk stops at the window and records each child's index")
    }

    func testColumnIndexRangeWinsOverTheRowIndex() {
        let (_, cell) = spreadsheet()
        cell.ranges["AXColumnIndexRange"] = 0
        let element = FocusedElementReader.read(focused: cell, captureValues: false)
        XCTAssertEqual(element?.columnHeader, "Customer")
    }

    func testValueIsNotReadWithoutTheFlag() {
        let (_, cell) = spreadsheet()
        let element = FocusedElementReader.read(focused: cell, captureValues: false)
        XCTAssertNil(element?.value)
        XCTAssertFalse(cell.reads.contains("AXValue"), "AXValue must not even be read")
        XCTAssertEqual(element?.columnHeader, "Amount", "labels are still read")
    }

    func testSecureFieldIsNeverRead() {
        let form = FakeNode(["AXRole": "AXGroup", "AXTitle": "Sign in"])
        let password = form.add(FakeNode([
            "AXRole": "AXTextField", "AXSubrole": "AXSecureTextField",
            "AXTitle": "Password", "AXValue": "hunter2",
        ]))
        XCTAssertNil(FocusedElementReader.read(focused: password, captureValues: true))
        XCTAssertEqual(password.reads, ["AXSubrole"], "nothing but the subrole is touched")
    }

    func testUnlabelledFieldTakesItsTitleElementAndInheritsItsGroup() {
        let window = FakeNode(["AXRole": "AXWindow"])
        let toolbar = window.add(FakeNode(["AXRole": "AXToolbar", "AXDescription": "Search"]))
        let field = toolbar.add(FakeNode(["AXRole": "AXTextField", "AXValue": "invoice 1042"]))
        field.related["AXTitleUIElement"] = FakeNode(["AXRole": "AXStaticText", "AXValue": "Find"])

        let element = FocusedElementReader.read(focused: field, captureValues: false)
        XCTAssertEqual(element?.title, "Find")
        XCTAssertNil(element?.value)
        XCTAssertEqual(element?.ancestors.first?.description, "Search")
        XCTAssertNil(element?.columnHeader, "only cells look up a column header")
    }

    func testAncestorChainIsCappedAtSixLevels() {
        let window = FakeNode(["AXRole": "AXWindow"])  // holds the chain (parents are weak)
        var node = window
        for level in 0..<10 {
            node = node.add(FakeNode(["AXRole": "AXGroup", "AXTitle": "g\(level)"]))
        }
        let leaf = node.add(FakeNode(["AXRole": "AXButton", "AXTitle": "OK"]))
        let element = FocusedElementReader.read(focused: leaf, captureValues: false)
        XCTAssertEqual(element?.ancestors.count, FocusedElementReader.maxAncestors)
        XCTAssertEqual(element?.ancestors.first?.title, "g9")
        withExtendedLifetime(window) {}
    }

    func testWebAreaUrlLosesQueryAndFragment() {
        let window = FakeNode(["AXRole": "AXWindow"])
        let web = window.add(FakeNode([
            "AXRole": "AXWebArea",
            "AXURL": "https://user:pw@app.example.com/orders/42?token=abc#row",
        ]))
        let button = web.add(FakeNode(["AXRole": "AXButton", "AXTitle": "Save"]))
        let element = FocusedElementReader.read(focused: button, captureValues: false)
        XCTAssertEqual(element?.url, "https://app.example.com/orders/42")
    }

    func testLongTextIsClipped() {
        let field = FakeNode([
            "AXRole": "AXTextArea", "AXValue": String(repeating: "x", count: 1000),
        ])
        let element = FocusedElementReader.read(focused: field, captureValues: true)
        XCTAssertEqual(element?.value?.count, FocusedElementReader.maxTextLength)
    }

    func testReplyEncodesTheWireShape() throws {
        var element = ObservationFocusedElement(role: "AXCell")
        element.columnHeader = "Amount"
        let data = try JSONEncoder().encode(FocusedElementReply(focusedElement: element))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let body = try XCTUnwrap(json["focused_element"] as? [String: Any])
        XCTAssertEqual(body["role"] as? String, "AXCell")
        XCTAssertEqual(body["column_header"] as? String, "Amount")
        XCTAssertNil(body["value"], "absent fields are omitted")

        let none = try JSONEncoder().encode(FocusedElementReply(focusedElement: nil))
        XCTAssertEqual(String(data: none, encoding: .utf8), #"{"focused_element":null}"#)
    }

    func testSleepMonitorRecordsOnlyACompleteSleep() {
        let monitor = SleepMonitor()
        monitor.didWake(at: Date(timeIntervalSince1970: 100))
        XCTAssertNil(monitor.lastSleep, "a wake without a sleep has no start")

        monitor.willSleep(at: Date(timeIntervalSince1970: 1_790_000_000))
        XCTAssertNil(monitor.lastSleep)
        monitor.didWake(at: Date(timeIntervalSince1970: 1_790_003_600))
        XCTAssertEqual(monitor.lastSleep, ObservationSleepRecord(
            sleptAt: "2026-09-21T14:13:20Z", wokeAt: "2026-09-21T15:13:20Z"))
    }
}
