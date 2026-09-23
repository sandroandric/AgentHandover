import ApplicationServices
import Foundation

/// One ancestor of the focused element (`agenthandover_common::event::AxAncestor`).
struct ObservationAXAncestor: Codable, Equatable {
    let role: String
    var title: String?
    var description: String?
    /// Position, among this ancestor's children, of the child on the path
    /// down to the focused element.
    var index: Int?
}

/// The focused UI element (`agenthandover_common::event::FocusedElement`).
struct ObservationFocusedElement: Codable, Equatable {
    let role: String
    var subrole: String?
    var title: String?
    var description: String?
    var value: String?
    var columnHeader: String?
    var document: String?
    var url: String?
    var ancestors: [ObservationAXAncestor] = []

    enum CodingKeys: String, CodingKey {
        case role, subrole, title, description, value
        case columnHeader = "column_header"
        case document, url, ancestors
    }
}

/// The Accessibility reads the reader needs, so the walk can be tested
/// against a fake tree.  `AXElementNode` wraps a live `AXUIElement`.
protocol AXNode {
    /// A string attribute (a CFString, a CFURL's absolute string, or a
    /// number's description); nil when absent or empty.
    func string(_ attribute: String) -> String?
    func element(_ attribute: String) -> Self?
    func elements(_ attribute: String) -> [Self]
    /// The location of a CFRange attribute (e.g. `AXColumnIndexRange`).
    func rangeLocation(_ attribute: String) -> Int?
    /// Position of `child` among this node's `AXChildren`; nil when absent
    /// or when the node has more children than is cheap to scan.
    func childIndex(of child: Self) -> Int?
}

/// Reads the focused element, its ancestor chain and its window's document.
///
/// Privacy: a secure (password) field is never read (nil is returned
/// before any other attribute is touched), and `AXValue` of the focused
/// element is read only when `captureValues` (`[privacy]
/// capture_input_values`) is on.  Ancestors contribute role, title,
/// description and index only, never values.
enum FocusedElementReader {
    static let maxAncestors = 6
    static let maxTextLength = 300

    static func read<N: AXNode>(focused: N, captureValues: Bool) -> ObservationFocusedElement? {
        let subrole = focused.string(kAXSubroleAttribute as String)
        if subrole == (kAXSecureTextFieldSubrole as String) { return nil }
        guard let role = focused.string(kAXRoleAttribute as String) else { return nil }

        var element = ObservationFocusedElement(role: role)
        element.subrole = subrole
        element.title = clip(
            focused.string(kAXTitleAttribute as String)
                // An unlabelled field's visible label (a static text).
                ?? focused.element(kAXTitleUIElementAttribute as String)?
                    .string(kAXValueAttribute as String))
        element.description = clip(focused.string(kAXDescriptionAttribute as String))
        if captureValues {
            element.value = clip(focused.string(kAXValueAttribute as String))
        }
        element.document = focused.element(kAXWindowAttribute as String)?
            .string(kAXDocumentAttribute as String)

        var child = focused
        var rowIndex: Int?
        var table: N?
        for _ in 0..<maxAncestors {
            guard let parent = child.element(kAXParentAttribute as String),
                  let parentRole = parent.string(kAXRoleAttribute as String),
                  parentRole != (kAXWindowRole as String),
                  parentRole != (kAXApplicationRole as String) else { break }
            let index = parent.childIndex(of: child)
            element.ancestors.append(ObservationAXAncestor(
                role: parentRole,
                title: clip(parent.string(kAXTitleAttribute as String)),
                description: clip(parent.string(kAXDescriptionAttribute as String)),
                index: index))
            if element.ancestors.count == 1 && parentRole == (kAXRowRole as String) {
                rowIndex = index
            }
            if table == nil && tableRoles.contains(parentRole) { table = parent }
            if element.url == nil && parentRole == "AXWebArea" {
                element.url = parent.string(kAXURLAttribute as String).map(stripQuery)
            }
            child = parent
        }

        if role == (kAXCellRole as String) {
            element.columnHeader = columnHeader(cell: focused, rowIndex: rowIndex, table: table)
        }
        return element
    }

    private static let tableRoles: Set<String> = [
        kAXTableRole as String, kAXOutlineRole as String, "AXGrid",
    ]

    /// The cell's column header: the cell's own `AXColumnHeaderUIElements`,
    /// else the table's header at the cell's column (from
    /// `AXColumnIndexRange`, else the cell's index in its row).
    private static func columnHeader<N: AXNode>(cell: N, rowIndex: Int?, table: N?) -> String? {
        if let header = cell.elements(kAXColumnHeaderUIElementsAttribute as String).first {
            return headerText(header)
        }
        guard let table,
              let column = cell.rangeLocation(kAXColumnIndexRangeAttribute as String) ?? rowIndex
        else { return nil }
        let headers = table.elements(kAXColumnHeaderUIElementsAttribute as String)
        return column < headers.count ? headerText(headers[column]) : nil
    }

    private static func headerText<N: AXNode>(_ header: N) -> String? {
        clip(header.string(kAXTitleAttribute as String)
            ?? header.string(kAXValueAttribute as String)
            ?? header.string(kAXDescriptionAttribute as String))
    }

    static func clip(_ text: String?) -> String? {
        guard let text = text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty
        else { return nil }
        return String(text.prefix(maxTextLength))
    }

    /// Origin and path only, as the extension stores page URLs: query
    /// strings and fragments carry search terms and tokens.
    static func stripQuery(_ url: String) -> String {
        guard var parts = URLComponents(string: url) else { return url }
        parts.query = nil
        parts.fragment = nil
        parts.user = nil
        parts.password = nil
        return parts.string ?? url
    }
}

/// An `AXUIElement` read through `AXNode`.
struct AXElementNode: AXNode {
    let element: AXUIElement

    /// Scans at most this many children (a spreadsheet grid can expose
    /// thousands; its row, where the index matters, has few).
    private static let maxChildrenScanned = 1000

    private func copy(_ attribute: String) -> CFTypeRef? {
        var value: CFTypeRef?
        guard AXUIElementCopyAttributeValue(element, attribute as CFString, &value) == .success
        else { return nil }
        return value
    }

    func string(_ attribute: String) -> String? {
        guard let value = copy(attribute) else { return nil }
        let text: String?
        switch CFGetTypeID(value) {
        case CFStringGetTypeID(): text = (value as! CFString) as String
        case CFURLGetTypeID(): text = ((value as! CFURL) as URL).absoluteString
        case CFNumberGetTypeID(): text = (value as! NSNumber).stringValue
        default: text = nil
        }
        return text?.isEmpty == false ? text : nil
    }

    func element(_ attribute: String) -> AXElementNode? {
        guard let value = copy(attribute), CFGetTypeID(value) == AXUIElementGetTypeID()
        else { return nil }
        return AXElementNode(element: value as! AXUIElement)
    }

    func elements(_ attribute: String) -> [AXElementNode] {
        guard let value = copy(attribute), CFGetTypeID(value) == CFArrayGetTypeID() else {
            return []
        }
        return ((value as! CFArray) as [AnyObject]).compactMap {
            CFGetTypeID($0) == AXUIElementGetTypeID()
                ? AXElementNode(element: $0 as! AXUIElement) : nil
        }
    }

    func rangeLocation(_ attribute: String) -> Int? {
        guard let value = copy(attribute), CFGetTypeID(value) == AXValueGetTypeID() else {
            return nil
        }
        var range = CFRange()
        guard AXValueGetValue(value as! AXValue, .cfRange, &range) else { return nil }
        return range.location
    }

    func childIndex(of child: AXElementNode) -> Int? {
        var count: CFIndex = 0
        guard AXUIElementGetAttributeValueCount(
                element, kAXChildrenAttribute as CFString, &count) == .success,
              count <= Self.maxChildrenScanned else { return nil }
        return elements(kAXChildrenAttribute as String)
            .firstIndex { CFEqual($0.element, child.element) }
    }
}
