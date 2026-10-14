import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

struct ObservationCursorPosition: Codable {
    let x: Int
    let y: Int
}

struct ObservationWindowInfo: Codable {
    let windowId: String
    let appId: String
    let title: String
    let boundsGlobalPx: [Int]
    let zOrder: UInt32
    let isFullscreen: Bool

    enum CodingKeys: String, CodingKey {
        case windowId = "window_id"
        case appId = "app_id"
        case title
        case boundsGlobalPx = "bounds_global_px"
        case zOrder = "z_order"
        case isFullscreen = "is_fullscreen"
    }
}

struct ObservationDisplayInfo: Codable {
    let displayId: String
    let boundsGlobalPx: [Int]
    let scaleFactor: Double
    let orientation: UInt32

    enum CodingKeys: String, CodingKey {
        case displayId = "display_id"
        case boundsGlobalPx = "bounds_global_px"
        case scaleFactor = "scale_factor"
        case orientation
    }
}

/// One sleep of the Mac (`agenthandover_common::observation::SleepRecord`),
/// timestamps in ISO 8601 UTC.
struct ObservationSleepRecord: Codable, Equatable {
    let sleptAt: String
    let wokeAt: String

    enum CodingKeys: String, CodingKey {
        case sleptAt = "slept_at"
        case wokeAt = "woke_at"
    }
}

struct ObservationSnapshot: Codable {
    let accessibilityGranted: Bool
    let screenRecordingGranted: Bool
    let secureFieldFocused: Bool
    let focusedWindow: ObservationWindowInfo?
    let displayTopology: [ObservationDisplayInfo]
    let cursorGlobalPx: ObservationCursorPosition?
    var lastSleep: ObservationSleepRecord? = nil

    enum CodingKeys: String, CodingKey {
        case accessibilityGranted = "accessibility_granted"
        case screenRecordingGranted = "screen_recording_granted"
        case secureFieldFocused = "secure_field_focused"
        case focusedWindow = "focused_window"
        case displayTopology = "display_topology"
        case cursorGlobalPx = "cursor_global_px"
        case lastSleep = "last_sleep"
    }
}

/// Records the Mac's most recent sleep from NSWorkspace's `willSleep` and
/// `didWake` notifications.  The daemon turns a new record into a
/// `SystemSleep` boundary event.
final class SleepMonitor {
    private let lock = NSLock()
    private var sleptAt: Date?
    private var record: ObservationSleepRecord?
    private var tokens: [NSObjectProtocol] = []

    /// A monitor observing the shared workspace's notifications.
    static func started() -> SleepMonitor {
        let monitor = SleepMonitor()
        let center = NSWorkspace.shared.notificationCenter
        monitor.tokens = [
            center.addObserver(
                forName: NSWorkspace.willSleepNotification, object: nil, queue: nil
            ) { [weak monitor] _ in monitor?.willSleep(at: Date()) },
            center.addObserver(
                forName: NSWorkspace.didWakeNotification, object: nil, queue: nil
            ) { [weak monitor] _ in monitor?.didWake(at: Date()) },
        ]
        return monitor
    }

    deinit {
        tokens.forEach { NSWorkspace.shared.notificationCenter.removeObserver($0) }
    }

    func willSleep(at date: Date) {
        lock.lock(); defer { lock.unlock() }
        sleptAt = date
    }

    /// A wake without a preceding sleep notification (the app started
    /// while asleep) records nothing: there is no honest start time.
    func didWake(at date: Date) {
        lock.lock(); defer { lock.unlock() }
        guard let start = sleptAt else { return }
        let format = ISO8601DateFormatter()
        record = ObservationSleepRecord(
            sleptAt: format.string(from: start), wokeAt: format.string(from: date))
        sleptAt = nil
    }

    var lastSleep: ObservationSleepRecord? {
        lock.lock(); defer { lock.unlock() }
        return record
    }
}

/// App-owned observation service.
///
/// This keeps all TCC-sensitive reads in AgentHandover.app so the daemon can
/// consume observation metadata without directly touching Accessibility APIs.
final class ObservationService {
    private let sleepMonitor: SleepMonitor?

    init(sleepMonitor: SleepMonitor? = nil) {
        self.sleepMonitor = sleepMonitor
    }

    func snapshot() -> ObservationSnapshot {
        let accessibilityGranted = PermissionChecker.isAccessibilityGranted()
        let screenRecordingGranted = PermissionChecker.isScreenRecordingGranted()
        let displays = getDisplayTopology()

        return ObservationSnapshot(
            accessibilityGranted: accessibilityGranted,
            screenRecordingGranted: screenRecordingGranted,
            secureFieldFocused: accessibilityGranted ? isSecureFieldFocused() : false,
            focusedWindow: accessibilityGranted ? getFocusedWindow(displays: displays) : nil,
            displayTopology: displays,
            cursorGlobalPx: getCursorPosition(),
            lastSleep: sleepMonitor?.lastSleep
        )
    }

    /// The system-wide focused element with its ancestor chain, or nil when
    /// Accessibility is not granted, nothing is focused, or the focused
    /// element is a secure field (see `FocusedElementReader`).
    func focusedElement(captureValues: Bool) -> ObservationFocusedElement? {
        guard PermissionChecker.isAccessibilityGranted() else { return nil }
        let systemWide = AXUIElementCreateSystemWide()
        // Set on the system-wide element, the timeout applies to every AX
        // call this app makes, so a hung app costs at most 100 ms per read.
        AXUIElementSetMessagingTimeout(systemWide, 0.1)
        guard let focused = AXElementNode(element: systemWide)
            .element(kAXFocusedUIElementAttribute as String) else { return nil }
        return FocusedElementReader.read(focused: focused, captureValues: captureValues)
    }

    private func getDisplayTopology() -> [ObservationDisplayInfo] {
        var count: UInt32 = 0
        guard CGGetActiveDisplayList(0, nil, &count) == .success, count > 0 else {
            return []
        }

        var ids = [CGDirectDisplayID](repeating: 0, count: Int(count))
        guard CGGetActiveDisplayList(count, &ids, &count) == .success else {
            return []
        }

        return ids.map { id in
            let bounds = CGDisplayBounds(id)
            let pixelsWide = Int(CGDisplayPixelsWide(id))
            let logicalWidth = Int(bounds.width)
            let scaleFactor = logicalWidth > 0 ? Double(pixelsWide) / Double(logicalWidth) : 1.0

            return ObservationDisplayInfo(
                displayId: String(id),
                boundsGlobalPx: [
                    Int(bounds.origin.x),
                    Int(bounds.origin.y),
                    Int(bounds.width),
                    Int(bounds.height)
                ],
                scaleFactor: scaleFactor,
                orientation: UInt32(CGDisplayRotation(id))
            )
        }
    }

    func getFocusedWindow(displays: [ObservationDisplayInfo]) -> ObservationWindowInfo? {
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let windowList = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]],
              let dict = Self.focusedWindow(
                  in: windowList,
                  frontmostPID: NSWorkspace.shared.frontmostApplication.map { Int($0.processIdentifier) }
              ) else {
            return nil
        }

        let ownerName = dict[kCGWindowOwnerName as String] as? String ?? ""
        let windowNumber = dict[kCGWindowNumber as String] as? Int ?? 0
        let title = dict[kCGWindowName as String] as? String ?? ""
        let ownerPID = dict[kCGWindowOwnerPID as String] as? Int ?? 0
        let bounds = Self.bounds(of: dict)

        return ObservationWindowInfo(
            windowId: String(windowNumber),
            appId: "pid:\(ownerPID):\(ownerName)",
            title: title,
            boundsGlobalPx: bounds,
            zOrder: 0,
            isFullscreen: isWindowFullscreen(bounds: bounds, displays: displays)
        )
    }

    /// Narrower or shorter than this (points), a window is a bar or a strip,
    /// not one you work in.
    static let minimumWindowSide = 100

    /// The window you are working in, from `windowList` (front to back, as
    /// `CGWindowListCopyWindowInfo` returns it), among ordinary windows
    /// (layer 0, an owner, at least `minimumWindowSide` each way): of the
    /// frontmost app (else of every app), the frontmost one with a title,
    /// else the largest.  Taking simply the topmost layer-0 window caught
    /// toolbars: Comet draws its tab strip and toolbar as untitled windows
    /// (1512 x 43, x 42 and x 121 on a MacBook Pro) above the titled page
    /// window, and every Comet frame was a black band.
    static func focusedWindow(in windowList: [[String: Any]], frontmostPID: Int?) -> [String: Any]? {
        let ordinary = windowList.filter { dict in
            guard let layer = dict[kCGWindowLayer as String] as? Int, layer == 0,
                  let owner = dict[kCGWindowOwnerName as String] as? String, !owner.isEmpty else {
                return false
            }
            let b = bounds(of: dict)
            return b[2] >= minimumWindowSide && b[3] >= minimumWindowSide
        }
        let own = frontmostPID.map { pid in
            ordinary.filter { ($0[kCGWindowOwnerPID as String] as? Int) == pid }
        } ?? []
        let candidates = own.isEmpty ? ordinary : own
        if let titled = candidates.first(where: { !(($0[kCGWindowName as String] as? String) ?? "").isEmpty }) {
            return titled
        }
        return candidates.max { area($0) < area($1) }
    }

    private static func area(_ dict: [String: Any]) -> Int {
        let b = bounds(of: dict)
        return b[2] * b[3]
    }

    static func bounds(of dict: [String: Any]) -> [Int] {
        let boundsDict = dict[kCGWindowBounds as String] as? [String: Any] ?? [:]
        return [
            Int((boundsDict["X"] as? Double) ?? 0),
            Int((boundsDict["Y"] as? Double) ?? 0),
            Int((boundsDict["Width"] as? Double) ?? 0),
            Int((boundsDict["Height"] as? Double) ?? 0)
        ]
    }

    private func getCursorPosition() -> ObservationCursorPosition? {
        guard let event = CGEvent(source: nil) else { return nil }
        let location = event.location
        return ObservationCursorPosition(x: Int(location.x), y: Int(location.y))
    }

    private func isWindowFullscreen(bounds: [Int], displays: [ObservationDisplayInfo]) -> Bool {
        guard bounds.count == 4 else { return false }
        for display in displays {
            let db = display.boundsGlobalPx
            if bounds[0] == db[0]
                && bounds[1] == db[1]
                && bounds[2] == db[2]
                && bounds[3] == db[3] {
                return true
            }
        }
        return false
    }

    private func isSecureFieldFocused() -> Bool {
        let systemWide = AXUIElementCreateSystemWide()
        AXUIElementSetMessagingTimeout(systemWide, 0.1)

        var focusedValue: CFTypeRef?
        let focusedError = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedValue
        )
        guard focusedError == .success, let focusedValue else {
            return false
        }

        let focusedElement = unsafeBitCast(focusedValue, to: AXUIElement.self)
        var subroleValue: CFTypeRef?
        let subroleError = AXUIElementCopyAttributeValue(
            focusedElement,
            kAXSubroleAttribute as CFString,
            &subroleValue
        )
        guard subroleError == .success, let subroleValue else {
            return false
        }

        let subrole = unsafeBitCast(subroleValue, to: CFString.self) as String
        return subrole == (kAXSecureTextFieldSubrole as String)
    }
}
