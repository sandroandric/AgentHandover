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

struct ObservationSnapshot: Codable {
    let accessibilityGranted: Bool
    let screenRecordingGranted: Bool
    let secureFieldFocused: Bool
    let focusedWindow: ObservationWindowInfo?
    let displayTopology: [ObservationDisplayInfo]
    let cursorGlobalPx: ObservationCursorPosition?

    enum CodingKeys: String, CodingKey {
        case accessibilityGranted = "accessibility_granted"
        case screenRecordingGranted = "screen_recording_granted"
        case secureFieldFocused = "secure_field_focused"
        case focusedWindow = "focused_window"
        case displayTopology = "display_topology"
        case cursorGlobalPx = "cursor_global_px"
    }
}

/// App-owned observation service.
///
/// This keeps all TCC-sensitive reads in AgentHandover.app so the daemon can
/// consume observation metadata without directly touching Accessibility APIs.
final class ObservationService {

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
            cursorGlobalPx: getCursorPosition()
        )
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

    private func getFocusedWindow(displays: [ObservationDisplayInfo]) -> ObservationWindowInfo? {
        let options: CGWindowListOption = [.optionOnScreenOnly, .excludeDesktopElements]
        guard let windowList = CGWindowListCopyWindowInfo(options, kCGNullWindowID) as? [[String: Any]] else {
            return nil
        }

        for dict in windowList {
            guard let layer = dict[kCGWindowLayer as String] as? Int, layer == 0 else {
                continue
            }

            let ownerName = dict[kCGWindowOwnerName as String] as? String ?? ""
            guard !ownerName.isEmpty else { continue }

            let windowNumber = dict[kCGWindowNumber as String] as? Int ?? 0
            let title = dict[kCGWindowName as String] as? String ?? ""
            let ownerPID = dict[kCGWindowOwnerPID as String] as? Int ?? 0

            let boundsDict = dict[kCGWindowBounds as String] as? [String: Any] ?? [:]
            let bounds = [
                Int((boundsDict["X"] as? Double) ?? 0),
                Int((boundsDict["Y"] as? Double) ?? 0),
                Int((boundsDict["Width"] as? Double) ?? 0),
                Int((boundsDict["Height"] as? Double) ?? 0)
            ]

            return ObservationWindowInfo(
                windowId: String(windowNumber),
                appId: "pid:\(ownerPID):\(ownerName)",
                title: title,
                boundsGlobalPx: bounds,
                zOrder: 0,
                isFullscreen: isWindowFullscreen(bounds: bounds, displays: displays)
            )
        }

        return nil
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
