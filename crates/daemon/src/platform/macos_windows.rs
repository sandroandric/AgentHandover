use core_foundation::base::{CFType, TCFType};
use core_foundation::boolean::CFBoolean;
use core_foundation::dictionary::CFDictionaryRef;
use core_foundation::number::CFNumber;
use core_foundation::string::{CFString, CFStringRef};
use core_graphics::display::CGDisplay;
use core_graphics::window::{
    copy_window_info, kCGNullWindowID, kCGWindowBounds, kCGWindowLayer,
    kCGWindowListExcludeDesktopElements, kCGWindowListOptionOnScreenOnly, kCGWindowName,
    kCGWindowNumber, kCGWindowOwnerName, kCGWindowOwnerPID,
};
use agenthandover_common::event::{CursorPosition, DisplayInfo, WindowInfo};
use tracing::debug;

/// Get all active display information (multi-monitor topology).
pub fn get_display_topology() -> Vec<DisplayInfo> {
    let display_ids = CGDisplay::active_displays().unwrap_or_default();

    display_ids
        .iter()
        .map(|&id| {
            let display = CGDisplay::new(id);
            let bounds = display.bounds();
            let scale = if display.pixels_wide() > 0 && bounds.size.width > 0.0 {
                display.pixels_wide() as f64 / bounds.size.width
            } else {
                1.0
            };

            DisplayInfo {
                display_id: id.to_string(),
                bounds_global_px: [
                    bounds.origin.x as i32,
                    bounds.origin.y as i32,
                    bounds.size.width as i32,
                    bounds.size.height as i32,
                ],
                scale_factor: scale,
                orientation: display.rotation() as u32,
            }
        })
        .collect()
}

/// Get the focused (frontmost) window info.
/// Uses CGWindowListCopyWindowInfo to find the topmost on-screen window at layer 0
/// (normal window layer). Extracts window title, owner name, bounds, and window ID.
///
/// Returns None if no window is focused or in headless environments.
pub fn get_focused_window() -> Option<WindowInfo> {
    let options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements;
    let window_list = copy_window_info(options, kCGNullWindowID)?;

    let count = window_list.len();
    for i in 0..count {
        // Each entry in the CFArray is a *const c_void pointing to a CFDictionary.
        let dict_ref: CFDictionaryRef = unsafe {
            let ptr = window_list.get_unchecked(i);
            *(&ptr as *const _ as *const CFDictionaryRef)
        };

        // Only consider normal windows (layer == 0).
        let layer = get_dict_number_cfstr(dict_ref, unsafe { kCGWindowLayer });
        match layer {
            Some(0) => {}
            Some(_) => continue,
            None => continue,
        }

        // Extract window number (ID).
        let window_number =
            get_dict_number_cfstr(dict_ref, unsafe { kCGWindowNumber }).unwrap_or(0);

        // Extract owner name (application name).
        let owner_name =
            get_dict_string_cfstr(dict_ref, unsafe { kCGWindowOwnerName }).unwrap_or_default();

        // Extract window title. Some windows (e.g. menu bar owners) lack a title.
        let title =
            get_dict_string_cfstr(dict_ref, unsafe { kCGWindowName }).unwrap_or_default();

        // Extract window bounds.
        let bounds = get_dict_bounds_cfstr(dict_ref, unsafe { kCGWindowBounds });

        // Extract owner PID for the app_id field.
        let owner_pid =
            get_dict_number_cfstr(dict_ref, unsafe { kCGWindowOwnerPID }).unwrap_or(0);

        // Skip windows with empty owner (system artifacts).
        if owner_name.is_empty() {
            continue;
        }

        // Determine if the window is fullscreen by comparing bounds to display bounds.
        let is_fullscreen = is_window_fullscreen(&bounds);

        debug!(
            window_id = window_number,
            app = %owner_name,
            title = %title,
            "Focused window detected"
        );

        return Some(WindowInfo {
            window_id: window_number.to_string(),
            app_id: format!("pid:{owner_pid}:{owner_name}"),
            title,
            bounds_global_px: bounds,
            z_order: 0, // Topmost on-screen window in the list
            is_fullscreen,
        });
    }

    None
}

/// Get the current mouse cursor position using CoreGraphics.
///
/// Uses CGEventCreate with a null source and CGEventGetLocation to read the
/// cursor's global position without intercepting any events.
pub fn get_cursor_position() -> Option<CursorPosition> {
    use core_graphics::event::CGEvent;
    use core_graphics::event_source::CGEventSource;
    use core_graphics::event_source::CGEventSourceStateID;

    // Create a dummy event from the HID system state to read the cursor position.
    let source = CGEventSource::new(CGEventSourceStateID::HIDSystemState).ok()?;
    let event = CGEvent::new(source).ok()?;
    let location = event.location();

    Some(CursorPosition {
        x: location.x as i32,
        y: location.y as i32,
    })
}

/// Determine the ui_scale factor for a given cursor position.
/// Finds which display the cursor is on and returns that display's scale_factor.
pub fn get_ui_scale_for_position(
    cursor: &CursorPosition,
    displays: &[DisplayInfo],
) -> Option<f64> {
    for display in displays {
        let [dx, dy, dw, dh] = display.bounds_global_px;
        if cursor.x >= dx && cursor.x < dx + dw && cursor.y >= dy && cursor.y < dy + dh {
            return Some(display.scale_factor);
        }
    }
    // Fallback: return primary display scale.
    displays.first().map(|d| d.scale_factor)
}

// --- Internal helpers ---

/// Extract an i64 from a CFDictionary value keyed by a raw CFStringRef.
fn get_dict_number_cfstr(dict: CFDictionaryRef, key: CFStringRef) -> Option<i64> {
    unsafe {
        let mut value = std::ptr::null();
        if core_foundation::dictionary::CFDictionaryGetValueIfPresent(
            dict,
            key as *const _,
            &mut value,
        ) != 0
            && !value.is_null()
        {
            let cf_type: CFType =
                TCFType::wrap_under_get_rule(value as core_foundation::base::CFTypeRef);
            if let Some(cf_num) = cf_type.downcast::<CFNumber>() {
                return cf_num.to_i64();
            }
        }
    }
    None
}

/// Extract a String from a CFDictionary value keyed by a raw CFStringRef.
fn get_dict_string_cfstr(dict: CFDictionaryRef, key: CFStringRef) -> Option<String> {
    unsafe {
        let mut value = std::ptr::null();
        if core_foundation::dictionary::CFDictionaryGetValueIfPresent(
            dict,
            key as *const _,
            &mut value,
        ) != 0
            && !value.is_null()
        {
            let cf_type: CFType =
                TCFType::wrap_under_get_rule(value as core_foundation::base::CFTypeRef);
            if let Some(cf_str) = cf_type.downcast::<CFString>() {
                return Some(cf_str.to_string());
            }
        }
    }
    None
}

/// Extract an i64 from a CFDictionary value keyed by a Rust CFString.
fn get_dict_number(dict: CFDictionaryRef, key: &CFString) -> Option<i64> {
    get_dict_number_cfstr(dict, key.as_concrete_TypeRef())
}

/// Extract window bounds [x, y, width, height] from a CFDictionary.
/// The kCGWindowBounds value is itself a CFDictionary with X, Y, Width, Height keys.
fn get_dict_bounds_cfstr(dict: CFDictionaryRef, key: CFStringRef) -> [i32; 4] {
    unsafe {
        let mut value = std::ptr::null();
        if core_foundation::dictionary::CFDictionaryGetValueIfPresent(
            dict,
            key as *const _,
            &mut value,
        ) != 0
            && !value.is_null()
        {
            // The bounds value is a CFDictionary with keys "X", "Y", "Width", "Height".
            let bounds_dict = value as CFDictionaryRef;

            let x = get_bounds_component(bounds_dict, "X");
            let y = get_bounds_component(bounds_dict, "Y");
            let w = get_bounds_component(bounds_dict, "Width");
            let h = get_bounds_component(bounds_dict, "Height");

            return [x, y, w, h];
        }
    }
    [0, 0, 0, 0]
}

/// Extract a numeric component from a bounds CFDictionary by string key name.
fn get_bounds_component(dict: CFDictionaryRef, key_name: &str) -> i32 {
    let key = CFString::new(key_name);
    get_dict_number(dict, &key).unwrap_or(0) as i32
}

/// Check if a window is fullscreen by comparing its bounds to any active display.
fn is_window_fullscreen(bounds: &[i32; 4]) -> bool {
    let display_ids = CGDisplay::active_displays().unwrap_or_default();
    for &id in &display_ids {
        let display = CGDisplay::new(id);
        let db = display.bounds();
        if bounds[0] == db.origin.x as i32
            && bounds[1] == db.origin.y as i32
            && bounds[2] == db.size.width as i32
            && bounds[3] == db.size.height as i32
        {
            return true;
        }
    }
    false
}

/// Retrieve a CFBoolean from a CFDictionary. Not currently used but available for
/// window-property queries that return booleans (e.g. kCGWindowIsOnscreen).
#[allow(dead_code)]
fn get_dict_bool_cfstr(dict: CFDictionaryRef, key: CFStringRef) -> Option<bool> {
    unsafe {
        let mut value = std::ptr::null();
        if core_foundation::dictionary::CFDictionaryGetValueIfPresent(
            dict,
            key as *const _,
            &mut value,
        ) != 0
            && !value.is_null()
        {
            let cf_type: CFType =
                TCFType::wrap_under_get_rule(value as core_foundation::base::CFTypeRef);
            if let Some(cf_bool) = cf_type.downcast::<CFBoolean>() {
                return Some(cf_bool == CFBoolean::true_value());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial(macos_ffi)]
    fn test_get_display_topology_returns_at_least_one() {
        let displays = get_display_topology();
        assert!(!displays.is_empty(), "Should detect at least one display");
        let primary = &displays[0];
        assert!(primary.scale_factor >= 1.0);
        assert!(primary.bounds_global_px[2] > 0, "Width should be positive");
        assert!(primary.bounds_global_px[3] > 0, "Height should be positive");
    }

    #[test]
    #[serial(macos_ffi)]
    fn test_get_focused_window_returns_some_info() {
        // In a desktop environment, there should be at least one window.
        // In CI/headless this may return None, which is acceptable.
        let window = get_focused_window();
        if let Some(w) = window {
            assert!(!w.app_id.is_empty(), "app_id should not be empty");
            assert!(!w.window_id.is_empty(), "window_id should not be empty");
        }
    }

    #[test]
    #[serial(macos_ffi)]
    fn test_get_cursor_position_returns_coordinates() {
        let pos = get_cursor_position();
        // In a desktop environment this should always succeed.
        if let Some(p) = pos {
            assert!(p.x > -100_000 && p.x < 100_000);
            assert!(p.y > -100_000 && p.y < 100_000);
        }
    }

    #[test]
    fn test_ui_scale_for_position() {
        let displays = vec![
            DisplayInfo {
                display_id: "1".to_string(),
                bounds_global_px: [0, 0, 1920, 1080],
                scale_factor: 2.0,
                orientation: 0,
            },
            DisplayInfo {
                display_id: "2".to_string(),
                bounds_global_px: [1920, 0, 1920, 1080],
                scale_factor: 1.0,
                orientation: 0,
            },
        ];

        let cursor_on_first = CursorPosition { x: 500, y: 500 };
        assert_eq!(
            get_ui_scale_for_position(&cursor_on_first, &displays),
            Some(2.0)
        );

        let cursor_on_second = CursorPosition { x: 2000, y: 500 };
        assert_eq!(
            get_ui_scale_for_position(&cursor_on_second, &displays),
            Some(1.0)
        );

        // Cursor outside all displays -- falls back to primary.
        let cursor_offscreen = CursorPosition { x: -500, y: -500 };
        assert_eq!(
            get_ui_scale_for_position(&cursor_offscreen, &displays),
            Some(2.0)
        );

        // Empty display list.
        let cursor = CursorPosition { x: 0, y: 0 };
        assert_eq!(get_ui_scale_for_position(&cursor, &[]), None);
    }
}
