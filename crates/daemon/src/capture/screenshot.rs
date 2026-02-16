use core_graphics::display::CGDisplay;

/// Capture the main display as raw BGRA pixel data.
/// Returns (width, height, bytes) or None if capture fails.
///
/// On macOS 10.15+, this requires Screen Recording permission.
/// Without permission, `CGDisplay::image()` returns None.
///
/// Full PNG encoding will be added when the image crate is integrated.
pub fn capture_main_display() -> Option<(usize, usize, Vec<u8>)> {
    let display = CGDisplay::main();
    let image = display.image()?;

    let width = image.width();
    let height = image.height();
    let data = image.data();
    let bytes = data.bytes().to_vec();

    Some((width, height, bytes))
}

/// Check if screen recording permission is granted.
/// On macOS 10.15+, `CGDisplayCreateImage` returns NULL (mapped to None)
/// without the Screen Recording permission.
pub fn has_screen_recording_permission() -> bool {
    let display = CGDisplay::main();
    display.image().is_some()
}
