use core_graphics::display::CGDisplay;

/// Capture the main display as raw BGRA pixel data.
/// Returns (width, height, bytes) or None if capture fails.
///
/// On macOS 10.15+, this requires Screen Recording permission.
/// Without permission, `CGDisplay::image()` returns None.
///
/// The returned pixel data is guaranteed to be contiguous (stride = width * 4)
/// with any row padding stripped, so callers can index as `(y * width + x) * 4`.
pub fn capture_main_display() -> Option<(usize, usize, Vec<u8>)> {
    let display = CGDisplay::main();
    let image = display.image()?;

    let width = image.width();
    let height = image.height();
    let bytes_per_row = image.bytes_per_row();
    let data = image.data();
    let raw_bytes = data.bytes();

    let pixel_stride = width * 4;

    if bytes_per_row == pixel_stride {
        // No row padding — return as-is (fast path)
        Some((width, height, raw_bytes.to_vec()))
    } else {
        // Strip row padding to produce contiguous BGRA data.
        // Some display configurations pad rows for alignment.
        let mut pixels = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            let row_start = y * bytes_per_row;
            let row_end = row_start + pixel_stride;
            if row_end <= raw_bytes.len() {
                pixels.extend_from_slice(&raw_bytes[row_start..row_end]);
            }
        }
        Some((width, height, pixels))
    }
}

/// Check if screen recording permission is granted.
/// On macOS 10.15+, `CGDisplayCreateImage` returns NULL (mapped to None)
/// without the Screen Recording permission.
pub fn has_screen_recording_permission() -> bool {
    let display = CGDisplay::main();
    display.image().is_some()
}
