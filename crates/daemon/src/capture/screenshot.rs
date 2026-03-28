/// Capture the main display as raw BGRA pixel data.
/// Returns (width, height, bytes) or None if capture fails.
///
/// Connects to the Swift app's capture socket at
/// `~/Library/Application Support/agenthandover/capture.sock`,
/// sends a JSON capture command, and reads back a binary frame
/// (8-byte header: width u32 LE + height u32 LE, then width*height*4 BGRA bytes).
pub fn capture_main_display() -> Option<(usize, usize, Vec<u8>)> {
    use std::io::{Read, Write};

    // Connect to the Swift app's capture socket
    let home = std::env::var("HOME").ok()?;
    let socket_path = format!(
        "{}/Library/Application Support/agenthandover/capture.sock",
        home
    );

    let stream = std::os::unix::net::UnixStream::connect(&socket_path).ok()?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .ok()?;
    stream
        .set_write_timeout(Some(std::time::Duration::from_secs(10)))
        .ok()?;

    // Send capture request
    (&stream).write_all(b"{\"command\":\"capture\"}\n").ok()?;

    // Read response header (8 bytes: width u32 LE + height u32 LE)
    let mut header = [0u8; 8];
    (&stream).read_exact(&mut header).ok()?;

    // Check if it's a JSON error (starts with '{')
    if header[0] == b'{' {
        // Error response, read rest and return None
        return None;
    }

    let width = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let height = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    if width == 0 || height == 0 || width > 10000 || height > 10000 {
        return None;
    }

    // Read pixel data
    let pixel_count = width * height * 4;
    let mut pixels = vec![0u8; pixel_count];
    (&stream).read_exact(&mut pixels).ok()?;

    Some((width, height, pixels))
}

/// Check if the capture socket exists, indicating the Swift app is running
/// and ready to serve screenshots.
pub fn has_screen_recording_permission() -> bool {
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => return false,
    };
    let socket_path = format!(
        "{}/Library/Application Support/agenthandover/capture.sock",
        home
    );
    if !std::path::Path::new(&socket_path).exists() {
        return false;
    }

    crate::observation::snapshot()
        .map(|snapshot| snapshot.screen_recording_granted)
        .unwrap_or(false)
}
