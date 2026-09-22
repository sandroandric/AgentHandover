//! Client for the Swift app's capture socket (`capture.sock`).
//!
//! The app captures the focused window (or the display under it) at native
//! pixel resolution, downscales by the requested `scale`, JPEG-encodes it and
//! sends the encoded bytes. The daemon stores those bytes as-is for the VLM
//! pipeline and decodes them once for dHash and OCR.
//!
//! Protocol:
//! - request: `{"command":"capture","scale":0.5,"format":"jpeg","quality":70}\n`
//! - success: `AHJ1` + width u32 LE + height u32 LE + byte length u32 LE + JPEG bytes
//! - error:   `{"error":"..."}\n`

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Magic prefix of a successful frame response.
pub const FRAME_MAGIC: &[u8; 4] = b"AHJ1";

/// Bound on one capture round trip. The app answers a hung capture with
/// `capture timeout` after 8 s, so this only fires if the app itself is wedged.
pub const CAPTURE_TIMEOUT: Duration = Duration::from_secs(10);

/// Largest frame dimension accepted from the socket.
const MAX_DIMENSION: u32 = 16_384;
/// Largest JPEG payload accepted from the socket.
const MAX_JPEG_BYTES: u32 = 64 * 1024 * 1024;

/// One captured frame: the JPEG bytes as sent by the app plus the decoded
/// BGRA pixels used for dHash and OCR.
pub struct CapturedFrame {
    pub width: usize,
    pub height: usize,
    pub jpeg: Vec<u8>,
    pub bgra: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("capture socket unavailable: {0}")]
    Unavailable(std::io::Error),
    #[error("capture socket I/O: {0}")]
    Io(std::io::Error),
    #[error("app reported: {0}")]
    App(String),
    #[error("capture protocol: {0}")]
    Protocol(String),
    #[error("JPEG decode: {0}")]
    Decode(String),
    #[error("capture did not complete within {0:?}")]
    Timeout(Duration),
}

impl CaptureError {
    /// Transient failures worth an immediate retry (the app restarting its
    /// socket). Timeouts are never retried: a retry would stall the tick again.
    pub fn is_retryable(&self) -> bool {
        matches!(self, CaptureError::Unavailable(_) | CaptureError::Io(_))
    }
}

pub fn capture_socket_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    Some(PathBuf::from(home).join("Library/Application Support/agenthandover/capture.sock"))
}

/// The request line sent to the app.
pub fn request_line(scale: f64, quality: u8) -> String {
    let mut line = serde_json::json!({
        "command": "capture",
        "scale": scale,
        "format": "jpeg",
        "quality": quality,
    })
    .to_string();
    line.push('\n');
    line
}

/// A socket read/write that hit its timeout is a capture timeout, not a
/// retryable I/O error.
fn io_error(error: std::io::Error, io_timeout: Duration) -> CaptureError {
    match error.kind() {
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => {
            CaptureError::Timeout(io_timeout)
        }
        _ => CaptureError::Io(error),
    }
}

/// Blocking capture over the socket at `path`. Each socket operation is
/// bounded by `io_timeout`.
pub fn capture_frame_blocking(
    path: &Path,
    scale: f64,
    quality: u8,
    io_timeout: Duration,
) -> Result<CapturedFrame, CaptureError> {
    let stream =
        std::os::unix::net::UnixStream::connect(path).map_err(CaptureError::Unavailable)?;
    stream
        .set_read_timeout(Some(io_timeout))
        .map_err(CaptureError::Io)?;
    stream
        .set_write_timeout(Some(io_timeout))
        .map_err(CaptureError::Io)?;

    (&stream)
        .write_all(request_line(scale, quality).as_bytes())
        .map_err(|e| io_error(e, io_timeout))?;

    let mut magic = [0u8; 4];
    (&stream).read_exact(&mut magic).map_err(|e| io_error(e, io_timeout))?;
    if magic[0] == b'{' {
        let mut rest = String::new();
        let _ = (&stream).take(64 * 1024).read_to_string(&mut rest);
        let text = format!("{}{}", String::from_utf8_lossy(&magic), rest);
        let message = serde_json::from_str::<serde_json::Value>(text.trim())
            .ok()
            .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string))
            .unwrap_or_else(|| text.trim().to_string());
        return Err(CaptureError::App(message));
    }
    if &magic != FRAME_MAGIC {
        return Err(CaptureError::Protocol(format!(
            "unexpected response prefix {magic:02x?}"
        )));
    }

    let mut header = [0u8; 12];
    (&stream).read_exact(&mut header).map_err(|e| io_error(e, io_timeout))?;
    let field = |i: usize| u32::from_le_bytes([header[i], header[i + 1], header[i + 2], header[i + 3]]);
    let (width, height, length) = (field(0), field(4), field(8));

    if width == 0 || height == 0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(CaptureError::Protocol(format!("bad frame size {width}x{height}")));
    }
    if length == 0 || length > MAX_JPEG_BYTES {
        return Err(CaptureError::Protocol(format!("bad JPEG length {length}")));
    }

    let mut jpeg = vec![0u8; length as usize];
    (&stream).read_exact(&mut jpeg).map_err(|e| io_error(e, io_timeout))?;

    let (decoded_w, decoded_h, bgra) = decode_jpeg_to_bgra(&jpeg)?;
    if decoded_w != width as usize || decoded_h != height as usize {
        return Err(CaptureError::Protocol(format!(
            "header says {width}x{height}, JPEG is {decoded_w}x{decoded_h}"
        )));
    }

    Ok(CapturedFrame {
        width: decoded_w,
        height: decoded_h,
        jpeg,
        bgra,
    })
}

/// Decode a JPEG into tightly packed BGRA pixels (the layout dHash and the
/// Vision OCR bridge expect).
pub fn decode_jpeg_to_bgra(jpeg: &[u8]) -> Result<(usize, usize, Vec<u8>), CaptureError> {
    if !jpeg.starts_with(&[0xFF, 0xD8]) {
        return Err(CaptureError::Decode("payload is not a JPEG (no SOI marker)".into()));
    }
    let image = image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg)
        .map_err(|e| CaptureError::Decode(e.to_string()))?;
    let rgb = image.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    let mut bgra = Vec::with_capacity(width * height * 4);
    for px in rgb.pixels() {
        bgra.extend_from_slice(&[px[2], px[1], px[0], 0xFF]);
    }
    Ok((width, height, bgra))
}

/// Capture one frame without blocking the async runtime: the socket read and
/// JPEG decode run on the blocking pool, and the whole call is bounded by
/// `CAPTURE_TIMEOUT` so a wedged app cannot stall the observer tick.
pub async fn capture_frame(scale: f64, quality: u8) -> Result<CapturedFrame, CaptureError> {
    let path = capture_socket_path().ok_or_else(|| {
        CaptureError::Unavailable(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "HOME is not set",
        ))
    })?;
    capture_frame_at(path, scale, quality, CAPTURE_TIMEOUT).await
}

/// `capture_frame` against an explicit socket path and bound.
pub async fn capture_frame_at(
    path: PathBuf,
    scale: f64,
    quality: u8,
    timeout: Duration,
) -> Result<CapturedFrame, CaptureError> {
    let task = tokio::task::spawn_blocking(move || {
        capture_frame_blocking(&path, scale, quality, timeout)
    });
    match tokio::time::timeout(timeout, task).await {
        Ok(Ok(result)) => result,
        Ok(Err(join_error)) => Err(CaptureError::Protocol(format!(
            "capture task failed: {join_error}"
        ))),
        Err(_) => Err(CaptureError::Timeout(timeout)),
    }
}
