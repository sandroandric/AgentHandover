//! capture.sock client: protocol, JPEG payload, and timeouts, against a fake
//! app listener on a temp socket (no Screen Recording needed).
#![cfg(target_os = "macos")]

use agenthandover_daemon::capture::screenshot::{
    capture_frame_at, capture_frame_blocking, decode_jpeg_to_bgra, request_line,
    thumbnail_key_blocking, CaptureError, FRAME_MAGIC, THUMBNAIL_KEY_REQUEST,
};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = image::RgbImage::new(width, height);
    for (x, y, px) in rgb.enumerate_pixels_mut() {
        *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, 200]);
    }
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 70)
        .encode_image(&rgb)
        .unwrap();
    out
}

fn frame_response(width: u32, height: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = FRAME_MAGIC.to_vec();
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Serve one connection: record the request line, then write `response`.
fn fake_app(response: Vec<u8>) -> (tempfile::TempDir, PathBuf, std::thread::JoinHandle<String>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.sock");
    let listener = UnixListener::bind(&path).unwrap();
    let handle = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        reader.get_mut().write_all(&response).unwrap();
        line
    });
    (dir, path, handle)
}

#[test]
fn request_asks_for_jpeg_at_scale_and_quality() {
    let line = request_line(0.5, 70);
    assert!(line.ends_with('\n'));
    let json: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
    assert_eq!(json["command"], "capture");
    assert_eq!(json["format"], "jpeg");
    assert_eq!(json["scale"], 0.5);
    assert_eq!(json["quality"], 70);
}

#[test]
fn frame_is_jpeg_passed_through_and_decoded_once() {
    let payload = jpeg(64, 40);
    let (_dir, path, server) = fake_app(frame_response(64, 40, &payload));

    let frame = capture_frame_blocking(&path, 0.5, 70, Duration::from_secs(2)).unwrap();

    assert_eq!(server.join().unwrap(), request_line(0.5, 70));
    assert_eq!((frame.width, frame.height), (64, 40));
    assert_eq!(frame.jpeg, payload, "JPEG bytes are stored as sent, not re-encoded");
    assert!(frame.jpeg.starts_with(&[0xFF, 0xD8]));
    assert_eq!(frame.bgra.len(), 64 * 40 * 4);
}

#[test]
fn width_whose_low_byte_is_a_brace_is_not_mistaken_for_an_error() {
    // 379 = 0x17B; the old protocol treated a leading 0x7B ('{') as JSON.
    let payload = jpeg(379, 20);
    let (_dir, path, _server) = fake_app(frame_response(379, 20, &payload));
    let frame = capture_frame_blocking(&path, 0.5, 70, Duration::from_secs(2)).unwrap();
    assert_eq!(frame.width, 379);
}

#[test]
fn app_error_is_reported_with_its_message() {
    let (_dir, path, _server) = fake_app(b"{\"error\":\"capture timeout\"}\n".to_vec());
    match capture_frame_blocking(&path, 0.5, 70, Duration::from_secs(2)) {
        Err(CaptureError::App(message)) => assert_eq!(message, "capture timeout"),
        other => panic!("expected app error, got {:?}", other.err()),
    }
}

#[test]
fn raw_pixels_are_rejected_as_not_jpeg() {
    let bgra = vec![0u8; 8 * 8 * 4];
    let (_dir, path, _server) = fake_app(frame_response(8, 8, &bgra));
    assert!(matches!(
        capture_frame_blocking(&path, 0.5, 70, Duration::from_secs(2)),
        Err(CaptureError::Decode(_))
    ));
}

#[test]
fn header_that_disagrees_with_jpeg_is_rejected() {
    let payload = jpeg(32, 32);
    let (_dir, path, _server) = fake_app(frame_response(64, 64, &payload));
    assert!(matches!(
        capture_frame_blocking(&path, 0.5, 70, Duration::from_secs(2)),
        Err(CaptureError::Protocol(_))
    ));
}

#[test]
fn unknown_prefix_is_a_protocol_error() {
    let (_dir, path, _server) = fake_app(b"\x00\x05\x00\x00garbage".to_vec());
    assert!(matches!(
        capture_frame_blocking(&path, 0.5, 70, Duration::from_secs(2)),
        Err(CaptureError::Protocol(_))
    ));
}

#[test]
fn missing_socket_is_retryable_unavailable() {
    let dir = tempfile::tempdir().unwrap();
    let err = capture_frame_blocking(&dir.path().join("absent.sock"), 0.5, 70, Duration::from_secs(1))
        .err()
        .unwrap();
    assert!(matches!(err, CaptureError::Unavailable(_)));
    assert!(err.is_retryable());
}

#[test]
fn decode_rejects_non_jpeg() {
    assert!(decode_jpeg_to_bgra(b"not a jpeg").is_err());
}

#[test]
fn blocking_read_timeout_is_a_capture_timeout() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.sock");
    let listener = UnixListener::bind(&path).unwrap();
    std::thread::spawn(move || {
        let (_stream, _) = listener.accept().unwrap();
        std::thread::sleep(Duration::from_secs(3));
    });
    let started = Instant::now();
    let result = capture_frame_blocking(&path, 0.5, 70, Duration::from_millis(200));
    assert!(matches!(result, Err(CaptureError::Timeout(_))), "got {:?}", result.err());
    assert!(started.elapsed() < Duration::from_secs(1));
}

#[tokio::test(flavor = "current_thread")]
async fn wedged_app_times_out_without_blocking_the_runtime() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.sock");
    let listener = UnixListener::bind(&path).unwrap();
    // Accept and never answer.
    std::thread::spawn(move || {
        let (_stream, _) = listener.accept().unwrap();
        std::thread::sleep(Duration::from_secs(5));
    });

    // A ticker on the same single-threaded runtime keeps running while the
    // capture waits, proving the socket read is off the async executor.
    let ticks = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
    let counter = ticks.clone();
    let ticker = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(20)).await;
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });

    let started = Instant::now();
    let result = capture_frame_at(path, 0.5, 70, Duration::from_millis(500)).await;
    let elapsed = started.elapsed();
    ticker.abort();

    assert!(matches!(result, Err(CaptureError::Timeout(_))), "got {:?}", result.err());
    assert!(!CaptureError::Timeout(Duration::ZERO).is_retryable());
    assert!(elapsed < Duration::from_millis(900), "took {elapsed:?}");
    assert!(ticks.load(std::sync::atomic::Ordering::Relaxed) >= 10, "runtime was blocked");
}

#[test]
fn thumbnail_key_is_read_from_the_app() {
    let key: Vec<u8> = (0u8..32).collect();
    let mut reply = 32u32.to_le_bytes().to_vec();
    reply.extend_from_slice(&key);
    let (_dir, path, server) = fake_app(reply);

    let got = thumbnail_key_blocking(&path, Duration::from_secs(2)).unwrap();

    assert_eq!(server.join().unwrap(), THUMBNAIL_KEY_REQUEST);
    assert_eq!(got.to_vec(), key);
}

#[test]
fn thumbnail_key_refusal_or_wrong_length_is_an_error() {
    // An app that does not know the command (an older app) answers an error.
    let (_dir, path, _server) = fake_app(b"{\"error\":\"unknown command\"}\n".to_vec());
    match thumbnail_key_blocking(&path, Duration::from_secs(2)) {
        Err(CaptureError::App(message)) => assert_eq!(message, "unknown command"),
        other => panic!("expected an app error, got {:?}", other.map(|_| ())),
    }
    let mut short = 16u32.to_le_bytes().to_vec();
    short.extend_from_slice(&[1u8; 16]);
    let (_dir, path, _server) = fake_app(short);
    assert!(matches!(
        thumbnail_key_blocking(&path, Duration::from_secs(2)),
        Err(CaptureError::Protocol(_))
    ));
    // A peer the app refuses sees the connection closed without a key.
    let (_dir, path, _server) = fake_app(Vec::new());
    assert!(thumbnail_key_blocking(&path, Duration::from_secs(2)).is_err());
}
