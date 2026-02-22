//! macOS OCR via the Vision framework (VNRecognizeTextRequest).
//!
//! The actual Vision framework calls live in `objc_try_catch.m` (compiled via
//! build.rs) so that Objective-C exceptions are caught in ObjC-land and never
//! propagate through Rust stack frames (which would abort the process).

use serde::{Deserialize, Serialize};
use std::ffi::CStr;
use tracing::{debug, warn};

// ── C FFI types matching objc_try_catch.m ──────────────────────────────

#[repr(C)]
struct OcrElement {
    text: *mut i8,     // heap-allocated UTF-8, we must free
    confidence: f32,
    bbox_x: f64,
    bbox_y: f64,
    bbox_w: f64,
    bbox_h: f64,
}

#[repr(C)]
struct OcrCResult {
    elements: *mut OcrElement,
    count: i32,
    full_text: *mut i8,
    success: i32,
}

extern "C" {
    fn perform_ocr_safe(pixels: *const u8, width: usize, height: usize) -> OcrCResult;
    fn free_ocr_result(result: *mut OcrCResult);
}

// ── Public Rust types ──────────────────────────────────────────────────

/// A single recognized text element from OCR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrTextResult {
    pub text: String,
    pub confidence: f32,
    /// Normalized bounding box [x, y, width, height] in 0.0-1.0 range.
    pub bbox_normalized: [f64; 4],
}

/// Full OCR result from a single image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    pub elements: Vec<OcrTextResult>,
    pub full_text: String,
    pub processing_time_ms: u64,
}

/// Recognize text in raw BGRA pixel data using Apple Vision framework.
///
/// Returns `Some(OcrResult)` on success, `None` on failure.
/// The pixel data must be in BGRA format (as returned by `CGDisplay::image()`).
pub fn recognize_text(pixels: &[u8], width: usize, height: usize) -> Option<OcrResult> {
    if pixels.is_empty() || width == 0 || height == 0 {
        return None;
    }

    // Vision framework may throw an uncatchable ObjC exception on very
    // small images. Require at least 10x10 to avoid this.
    if width < 10 || height < 10 {
        debug!(width, height, "Image too small for OCR (minimum 10x10)");
        return None;
    }

    let expected_len = width * height * 4;
    if pixels.len() < expected_len {
        warn!(
            pixels_len = pixels.len(),
            expected = expected_len,
            "Pixel buffer too small for declared dimensions"
        );
        return None;
    }

    let start = std::time::Instant::now();

    // Call into the ObjC helper. All Vision framework work + exception
    // handling happens in ObjC, so no foreign exceptions reach Rust.
    let mut c_result = unsafe { perform_ocr_safe(pixels.as_ptr(), width, height) };

    if c_result.success == 0 {
        unsafe { free_ocr_result(&mut c_result) };
        return None;
    }

    // Convert C result to Rust types
    let mut elements = Vec::with_capacity(c_result.count as usize);
    for i in 0..c_result.count {
        let elem = unsafe { &*c_result.elements.add(i as usize) };
        if elem.text.is_null() {
            continue;
        }
        let text = unsafe { CStr::from_ptr(elem.text) }
            .to_str()
            .unwrap_or("")
            .to_string();
        elements.push(OcrTextResult {
            text,
            confidence: elem.confidence.clamp(0.0, 1.0),
            bbox_normalized: [elem.bbox_x, elem.bbox_y, elem.bbox_w, elem.bbox_h],
        });
    }

    let full_text = if c_result.full_text.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(c_result.full_text) }
            .to_str()
            .unwrap_or("")
            .to_string()
    };

    let processing_time_ms = start.elapsed().as_millis() as u64;

    // Free the C-allocated memory
    unsafe { free_ocr_result(&mut c_result) };

    debug!(
        elements = elements.len(),
        time_ms = processing_time_ms,
        "OCR completed"
    );

    Some(OcrResult {
        elements,
        full_text,
        processing_time_ms,
    })
}

/// Async wrapper that runs OCR in a blocking thread with a 500ms timeout.
pub async fn recognize_text_async(
    pixels: Vec<u8>,
    width: usize,
    height: usize,
) -> Option<OcrResult> {
    let task = tokio::task::spawn_blocking(move || recognize_text(&pixels, width, height));

    match tokio::time::timeout(std::time::Duration::from_millis(500), task).await {
        Ok(Ok(result)) => result,
        Ok(Err(e)) => {
            warn!(error = %e, "OCR task panicked");
            None
        }
        Err(_) => {
            warn!("OCR timed out after 500ms");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_ocr_result_serde_roundtrip() {
        let result = OcrResult {
            elements: vec![OcrTextResult {
                text: "Hello World".to_string(),
                confidence: 0.95,
                bbox_normalized: [0.1, 0.2, 0.5, 0.1],
            }],
            full_text: "Hello World".to_string(),
            processing_time_ms: 42,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: OcrResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.elements.len(), 1);
        assert_eq!(deserialized.full_text, "Hello World");
        assert_eq!(deserialized.processing_time_ms, 42);
    }

    #[test]
    #[serial(macos_ffi)]
    fn test_empty_pixels_returns_none() {
        let result = recognize_text(&[], 0, 0);
        assert!(result.is_none());
    }

    #[test]
    #[serial(macos_ffi)]
    fn test_small_image_returns_none() {
        // 1x1 BGRA pixel — below the 10x10 minimum for Vision framework
        let pixels = vec![0u8; 4];
        let result = recognize_text(&pixels, 1, 1);
        assert!(result.is_none(), "Images smaller than 10x10 should return None");
    }

    #[test]
    #[serial(macos_ffi)]
    fn test_9x9_below_minimum() {
        // 9x9 BGRA — just below the 10x10 minimum
        let pixels = vec![0u8; 9 * 9 * 4];
        let result = recognize_text(&pixels, 9, 9);
        assert!(result.is_none(), "9x9 should be below minimum threshold");
    }

    #[test]
    #[serial(macos_ffi)]
    fn test_undersized_buffer_returns_none() {
        // Claims 100x100 but only provides 4 bytes
        let pixels = vec![0u8; 4];
        let result = recognize_text(&pixels, 100, 100);
        assert!(result.is_none());
    }

    #[tokio::test]
    #[serial(macos_ffi)]
    async fn test_async_respects_timeout() {
        // Empty pixels should return quickly with None
        let result = recognize_text_async(vec![], 0, 0).await;
        assert!(result.is_none());
    }
}
