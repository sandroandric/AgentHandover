//! macOS OCR via the Vision framework (VNRecognizeTextRequest).
//!
//! Recognizes text in raw BGRA pixel buffers captured from screenshots.
//! Uses the Objective-C runtime FFI to call into Apple's Vision framework,
//! following the same pattern as `macos_clipboard.rs`.

use serde::{Deserialize, Serialize};
use std::ffi::c_void;
use tracing::{debug, warn};

// Objective-C runtime FFI (libobjc.dylib ships with macOS).
#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const u8) -> *mut c_void;
    fn sel_registerName(name: *const u8) -> *mut c_void;
    fn objc_msgSend(obj: *mut c_void, sel: *mut c_void, ...) -> *mut c_void;
}

// Link Vision and CoreGraphics frameworks.
#[link(name = "Vision", kind = "framework")]
extern "C" {}

#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGBitmapContextCreate(
        data: *mut c_void,
        width: usize,
        height: usize,
        bits_per_component: usize,
        bytes_per_row: usize,
        space: *mut c_void,
        bitmap_info: u32,
    ) -> *mut c_void;
    fn CGBitmapContextCreateImage(context: *mut c_void) -> *mut c_void;
    fn CGColorSpaceCreateDeviceRGB() -> *mut c_void;
    fn CGColorSpaceRelease(space: *mut c_void);
    fn CGContextRelease(context: *mut c_void);
    fn CGImageRelease(image: *mut c_void);
}

/// ObjC BOOL on arm64 is C `bool` (1 byte). On x86_64 it is `signed char`.
/// Either way, 0 = NO, non-zero = YES. We transmute objc_msgSend to return
/// u8 and compare against 0.
type MsgSendBool = unsafe extern "C" fn(
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut *mut c_void,
) -> u8;

/// For methods returning `float` / `VNConfidence` (f32).
/// On arm64 the value is returned in s0; on x86_64 in xmm0.
type MsgSendF32 = unsafe extern "C" fn(*mut c_void, *mut c_void) -> f32;

/// CGRect as a repr(C) struct so the compiler uses the correct ABI for
/// returning it from objc_msgSend (d0-d3 on arm64 since CGRect is an HFA).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct CGRect {
    origin_x: f64,
    origin_y: f64,
    size_width: f64,
    size_height: f64,
}

/// For methods returning CGRect.
/// On arm64, CGRect is a Homogeneous Floating-point Aggregate (4 × f64),
/// returned in d0-d3. On x86_64, structs > 16 bytes use objc_msgSend_stret.
#[cfg(target_arch = "aarch64")]
type MsgSendCGRect = unsafe extern "C" fn(*mut c_void, *mut c_void) -> CGRect;

#[cfg(target_arch = "x86_64")]
extern "C" {
    fn objc_msgSend_stret(ret: *mut c_void, obj: *mut c_void, sel: *mut c_void, ...);
}

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

/// Helper: send `release` to an ObjC object. Null-safe.
unsafe fn objc_release(obj: *mut c_void) {
    if !obj.is_null() {
        let sel = sel_registerName(b"release\0".as_ptr());
        objc_msgSend(obj, sel);
    }
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
    // small images. Require at least 10×10 to avoid this.
    if width < 10 || height < 10 {
        debug!(width, height, "Image too small for OCR (minimum 10×10)");
        return None;
    }

    let start = std::time::Instant::now();

    // Expected: 4 bytes per pixel (BGRA)
    let expected_len = width * height * 4;
    if pixels.len() < expected_len {
        warn!(
            pixels_len = pixels.len(),
            expected = expected_len,
            "Pixel buffer too small for declared dimensions"
        );
        return None;
    }

    unsafe {
        // Wrap entire function body in an autorelease pool so that any
        // autoreleased objects created on this (background) thread are
        // properly drained.
        let pool_class = objc_getClass(b"NSAutoreleasePool\0".as_ptr());
        let pool = if !pool_class.is_null() {
            let sel_new = sel_registerName(b"new\0".as_ptr());
            objc_msgSend(pool_class, sel_new)
        } else {
            std::ptr::null_mut()
        };

        let result = recognize_text_inner(pixels, width, height, expected_len, start);

        // Drain the autorelease pool
        if !pool.is_null() {
            let sel_drain = sel_registerName(b"drain\0".as_ptr());
            objc_msgSend(pool, sel_drain);
        }

        result
    }
}

/// Inner OCR implementation, called within an NSAutoreleasePool.
unsafe fn recognize_text_inner(
    pixels: &[u8],
    width: usize,
    height: usize,
    expected_len: usize,
    start: std::time::Instant,
) -> Option<OcrResult> {
    // 1. Create CGImage from raw pixels via CGBitmapContext
    let color_space = CGColorSpaceCreateDeviceRGB();
    if color_space.is_null() {
        return None;
    }

    // kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little = 0x2002
    let bitmap_info: u32 = 0x2002;
    let bytes_per_row = width * 4;

    // We need a mutable copy since CGBitmapContextCreate takes *mut.
    // The copy must live until after CGBitmapContextCreateImage.
    let mut pixel_copy = pixels[..expected_len].to_vec();

    let context = CGBitmapContextCreate(
        pixel_copy.as_mut_ptr() as *mut c_void,
        width,
        height,
        8, // bits per component
        bytes_per_row,
        color_space,
        bitmap_info,
    );
    CGColorSpaceRelease(color_space);

    if context.is_null() {
        return None;
    }

    let cg_image = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    // pixel_copy can be dropped now; CGImage retains its own data

    if cg_image.is_null() {
        return None;
    }

    // 2. Create VNImageRequestHandler with the CGImage
    let handler_class = objc_getClass(b"VNImageRequestHandler\0".as_ptr());
    if handler_class.is_null() {
        CGImageRelease(cg_image);
        return None;
    }

    let ns_dict_class = objc_getClass(b"NSDictionary\0".as_ptr());
    if ns_dict_class.is_null() {
        CGImageRelease(cg_image);
        return None;
    }

    let sel_alloc = sel_registerName(b"alloc\0".as_ptr());
    let sel_dict = sel_registerName(b"dictionary\0".as_ptr());
    let empty_dict = objc_msgSend(ns_dict_class, sel_dict);

    // alloc + initWithCGImage:options: — treat as one ownership step
    let handler_raw = objc_msgSend(handler_class, sel_alloc);
    if handler_raw.is_null() {
        CGImageRelease(cg_image);
        return None;
    }

    let sel_init = sel_registerName(b"initWithCGImage:options:\0".as_ptr());
    let handler = objc_msgSend(handler_raw, sel_init, cg_image, empty_dict);
    CGImageRelease(cg_image);

    if handler.is_null() {
        // init failed — alloc'd object is already consumed
        return None;
    }

    // 3. Create VNRecognizeTextRequest
    let request_class = objc_getClass(b"VNRecognizeTextRequest\0".as_ptr());
    if request_class.is_null() {
        objc_release(handler);
        return None;
    }

    let request_raw = objc_msgSend(request_class, sel_alloc);
    if request_raw.is_null() {
        objc_release(handler);
        return None;
    }

    let sel_init_simple = sel_registerName(b"init\0".as_ptr());
    let request = objc_msgSend(request_raw, sel_init_simple);
    if request.is_null() {
        objc_release(handler);
        return None;
    }

    // setRecognitionLevel: 1 (accurate)
    let sel_set_level = sel_registerName(b"setRecognitionLevel:\0".as_ptr());
    objc_msgSend(request, sel_set_level, 1i64);

    // 4. Perform the request
    let sel_perform = sel_registerName(b"performRequests:error:\0".as_ptr());

    // Wrap request in NSArray (autoreleased — pool will drain it)
    let ns_array_class = objc_getClass(b"NSArray\0".as_ptr());
    if ns_array_class.is_null() {
        objc_release(request);
        objc_release(handler);
        return None;
    }

    let sel_array_with = sel_registerName(b"arrayWithObject:\0".as_ptr());
    let requests_array = objc_msgSend(ns_array_class, sel_array_with, request);
    if requests_array.is_null() {
        objc_release(request);
        objc_release(handler);
        return None;
    }

    let mut error_ptr: *mut c_void = std::ptr::null_mut();

    // performRequests:error: returns BOOL. Use a correctly-typed function
    // pointer so the compiler reads the return from the right register.
    let perform_fn: MsgSendBool = std::mem::transmute(objc_msgSend as *const c_void);
    let success = perform_fn(
        handler,
        sel_perform,
        requests_array,
        &mut error_ptr as *mut *mut c_void,
    );

    if success == 0 || !error_ptr.is_null() {
        debug!("Vision OCR request failed");
        objc_release(request);
        objc_release(handler);
        return None;
    }

    // 5. Extract results from VNRecognizeTextRequest
    let result = extract_ocr_results(request, start);

    // Release owned objects
    objc_release(request);
    objc_release(handler);

    result
}

/// Extract recognized text observations from a completed VNRecognizeTextRequest.
unsafe fn extract_ocr_results(
    request: *mut c_void,
    start: std::time::Instant,
) -> Option<OcrResult> {
    let sel_results = sel_registerName(b"results\0".as_ptr());
    let observations = objc_msgSend(request, sel_results);
    if observations.is_null() {
        return Some(OcrResult {
            elements: vec![],
            full_text: String::new(),
            processing_time_ms: start.elapsed().as_millis() as u64,
        });
    }

    let sel_count = sel_registerName(b"count\0".as_ptr());
    let count = objc_msgSend(observations, sel_count) as usize;

    let mut elements = Vec::with_capacity(count);
    let mut full_text_parts: Vec<String> = Vec::with_capacity(count);

    let sel_object_at = sel_registerName(b"objectAtIndex:\0".as_ptr());

    for i in 0..count {
        let observation = objc_msgSend(observations, sel_object_at, i as u64);
        if observation.is_null() {
            continue;
        }

        // Get top candidate text
        let sel_top = sel_registerName(b"topCandidates:\0".as_ptr());
        let candidates = objc_msgSend(observation, sel_top, 1i64);
        if candidates.is_null() {
            continue;
        }

        let cand_count = objc_msgSend(candidates, sel_count) as usize;
        if cand_count == 0 {
            continue;
        }

        let top_candidate = objc_msgSend(candidates, sel_object_at, 0u64);
        if top_candidate.is_null() {
            continue;
        }

        // Get the text string
        let sel_string = sel_registerName(b"string\0".as_ptr());
        let ns_string = objc_msgSend(top_candidate, sel_string);
        if ns_string.is_null() {
            continue;
        }

        let sel_utf8 = sel_registerName(b"UTF8String\0".as_ptr());
        let cstr = objc_msgSend(ns_string, sel_utf8) as *const u8;
        if cstr.is_null() {
            continue;
        }

        let text = match std::ffi::CStr::from_ptr(cstr as *const _).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => continue,
        };

        // Get confidence from the candidate — returns VNConfidence (f32)
        let sel_confidence = sel_registerName(b"confidence\0".as_ptr());
        let confidence_fn: MsgSendF32 =
            std::mem::transmute(objc_msgSend as *const c_void);
        let confidence = confidence_fn(top_candidate, sel_confidence).clamp(0.0, 1.0);

        // Get bounding box from the observation (normalized 0.0-1.0)
        let sel_bbox = sel_registerName(b"boundingBox\0".as_ptr());
        let bbox = get_bounding_box(observation, sel_bbox);

        // Vision framework uses bottom-left origin; convert to top-left
        let bbox_normalized = [
            bbox.origin_x,
            1.0 - bbox.origin_y - bbox.size_height,
            bbox.size_width,
            bbox.size_height,
        ];

        full_text_parts.push(text.clone());
        elements.push(OcrTextResult {
            text,
            confidence,
            bbox_normalized,
        });
    }

    let full_text = full_text_parts.join("\n");
    let processing_time_ms = start.elapsed().as_millis() as u64;

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

/// Get the boundingBox CGRect from a VNRecognizedTextObservation.
///
/// Uses the correct ABI for CGRect returns:
/// - arm64: CGRect is an HFA (4 × f64), returned in d0-d3 via objc_msgSend
/// - x86_64: CGRect (32 bytes) is returned via objc_msgSend_stret
#[cfg(target_arch = "aarch64")]
unsafe fn get_bounding_box(observation: *mut c_void, sel_bbox: *mut c_void) -> CGRect {
    let bbox_fn: MsgSendCGRect = std::mem::transmute(objc_msgSend as *const c_void);
    bbox_fn(observation, sel_bbox)
}

#[cfg(target_arch = "x86_64")]
unsafe fn get_bounding_box(observation: *mut c_void, sel_bbox: *mut c_void) -> CGRect {
    let mut bbox = CGRect::default();
    objc_msgSend_stret(
        &mut bbox as *mut CGRect as *mut c_void,
        observation,
        sel_bbox,
    );
    bbox
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
    fn test_empty_pixels_returns_none() {
        let result = recognize_text(&[], 0, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_small_image_returns_none() {
        // 1x1 BGRA pixel — below the 10×10 minimum for Vision framework
        let pixels = vec![0u8; 4];
        let result = recognize_text(&pixels, 1, 1);
        assert!(result.is_none(), "Images smaller than 10x10 should return None");
    }

    #[test]
    fn test_9x9_below_minimum() {
        // 9x9 BGRA — just below the 10×10 minimum
        let pixels = vec![0u8; 9 * 9 * 4];
        let result = recognize_text(&pixels, 9, 9);
        assert!(result.is_none(), "9x9 should be below minimum threshold");
    }

    #[test]
    fn test_undersized_buffer_returns_none() {
        // Claims 100x100 but only provides 4 bytes
        let pixels = vec![0u8; 4];
        let result = recognize_text(&pixels, 100, 100);
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_async_respects_timeout() {
        // Empty pixels should return quickly with None
        let result = recognize_text_async(vec![], 0, 0).await;
        assert!(result.is_none());
    }

    #[test]
    fn test_cgrect_default_is_zero() {
        let rect = CGRect::default();
        assert_eq!(rect.origin_x, 0.0);
        assert_eq!(rect.origin_y, 0.0);
        assert_eq!(rect.size_width, 0.0);
        assert_eq!(rect.size_height, 0.0);
    }
}
