//! Convert raw BGRA screenshot pixels to a downscaled JPEG file.
//!
//! Used by the v2 VLM annotation pipeline: the daemon saves a plain (unencrypted)
//! half-resolution JPEG alongside the encrypted artifact so the Python worker can
//! read screenshots directly without needing the machine encryption key.
//!
//! Typical usage: 3024x1964 Retina screenshot → 1512x982 JPEG at quality 70 → ~270KB.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, ImageEncoder, Rgba};

/// Convert BGRA pixel data to a downscaled JPEG and write it to disk.
///
/// Returns the file size in bytes on success.
///
/// # Arguments
/// * `bgra_pixels` — Raw BGRA pixel data (4 bytes per pixel, row-major, no padding)
/// * `width` — Source image width in pixels
/// * `height` — Source image height in pixels
/// * `scale` — Scale factor (0.5 = half resolution)
/// * `quality` — JPEG quality (1-100, 70 recommended)
/// * `output_path` — Where to write the JPEG file
pub fn save_screenshot_jpeg(
    bgra_pixels: &[u8],
    width: u32,
    height: u32,
    scale: f64,
    quality: u8,
    output_path: &Path,
) -> Result<u64, JpegConvertError> {
    if width == 0 || height == 0 {
        return Err(JpegConvertError::InvalidPixelData {
            expected: 4, // At least one pixel needed
            actual: 0,
        });
    }

    let expected_len = (width as usize) * (height as usize) * 4;
    if bgra_pixels.len() < expected_len {
        return Err(JpegConvertError::InvalidPixelData {
            expected: expected_len,
            actual: bgra_pixels.len(),
        });
    }

    // Convert BGRA → RGBA (swap bytes 0 and 2 in each 4-byte pixel)
    let mut rgba = Vec::with_capacity(expected_len);
    for chunk in bgra_pixels[..expected_len].chunks_exact(4) {
        rgba.push(chunk[2]); // R (was at index 2 in BGRA)
        rgba.push(chunk[1]); // G
        rgba.push(chunk[0]); // B (was at index 0 in BGRA)
        rgba.push(chunk[3]); // A
    }

    let img = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, rgba)
        .ok_or(JpegConvertError::BufferCreationFailed)?;

    // Downscale
    let new_width = ((width as f64) * scale).max(1.0) as u32;
    let new_height = ((height as f64) * scale).max(1.0) as u32;
    let resized = image::imageops::resize(&img, new_width, new_height, FilterType::Triangle);

    // Convert RGBA → RGB (JPEG doesn't support alpha channel)
    let rgb_img = DynamicImage::ImageRgba8(resized).to_rgb8();

    // Encode as JPEG with specified quality
    let file = File::create(output_path).map_err(JpegConvertError::Io)?;
    let writer = BufWriter::new(file);
    let encoder = JpegEncoder::new_with_quality(writer, quality);
    encoder
        .write_image(
            rgb_img.as_raw(),
            rgb_img.width(),
            rgb_img.height(),
            image::ExtendedColorType::Rgb8,
        )
        .map_err(JpegConvertError::Encode)?;

    let metadata = std::fs::metadata(output_path).map_err(JpegConvertError::Io)?;
    Ok(metadata.len())
}

/// Errors that can occur during JPEG conversion.
#[derive(Debug)]
pub enum JpegConvertError {
    /// Pixel buffer is shorter than expected for the given dimensions.
    InvalidPixelData { expected: usize, actual: usize },
    /// Failed to create an image buffer from the pixel data.
    BufferCreationFailed,
    /// File I/O error.
    Io(std::io::Error),
    /// JPEG encoding error.
    Encode(image::ImageError),
}

impl std::fmt::Display for JpegConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPixelData { expected, actual } => {
                write!(
                    f,
                    "Pixel data too short: expected {} bytes, got {}",
                    expected, actual
                )
            }
            Self::BufferCreationFailed => {
                write!(f, "Failed to create image buffer from pixel data")
            }
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Encode(e) => write!(f, "JPEG encoding error: {}", e),
        }
    }
}

impl std::error::Error for JpegConvertError {}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a solid-color BGRA image for testing.
    fn make_solid_bgra(width: u32, height: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut pixels = vec![0u8; (width as usize) * (height as usize) * 4];
        for chunk in pixels.chunks_exact_mut(4) {
            chunk[0] = b; // B
            chunk[1] = g; // G
            chunk[2] = r; // R
            chunk[3] = 255; // A
        }
        pixels
    }

    /// Decode JPEG file dimensions using the JPEG decoder directly.
    fn jpeg_dimensions(path: &Path) -> (u32, u32) {
        use image::codecs::jpeg::JpegDecoder;
        use image::ImageDecoder;
        let file = File::open(path).unwrap();
        let reader = std::io::BufReader::new(file);
        let decoder = JpegDecoder::new(reader).unwrap();
        decoder.dimensions()
    }

    #[test]
    fn test_save_jpeg_basic() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.jpg");
        let pixels = make_solid_bgra(100, 100, 128, 64, 32);
        let size = save_screenshot_jpeg(&pixels, 100, 100, 0.5, 70, &path).unwrap();
        assert!(path.exists());
        assert!(size > 0);
    }

    #[test]
    fn test_save_jpeg_half_res_dimensions() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.jpg");
        let pixels = make_solid_bgra(200, 100, 0, 0, 0);
        save_screenshot_jpeg(&pixels, 200, 100, 0.5, 70, &path).unwrap();

        let (w, h) = jpeg_dimensions(&path);
        assert_eq!(w, 100);
        assert_eq!(h, 50);
    }

    #[test]
    fn test_save_jpeg_invalid_pixel_data() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.jpg");
        let too_short = vec![0u8; 10];
        let result = save_screenshot_jpeg(&too_short, 100, 100, 0.5, 70, &path);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_jpeg_scale_one_preserves_dimensions() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("full.jpg");
        let pixels = make_solid_bgra(100, 80, 0, 0, 0);
        save_screenshot_jpeg(&pixels, 100, 80, 1.0, 70, &path).unwrap();

        let (w, h) = jpeg_dimensions(&path);
        assert_eq!(w, 100);
        assert_eq!(h, 80);
    }

    #[test]
    fn test_bgra_to_rgba_channel_order() {
        // Create a single pixel: BGRA = [B=10, G=20, R=30, A=255]
        // After conversion, JPEG should contain RGB ≈ [R=30, G=20, B=10]
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("pixel.jpg");
        // Use a larger image to avoid JPEG compression artifacts on tiny images
        let pixels = make_solid_bgra(8, 8, 30, 20, 10);
        save_screenshot_jpeg(&pixels, 8, 8, 1.0, 100, &path).unwrap();

        // Read back RGB pixels
        use image::codecs::jpeg::JpegDecoder;
        use image::ImageDecoder;
        let file = File::open(&path).unwrap();
        let decoder = JpegDecoder::new(std::io::BufReader::new(file)).unwrap();
        let mut buf = vec![0u8; decoder.total_bytes() as usize];
        decoder.read_image(&mut buf).unwrap();

        // Check center pixel (avoids edge artifacts). RGB data at stride = 8*3.
        let center = 4 * 8 * 3 + 4 * 3; // row=4, col=4, 3 bytes per pixel
        let r = buf[center] as i32;
        let g = buf[center + 1] as i32;
        let b = buf[center + 2] as i32;
        // JPEG is lossy, allow tolerance
        assert!((r - 30).unsigned_abs() < 15, "R: expected ~30, got {}", r);
        assert!((g - 20).unsigned_abs() < 15, "G: expected ~20, got {}", g);
        assert!((b - 10).unsigned_abs() < 15, "B: expected ~10, got {}", b);
    }

    #[test]
    fn test_save_jpeg_large_image_realistic() {
        // Simulate a realistic Retina display capture (scaled down for test speed)
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("large.jpg");
        let pixels = make_solid_bgra(1024, 768, 100, 150, 200);
        let size = save_screenshot_jpeg(&pixels, 1024, 768, 0.5, 70, &path).unwrap();

        let (w, h) = jpeg_dimensions(&path);
        assert_eq!(w, 512);
        assert_eq!(h, 384);
        assert!(size > 0);
        assert!(size < 1024 * 768 * 3); // JPEG should be much smaller than raw
    }

    #[test]
    fn test_save_jpeg_zero_dimensions() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("zero.jpg");
        let result = save_screenshot_jpeg(&[], 0, 0, 0.5, 70, &path);
        assert!(result.is_err());
    }
}
