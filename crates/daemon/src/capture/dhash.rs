//! Perceptual hash (dHash) for screenshot change detection.
//!
//! Computes an 8×8 = 64-bit difference hash from raw BGRA pixel data.
//! Two screenshots are considered "similar" when their hamming distance
//! (number of differing bits) is below a configurable threshold.
//!
//! dHash is orientation-preserving and fast (~<1ms in Rust).

/// Compute the 64-bit dHash of raw BGRA pixel data.
///
/// The algorithm:
/// 1. Downscale to 9×8 grayscale (using bilinear-ish sampling)
/// 2. For each row, compare adjacent pixels: bit=1 if left > right
/// 3. This produces 8×8 = 64 comparison bits
pub fn compute_dhash(pixels: &[u8], width: usize, height: usize) -> u64 {
    if width == 0 || height == 0 || pixels.len() < width * height * 4 {
        return 0;
    }

    // Downsample to 9×8 grayscale
    let mut gray_9x8 = [0u8; 9 * 8];

    for row in 0..8u32 {
        for col in 0..9u32 {
            // Map (col, row) in 9×8 space to source image coordinates
            let src_x = (col as f64 / 9.0 * width as f64) as usize;
            let src_y = (row as f64 / 8.0 * height as f64) as usize;

            // Clamp to valid range
            let src_x = src_x.min(width - 1);
            let src_y = src_y.min(height - 1);

            // Read BGRA pixel and convert to grayscale (BT.601 luma)
            let idx = (src_y * width + src_x) * 4;
            if idx + 2 < pixels.len() {
                let b = pixels[idx] as u32;
                let g = pixels[idx + 1] as u32;
                let r = pixels[idx + 2] as u32;
                // Y = 0.299*R + 0.587*G + 0.114*B (integer approximation)
                let gray = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
                gray_9x8[(row as usize) * 9 + (col as usize)] = gray;
            }
        }
    }

    // Compute dHash: compare adjacent pixels in each row
    let mut hash: u64 = 0;
    let mut bit = 0u32;

    for row in 0..8usize {
        for col in 0..8usize {
            let left = gray_9x8[row * 9 + col] as u32;
            let right = gray_9x8[row * 9 + col + 1] as u32;
            if left > right {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }

    hash
}

/// Compute the hamming distance between two dHash values.
///
/// Returns the number of differing bits (0 = identical, 64 = completely different).
#[inline]
pub fn hamming_distance(hash_a: u64, hash_b: u64) -> u32 {
    (hash_a ^ hash_b).count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a solid-color BGRA image.
    fn solid_image(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut pixels = vec![0u8; width * height * 4];
        for i in (0..pixels.len()).step_by(4) {
            pixels[i] = b;     // B
            pixels[i + 1] = g; // G
            pixels[i + 2] = r; // R
            pixels[i + 3] = 255; // A
        }
        pixels
    }

    /// Create a simple gradient image (varies horizontally).
    fn gradient_image(width: usize, height: usize) -> Vec<u8> {
        let mut pixels = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let val = ((x as f64 / width as f64) * 255.0) as u8;
                let idx = (y * width + x) * 4;
                pixels[idx] = val;     // B
                pixels[idx + 1] = val; // G
                pixels[idx + 2] = val; // R
                pixels[idx + 3] = 255; // A
            }
        }
        pixels
    }

    #[test]
    fn test_identical_images_zero_distance() {
        let img = solid_image(100, 100, 128, 128, 128);
        let h1 = compute_dhash(&img, 100, 100);
        let h2 = compute_dhash(&img, 100, 100);
        assert_eq!(hamming_distance(h1, h2), 0);
    }

    #[test]
    fn test_same_color_images_zero_distance() {
        let img1 = solid_image(100, 100, 50, 50, 50);
        let img2 = solid_image(200, 200, 50, 50, 50);
        let h1 = compute_dhash(&img1, 100, 100);
        let h2 = compute_dhash(&img2, 200, 200);
        assert_eq!(hamming_distance(h1, h2), 0);
    }

    /// Create a right-to-left gradient (dark on right, bright on left).
    /// dHash: left > right for all pairs → all bits set → hash = u64::MAX.
    fn reverse_gradient_image(width: usize, height: usize) -> Vec<u8> {
        let mut pixels = vec![0u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let val = (((width - 1 - x) as f64 / width as f64) * 255.0) as u8;
                let idx = (y * width + x) * 4;
                pixels[idx] = val;     // B
                pixels[idx + 1] = val; // G
                pixels[idx + 2] = val; // R
                pixels[idx + 3] = 255; // A
            }
        }
        pixels
    }

    #[test]
    fn test_different_images_nonzero_distance() {
        // Left-to-right gradient: left < right for all pairs → hash ≈ 0
        let img1 = gradient_image(100, 100);
        // Right-to-left gradient: left > right for all pairs → hash ≈ u64::MAX
        let img2 = reverse_gradient_image(100, 100);
        let h1 = compute_dhash(&img1, 100, 100);
        let h2 = compute_dhash(&img2, 100, 100);
        assert!(hamming_distance(h1, h2) > 0, "Opposite gradients should produce different hashes");
    }

    #[test]
    fn test_gradient_consistency() {
        let img = gradient_image(200, 200);
        let h1 = compute_dhash(&img, 200, 200);
        let h2 = compute_dhash(&img, 200, 200);
        assert_eq!(hamming_distance(h1, h2), 0);
    }

    #[test]
    fn test_empty_image() {
        assert_eq!(compute_dhash(&[], 0, 0), 0);
    }

    #[test]
    fn test_threshold_comparison() {
        let img1 = gradient_image(100, 100);
        let img2 = gradient_image(100, 100);
        let h1 = compute_dhash(&img1, 100, 100);
        let h2 = compute_dhash(&img2, 100, 100);
        // Same image → distance should be below any reasonable threshold
        assert!(hamming_distance(h1, h2) < 10);
    }

    #[test]
    fn test_hamming_edge_cases() {
        assert_eq!(hamming_distance(0, 0), 0);
        assert_eq!(hamming_distance(u64::MAX, 0), 64);
        assert_eq!(hamming_distance(0, u64::MAX), 64);
        assert_eq!(hamming_distance(u64::MAX, u64::MAX), 0);
    }
}
