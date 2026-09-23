//! Frame change detection by block averaging (the observer's "dHash" gate).
//!
//! A frame is reduced to a 16×12 grid of blocks, each holding the mean luma
//! of **every** pixel in it, so a small edit anywhere in the frame moves the
//! mean of the block it falls in. (This module used to sample one pixel per
//! cell of a 9×8 grid: it saw 72 pixels of a million and missed text edits,
//! dropdowns and most dialogs.)
//!
//! Two signatures are compared block by block: the distance is the number of
//! blocks whose mean moved by more than [`DEAD_ZONE`] gray levels. Comparing
//! the means directly, rather than the sign of neighbouring differences as a
//! classic dHash does, is what makes a small edit reliably visible: a sign
//! bit flips only when a neighbour comparison happens to cross zero, so a
//! line of text typed next to a darker sidebar can leave every bit as it was.
//!
//! Two frames are "the same screen" when their distance is below the
//! configured threshold (`[observer] dhash_threshold`, default 1: one changed
//! block is a change).

/// Grid columns.
const GRID_COLS: usize = 16;
/// Grid rows.
const GRID_ROWS: usize = 12;
/// Number of blocks in a signature.
pub const BLOCKS: usize = GRID_COLS * GRID_ROWS;

/// Minimum change of a block's mean luma (0-255) that counts as a change.
///
/// On a 1440×900 frame a block is 90×75 pixels. The mouse pointer (about
/// 8×12 pixels at half scale) moves a block mean by about 2 levels and a
/// blinking caret by less than 1; a 200×20 line of text typed into an empty
/// field moves it by about 12, or by about 6 in each of two blocks when the
/// line straddles a block boundary.
pub const DEAD_ZONE: u8 = 4;

/// Mean luma of each block of the grid, row-major.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSignature(pub [u8; BLOCKS]);

/// Compute the signature of tightly packed BGRA pixel data.
///
/// Returns `None` for an empty or undersized buffer.
pub fn compute_signature(pixels: &[u8], width: usize, height: usize) -> Option<FrameSignature> {
    if width == 0 || height == 0 || pixels.len() < width * height * 4 {
        return None;
    }

    // Every pixel belongs to exactly one block: pixel column `x` falls in
    // block column `x * GRID_COLS / width`, so blocks differ in width by at
    // most one pixel. Frames smaller than the grid leave some blocks empty
    // (read as 0), identically in both frames being compared.
    let col_of: Vec<usize> = (0..width).map(|x| x * GRID_COLS / width).collect();
    let mut sums = [0u64; BLOCKS];
    let mut counts = [0u64; BLOCKS];

    for y in 0..height {
        let block_row = y * GRID_ROWS / height;
        let row = &pixels[y * width * 4..(y + 1) * width * 4];
        for (x, px) in row.chunks_exact(4).enumerate() {
            let (b, g, r) = (px[0] as u64, px[1] as u64, px[2] as u64);
            let block = block_row * GRID_COLS + col_of[x];
            // BT.601 luma, integer approximation.
            sums[block] += (r * 77 + g * 150 + b * 29) >> 8;
            counts[block] += 1;
        }
    }

    let mut means = [0u8; BLOCKS];
    for i in 0..BLOCKS {
        if counts[i] > 0 {
            means[i] = (sums[i] / counts[i]) as u8;
        }
    }
    Some(FrameSignature(means))
}

/// Number of blocks whose mean moved by more than [`DEAD_ZONE`]
/// (0 = the same screen, [`BLOCKS`] = everything changed).
pub fn distance(a: &FrameSignature, b: &FrameSignature) -> u32 {
    a.0.iter()
        .zip(b.0.iter())
        .filter(|(x, y)| x.abs_diff(**y) > DEAD_ZONE)
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame the size of a half-scale 2880×1800 display capture.
    const W: usize = 1440;
    const H: usize = 900;

    fn solid_image(width: usize, height: usize, gray: u8) -> Vec<u8> {
        let mut pixels = vec![0u8; width * height * 4];
        for px in pixels.chunks_exact_mut(4) {
            px.copy_from_slice(&[gray, gray, gray, 255]);
        }
        pixels
    }

    fn fill_rect(pixels: &mut [u8], width: usize, x0: usize, y0: usize, w: usize, h: usize, gray: u8) {
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                let i = (y * width + x) * 4;
                pixels[i..i + 4].copy_from_slice(&[gray, gray, gray, 255]);
            }
        }
    }

    /// Draw glyph-like strokes into a `w`×`h` box: 2-pixel vertical strokes
    /// every 6 pixels over the middle 12 rows (about 20% ink, like a line of
    /// 13-px text).
    fn draw_text_line(pixels: &mut [u8], width: usize, x0: usize, y0: usize, w: usize, h: usize) {
        for y in y0 + 4..y0 + h - 4 {
            for x in x0..x0 + w {
                if (x - x0) % 6 < 2 {
                    let i = (y * width + x) * 4;
                    pixels[i..i + 4].copy_from_slice(&[25, 25, 25, 255]);
                }
            }
        }
    }

    /// A plausible application window: white page, gray sidebar, dark
    /// toolbar, a few paragraphs of text.
    fn document_window() -> Vec<u8> {
        let mut img = solid_image(W, H, 250);
        fill_rect(&mut img, W, 0, 0, W, 60, 60);
        fill_rect(&mut img, W, 0, 60, 260, H - 60, 225);
        for line in 0..12 {
            draw_text_line(&mut img, W, 320, 100 + line * 28, 900, 20);
        }
        img
    }

    fn sig(pixels: &[u8]) -> FrameSignature {
        compute_signature(pixels, W, H).expect("valid frame")
    }

    #[test]
    fn identical_frames_have_zero_distance() {
        let img = document_window();
        assert_eq!(distance(&sig(&img), &sig(&img)), 0);
    }

    #[test]
    fn same_color_frames_of_different_size_match() {
        let a = compute_signature(&solid_image(100, 100, 50), 100, 100).unwrap();
        let b = compute_signature(&solid_image(200, 200, 50), 200, 200).unwrap();
        assert_eq!(distance(&a, &b), 0);
    }

    #[test]
    fn empty_or_short_buffer_has_no_signature() {
        assert!(compute_signature(&[], 0, 0).is_none());
        assert!(compute_signature(&[0u8; 16], 100, 100).is_none());
    }

    #[test]
    fn frames_smaller_than_the_grid_do_not_panic() {
        let img = solid_image(3, 2, 200);
        assert!(compute_signature(&img, 3, 2).is_some());
    }

    #[test]
    fn a_full_screen_change_changes_every_block() {
        let a = sig(&solid_image(W, H, 20));
        let b = sig(&solid_image(W, H, 230));
        assert_eq!(distance(&a, &b), BLOCKS as u32);
    }

    /// Typing one line of text (200×20 frame pixels) into an empty area of
    /// an otherwise unchanged window is a change at the default threshold,
    /// wherever it sits: next to the darker sidebar, straddling a block
    /// boundary, in a corner.
    #[test]
    fn a_200x20_text_edit_is_a_change() {
        let before = document_window();
        let s_before = sig(&before);
        // y = 440 puts the 12 inked rows (444..456) evenly across the
        // block-row boundary at 450.
        for &(x, y) in &[(320usize, 520usize), (700, 440), (1200, 800), (10, 820), (1230, 870), (0, 64)] {
            let mut after = before.clone();
            draw_text_line(&mut after, W, x, y, 200, 20);
            let d = distance(&s_before, &sig(&after));
            assert!(d >= 1, "200x20 text edit at ({x},{y}) was invisible to the gate");
        }
    }

    /// Opening a dropdown (a 220×300 menu with items over the page) changes
    /// several blocks.
    #[test]
    fn a_dropdown_is_a_change() {
        let before = document_window();
        let mut after = before.clone();
        fill_rect(&mut after, W, 500, 300, 220, 300, 255);
        fill_rect(&mut after, W, 500, 300, 220, 2, 120);
        fill_rect(&mut after, W, 500, 598, 220, 2, 120);
        for item in 0..9 {
            draw_text_line(&mut after, W, 512, 310 + item * 32, 150, 20);
        }
        // Highlighted item.
        fill_rect(&mut after, W, 502, 342, 216, 30, 70);
        let d = distance(&sig(&before), &sig(&after));
        assert!(d >= 4, "dropdown changed only {d} blocks");
    }

    /// The mouse pointer or a blinking caret is not a change.
    #[test]
    fn pointer_and_caret_are_not_a_change() {
        let before = document_window();
        let mut after = before.clone();
        fill_rect(&mut after, W, 800, 452, 8, 12, 0);
        fill_rect(&mut after, W, 1100, 700, 1, 16, 0);
        let d = distance(&sig(&before), &sig(&after));
        assert_eq!(d, 0);
    }
}
