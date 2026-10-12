//! Encrypted thumbnails of kept frames (v0.6 Memory).
//!
//! For each kept frame, while `[memory]` is on and not paused, the daemon
//! writes `<data dir>/thumbs/<event_id>.ahthumb`: a JPEG of the frame
//! (longest side [`MAX_SIDE`] px, quality [`QUALITY`]) sealed with
//! AES-256-GCM under the app's thumbnail key. The file is
//! `nonce (12 bytes) || ciphertext || tag (16 bytes)`, the layout CryptoKit's
//! `AES.GCM.SealedBox(combined:)` reads, so the app decrypts it for display.
//! The plaintext JPEG exists only in memory; the key is fetched from the
//! app over the capture socket and never written anywhere.

use aes_gcm::aead::{Aead, AeadCore, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use std::io::Write;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Longest side of a thumbnail, in pixels.
pub const MAX_SIDE: u32 = 480;
/// JPEG quality of a thumbnail.
pub const QUALITY: u8 = 60;
/// Key length (AES-256).
pub const KEY_LEN: usize = 32;
/// AES-GCM nonce length.
pub const NONCE_LEN: usize = 12;
/// AES-GCM tag length.
pub const TAG_LEN: usize = 16;
/// File extension of an encrypted thumbnail.
pub const EXTENSION: &str = "ahthumb";

/// The app's thumbnail key.
pub type ThumbnailKey = [u8; KEY_LEN];

#[derive(Debug, thiserror::Error)]
pub enum ThumbnailError {
    #[error("frame has no pixels")]
    Empty,
    #[error("JPEG encode: {0}")]
    Encode(String),
    #[error("sealed thumbnail is malformed or the key is wrong")]
    Open,
    #[error("thumbnail I/O: {0}")]
    Io(#[from] std::io::Error),
}

/// `<thumbs_dir>/<event_id>.ahthumb`.
pub fn thumbnail_path(thumbs_dir: &Path, event_id: Uuid) -> PathBuf {
    thumbs_dir.join(format!("{event_id}.{EXTENSION}"))
}

/// The size a `width` x `height` frame is scaled to: longest side
/// [`MAX_SIDE`], aspect kept, never enlarged, never zero.
pub fn thumbnail_size(width: u32, height: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= MAX_SIDE {
        return (width, height);
    }
    let scale = |side: u32| ((u64::from(side) * u64::from(MAX_SIDE) / u64::from(longest)) as u32).max(1);
    (scale(width), scale(height))
}

/// Scale tightly packed BGRA pixels (the observer's decoded frame) down to
/// a thumbnail and JPEG-encode it.
pub fn encode_thumbnail(bgra: &[u8], width: usize, height: usize) -> Result<Vec<u8>, ThumbnailError> {
    if width == 0 || height == 0 || bgra.len() < width * height * 4 {
        return Err(ThumbnailError::Empty);
    }
    let rgb: Vec<u8> = bgra[..width * height * 4]
        .chunks_exact(4)
        .flat_map(|px| [px[2], px[1], px[0]])
        .collect();
    let frame = image::RgbImage::from_raw(width as u32, height as u32, rgb).ok_or(ThumbnailError::Empty)?;
    let (w, h) = thumbnail_size(frame.width(), frame.height());
    let small = if (w, h) == frame.dimensions() {
        frame
    } else {
        image::imageops::resize(&frame, w, h, image::imageops::FilterType::Triangle)
    };
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, QUALITY)
        .encode_image(&small)
        .map_err(|e| ThumbnailError::Encode(e.to_string()))?;
    Ok(jpeg)
}

/// AES-256-GCM seal with a fresh random nonce: `nonce || ciphertext || tag`.
pub fn seal(key: &ThumbnailKey, plaintext: &[u8]) -> Vec<u8> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .expect("AES-GCM encryption of an in-memory buffer cannot fail");
    let mut sealed = Vec::with_capacity(NONCE_LEN + ciphertext.len());
    sealed.extend_from_slice(&nonce);
    sealed.extend_from_slice(&ciphertext);
    sealed
}

/// Open what [`seal`] produced. Fails on a wrong key or any altered byte.
pub fn open(key: &ThumbnailKey, sealed: &[u8]) -> Result<Vec<u8>, ThumbnailError> {
    if sealed.len() < NONCE_LEN + TAG_LEN {
        return Err(ThumbnailError::Open);
    }
    let (nonce, ciphertext) = sealed.split_at(NONCE_LEN);
    Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key))
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|_| ThumbnailError::Open)
}

/// Seal `jpeg` and write it as `<thumbs_dir>/<event_id>.ahthumb` (dir 0700,
/// file 0600, written to a temp name and renamed so a reader never sees a
/// partial file).
pub fn write_thumbnail(
    thumbs_dir: &Path,
    event_id: Uuid,
    key: &ThumbnailKey,
    jpeg: &[u8],
) -> Result<PathBuf, ThumbnailError> {
    use std::os::unix::fs::OpenOptionsExt;

    agenthandover_common::status::ensure_private_dir(thumbs_dir)?;
    let path = thumbnail_path(thumbs_dir, event_id);
    let tmp = thumbs_dir.join(format!(".{event_id}.{EXTENSION}.tmp"));
    let written = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(&seal(key, jpeg))?;
        file.sync_all()?;
        std::fs::rename(&tmp, &path)
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.into());
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: ThumbnailKey = [7u8; KEY_LEN];

    fn bgra_frame(width: usize, height: usize) -> Vec<u8> {
        let mut px = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                px.extend_from_slice(&[(x % 256) as u8, (y % 256) as u8, 180, 0xFF]);
            }
        }
        px
    }

    #[test]
    fn size_keeps_aspect_and_never_enlarges() {
        assert_eq!(thumbnail_size(1728, 1117), (480, 310));
        assert_eq!(thumbnail_size(1117, 1728), (310, 480));
        assert_eq!(thumbnail_size(320, 200), (320, 200));
        assert_eq!(thumbnail_size(10_000, 1), (480, 1));
    }

    #[test]
    fn encode_scales_the_frame_to_480_px() {
        let jpeg = encode_thumbnail(&bgra_frame(960, 600), 960, 600).unwrap();
        let decoded = image::load_from_memory_with_format(&jpeg, image::ImageFormat::Jpeg).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (480, 300));
        assert!(encode_thumbnail(&[], 0, 0).is_err());
    }

    #[test]
    fn seal_round_trips_and_uses_a_fresh_nonce() {
        let plaintext = b"\xFF\xD8 a jpeg".to_vec();
        let a = seal(&KEY, &plaintext);
        let b = seal(&KEY, &plaintext);
        assert_eq!(a.len(), NONCE_LEN + plaintext.len() + TAG_LEN);
        assert_ne!(a[..NONCE_LEN], b[..NONCE_LEN], "a new nonce per seal");
        assert_eq!(open(&KEY, &a).unwrap(), plaintext);
        assert_eq!(open(&KEY, &b).unwrap(), plaintext);
    }

    #[test]
    fn wrong_key_or_altered_bytes_fail() {
        let sealed = seal(&KEY, b"secret frame");
        assert!(open(&[8u8; KEY_LEN], &sealed).is_err(), "wrong key");
        let mut altered = sealed.clone();
        *altered.last_mut().unwrap() ^= 1;
        assert!(open(&KEY, &altered).is_err(), "altered tag");
        assert!(open(&KEY, &sealed[..NONCE_LEN + TAG_LEN - 1]).is_err(), "truncated");
    }

    #[test]
    fn written_file_holds_no_plaintext_and_opens_with_the_key() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let thumbs = dir.path().join("thumbs");
        let jpeg = encode_thumbnail(&bgra_frame(800, 500), 800, 500).unwrap();
        let id = Uuid::new_v4();
        let path = write_thumbnail(&thumbs, id, &KEY, &jpeg).unwrap();

        assert_eq!(path, thumbs.join(format!("{id}.ahthumb")));
        let on_disk = std::fs::read(&path).unwrap();
        assert!(!on_disk.starts_with(&[0xFF, 0xD8]), "not a JPEG on disk");
        for chunk in jpeg.chunks(32).filter(|c| c.len() == 32) {
            assert!(
                !on_disk.windows(32).any(|w| w == chunk),
                "no 32-byte run of the plaintext JPEG is on disk"
            );
        }
        assert_eq!(open(&KEY, &on_disk).unwrap(), jpeg);
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(std::fs::metadata(&thumbs).unwrap().permissions().mode() & 0o777, 0o700);
        // Only the thumbnail: no temp file left behind.
        assert_eq!(std::fs::read_dir(&thumbs).unwrap().count(), 1);
    }
}
