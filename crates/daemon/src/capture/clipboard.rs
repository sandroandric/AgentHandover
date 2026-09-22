use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipboardMeta {
    pub content_types: Vec<String>,
    pub byte_size: u64,
    pub high_entropy: bool,
    pub content_hash: String,
}

/// SHA-256 hash of content, returned as hex string.
pub fn hash_content(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Build the text preview stored with a clipboard event.
///
/// `max_chars` is `None` unless the user turned on
/// `[privacy] enable_clipboard_preview` (off by default); it then carries
/// `clipboard_preview_max_chars`.  The preview exists only for plaintext
/// that is valid UTF-8 and not high-entropy: the entropy gate is a second
/// line of defence behind the opt-in, and the storage writer's redactor is
/// the third.  Truncation counts chars, so multi-byte UTF-8 never panics.
pub fn clipboard_preview(
    data: &[u8],
    is_plaintext: bool,
    high_entropy: bool,
    max_chars: Option<usize>,
) -> Option<String> {
    let max_chars = max_chars?;
    if !is_plaintext || high_entropy || max_chars == 0 {
        return None;
    }
    let text = std::str::from_utf8(data).ok()?.trim();
    let preview: String = text.chars().take(max_chars).collect();
    (!preview.is_empty()).then_some(preview)
}

/// Estimate if content is high-entropy (potential secret).
/// Uses Shannon entropy calculation.
pub fn is_high_entropy(data: &[u8]) -> bool {
    if data.len() < 16 {
        return false;
    }
    let entropy = shannon_entropy(data);
    entropy > 4.5
}

fn shannon_entropy(data: &[u8]) -> f64 {
    let mut freq = [0u64; 256];
    for &byte in data {
        freq[byte as usize] += 1;
    }
    let len = data.len() as f64;
    freq.iter()
        .filter(|&&f| f > 0)
        .map(|&f| {
            let p = f as f64 / len;
            -p * p.log2()
        })
        .sum()
}
