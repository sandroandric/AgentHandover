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
