use anyhow::Result;
use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    XChaCha20Poly1305, XNonce,
};
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const HEADER_MAGIC: &[u8; 4] = b"OCAA"; // OpenClaw Apprentice Artifact
const ARTIFACT_VERSION: u8 = 1;
const NONCE_SIZE: usize = 24; // XChaCha20 uses 24-byte nonces

pub struct ArtifactStore {
    base_path: PathBuf,
    key: [u8; 32],
}

impl ArtifactStore {
    pub fn new(base_path: PathBuf, key: [u8; 32]) -> Self {
        Self { base_path, key }
    }

    /// Store: capture -> compress -> encrypt -> write (spec order from §6.2)
    pub fn store(&self, data: &[u8], artifact_type: &str) -> Result<String> {
        // 1. Compress with zstd (level 3 — balanced speed/ratio)
        let compressed = zstd::encode_all(data, 3)?;

        // 2. Encrypt with XChaCha20-Poly1305
        let cipher = XChaCha20Poly1305::new((&self.key).into());
        let nonce = XChaCha20Poly1305::generate_nonce(&mut OsRng);
        let encrypted = cipher
            .encrypt(&nonce, compressed.as_ref())
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

        // 3. Generate artifact ID from content hash + timestamp
        let mut hasher = Sha256::new();
        hasher.update(data);
        hasher.update(&Utc::now().timestamp_nanos_opt().unwrap_or(0).to_le_bytes());
        let hash = hex::encode(&hasher.finalize()[..8]);
        let artifact_id = format!("{}_{}", artifact_type, hash);

        // 4. Build date-based path: base/yyyy/mm/dd/
        let now = Utc::now();
        let dir = self
            .base_path
            .join(now.format("%Y").to_string())
            .join(now.format("%m").to_string())
            .join(now.format("%d").to_string());
        fs::create_dir_all(&dir)?;

        // 5. Write atomically: tmp file -> fsync -> rename
        let final_path = dir.join(format!("{}.bin", artifact_id));
        let tmp_path = dir.join(format!("{}.bin.tmp", artifact_id));

        let nonce_bytes: [u8; NONCE_SIZE] = nonce.into();

        let mut file = fs::File::create(&tmp_path)?;
        // Write header
        file.write_all(HEADER_MAGIC)?;
        file.write_all(&[ARTIFACT_VERSION])?;
        file.write_all(&(NONCE_SIZE as u16).to_le_bytes())?;
        file.write_all(&nonce_bytes)?;
        file.write_all(&(data.len() as u64).to_le_bytes())?; // original size for verification
        // Write encrypted payload
        file.write_all(&encrypted)?;
        file.flush()?;
        file.sync_all()?;

        fs::rename(&tmp_path, &final_path)?;

        Ok(artifact_id)
    }

    /// Retrieve: read -> decrypt -> decompress (reverse of store)
    pub fn retrieve(&self, artifact_id: &str) -> Result<Vec<u8>> {
        let path = self.artifact_path(artifact_id);
        let raw = fs::read(&path)?;

        // Parse header
        if raw.len() < 15 {
            anyhow::bail!("Artifact file too small to contain header");
        }
        if &raw[0..4] != HEADER_MAGIC {
            anyhow::bail!("Invalid artifact magic bytes");
        }
        let _version = raw[4];
        let nonce_len = u16::from_le_bytes([raw[5], raw[6]]) as usize;

        let header_size = 7 + nonce_len + 8; // magic(4) + version(1) + nonce_len_field(2) + nonce + original_size(8)
        if raw.len() < header_size {
            anyhow::bail!("Artifact file too small for declared nonce length");
        }

        let nonce_bytes = &raw[7..7 + nonce_len];
        let _original_size = u64::from_le_bytes(
            raw[7 + nonce_len..15 + nonce_len]
                .try_into()
                .map_err(|_| anyhow::anyhow!("Failed to parse original size"))?,
        );
        let encrypted = &raw[header_size..];

        // Decrypt
        let cipher = XChaCha20Poly1305::new((&self.key).into());
        let nonce = XNonce::from_slice(nonce_bytes);
        let compressed = cipher
            .decrypt(nonce, encrypted)
            .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))?;

        // Decompress
        let data = zstd::decode_all(compressed.as_slice())?;
        Ok(data)
    }

    /// Get the filesystem path for a given artifact ID.
    /// Searches the date hierarchy; falls back to today's date directory if not found.
    pub fn artifact_path(&self, artifact_id: &str) -> PathBuf {
        self.find_artifact(artifact_id).unwrap_or_else(|| {
            let now = Utc::now();
            self.base_path
                .join(now.format("%Y").to_string())
                .join(now.format("%m").to_string())
                .join(now.format("%d").to_string())
                .join(format!("{}.bin", artifact_id))
        })
    }

    fn find_artifact(&self, artifact_id: &str) -> Option<PathBuf> {
        let filename = format!("{}.bin", artifact_id);
        find_file_recursive(&self.base_path, &filename)
    }
}

fn find_file_recursive(dir: &Path, filename: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file_recursive(&path, filename) {
                return Some(found);
            }
        } else if path.file_name().map(|n| n == filename).unwrap_or(false) {
            return Some(path);
        }
    }
    None
}
