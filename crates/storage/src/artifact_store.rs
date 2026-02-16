use anyhow::Result;
use chacha20poly1305::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    XChaCha20Poly1305, XNonce,
};
use chrono::Utc;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::RwLock;

const HEADER_MAGIC: &[u8; 4] = b"OCAA"; // OpenClaw Apprentice Artifact
const ARTIFACT_VERSION: u8 = 2;
const NONCE_SIZE: usize = 24; // XChaCha20 uses 24-byte nonces

// Algorithm identifiers for the binary header
const COMPRESSION_ZSTD: u8 = 1;
const ENCRYPTION_XCHACHA20POLY1305: u8 = 1;

pub struct ArtifactStore {
    base_path: PathBuf,
    key: [u8; 32],
    /// Cache of artifact_id -> file path to avoid repeated recursive directory scans.
    path_cache: RwLock<HashMap<String, PathBuf>>,
}

impl ArtifactStore {
    pub fn new(base_path: PathBuf, key: [u8; 32]) -> Self {
        Self {
            base_path,
            key,
            path_cache: RwLock::new(HashMap::new()),
        }
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
        // Write header (v2: magic + version + compression_algo + encryption_algo + nonce_len + nonce + original_size)
        file.write_all(HEADER_MAGIC)?;
        file.write_all(&[ARTIFACT_VERSION])?;
        file.write_all(&[COMPRESSION_ZSTD])?;
        file.write_all(&[ENCRYPTION_XCHACHA20POLY1305])?;
        file.write_all(&(NONCE_SIZE as u16).to_le_bytes())?;
        file.write_all(&nonce_bytes)?;
        file.write_all(&(data.len() as u64).to_le_bytes())?; // original size for verification
        // Write encrypted payload
        file.write_all(&encrypted)?;
        file.flush()?;
        file.sync_all()?;

        fs::rename(&tmp_path, &final_path)?;

        // Cache the path for fast retrieval
        if let Ok(mut cache) = self.path_cache.write() {
            cache.insert(artifact_id.clone(), final_path);
        }

        Ok(artifact_id)
    }

    /// Retrieve: read -> decrypt -> decompress (reverse of store)
    /// Handles both v1 (no algo bytes) and v2 (with algo bytes) headers.
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
        let version = raw[4];

        // v1 header: magic(4) + version(1) + nonce_len(2) + nonce + original_size(8)
        // v2 header: magic(4) + version(1) + compression_algo(1) + encryption_algo(1) + nonce_len(2) + nonce + original_size(8)
        let (nonce_offset, nonce_len) = if version >= 2 {
            let _compression_algo = raw[5];
            let _encryption_algo = raw[6];
            let nlen = u16::from_le_bytes([raw[7], raw[8]]) as usize;
            (9, nlen)
        } else {
            let nlen = u16::from_le_bytes([raw[5], raw[6]]) as usize;
            (7, nlen)
        };

        let header_size = nonce_offset + nonce_len + 8;
        if raw.len() < header_size {
            anyhow::bail!("Artifact file too small for declared nonce length");
        }

        let nonce_bytes = &raw[nonce_offset..nonce_offset + nonce_len];
        let _original_size = u64::from_le_bytes(
            raw[nonce_offset + nonce_len..nonce_offset + nonce_len + 8]
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
    /// Checks the in-memory cache first, then searches the date hierarchy,
    /// and falls back to today's date directory if not found.
    pub fn artifact_path(&self, artifact_id: &str) -> PathBuf {
        // Check cache first
        if let Ok(cache) = self.path_cache.read() {
            if let Some(path) = cache.get(artifact_id) {
                return path.clone();
            }
        }

        // Fall back to recursive search and cache the result
        match self.find_artifact(artifact_id) {
            Some(path) => {
                if let Ok(mut cache) = self.path_cache.write() {
                    cache.insert(artifact_id.to_string(), path.clone());
                }
                path
            }
            None => {
                let now = Utc::now();
                self.base_path
                    .join(now.format("%Y").to_string())
                    .join(now.format("%m").to_string())
                    .join(now.format("%d").to_string())
                    .join(format!("{}.bin", artifact_id))
            }
        }
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
