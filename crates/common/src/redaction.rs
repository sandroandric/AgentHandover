use regex::Regex;

pub struct Redactor {
    patterns: Vec<(Regex, &'static str)>,
    high_entropy_pattern: Regex,
}

impl Redactor {
    pub fn new() -> Self {
        let patterns = vec![
            // AWS Access Key ID (starts with AKIA)
            (Regex::new(r"(?i)(AKIA[0-9A-Z]{16})").unwrap(), "[REDACTED_AWS_KEY]"),
            // AWS Secret Access Key (40 char base64-ish after = or :)
            (Regex::new(r"(?i)(?:aws_secret_access_key|secret_key)\s*[=:]\s*([A-Za-z0-9/+=]{30,})").unwrap(), "[REDACTED_SECRET]"),
            // Generic API keys/tokens (long alphanumeric after common key words)
            (Regex::new(r"(?i)(?:api[_-]?key|api[_-]?token|auth[_-]?token|bearer)\s*[=:]\s*['\x22]?([A-Za-z0-9_\-]{20,})['\x22]?").unwrap(), "[REDACTED_API_KEY]"),
            // Credit card numbers (Visa, MC, Amex, Discover with optional dashes/spaces)
            (Regex::new(r"\b([3-6]\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3,4})\b").unwrap(), "[REDACTED_CC]"),
            // SSN
            (Regex::new(r"\b(\d{3}-\d{2}-\d{4})\b").unwrap(), "[REDACTED_SSN]"),
            // Private keys (PEM format)
            (Regex::new(r"(?s)(-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+)?PRIVATE KEY-----.*?-----END\s+(?:RSA\s+|EC\s+|DSA\s+)?PRIVATE KEY-----)").unwrap(), "[REDACTED_PRIVATE_KEY]"),
            // GitHub tokens
            (Regex::new(r"(ghp_[A-Za-z0-9]{36,})").unwrap(), "[REDACTED_GITHUB_TOKEN]"),
            (Regex::new(r"(gho_[A-Za-z0-9]{36,})").unwrap(), "[REDACTED_GITHUB_TOKEN]"),
            // Slack tokens
            (Regex::new(r"(xox[bpors]-[A-Za-z0-9\-]{10,})").unwrap(), "[REDACTED_SLACK_TOKEN]"),
        ];

        let high_entropy_pattern = Regex::new(r"\b([a-f0-9]{48,})\b").unwrap();

        Self { patterns, high_entropy_pattern }
    }

    pub fn redact(&self, input: &str) -> String {
        let mut output = input.to_string();

        for (pattern, replacement) in &self.patterns {
            output = pattern.replace_all(&output, *replacement).to_string();
        }

        // High-entropy hex strings (potential secrets/hashes)
        output = self.high_entropy_pattern.replace_all(&output, "[REDACTED_HIGH_ENTROPY]").to_string();

        output
    }

    pub fn contains_sensitive(&self, input: &str) -> bool {
        for (pattern, _) in &self.patterns {
            if pattern.is_match(input) {
                return true;
            }
        }
        self.high_entropy_pattern.is_match(input)
    }
}

impl Default for Redactor {
    fn default() -> Self {
        Self::new()
    }
}
