use regex::Regex;

pub struct Redactor {
    patterns: Vec<(Regex, &'static str)>,
    cc_pattern: Regex,
    high_entropy_pattern: Regex,
}

/// Validate a number string using the Luhn algorithm.
/// Returns true if the number passes the Luhn check (i.e., is a plausible card number).
fn luhn_check(number: &str) -> bool {
    let digits: Vec<u32> = number
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c.to_digit(10).unwrap())
        .collect();

    if digits.len() < 13 {
        return false;
    }

    let mut sum = 0u32;
    let mut double = false;
    for &d in digits.iter().rev() {
        let mut val = d;
        if double {
            val *= 2;
            if val > 9 {
                val -= 9;
            }
        }
        sum += val;
        double = !double;
    }
    sum % 10 == 0
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

        // Credit card pattern — matches are validated with Luhn before redacting.
        let cc_pattern = Regex::new(r"\b([3-6]\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3,4})\b").unwrap();

        // High-entropy hex strings: 80+ chars to avoid matching common hashes
        // (SHA-256 = 64 hex chars, SHA-512 = 128 hex chars). 80 is above
        // standard hash lengths but catches genuinely suspicious long hex secrets.
        let high_entropy_pattern = Regex::new(r"\b([a-f0-9]{80,})\b").unwrap();

        Self { patterns, cc_pattern, high_entropy_pattern }
    }

    pub fn redact(&self, input: &str) -> String {
        let mut output = input.to_string();

        for (pattern, replacement) in &self.patterns {
            output = pattern.replace_all(&output, *replacement).to_string();
        }

        // Credit card numbers — only redact if they pass the Luhn check
        output = self.cc_pattern.replace_all(&output, |caps: &regex::Captures| {
            let matched = &caps[0];
            if luhn_check(matched) {
                "[REDACTED_CC]".to_string()
            } else {
                matched.to_string()
            }
        }).to_string();

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
        // Check CC with Luhn validation
        if let Some(caps) = self.cc_pattern.captures(input) {
            if luhn_check(&caps[0]) {
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
