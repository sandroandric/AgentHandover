use regex::Regex;

pub struct Redactor {
    patterns: Vec<(Regex, &'static str)>,
    passwords: PasswordRules,
    cc_pattern: Regex,
    high_entropy_pattern: Regex,
}

/// Passwords (the worker's `redaction.py` has the same rules). A value is
/// dropped, its label kept, when it follows
/// - a label and `=` (`pwd=x`, stopping at `&` as in a query) or `:`
///   (`password: x`, `pw: x`);
/// - a PIN label (`PIN 1234`, `passcode: 482913`): 4-8 digits;
/// - `password is x`, `my password's x`, or the bare `passcode x` (a
///   password manager's `password  x` column): only when the value looks
///   like a secret (a digit, a symbol inside it, or an inner capital), so
///   prose that mentions passwords ("reset your password", "password is
///   required") is left alone.
struct PasswordRules {
    equals: Regex,
    colon: Regex,
    pin: Regex,
    phrase: Regex,
    secret_like: Regex,
}

const PW_LABEL: &str = r"(?:pass(?:word|wd|phrase|code)|pwd|pw)";
const PW_TOKEN: &str = "[REDACTED_PASSWORD]";
const WRAP_OPEN: &[char] = &['(', '[', '{', '<', '\'', '"', '\u{201c}', '\u{2018}'];
const WRAP_CLOSE: &[char] = &['.', ',', ';', ':', ')', ']', '}', '>', '\'', '"', '\u{201d}', '\u{2019}'];

impl PasswordRules {
    fn new() -> Self {
        Self {
            equals: Regex::new(&format!(r#"(?i)\b({PW_LABEL}[ \t]*=[ \t]*)['"]?[^\s'"&,;]+['"]?"#)).unwrap(),
            colon: Regex::new(&format!(r#"(?i)\b({PW_LABEL}[ \t]*:[ \t]*)['"]?[^\s'",;]+['"]?"#)).unwrap(),
            pin: Regex::new(
                r"(?i)\b((?:pin|passcode)(?:[ \t]+(?:code|number))?(?:[ \t]+is)?[ \t]*[:=#]?[ \t]*)\d{4,8}\b",
            )
            .unwrap(),
            phrase: Regex::new(&format!(
                r"(?i)\b({PW_LABEL}(?:['\x{{2019}}]s|[ \t]+(?:is|was))?:?[ \t]+)(\S+)"
            ))
            .unwrap(),
            secret_like: Regex::new(r"\d|[^\w]|[a-z][A-Z]").unwrap(),
        }
    }

    fn redact(&self, input: &str) -> String {
        let text = self.equals.replace_all(input, format!("${{1}}{PW_TOKEN}"));
        let text = self.colon.replace_all(&text, format!("${{1}}{PW_TOKEN}"));
        let text = self.pin.replace_all(&text, "${1}[REDACTED_PIN]");
        self.phrase
            .replace_all(&text, |caps: &regex::Captures| {
                let value = &caps[2];
                let core = value.trim_start_matches(WRAP_OPEN);
                let lead = &value[..value.len() - core.len()];
                let stripped = core.trim_end_matches(WRAP_CLOSE);
                let trail = &core[stripped.len()..];
                if stripped.chars().count() < 4
                    || stripped.starts_with("[REDACTED")
                    || !self.secret_like.is_match(stripped)
                {
                    caps[0].to_string()
                } else {
                    format!("{}{lead}{PW_TOKEN}{trail}", &caps[1])
                }
            })
            .into_owned()
    }
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
            // GitHub tokens: classic (ghp_/gho_/ghu_/ghs_/ghr_) and fine-grained PATs
            (Regex::new(r"(gh[pousr]_[A-Za-z0-9]{36,})").unwrap(), "[REDACTED_GITHUB_TOKEN]"),
            (Regex::new(r"(github_pat_[A-Za-z0-9_]{22,})").unwrap(), "[REDACTED_GITHUB_TOKEN]"),
            // Slack tokens
            (Regex::new(r"(xox[bpors]-[A-Za-z0-9\-]{10,})").unwrap(), "[REDACTED_SLACK_TOKEN]"),
            // Anthropic API keys (must run before the OpenAI pattern)
            (Regex::new(r"(sk-ant-[A-Za-z0-9_\-]{20,})").unwrap(), "[REDACTED_ANTHROPIC_KEY]"),
            // OpenAI project keys
            (Regex::new(r"(sk-proj-[A-Za-z0-9_\-]{20,})").unwrap(), "[REDACTED_OPENAI_KEY]"),
            // Stripe secret and restricted keys (live and test)
            (Regex::new(r"\b([sr]k_(?:live|test)_[A-Za-z0-9]{16,})").unwrap(), "[REDACTED_STRIPE_KEY]"),
            // Google API keys
            (Regex::new(r"(AIza[0-9A-Za-z_\-]{35})").unwrap(), "[REDACTED_GOOGLE_API_KEY]"),
            // GitLab personal access tokens
            (Regex::new(r"(glpat-[A-Za-z0-9_\-]{20,})").unwrap(), "[REDACTED_GITLAB_TOKEN]"),
            // Hugging Face tokens
            (Regex::new(r"\b(hf_[A-Za-z0-9]{30,})").unwrap(), "[REDACTED_HF_TOKEN]"),
            // JSON Web Tokens: three base64url segments, header starts with `{"`
            (Regex::new(r"\b(eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})").unwrap(), "[REDACTED_JWT]"),
        ];

        // Credit card pattern — matches are validated with Luhn before redacting.
        let cc_pattern = Regex::new(r"\b([3-6]\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{3,4})\b").unwrap();

        // High-entropy hex strings: 80+ chars to avoid matching common hashes
        // (SHA-256 = 64 hex chars, SHA-512 = 128 hex chars). 80 is above
        // standard hash lengths but catches genuinely suspicious long hex secrets.
        let high_entropy_pattern = Regex::new(r"\b([a-f0-9]{80,})\b").unwrap();

        Self { patterns, passwords: PasswordRules::new(), cc_pattern, high_entropy_pattern }
    }

    pub fn redact(&self, input: &str) -> String {
        let mut output = input.to_string();

        for (pattern, replacement) in &self.patterns {
            output = pattern.replace_all(&output, *replacement).to_string();
        }
        output = self.passwords.redact(&output);

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
        if self.passwords.redact(input) != input {
            return true;
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
