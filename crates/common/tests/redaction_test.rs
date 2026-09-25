use agenthandover_common::redaction::Redactor;

#[test]
fn test_redacts_aws_access_key() {
    let r = Redactor::new();
    let input = "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE";
    let output = r.redact(input);
    assert!(!output.contains(concat!("AKIA", "IOSFODNN7EXAMPLE")));
    assert!(output.contains("[REDACTED_AWS_KEY]"));
}

#[test]
fn test_redacts_aws_secret_key() {
    let r = Redactor::new();
    let input = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
    let output = r.redact(input);
    assert!(!output.contains("wJalrXUtnFEMI"));
    assert!(output.contains("[REDACTED_SECRET]"));
}

#[test]
fn test_redacts_credit_card_number() {
    let r = Redactor::new();
    let input = "Card: 4111-1111-1111-1111 expires 12/25";
    let output = r.redact(input);
    assert!(!output.contains("4111-1111-1111-1111"));
    assert!(output.contains("[REDACTED_CC]"));
}

#[test]
fn test_redacts_private_key() {
    let r = Redactor::new();
    let input = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKC...\n-----END RSA PRIVATE KEY-----";
    let output = r.redact(input);
    assert!(!output.contains("MIIEowIBAAKC"));
    assert!(output.contains("[REDACTED_PRIVATE_KEY]"));
}

#[test]
fn test_redacts_high_entropy_hex_strings() {
    let r = Redactor::new();
    // 80+ hex chars triggers redaction (below 80 is treated as a normal hash)
    let input = "token: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6";
    let output = r.redact(input);
    assert!(output.contains("[REDACTED_HIGH_ENTROPY]"));
}

#[test]
fn test_does_not_redact_sha256_hash() {
    let r = Redactor::new();
    // 64 hex chars (SHA-256 hash) should NOT be redacted
    let input = "commit: a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2";
    let output = r.redact(input);
    assert!(!output.contains("[REDACTED_HIGH_ENTROPY]"), "SHA-256 hashes should not be redacted");
}

#[test]
fn test_does_not_redact_version_number_as_cc() {
    let r = Redactor::new();
    // Version numbers like 3.12.4567.8901.2345 should not match CC pattern
    let input = "version: 3.1.4.1.5.9.2.6.5.3.5.8.9.7.9.3";
    let output = r.redact(input);
    assert!(!output.contains("[REDACTED_CC]"), "Version numbers should not be redacted as CC");
}

#[test]
fn test_does_not_redact_normal_text() {
    let r = Redactor::new();
    let input = "Hello world, this is a normal sentence about coding.";
    let output = r.redact(input);
    assert_eq!(output, input);
}

#[test]
fn test_detects_sensitive_content() {
    let r = Redactor::new();
    assert!(r.contains_sensitive("my key is AKIAIOSFODNN7EXAMPLE"));
    assert!(!r.contains_sensitive("hello world"));
}

/// Each provider-specific secret shape is redacted and replaced by its marker.
#[test]
fn test_redacts_provider_secret_shapes() {
    let r = Redactor::new();
    let cases = [
        (concat!("sk-ant-", "api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789_-abcd"), "[REDACTED_ANTHROPIC_KEY]"),
        (concat!("sk-proj-", "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"), "[REDACTED_OPENAI_KEY]"),
        (concat!("sk_", "live_51HqLyjWDarjtT1zdp7dcXYZ"), "[REDACTED_STRIPE_KEY]"),
        (concat!("sk_", "test_4eC39HqLyjWDarjtT1zdp7dc"), "[REDACTED_STRIPE_KEY]"),
        (concat!("rk_", "live_51HqLyjWDarjtT1zdp7dcXYZ"), "[REDACTED_STRIPE_KEY]"),
        (concat!("AIza", "SyA-1234567890abcdefghijklmnopqrstu"), "[REDACTED_GOOGLE_API_KEY]"),
        (concat!("github_pat_", "11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz"), "[REDACTED_GITHUB_TOKEN]"),
        (concat!("ghp_", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"), "[REDACTED_GITHUB_TOKEN]"),
        ("gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij", "[REDACTED_GITHUB_TOKEN]"),
        ("ghs_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij", "[REDACTED_GITHUB_TOKEN]"),
        (concat!("glpat-", "AbCdEfGhIjKlMnOpQrSt"), "[REDACTED_GITLAB_TOKEN]"),
        (concat!("hf_", "AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh"), "[REDACTED_HF_TOKEN]"),
        (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            "[REDACTED_JWT]",
        ),
    ];
    for (secret, marker) in cases {
        let input = format!("copied {} into the form", secret);
        let output = r.redact(&input);
        assert!(!output.contains(secret), "{} should be redacted, got {}", secret, output);
        assert_eq!(output, format!("copied {} into the form", marker), "wrong marker for {}", secret);
        assert!(r.contains_sensitive(&input), "{} should be flagged sensitive", secret);
    }
}

/// Labelled secrets keep their label and lose their value; the rest of the
/// text survives.
#[test]
fn test_redacts_labelled_passwords() {
    let r = Redactor::new();
    assert_eq!(r.redact("password: hunter2 please"), "password: [REDACTED_PASSWORD] please");
    assert_eq!(r.redact("Password=hunter2"), "Password=[REDACTED_PASSWORD]");
    assert_eq!(
        r.redact("https://h.example/login?user=a&pwd=s3cret&x=1"),
        "https://h.example/login?user=a&pwd=[REDACTED_PASSWORD]&x=1"
    );
    assert_eq!(r.redact("passwd = 'quoted'"), "passwd = [REDACTED_PASSWORD]");
    // A bare label with no value is left alone.
    assert_eq!(r.redact("Password:"), "Password:");
    // Redaction is idempotent: stored text redacted twice does not change.
    let once = r.redact("pwd=hunter2");
    assert_eq!(r.redact(&once), once);
}

/// Near-miss strings that look like prefixes but are ordinary text stay intact.
#[test]
fn test_does_not_redact_prefix_lookalikes() {
    let r = Redactor::new();
    for input in [
        "the sk-ant prefix is used by Anthropic",
        "use hf_ tokens for Hugging Face",
        "risk_test_plan.md",
        "eyJ is how base64 JSON starts",
        "AIza is a Google prefix",
    ] {
        assert_eq!(r.redact(input), input);
    }
}

/// WP-F1: password phrasing and PINs (the worker's `redaction.py` has the
/// same rules and the same cases in `test_redaction.py`).
#[test]
fn test_redacts_password_phrasing_and_pins() {
    let r = Redactor::new();
    for (input, expected) in [
        ("my password is hunter22", "my password is [REDACTED_PASSWORD]"),
        (
            "The staging GitHub password is Tr0ub4dor&3-horse-battery",
            "The staging GitHub password is [REDACTED_PASSWORD]",
        ),
        ("pw: abc", "pw: [REDACTED_PASSWORD]"),
        ("PW=abc", "PW=[REDACTED_PASSWORD]"),
        ("passcode 482913", "passcode [REDACTED_PIN]"),
        ("passcode Zq9-x", "passcode [REDACTED_PASSWORD]"),
        ("my password's S3cret!x", "my password's [REDACTED_PASSWORD]"),
        ("my password\u{2019}s S3cret", "my password\u{2019}s [REDACTED_PASSWORD]"),
        ("PIN 1234", "PIN [REDACTED_PIN]"),
        ("Your pin code is 90210.", "Your pin code is [REDACTED_PIN]."),
        ("the wifi password was CorrectHorse, ok", "the wifi password was [REDACTED_PASSWORD], ok"),
        ("password is \"hunter22\".", "password is \"[REDACTED_PASSWORD]\"."),
        ("password  Tr0ub4dor&3-horse-battery", "password  [REDACTED_PASSWORD]"),
        ("password: Tr0ub4dor&3 ok", "password: [REDACTED_PASSWORD] ok"),
        ("password=hunter2&x=1", "password=[REDACTED_PASSWORD]&x=1"),
    ] {
        assert_eq!(r.redact(input), expected, "{input}");
        assert!(r.contains_sensitive(input), "{input}");
        assert_eq!(r.redact(expected), expected, "idempotent: {expected}");
    }
}

/// Prose that merely mentions passwords is left alone.
#[test]
fn test_prose_about_passwords_is_left_alone() {
    let r = Redactor::new();
    for input in [
        "reset your password",
        "Forgot password?",
        "Password is required.",
        "Your password is incorrect",
        "New password\nConfirm new password",
        "Password (optional)",
        "Change your passcode settings",
        "password reset link sent",
        "Enter the PIN you received",
        "pinned 1234 items",
        "Passwords",
    ] {
        assert_eq!(r.redact(input), input);
        assert!(!r.contains_sensitive(input), "{input}");
    }
}
