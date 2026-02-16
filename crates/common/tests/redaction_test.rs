use oc_apprentice_common::redaction::Redactor;

#[test]
fn test_redacts_aws_access_key() {
    let r = Redactor::new();
    let input = "export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE";
    let output = r.redact(input);
    assert!(!output.contains("AKIAIOSFODNN7EXAMPLE"));
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
