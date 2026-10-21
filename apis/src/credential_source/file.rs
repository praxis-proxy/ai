// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Bounded, fail-closed projected-file token read.

use std::{fs::File, io::Read as _, path::Path};

use praxis_filter::FilterError;
use zeroize::Zeroizing;

use super::{DiagnosticContext, MAX_SOURCE_LEN, MAX_TOKEN_BYTES};

/// One more than the token cap, so a `.take()` read can detect overflow.
const MAX_TOKEN_READ_BYTES: u64 = MAX_TOKEN_BYTES as u64 + 1;

/// Format the trailing `for '{reference}'` clause only when a reference exists.
fn for_reference(ctx: &DiagnosticContext) -> String {
    match &ctx.reference {
        Some(reference) => format!(" for '{reference}'"),
        None => String::new(),
    }
}

/// Reject an empty (trimmed) or oversized token. `owner` qualifies the error
/// with the owning filter name.
pub(crate) fn validate_token(token: &str, owner: &str) -> Result<(), FilterError> {
    if token.trim().is_empty() || token.len() > MAX_TOKEN_BYTES {
        return Err(format!("{owner}: credential token must contain 1-{MAX_TOKEN_BYTES} bytes").into());
    }
    Ok(())
}

/// Reject an empty or oversized env-var name / file path. `owner` qualifies the
/// error with the owning filter name.
pub(crate) fn validate_source_len(kind: &str, value: &str, owner: &str) -> Result<(), FilterError> {
    if value.trim().is_empty() || value.len() > MAX_SOURCE_LEN {
        return Err(format!("{owner}: {kind} must contain 1-{MAX_SOURCE_LEN} bytes").into());
    }
    Ok(())
}

/// Read a projected credential file with a hard byte cap; trim and validate.
/// Secret bytes never appear in any error.
pub(crate) fn read_projected_token(path: &str, ctx: &DiagnosticContext) -> Result<Zeroizing<String>, FilterError> {
    let owner = ctx.owner;
    let file = File::open(Path::new(path))
        .map_err(|error| format!("{owner}: cannot read file '{path}'{}: {error}", for_reference(ctx)))?;
    let mut content = Zeroizing::new(String::new());
    file.take(MAX_TOKEN_READ_BYTES)
        .read_to_string(&mut content)
        .map_err(|error| {
            format!(
                "{owner}: cannot read projected credential '{path}'{}: {error}",
                for_reference(ctx)
            )
        })?;
    if content.len() > MAX_TOKEN_BYTES {
        return Err(format!("{owner}: projected credential '{path}' exceeds {MAX_TOKEN_BYTES} bytes").into());
    }
    let token = Zeroizing::new(content.trim().to_owned());
    validate_token(&token, owner)?;
    Ok(token)
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use std::io::Write as _;

    use super::*;

    fn ctx_with_ref() -> DiagnosticContext {
        DiagnosticContext {
            owner: "credential_inject",
            reference: Some("name/namespace/key".to_owned()),
        }
    }
    fn ctx_no_ref() -> DiagnosticContext {
        DiagnosticContext {
            owner: "anthropic_web_search",
            reference: None,
        }
    }

    #[test]
    fn reads_and_trims_token() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "  tok-value").unwrap();
        let token = read_projected_token(f.path().to_str().unwrap(), &ctx_with_ref()).unwrap();
        assert_eq!(token.as_str(), "tok-value");
    }

    #[test]
    fn missing_file_error_names_reference_not_token() {
        let err = read_projected_token("/no/such/projected/file", &ctx_with_ref()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("credential_inject"));
        assert!(msg.contains("name/namespace/key"), "{msg}");
        assert!(msg.contains("cannot read file"), "{msg}");
    }

    #[test]
    fn missing_file_error_owner_only_when_no_reference() {
        let err = read_projected_token("/no/such/projected/file", &ctx_no_ref()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("anthropic_web_search"), "{msg}");
        assert!(
            !msg.contains(" for '"),
            "owner-only form must omit the reference clause: {msg}"
        );
    }

    #[test]
    fn empty_file_rejected() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, "   ").unwrap();
        assert!(read_projected_token(f.path().to_str().unwrap(), &ctx_with_ref()).is_err());
    }

    #[test]
    fn oversized_file_rejected() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[b'a'; MAX_TOKEN_BYTES + 1]).unwrap();
        assert!(read_projected_token(f.path().to_str().unwrap(), &ctx_with_ref()).is_err());
    }

    #[test]
    fn invalid_utf8_file_rejected_without_leaking_bytes() {
        // A projected file whose bytes are not valid UTF-8 must fail closed, and
        // the underlying `read_to_string` error (`ErrorKind::InvalidData`) must
        // never surface the raw non-UTF-8 bytes in the diagnostic.
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&[0xFF, 0xFE]).unwrap();
        let err = read_projected_token(f.path().to_str().unwrap(), &ctx_with_ref()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("credential_inject"), "{msg}");
        assert!(msg.contains("cannot read projected credential"), "{msg}");
        assert!(!msg.as_bytes().contains(&0xFF), "{msg}");
        assert!(!msg.as_bytes().contains(&0xFE), "{msg}");
    }

    #[test]
    fn validate_token_bounds_error_is_owner_qualified() {
        let empty = validate_token("", "credential_inject").unwrap_err();
        assert_eq!(
            empty.to_string(),
            "credential_inject: credential token must contain 1-16384 bytes"
        );
        let oversized_input = "a".repeat(MAX_TOKEN_BYTES + 1);
        let oversized = validate_token(&oversized_input, "credential_inject").unwrap_err();
        assert_eq!(
            oversized.to_string(),
            "credential_inject: credential token must contain 1-16384 bytes"
        );
    }

    #[test]
    fn validate_source_len_bounds_error_is_owner_qualified() {
        let env_var = validate_source_len("env_var", "", "credential_inject").unwrap_err();
        assert_eq!(
            env_var.to_string(),
            "credential_inject: env_var must contain 1-4096 bytes"
        );
        let file = validate_source_len("file", "", "credential_inject").unwrap_err();
        assert_eq!(file.to_string(), "credential_inject: file must contain 1-4096 bytes");
    }

    #[test]
    fn bounds_error_prefix_tracks_owner() {
        let err = validate_source_len("file", "", "web_search").unwrap_err();
        assert_eq!(err.to_string(), "web_search: file must contain 1-4096 bytes");
    }
}
