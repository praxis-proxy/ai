// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared `PostgreSQL` TLS and certificate-authentication configuration.
//!
//! [`PgTlsConfig`] is the single carrier for the TLS-related connection
//! settings used by both the `openai_response_store` and `openai_conversations`
//! filters. It is validated once at filter construction (fail-before-serving)
//! and then handed to the store constructor, so the two filters cannot drift in
//! how they interpret TLS options or the certificate-authentication compliance
//! profile.
//!
//! # Cryptographic boundary
//!
//! TLS transport (including client-certificate authentication) runs inside the
//! platform TLS library that `SQLx` links through the `tls-native-tls` feature —
//! `OpenSSL` on Linux and Security.framework (Secure Transport) on macOS.
//! Password authentication, by contrast, runs application-side `RustCrypto`
//! primitives (`hmac`/`sha2` for SCRAM-SHA-256, `md-5` for MD5) that sit
//! *outside* that boundary.
//!
//! The [`require_certificate_authentication`] compliance profile steers a
//! deployment onto certificate authentication so the server answers the TLS
//! handshake with `Authentication::Ok` and no password primitive executes.
//!
//! This profile can only enforce the parts of that contract that are visible to
//! the client: it fails closed when the URL, environment, or TLS settings would
//! permit a password path. It CANNOT enforce the decisive half — the `PostgreSQL`
//! server selects the authentication method via `pg_hba.conf` and delivers it
//! after the handshake, and `SQLx` obeys whatever arrives. The server MUST be
//! configured with `hostssl <db> <user> <cidr> cert` (or otherwise never issue a
//! password challenge); operators must verify that out of band.
//!
//! [`require_certificate_authentication`]: PgTlsConfig::require_certificate_authentication

use praxis_filter::{FilterError, has_dot_dot_traversal};

use super::{
    SslMode,
    postgres_url::{
        has_postgres_url_ssl_root_cert, is_verified_postgres_sslmode, postgres_url_dropped_connection_param,
        postgres_url_has_password, postgres_url_has_tls_params, postgres_url_sslmode,
        validate_postgres_url_tls_file_params,
    },
};

/// TLS and certificate-authentication settings for a `PostgreSQL` connection.
///
/// Borrows its string fields from the owning filter configuration so no secret
/// material is copied. Construct it from the filter config, [`validate`] it, and
/// pass it to [`PostgresResponseStore::new`].
///
/// [`validate`]: PgTlsConfig::validate
/// [`PostgresResponseStore::new`]: super::PostgresResponseStore::new
#[derive(Clone, Copy, Debug, Default)]
pub struct PgTlsConfig<'a> {
    /// Enforce the certificate-authentication compliance profile.
    pub require_certificate_authentication: bool,

    /// Path to a PEM-encoded client certificate for mutual TLS.
    pub ssl_client_cert: Option<&'a str>,

    /// Path to the PEM-encoded private key for [`ssl_client_cert`].
    ///
    /// [`ssl_client_cert`]: PgTlsConfig::ssl_client_cert
    pub ssl_client_key: Option<&'a str>,

    /// TLS mode override. Always overrides any `sslmode` in the URL.
    pub ssl_mode: Option<SslMode>,

    /// Path to a PEM-encoded root CA certificate for server verification.
    pub ssl_root_cert: Option<&'a str>,
}

impl PgTlsConfig<'_> {
    /// Return whether a root CA is configured (via field or URL).
    fn has_root_cert(&self, database_url: &str) -> bool {
        self.ssl_root_cert.is_some() || has_postgres_url_ssl_root_cert(database_url)
    }

    /// Return whether the effective SSL mode verifies certificates.
    ///
    /// With no explicit `ssl_mode`, the runtime default is
    /// [`SslMode::VerifyFull`], so the `None` case is verified unless the URL
    /// carries a non-verifying `sslmode`.
    fn has_verified_ssl_mode(&self, database_url: &str) -> bool {
        match self.ssl_mode {
            Some(SslMode::VerifyCa | SslMode::VerifyFull) => true,
            Some(SslMode::Disable | SslMode::Prefer | SslMode::Require) => false,
            None => postgres_url_sslmode(database_url)
                .as_deref()
                .is_none_or(is_verified_postgres_sslmode),
        }
    }

    /// Reject `..` traversal in any configured certificate path.
    fn reject_path_traversal(&self, filter_name: &str) -> Result<(), FilterError> {
        for (label, path) in [
            ("ssl_root_cert", self.ssl_root_cert),
            ("ssl_client_cert", self.ssl_client_cert),
            ("ssl_client_key", self.ssl_client_key),
        ] {
            if let Some(path) = path
                && has_dot_dot_traversal(path)
            {
                return Err(format!("{filter_name}: {label} must not contain '..' path traversal").into());
            }
        }
        Ok(())
    }

    /// Validate the TLS configuration against a `PostgreSQL` connection URL.
    ///
    /// Runs at filter construction so an unsupported authentication path fails
    /// closed before any traffic is served.
    ///
    /// # Errors
    ///
    /// Returns a [`FilterError`] carrying `filter_name` when any path contains
    /// `..` traversal, when the client certificate and key are not supplied
    /// together, when a certificate is configured without a verifying SSL mode,
    /// when the client private key is group- or world-accessible, or when the
    /// compliance profile is enabled but a client-detectable password path
    /// remains open.
    pub fn validate(&self, filter_name: &str, database_url: &str) -> Result<(), FilterError> {
        let pgpassword_present = std::env::var_os("PGPASSWORD").is_some();
        self.validate_impl(filter_name, database_url, pgpassword_present)?;
        self.validate_client_key_permissions(filter_name)
    }

    /// Fail closed when the client private key is group- or world-accessible.
    ///
    /// The key authenticates the proxy's `PostgreSQL` identity, so a key readable
    /// by other local users leaks that identity. The configuration contract
    /// documents mode `0600`; this enforces it. `std::fs::metadata` follows
    /// symlinks, so the check inspects the mode of the actual key bytes. A key
    /// that cannot be stat'd is left to surface at connect time with a clearer
    /// error, and non-regular files are skipped.
    #[cfg(unix)]
    fn validate_client_key_permissions(&self, filter_name: &str) -> Result<(), FilterError> {
        use std::os::unix::fs::PermissionsExt as _;

        let Some(path) = self.ssl_client_key else {
            return Ok(());
        };
        let Ok(metadata) = std::fs::metadata(path) else {
            return Ok(());
        };
        if !metadata.is_file() {
            return Ok(());
        }
        reject_insecure_key_mode(filter_name, metadata.permissions().mode())
    }

    /// No-op on non-Unix targets, which do not expose POSIX permission bits.
    #[cfg(not(unix))]
    fn validate_client_key_permissions(&self, _filter_name: &str) -> Result<(), FilterError> {
        Ok(())
    }

    /// Enforce the certificate-authentication compliance profile.
    ///
    /// Each rule below closes a client-visible way for a password to reach the
    /// effective connection options (which would drive application-side
    /// `RustCrypto` SCRAM/MD5). The remaining, decisive requirement — that the
    /// server's `pg_hba.conf` uses `cert` so it never issues a password
    /// challenge — is documented as a deployment prerequisite and cannot be
    /// enforced by the client.
    #[expect(
        clippy::too_many_lines,
        reason = "linear sequence of independent fail-closed guards, each with its own diagnostic"
    )]
    fn validate_compliance_profile(
        &self,
        filter_name: &str,
        database_url: &str,
        pgpassword_present: bool,
    ) -> Result<(), FilterError> {
        if !matches!(self.ssl_mode, Some(SslMode::VerifyFull)) {
            return Err(
                format!("{filter_name}: 'require_certificate_authentication' requires ssl_mode 'verify-full'").into(),
            );
        }
        if self.ssl_client_cert.is_none() || self.ssl_client_key.is_none() {
            return Err(format!(
                "{filter_name}: 'require_certificate_authentication' requires both 'ssl_client_cert' and 'ssl_client_key'"
            )
            .into());
        }
        if postgres_url_has_password(database_url) {
            return Err(format!(
                "{filter_name}: 'require_certificate_authentication' forbids a password in database_url; \
                 authenticate with the client certificate instead"
            )
            .into());
        }
        if postgres_url_has_tls_params(database_url) {
            return Err(format!(
                "{filter_name}: 'require_certificate_authentication' forbids TLS parameters in database_url; \
                 configure TLS through the filter fields so the settings are authoritative"
            )
            .into());
        }
        if let Some(param) = postgres_url_dropped_connection_param(database_url) {
            return Err(format!(
                "{filter_name}: 'require_certificate_authentication' forbids the '{param}' connection parameter in \
                 database_url; the certificate-authentication path reconstructs the connection with only addressing \
                 and TLS fields, so this parameter would be silently dropped. Set connection defaults such as \
                 search_path on the database role instead (ALTER ROLE ... SET ...)"
            )
            .into());
        }
        if pgpassword_present {
            return Err(format!(
                "{filter_name}: 'require_certificate_authentication' forbids the PGPASSWORD environment variable"
            )
            .into());
        }
        Ok(())
    }

    /// Validation core with the ambient `PGPASSWORD` presence injected.
    ///
    /// Splitting the environment read out of the logic keeps every rule
    /// deterministically testable without mutating process-wide state (which
    /// would require `unsafe` under Rust 2024).
    fn validate_impl(
        &self,
        filter_name: &str,
        database_url: &str,
        pgpassword_present: bool,
    ) -> Result<(), FilterError> {
        #[cfg(all(feature = "_store-postgres", not(feature = "store-postgres")))]
        if !self.require_certificate_authentication {
            return Err(format!(
                "{filter_name}: this build supports certificate-authenticated PostgreSQL only; \
                 set 'require_certificate_authentication' to true"
            )
            .into());
        }

        validate_postgres_url_tls_file_params(filter_name, database_url)?;
        self.reject_path_traversal(filter_name)?;

        if self.ssl_client_cert.is_some() != self.ssl_client_key.is_some() {
            return Err(
                format!("{filter_name}: 'ssl_client_cert' and 'ssl_client_key' must be configured together").into(),
            );
        }

        if self.has_root_cert(database_url) && !self.has_verified_ssl_mode(database_url) {
            return Err(
                format!("{filter_name}: 'ssl_root_cert' requires ssl_mode 'verify-ca' or 'verify-full'").into(),
            );
        }

        if self.ssl_client_cert.is_some() && !self.has_verified_ssl_mode(database_url) {
            return Err(format!(
                "{filter_name}: 'ssl_client_cert'/'ssl_client_key' require ssl_mode 'verify-ca' or 'verify-full'"
            )
            .into());
        }

        if self.require_certificate_authentication {
            self.validate_compliance_profile(filter_name, database_url, pgpassword_present)?;
        }
        Ok(())
    }
}

/// Reject a client-key file mode that grants group or world access.
///
/// `mode & 0o077` is non-zero when any group or other permission bit is set; a
/// compliant key is `0600` (owner read/write only). Kept as a pure function so
/// the boundary is testable without materializing files at every mode.
#[cfg(unix)]
fn reject_insecure_key_mode(filter_name: &str, mode: u32) -> Result<(), FilterError> {
    if mode & 0o077 != 0 {
        return Err(format!(
            "{filter_name}: 'ssl_client_key' must not be group- or world-accessible; \
             restrict it to mode 0600 (found {:04o})",
            mode & 0o7777
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]
mod tests {
    use super::*;

    const FILTER: &str = "test_filter";
    const URL: &str = "postgres://cert-user@db.internal:5432/praxis";

    #[test]
    fn accepts_compliant_profile() {
        validate_no_env(&compliant(), URL).expect("compliant profile should validate");
    }

    #[cfg(unix)]
    #[test]
    fn client_key_permissions_enforced_on_real_file() {
        use std::os::unix::fs::PermissionsExt as _;

        let key = tempfile::NamedTempFile::new().expect("create temp key");
        let path = key.path().to_str().expect("utf-8 path").to_owned();

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).expect("chmod 0600");
        let cfg = PgTlsConfig {
            ssl_client_key: Some(&path),
            ..compliant()
        };
        cfg.validate_client_key_permissions(FILTER)
            .expect("0600 key should be accepted");

        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).expect("chmod 0644");
        let err = cfg.validate_client_key_permissions(FILTER).unwrap_err().to_string();
        assert!(err.contains("group- or world-accessible"), "got: {err}");
    }

    #[cfg(unix)]
    #[test]
    fn client_key_permissions_no_op_without_key() {
        let cfg = PgTlsConfig {
            ssl_client_key: None,
            ..compliant()
        };
        cfg.validate_client_key_permissions(FILTER)
            .expect("no key configured means nothing to enforce");
    }

    #[test]
    fn compliance_rejects_dropped_connection_param_in_url() {
        // A connection parameter such as options[search_path] would be silently
        // dropped by the certificate-authentication rebuild, so it must fail
        // closed rather than run DDL/queries against an unexpected schema.
        let err = validate_no_env(
            &compliant(),
            "postgres://cert-user@db.internal:5432/praxis?options[search_path]=audit",
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("search_path") && err.contains("silently dropped"),
            "got: {err}"
        );
    }

    #[test]
    fn compliance_rejects_password_in_url() {
        let err = validate_no_env(&compliant(), "postgres://cert-user:secret@db.internal:5432/praxis")
            .unwrap_err()
            .to_string();
        assert!(err.contains("forbids a password"), "got: {err}");
    }

    #[test]
    fn compliance_rejects_pgpassword_env() {
        // The compliance profile fails closed when PGPASSWORD is present, even
        // though every other field is well-formed.
        let err = compliant().validate_impl(FILTER, URL, true).unwrap_err().to_string();
        assert!(err.contains("PGPASSWORD"), "got: {err}");
    }

    #[test]
    fn compliance_rejects_tls_params_in_url() {
        let err = validate_no_env(
            &compliant(),
            "postgres://cert-user@db.internal:5432/praxis?sslmode=verify-full",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("forbids TLS parameters"), "got: {err}");
    }

    #[test]
    fn compliance_requires_client_cert_and_key() {
        let cfg = PgTlsConfig {
            ssl_client_cert: None,
            ssl_client_key: None,
            ..compliant()
        };
        let err = validate_no_env(&cfg, URL).unwrap_err().to_string();
        assert!(err.contains("requires both"), "got: {err}");
    }

    #[test]
    fn compliance_requires_verify_full() {
        let cfg = PgTlsConfig {
            ssl_mode: Some(SslMode::VerifyCa),
            ..compliant()
        };
        let err = validate_no_env(&cfg, URL).unwrap_err().to_string();
        assert!(err.contains("requires ssl_mode 'verify-full'"), "got: {err}");
    }

    #[cfg(unix)]
    #[test]
    fn key_mode_group_or_world_accessible_is_rejected() {
        for mode in [0o640, 0o644, 0o604, 0o660, 0o666] {
            let err = reject_insecure_key_mode(FILTER, mode).unwrap_err().to_string();
            assert!(err.contains("group- or world-accessible"), "mode {mode:o} -> {err}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn key_mode_owner_only_is_accepted() {
        reject_insecure_key_mode(FILTER, 0o600).expect("0600 is owner-only");
    }

    #[cfg(feature = "store-postgres")]
    #[test]
    fn non_compliance_leaves_password_urls_alone() {
        // Without the compliance profile, a password URL with verify-full and no
        // client cert is still a valid (non-compliance) configuration.
        let cfg = PgTlsConfig {
            ssl_mode: Some(SslMode::VerifyFull),
            ssl_root_cert: None,
            ssl_client_cert: None,
            ssl_client_key: None,
            require_certificate_authentication: false,
        };
        validate_no_env(&cfg, "postgres://user:secret@db.internal:5432/praxis")
            .expect("non-compliance password URL should be allowed");
    }

    #[cfg(all(feature = "_store-postgres", not(feature = "store-postgres")))]
    #[test]
    fn certificate_only_build_rejects_non_compliance_profile() {
        let cfg = PgTlsConfig {
            ssl_mode: Some(SslMode::VerifyFull),
            ssl_root_cert: None,
            ssl_client_cert: None,
            ssl_client_key: None,
            require_certificate_authentication: false,
        };
        let err = validate_no_env(&cfg, URL).unwrap_err().to_string();
        assert!(
            err.contains("supports certificate-authenticated PostgreSQL only")
                && err.contains("require_certificate_authentication"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_client_cert_with_unverified_mode() {
        let cfg = PgTlsConfig {
            ssl_mode: Some(SslMode::Require),
            require_certificate_authentication: true,
            ..compliant()
        };
        let err = validate_no_env(&cfg, URL).unwrap_err().to_string();
        assert!(err.contains("verify-ca") && err.contains("verify-full"), "got: {err}");
    }

    #[test]
    fn rejects_client_cert_without_key() {
        let cfg = PgTlsConfig {
            ssl_client_key: None,
            require_certificate_authentication: true,
            ..compliant()
        };
        let err = validate_no_env(&cfg, URL).unwrap_err().to_string();
        assert!(err.contains("configured together"), "got: {err}");
    }

    #[test]
    fn rejects_path_traversal_in_client_key() {
        let cfg = PgTlsConfig {
            ssl_client_key: Some("../../etc/shadow"),
            ..compliant()
        };
        let err = validate_no_env(&cfg, URL).unwrap_err().to_string();
        assert!(err.contains("path traversal"), "got: {err}");
    }

    // Test Utilities

    /// A fully valid compliance-profile configuration.
    fn compliant() -> PgTlsConfig<'static> {
        PgTlsConfig {
            ssl_mode: Some(SslMode::VerifyFull),
            ssl_root_cert: Some("/etc/pki/ca.pem"),
            ssl_client_cert: Some("/etc/pki/client.pem"),
            ssl_client_key: Some("/etc/pki/client.key"),
            require_certificate_authentication: true,
        }
    }

    /// Validate with `PGPASSWORD` treated as absent, so results do not depend on
    /// the ambient test environment.
    fn validate_no_env(cfg: &PgTlsConfig<'_>, url: &str) -> Result<(), FilterError> {
        cfg.validate_impl(FILTER, url, false)
    }
}
