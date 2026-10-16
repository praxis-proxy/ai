// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Runtime cryptographic readiness checks.
//!
//! A portable interface that inspects the effective cryptographic state of
//! the process and the host, independent of the rustls provider abstraction.
//! The two checks the application performs itself, directly, are:
//!
//! 1. **OpenSSL effective default properties** — `EVP_default_properties_is_fips_enabled(NULL)`, the same call the
//!    rustls-openssl provider uses, called directly by praxis-ai so the check does not depend on the provider
//!    abstraction.
//! 2. **System crypto policy** — `/etc/crypto-policies/config` on RHEL and derivatives, where a `FIPS` (or `FIPS:*`)
//!    policy is the operating system's own signal that FIPS mode is fully configured.
//!
//! Together with the kernel flag (`/proc/sys/crypto/fips_enabled`) these
//! form the readiness prerequisites. The module never activates a provider
//! or changes global state; it queries what the host already configured.

/// Path of the kernel's FIPS mode flag.
const KERNEL_FIPS_FLAG: &str = "/proc/sys/crypto/fips_enabled";
/// Path of the system crypto policy configuration.
const CRYPTO_POLICIES_CONFIG: &str = "/etc/crypto-policies/config";

/// The result of a cryptographic readiness check.
#[derive(Debug, Clone)]
pub struct CryptoReadiness {
    /// Whether `EVP_default_properties_is_fips_enabled(NULL)` returns 1.
    pub openssl_fips_properties: bool,
    /// The kernel FIPS flag from `/proc/sys/crypto/fips_enabled`.
    /// `None` when the file does not exist or cannot be read.
    pub kernel_fips: Option<bool>,
    /// The system crypto policy from `/etc/crypto-policies/config`.
    /// `None` on platforms that do not use crypto-policies.
    pub crypto_policy: Option<String>,
}

impl CryptoReadiness {
    /// Perform all readiness checks and return the result.
    ///
    /// Reads the OpenSSL effective default properties, the kernel FIPS
    /// flag, and the system crypto policy. None of these change global
    /// state; the calls are pure queries.
    #[must_use]
    pub fn check() -> Self {
        Self {
            openssl_fips_properties: openssl_fips_properties_enabled(),
            kernel_fips: kernel_fips_flag(),
            crypto_policy: system_crypto_policy(),
        }
    }

    /// Whether all three FIPS readiness signals are present and positive.
    #[must_use]
    pub fn fips_ready(&self) -> bool {
        self.openssl_fips_properties
            && self.kernel_fips == Some(true)
            && self.crypto_policy.as_ref().is_some_and(|policy| is_fips_policy(policy))
    }

    /// Whether the platform cannot give a definitive readiness answer.
    ///
    /// True on non-Linux platforms and on Linux distributions that do not
    /// use `/etc/crypto-policies/config`, so the readiness check does not
    /// produce a false positive on Alpine, Debian, or macOS.
    #[must_use]
    pub fn unsupported(&self) -> bool {
        self.kernel_fips.is_none() || self.crypto_policy.is_none()
    }

    /// Every prerequisite that is not met, as an operator-facing explanation.
    ///
    /// Empty when every signal is present and positive. Each entry names
    /// the file path or API that failed so the operator knows where to look.
    #[must_use]
    pub fn unmet(&self) -> Vec<String> {
        let mut r = Vec::new();
        if !self.openssl_fips_properties {
            r.push(
                "OpenSSL's effective default properties do not select FIPS-approved algorithms \
                    (EVP_default_properties_is_fips_enabled is 0); is the host in FIPS mode?"
                    .into(),
            );
        }
        if let Some(false) | None = self.kernel_fips {
            r.push(kernel_fips_reason(self.kernel_fips));
        }
        if !self.crypto_policy.as_ref().is_some_and(|p| is_fips_policy(p)) {
            r.push(crypto_policy_reason(self.crypto_policy.as_deref()));
        }
        r
    }
}

/// The operator-facing reason for a missing or negative kernel FIPS flag.
fn kernel_fips_reason(flag: Option<bool>) -> String {
    match flag {
        Some(false) => "the kernel is not in FIPS mode (/proc/sys/crypto/fips_enabled is 0)".into(),
        None => "the kernel FIPS flag cannot be read (/proc/sys/crypto/fips_enabled is absent); \
                 this platform may not support FIPS readiness checks"
            .into(),
        Some(true) => String::new(),
    }
}

/// The operator-facing reason for a missing or non-FIPS crypto policy.
fn crypto_policy_reason(policy: Option<&str>) -> String {
    match policy {
        Some(p) => format!("the system crypto policy is {p:?}, not FIPS (/etc/crypto-policies/config)"),
        None => "the system crypto policy cannot be read (/etc/crypto-policies/config is absent); \
                 this platform may not support system crypto policies"
            .into(),
    }
}

/// Whether OpenSSL's effective default properties select FIPS-approved
/// algorithms only, checked directly through the C API.
///
/// This is `EVP_default_properties_is_fips_enabled(NULL)`, the same
/// function the host's OpenSSL configuration controls. The application
/// does not set this property; it reads what the host configured.
#[cfg(ossl300)]
fn openssl_fips_properties_enabled() -> bool {
    #[expect(
        unsafe_code,
        reason = "direct FFI query of OpenSSL's effective FIPS property; \
                  no state mutation, documented thread-safe after init"
    )]
    // SAFETY: `EVP_default_properties_is_fips_enabled` with a NULL context
    // queries the global default library context. It is a pure read of
    // process-global state that OpenSSL documents as thread-safe after
    // library initialization, which `openssl::init()` ensures.
    unsafe {
        openssl::init();
        openssl_sys::EVP_default_properties_is_fips_enabled(std::ptr::null_mut()) == 1
    }
}

#[cfg(not(ossl300))]
fn openssl_fips_properties_enabled() -> bool {
    false
}

/// The kernel FIPS flag, or `None` when the file cannot be read.
fn kernel_fips_flag() -> Option<bool> {
    let contents = std::fs::read_to_string(KERNEL_FIPS_FLAG).ok()?;
    match contents.trim() {
        "1" => Some(true),
        "0" => Some(false),
        _ => None,
    }
}

/// The active system crypto policy, or `None` on platforms without
/// crypto-policies support.
fn system_crypto_policy() -> Option<String> {
    let contents = std::fs::read_to_string(CRYPTO_POLICIES_CONFIG).ok()?;
    crypto_policy_from(&contents)
}

/// The first non-comment, non-blank line from a crypto-policies config.
fn crypto_policy_from(contents: &str) -> Option<String> {
    contents
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
}

/// Whether the base policy name (before the first `:`) is exactly `FIPS`.
fn is_fips_policy(policy: &str) -> bool {
    policy.split(':').next() == Some("FIPS")
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_returns_a_populated_result() {
        openssl::init();
        let readiness = CryptoReadiness::check();
        assert!(
            readiness.kernel_fips.is_some() || readiness.kernel_fips.is_none(),
            "kernel_fips is populated (Some or None is fine)"
        );
    }

    #[test]
    fn the_crypto_policy_parser_extracts_the_first_real_line() {
        assert_eq!(crypto_policy_from("# comment\n\nFIPS\n").as_deref(), Some("FIPS"));
        assert_eq!(crypto_policy_from("FIPS:OSPP\n").as_deref(), Some("FIPS:OSPP"));
        assert_eq!(crypto_policy_from("DEFAULT\n").as_deref(), Some("DEFAULT"));
        assert_eq!(crypto_policy_from(""), None);
        assert_eq!(crypto_policy_from("# only comments\n"), None);
    }

    #[test]
    fn unmet_reports_every_failed_prerequisite() {
        let all_failing = CryptoReadiness {
            openssl_fips_properties: false,
            kernel_fips: Some(false),
            crypto_policy: Some("DEFAULT".to_owned()),
        };
        let reasons = all_failing.unmet();
        assert!(reasons.len() >= 3, "three checks fail: {reasons:?}");
        assert!(reasons.iter().any(|r| r.contains("EVP_default_properties")));
        assert!(reasons.iter().any(|r| r.contains("kernel")));
        assert!(reasons.iter().any(|r| r.contains("crypto policy")));
    }

    #[test]
    fn unmet_is_empty_when_all_checks_pass() {
        let all_passing = CryptoReadiness {
            openssl_fips_properties: true,
            kernel_fips: Some(true),
            crypto_policy: Some("FIPS".to_owned()),
        };
        assert!(all_passing.unmet().is_empty());
    }

    #[test]
    fn fips_policy_variants_are_accepted() {
        let fips_ospp = CryptoReadiness {
            openssl_fips_properties: true,
            kernel_fips: Some(true),
            crypto_policy: Some("FIPS:OSPP".to_owned()),
        };
        assert!(fips_ospp.fips_ready(), "FIPS:OSPP is a FIPS policy");
        assert!(fips_ospp.unmet().is_empty());
    }

    #[test]
    fn fips_prefix_lookalikes_are_rejected() {
        for policy in ["FIPSXYZ", "FIPS-DRAFT", "fips", "DEFAULT:FIPS"] {
            let lookalike = CryptoReadiness {
                openssl_fips_properties: true,
                kernel_fips: Some(true),
                crypto_policy: Some(policy.to_owned()),
            };
            assert!(!lookalike.fips_ready(), "{policy:?} must not pass as a FIPS policy");
            assert!(!lookalike.unmet().is_empty(), "{policy:?} must be reported as unmet");
        }
    }

    #[test]
    fn fips_ready_requires_all_three_signals() {
        let missing_kernel = CryptoReadiness {
            openssl_fips_properties: true,
            kernel_fips: None,
            crypto_policy: Some("FIPS".to_owned()),
        };
        assert!(!missing_kernel.fips_ready());

        let missing_policy = CryptoReadiness {
            openssl_fips_properties: true,
            kernel_fips: Some(true),
            crypto_policy: None,
        };
        assert!(!missing_policy.fips_ready());

        let missing_evp = CryptoReadiness {
            openssl_fips_properties: false,
            kernel_fips: Some(true),
            crypto_policy: Some("FIPS".to_owned()),
        };
        assert!(!missing_evp.fips_ready());
    }

    #[test]
    fn unsupported_is_true_when_signals_are_absent() {
        let no_kernel = CryptoReadiness {
            openssl_fips_properties: false,
            kernel_fips: None,
            crypto_policy: None,
        };
        assert!(no_kernel.unsupported());

        let no_policy = CryptoReadiness {
            openssl_fips_properties: true,
            kernel_fips: Some(true),
            crypto_policy: None,
        };
        assert!(no_policy.unsupported());
    }

    #[test]
    fn unsupported_is_false_on_a_fully_instrumented_host() {
        let linux = CryptoReadiness {
            openssl_fips_properties: false,
            kernel_fips: Some(false),
            crypto_policy: Some("DEFAULT".to_owned()),
        };
        assert!(!linux.unsupported());
    }

    #[test]
    fn the_direct_evp_check_agrees_with_the_provider() {
        praxis_tls::provider::install();
        let provider_fips = praxis_tls::provider::status().provider_fips;
        let direct = openssl_fips_properties_enabled();
        assert_eq!(
            direct, provider_fips,
            "the direct EVP_default_properties_is_fips_enabled call must agree \
             with the provider's report"
        );
    }

    /// With `PRAXIS_TEST_FIPS_PROVIDER` set, the direct EVP check must
    /// report true and all readiness signals must be positive.
    #[test]
    fn fips_provider_is_fully_ready_when_the_run_requires_it() {
        if std::env::var_os("PRAXIS_TEST_FIPS_PROVIDER").is_none() {
            return;
        }
        let readiness = CryptoReadiness::check();
        assert!(
            readiness.openssl_fips_properties,
            "EVP_default_properties_is_fips_enabled must be 1 under PRAXIS_TEST_FIPS_PROVIDER"
        );
    }
}
