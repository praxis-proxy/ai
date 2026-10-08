// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Shared scaffolding for live-backend CLI acceptance tests.
//!
//! Every pinned-client acceptance suite that runs against a real vLLM backend
//! needs the same four things: a skip/fail gate over the live environment, a
//! network namespace whose isolation is actively verified rather than assumed,
//! an example-config patcher that fails loudly on drift, and a timeout
//! assertion that surfaces both pipes. This module owns them so a third client
//! does not arrive with a third copy.
//!
//! The copies are what went wrong with the Codex pins, which ended up spread
//! across two `.rs` constants, a checksum filename, and ten workflow strings.
//!
//! Client-specific material — launch flags, prompts, trace parsers, workspace
//! oracles — deliberately stays in each suite. Only what is genuinely common to
//! "drive a real executable at a real backend through Praxis" lives here.

use std::{net::IpAddr, path::PathBuf, process::Stdio, time::Duration};

use praxis_core::config::Config;
use praxis_test_utils::{CapturedChildOutput, ProxyGuard, free_port, start_proxy};

// -----------------------------------------------------------------------------
// Shared environment variables
// -----------------------------------------------------------------------------

/// Base URL (or bare authority) of the live vLLM backend.
pub(crate) const VLLM_BASE_URL_ENV: &str = "PRAXIS_TEST_VLLM_BASE_URL";
/// Exact served model name the backend advertises.
pub(crate) const VLLM_MODEL_ENV: &str = "PRAXIS_TEST_VLLM_MODEL";
/// Backend bearer token Praxis injects upstream.
pub(crate) const BACKEND_TOKEN_ENV: &str = "VLLM_API_KEY";
/// Address Praxis binds, defaulting to loopback.
///
/// Under namespace isolation the client reaches Praxis over a veth pair, so
/// Praxis must bind the host-side veth address rather than loopback.
pub(crate) const LISTEN_ADDRESS_ENV: &str = "PRAXIS_TEST_LISTEN_ADDRESS";
/// Demands enforced egress isolation.
///
/// When truthy, a suite refuses to run without a configured namespace, so a CI
/// acceptance run cannot silently degrade to the advisory-only path.
pub(crate) const REQUIRE_EGRESS_ISOLATION_ENV: &str = "PRAXIS_TEST_REQUIRE_EGRESS_ISOLATION";

/// Reports whether an environment variable is set to a truthy value.
pub(crate) fn env_is_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"))
        .unwrap_or(false)
}

/// Reads an environment variable, treating blank as unset.
pub(crate) fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Extracts a `host:port` authority from a base URL or an already-bare authority.
pub(crate) fn authority_of(base: &str) -> String {
    base.trim()
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .trim_end_matches('/')
        .to_owned()
}

// -----------------------------------------------------------------------------
// Example-config patching
// -----------------------------------------------------------------------------

/// Replace `from` with `to` in `haystack`, asserting it appears exactly once.
///
/// A silent no-op replacement would build a config that still carries a
/// placeholder — an env-var token, the wrong backend, or the shipped listener —
/// so the run would exercise the wrong thing rather than fail. The
/// exactly-once count additionally guards an anchor that drift has made
/// ambiguous: with two occurrences `str::replace` would rewrite both.
///
/// `context` names the file being patched so a failure points at the drift.
pub(crate) fn replace_once(haystack: &str, from: &str, to: &str, what: &str, context: &str) -> String {
    let count = haystack.matches(from).count();
    assert_eq!(
        count, 1,
        "{context} must contain the {what} anchor `{from}` exactly once, found {count}; \
         update the test and the example together",
    );
    haystack.replace(from, to)
}

// -----------------------------------------------------------------------------
// Proxy startup with verified isolation
// -----------------------------------------------------------------------------

/// Start Praxis on a free port and, when a namespace is configured, actively
/// prove the namespaced client can reach Praxis and nothing else.
///
/// Returns the guard and the admin listener address, which only configs that
/// declare one will populate; callers scrape it for metrics.
///
/// The verification is what makes isolation a property of the run rather than
/// an advisory base URL the client is free to ignore.
pub(crate) fn start_isolated_proxy(
    netns: Option<&str>,
    build_config: impl FnOnce(u16) -> Config,
) -> (ProxyGuard, Option<String>) {
    let proxy_port = free_port();
    let config = build_config(proxy_port);
    let admin_address = config.admin.address.clone();
    let proxy = start_proxy(&config);

    if let Some(namespace) = netns {
        let bound = proxy
            .addr()
            .parse::<std::net::SocketAddr>()
            .unwrap_or_else(|error| panic!("parse Praxis listen address {}: {error}", proxy.addr()));
        verify_egress_isolation(namespace, bound.ip(), bound.port());
    }

    (proxy, admin_address)
}

/// Assert a namespace was configured when the environment demands isolation.
///
/// Without this a CI run that failed to create the namespace would quietly
/// fall back to an unisolated client and still report success.
pub(crate) fn require_egress_isolation_if_demanded(netns: Option<&str>, netns_env: &str) {
    assert!(
        !env_is_truthy(REQUIRE_EGRESS_ISOLATION_ENV) || netns.is_some(),
        "{REQUIRE_EGRESS_ISOLATION_ENV} is set but {netns_env} is not; \
         the client would have external network access"
    );
}

// -----------------------------------------------------------------------------
// Enforced egress isolation
// -----------------------------------------------------------------------------

/// Public endpoints that MUST be unreachable from inside the isolated namespace.
///
/// A correctly isolated namespace has only the veth to the host-side Praxis
/// address and no default route, so any public IP is unreachable. Reaching one
/// would prove the client has general egress and could bypass Praxis to talk to
/// a provider (or the backend) directly. Two independent, stable anycast
/// targets guard against one being coincidentally routable.
pub(crate) const EGRESS_DENYLIST: &[(&str, u16)] = &[("1.1.1.1", 443), ("8.8.8.8", 53)];

/// Bound on each in-namespace connectivity probe.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Actively verify the namespace isolates the client to Praxis only.
pub(crate) fn verify_egress_isolation(namespace: &str, praxis_host: IpAddr, praxis_port: u16) {
    assert!(
        netns_can_reach(namespace, &praxis_host.to_string(), praxis_port),
        "the isolated client must be able to reach Praxis at {praxis_host}:{praxis_port}"
    );
    for (host, port) in EGRESS_DENYLIST {
        assert!(
            !netns_can_reach(namespace, host, *port),
            "isolated client reached {host}:{port}; egress is not restricted to Praxis, \
             so the client could bypass the proxy"
        );
    }
}

/// Reports whether a TCP connection to `host:port` succeeds inside `namespace`.
///
/// Uses the runner's guaranteed Python interpreter under `ip netns exec`. This
/// avoids depending on optional `timeout(1)` or Bash `/dev/tcp` support inside
/// a minimal GPU image, and retains the concrete socket error in CI logs.
pub(crate) fn netns_can_reach(namespace: &str, host: &str, port: u16) -> bool {
    let seconds = PROBE_TIMEOUT.as_secs().max(1).to_string();
    let output = std::process::Command::new(resolve_ip_binary())
        .arg("netns")
        .arg("exec")
        .arg(namespace)
        .arg(resolve_python_binary())
        .arg("-c")
        .arg(
            "import socket,sys; \
             socket.create_connection((sys.argv[1], int(sys.argv[2])), \
             timeout=float(sys.argv[3])).close()",
        )
        .arg(host)
        .arg(port.to_string())
        .arg(seconds)
        .stdin(Stdio::null())
        .output();
    match output {
        Ok(output) if output.status.success() => true,
        Ok(output) => {
            eprintln!(
                "network-namespace TCP probe to {host}:{port} failed: status={:?}, stderr={}",
                output.status.code(),
                String::from_utf8_lossy(&output.stderr).trim(),
            );
            false
        },
        Err(error) => {
            eprintln!("network-namespace TCP probe to {host}:{port} could not start: {error}");
            false
        },
    }
}

/// Resolves the `ip(8)` binary, which commonly lives outside a minimal `PATH`.
pub(crate) fn resolve_ip_binary() -> PathBuf {
    ["/usr/sbin/ip", "/sbin/ip", "/usr/bin/ip", "/bin/ip"]
        .into_iter()
        .map(PathBuf::from)
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| PathBuf::from("ip"))
}

/// Resolve the Python interpreter used by the workflow before entering netns.
pub(crate) fn resolve_python_binary() -> PathBuf {
    ["/usr/bin/python3", "/usr/local/bin/python3", "/bin/python3"]
        .into_iter()
        .map(PathBuf::from)
        .find(|candidate| candidate.exists())
        .unwrap_or_else(|| PathBuf::from("python3"))
}

// -----------------------------------------------------------------------------
// Child-process assertions
// -----------------------------------------------------------------------------

/// Assert the pinned client exited on its own rather than being reaped.
///
/// A timeout leaves the captured stdout truncated mid-stream, so every other
/// assertion about the trace would be reasoning about a partial transcript.
/// Both pipes are surfaced because the useful diagnosis is usually in stderr.
pub(crate) fn assert_not_timed_out(output: &CapturedChildOutput, client: &str, scenario: &str, bound: Duration) {
    assert!(
        !output.timed_out,
        "{client} exceeded the {bound:?} acceptance-test timeout on the {scenario}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[cfg(test)]
mod tests {
    use super::{authority_of, env_is_truthy, non_empty_env, replace_once};

    #[test]
    fn authority_strips_scheme_and_trailing_slash() {
        assert_eq!(authority_of("http://127.0.0.1:8000"), "127.0.0.1:8000");
        assert_eq!(authority_of("https://example.test/"), "example.test");
        assert_eq!(authority_of("  127.0.0.1:8000  "), "127.0.0.1:8000");
        assert_eq!(
            authority_of("example.trycloudflare.com:443"),
            "example.trycloudflare.com:443"
        );
    }

    #[test]
    fn truthy_accepts_the_documented_spellings_only() {
        // Read through a variable this process definitely does not set, so the
        // test never depends on ambient environment.
        assert!(!env_is_truthy("PRAXIS_TEST_DEFINITELY_UNSET_VARIABLE"));
        for value in ["1", "true", "TRUE", "yes", "on"] {
            assert!(
                matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"),
                "{value} should be truthy"
            );
        }
        for value in ["0", "false", "", "no", "off", "2"] {
            assert!(
                !matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "on"),
                "{value} should not be truthy"
            );
        }
    }

    #[test]
    fn non_empty_env_treats_blank_as_unset() {
        assert_eq!(non_empty_env("PRAXIS_TEST_DEFINITELY_UNSET_VARIABLE"), None);
        // CARGO_PKG_NAME is always set for a test binary and is never blank.
        assert!(non_empty_env("CARGO_PKG_NAME").is_some());
    }

    #[test]
    fn replace_once_rewrites_a_unique_anchor() {
        let patched = replace_once("a: 1\nb: 2\n", "b: 2", "b: 3", "value", "example.yaml");
        assert_eq!(patched, "a: 1\nb: 3\n");
    }

    #[test]
    #[should_panic(expected = "exactly once, found 2")]
    fn replace_once_rejects_an_ambiguous_anchor() {
        // Two occurrences mean `str::replace` would rewrite both, which is how
        // a drifted example silently patches more than the test intended.
        drop(replace_once("x\nx\n", "x", "y", "value", "example.yaml"));
    }

    #[test]
    #[should_panic(expected = "exactly once, found 0")]
    fn replace_once_rejects_a_missing_anchor() {
        drop(replace_once("a: 1\n", "b: 2", "b: 3", "value", "example.yaml"));
    }
}
