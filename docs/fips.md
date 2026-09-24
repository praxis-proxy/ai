# FIPS 140-3

Praxis AI performs all of its cryptography in the system OpenSSL library. On
a Red Hat Enterprise Linux 9 host in FIPS mode that library's provider is the
validated module (Red Hat Enterprise Linux 9 OpenSSL FIPS Provider, CMVP
certificate #4857 at the time of writing; Red Hat keeps the current list at
<https://access.redhat.com/compliance/fips>), so a praxis-ai running there
does its TLS, hashing and random numbers inside a FIPS 140-3 validated
boundary. Praxis AI itself is not a validated module and never enables FIPS
mode on its own: FIPS mode comes from the host, and praxis-ai reports and, on
request, enforces it.

This page is for operators deploying the FIPS build. How the build is checked
and how the tooling works is in [FIPS Tooling](developing/fips.md).

## The FIPS build

The published image (`make release`, `make container`) enables every
non-experimental filter (`full`). The FIPS build turns off what is known not
to be FIPS compliant yet, so nobody has to know which features to pick:

| | Standard | FIPS |
|---|---|---|
| Make targets | `release`, `container` | `release-fips`, `container-fips` |
| Cargo features | `full` | `openai-responses,store-postgres-cert-auth` |
| Responses API kernel (`openai_responses_*`, `responses_to_chat_completions`, agentic loop, file and web search dispatch) | yes | yes |
| `aws_sigv4_sign` filter | yes | no: the `aws-sigv4` crate signs with pure-Rust `hmac`/`sha2` |
| `policy` filter (policy engine) | yes | no: its dependencies carry their own cryptography |
| PostgreSQL Responses store | yes | yes: certificate-only SQLx profile through system OpenSSL |
| SQLite store, Conversations API, context compaction, MCP tools | yes | no: their additional dependency boundaries have not been cleared for this profile |
| `openai_file_resolve`, `azure_ad`, `gcp_adc`, MCP tool dispatch | `full` / experimental | no: `reqwest` bundles its own TLS provider (`aws-lc-rs`) |
| Base image | Alpine, praxis-ai built with upstream Rust | `ubi9/ubi-minimal`, praxis-ai built with Red Hat's `rust-toolset` on `ubi9/ubi`, both pinned by digest and signature-verified |
| OpenSSL | Alpine's, dynamically linked | UBI's, dynamically linked (`openssl-libs` and `openssl-fips-provider-so`), the validated module on a FIPS host |
| Published tags | `<version>`, `<major>.<minor>`, `sha-<hash>` (see [image tags](release.md#image-tags)) | the same with a `-fips` suffix (`0.3.0-fips`) |

A configuration that names one of the excluded filters is rejected at
startup by the FIPS build, as any unknown filter is. Everything else,
including the Anthropic Messages translation, guardrails, metering, identity
and credential filters, the agentic protocol filters, TLS listeners, mTLS,
SNI, the admin API and hot reload, behaves as in the standard build. What
blocks each excluded feature, and how far away it is, is tracked in the
Makefile's FIPS section.

The binary links `libcrypto.so.3` and `libssl.so.3` dynamically. The only
crates performing security-relevant cryptography in the image are the rustls
protocol engine and the OpenSSL bindings that delegate every primitive to the
system library; the rest of the crypto-adjacent crates in the image are
listed in the exemption table at the end of this page.

The certificate-only store currently pins SQLx to commit `6736b97d` from a
temporary fork. That commit makes migrations, PostgreSQL password
authentication, and advisory-lock string hashing independently optional.
The pin can return to an upstream release after [transact-rs/sqlx#4417],
[transact-rs/sqlx#4420], and [transact-rs/sqlx#4421] land and are released.

## Host prerequisites

- A RHEL 9 host in FIPS mode, enabled at install time or with
  `fips-mode-setup --enable` and a reboot: `cat /proc/sys/crypto/fips_enabled`
  prints `1` and `openssl list -providers` lists `fips`. RHEL 9 is the
  validated operating environment; the module that runs inside the container
  is UBI's own `openssl-fips-provider-so`, the host contributes the kernel
  flag.
- A container runtime that passes the host's FIPS mode into the container,
  as podman and CRI-O on RHEL do. The `-fips` image then needs no flag,
  environment variable or config: the OpenSSL inside it reads the kernel flag
  and activates the validated provider by itself. On a host that is not in
  FIPS mode the same image runs with OpenSSL's default provider, and the
  startup log says so.

## Startup: what praxis-ai reports and enforces

At startup praxis-ai installs its one crypto provider and logs what it found:

```text
installed rustls crypto provider provider=openssl provider_fips=true kernel_fips=Some(true) fips_required=true
```

- `provider_fips`: whether OpenSSL's default properties select FIPS-approved
  algorithms only (`EVP_default_properties_is_fips_enabled`), which is what
  RHEL's FIPS mode configures.
- `kernel_fips`: `/proc/sys/crypto/fips_enabled`; `None` where the file does
  not exist (a container without `/proc`, a non-Linux host).

Set `PRAXIS_REQUIRE_FIPS=1` (also `true`, `yes`, `on`) in production FIPS
deployments. It is a check, never a switch: praxis-ai then refuses to start
unless both signals are present, naming each one that is missing, and
refuses any listener TLS configuration that rustls does not consider
FIPS-approved. Upstream connections always require Extended Master Secret
and share the same provider, so they are FIPS whenever the listeners are.
Without the variable praxis-ai starts either way and only logs the status.

```console
podman run --rm -e PRAXIS_REQUIRE_FIPS=1 ghcr.io/praxis-proxy/ai:0.3.0-fips
```

On a host that is not in FIPS mode this exits immediately with

```text
fatal: PRAXIS_REQUIRE_FIPS is set but FIPS mode is not in effect: the OpenSSL provider does not report FIPS-approved algorithms (is the fips provider active?); the kernel is not in FIPS mode (/proc/sys/crypto/fips_enabled is 0)
```

## What changes under FIPS mode

- **TLS 1.2 requires the Extended Master Secret extension (RFC 7627)** on
  listeners and on upstream connections, in every build and regardless of
  FIPS mode; a TLS 1.2 peer that cannot negotiate it fails the handshake.
  TLS 1.3 is unaffected. This is what FIPS 140-3 requires of TLS 1.2 key
  derivation, and rustls counts a configuration as FIPS only with it.
- **The provider offers only what the system OpenSSL can perform.** In FIPS
  mode non-approved algorithms are absent rather than failing later: for
  example the ChaCha20-Poly1305 cipher suites are not offered, MD5 does not
  exist, and keys or certificates the module refuses (short RSA keys, legacy
  signature algorithms) are rejected at load or at first use. Which
  algorithms are approved is decided by the validated module and the host's
  crypto policy, not by praxis-ai. A listener whose `cipher_suites` names
  only suites the module does not offer fails to build in every build and
  mode, since it would have nothing to negotiate.
- **Random numbers** come from OpenSSL's DRBG (`RAND_priv_bytes`) for every
  TLS operation. Non-security randomness (load-balancer picks, request ids)
  uses ordinary Rust RNGs, which is fine: they protect nothing.
- **Digests praxis-ai computes itself** (the Anthropic `user_id` that becomes
  `safety_identifier`, routing descriptor ids, MCP approval fingerprints,
  token-rate-limit bucket keys, overlay content hashes) go through OpenSSL's
  SHA-256 in every build. None of them is a security function.

## Verifying a deployment

On a developer machine (no FIPS host needed):

```console
make fips-signature-store  # once on Debian/Ubuntu: their podman has no entry for Red Hat's signature store
make fips-check    # build on UBI 9 with Red Hat's toolchain, print the compliance report
make container-fips
make fips-scanner  # build Red Hat's scanner (check-payload) at its pinned revision; needs Go
make fips-oc       # download the OpenShift CLI the scanner insists on, checksum-verified
make fips-scan     # run the scanner against the image, warnings fatal
```

On the FIPS host, the two things only it can prove:

```console
cat /proc/sys/crypto/fips_enabled                        # 1
podman run --rm -e PRAXIS_REQUIRE_FIPS=1 --entrypoint praxis-ai \
    ghcr.io/praxis-proxy/ai:<version>-fips --validate
```

The second command validates the built-in configuration (pass `-c` with a
mounted file to validate yours) and exits 0 only when the provider and the
kernel both report FIPS mode; it prints nothing in that case. The listener
TLS configurations are checked, and the startup line above is logged, when
the real workload starts, so run it the same way with `PRAXIS_REQUIRE_FIPS=1`
and keep that line as evidence.

What the local checks cannot prove, and only a FIPS host can: the kernel flag
and the positive `PRAXIS_REQUIRE_FIPS` path, RHEL's boot-time module
integrity self-tests, and behaviour under the host-wide `FIPS` crypto policy
(the local checks activate the provider, not the policy).

## Scope and exemptions

Every crypto-adjacent component in the FIPS image, and why it is compliant:

| Component | Use | Disposition |
|---|---|---|
| rustls, rustls-webpki, rustls-pki-types, rustls-pemfile, rustls-native-certs, tokio-rustls | TLS protocol engine, X.509 path building, PEM parsing, system roots; no cryptography of their own | compliant through the OpenSSL provider |
| rustls-openssl (published from the Pingora fork as `quixotic-plecostomus-rustls-openssl`), openssl, openssl-sys, openssl-probe | the provider and the bindings; dynamic link to the system `libcrypto.so.3` | compliant |
| rand, rand_chacha, rand_xoshiro, chacha20 | request ids, load-balancer picks, jitter (rand's ChaCha-based RNG) | not security functions |
| ahash, crc32fast, blake2, digest, crypto-common | hash maps, gzip checksums, Pingora cache keys | not security functions |
| x509-parser, asn1-rs, der-parser, oid-registry (parsing only, no `verify` feature) | peer certificate fields in the Pingora fork | parse only |
| subtle, zeroize, secrecy | constant-time comparison, wiping, secret wrappers | helpers |
| policy engine (`policy` filter) | JWT, OAuth, Valkey builtins carry aws-lc, sha2 and hmac | not in the FIPS build |
| `aws_sigv4_sign` filter | the `aws-sigv4` crate's hmac/sha2 | not in the FIPS build |
| PostgreSQL Responses store | SQL transport and client-certificate authentication | compliant through system OpenSSL; migrations, password authentication, and advisory-lock hashing are not compiled |
| SQLite store, Conversations, compaction, MCP tools | additional database or request-client dependencies | not in the FIPS build |
| `openai_file_resolve`, `azure_ad`, `gcp_adc`, MCP tool dispatch | reqwest's bundled rustls provider (aws-lc-rs) | not in the FIPS build |
| `basic_auth` filter (praxis core) | password hashing through OpenSSL's SHA-256 (EVP) | compliant; experimental in praxis-ai and off in every build unless enabled |
| sha2, rcgen, aws-lc-rs | test utilities, fixtures and xtask | development only, absent from the shipped binary and its manifest |

[transact-rs/sqlx#4417]: https://github.com/transact-rs/sqlx/pull/4417
[transact-rs/sqlx#4420]: https://github.com/transact-rs/sqlx/pull/4420
[transact-rs/sqlx#4421]: https://github.com/transact-rs/sqlx/issues/4421

The report and Red Hat's scanner both confirm the last row on every build:
the embedded crate manifest lists none of the denied crates, and the binary
defines no symbol of a bundled crypto backend.
