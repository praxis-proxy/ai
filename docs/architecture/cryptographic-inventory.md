<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright (c) 2026 Praxis Contributors -->

# Cryptographic operations inventory

A human-readable inventory of where the Praxis AI proxy performs cryptography in
production, which provider executes each operation, and its FIPS disposition
(#1220). This document is the findings record; it is **not** a generated artifact
and no checker re-derives it. The machine-enforced invariants that keep the
provider story from silently regressing live in two places:

- **Feature-level drift** (a pinned provider rerouting off OpenSSL) is caught by
  `[[bans.features]]` in [`deny.toml`](../../deny.toml), run on every PR by
  `cargo deny check` (Supply Chain workflow).
- **A non-OpenSSL crypto backend crate entering the FIPS image** is caught by
  `cargo xtask fips` (`xtask/src/fips/graph.rs::DENIED`), run on every PR via
  `make lint` (`fips-deps`) and the FIPS workflow.

Pinned crate versions below are documentary — `Cargo.lock` is the source of
truth. Keep this file current by hand when a cryptographic operation or provider
changes; drift here breaks no build (the guards above do the enforcing).

## Build profiles

The proxy ships two profiles and the inventory tracks both:

- **full** — the published container image (`PRAXIS_AI_FEATURES ?= full`:
  `standard` + `openai-all` + `store-postgres`).
- **fips** — the reduced FIPS runtime, built `--no-default-features` with the
  feature set shared from `xtask/src/fips.rs::FIPS_FEATURES`
  (`openai-responses,openai-file-resolve-filter,aws-sigv4-filter,store-postgres-cert-auth`).

`fips ⊆ full`: the FIPS runtime is a subset of the published image. FIPS itself
is **host-activated** (see [`docs/fips.md`](../fips.md)); no build feature
asserts validation.

## Provider selection

The central decision is **OpenSSL everywhere**:

- All TLS (data plane, listeners, AI callouts, the policy Valkey/Redis store, the
  PostgreSQL store) runs on **rustls with the installed `rustls_openssl`
  CryptoProvider** — `custom-provider`, no bundled `ring`/`aws-lc-rs` backend.
  The provider is installed process-wide at boot and every subsequent rustls
  primitive (suites, key exchange, signatures, digests, MACs, RNG) executes on
  OpenSSL `libcrypto`.
- All first-party application hashing (AWS SigV4, MCP approval fingerprints,
  routing digests, the Anthropic `safety_identifier`) goes through
  `apis::hash` = **OpenSSL EVP**.

Deliberate exceptions, each scoped and documented below:

- **aws-lc-rs** executes JWT signature verification only, and only in **full**
  (via `jsonwebtoken`); it is never the TLS executor and must never gain `fips`.
- **RustCrypto `sha2`/`hmac`** drive PostgreSQL SCRAM and some upstream policy
  digests (application-side, full-only) — not the OpenSSL EVP seam.
- **native-tls** backs the PostgreSQL store TLS channel (diverges from the
  rustls data plane; see #1217), resolving to OpenSSL on Linux and the OS TLS
  stack on macOS/Windows.

### Pinned providers

| Crate | Version | Profiles | Pinned constraint |
| --- | --- | --- | --- |
| `quixotic-plecostomus-rustls-openssl` | 0.4.1 | full, fips | requires `tls12`; forbids `fips` |
| `openssl` | 0.10.81 | full, fips | forbids `vendored` (`OPENSSL_NO_VENDOR=1`) |
| `rustls` | 0.23.45 | full, fips | requires `custom-provider`; forbids `ring`, `aws_lc_rs` |
| `aws-lc-rs` | 1.18.1 | full | requires `aws-lc-sys`; forbids `fips` |
| `jsonwebtoken` | 11.1.0 | full | requires `aws_lc_rs` |
| `native-tls` | 0.2.18 | full, fips | version pin only |

The `forbid`/`require` columns are exactly the `[[bans.features]]` entries in
`deny.toml`; unlisted extra features are allowed by design (feature sets grow
across patch releases).

## Operations

Disposition legend: **non-validated** = a FIPS-approvable algorithm, validated
only on a FIPS-activated host (remediation: `docs/fips.md`, unless noted);
**n/a** = encoding/identity/non-security use, no validated primitive required;
**upstream-owned** = implemented and maintained in a Praxis-core/policy crate.

### Transport TLS, certificate verification & handshake RNG

| ID | Operation | Algorithm | Provider | Profiles | Disposition |
| --- | --- | --- | --- | --- | --- |
| T1 | rustls CryptoProvider install (process-default) | `rustls_openssl` as the sole provider (no `ring`) | OpenSSL via `quixotic-plecostomus-rustls-openssl` | full, fips | non-validated |
| T2 | Upstream data-plane TLS (proxy → LLM) | TLS 1.2/1.3 client handshake | OpenSSL (installed provider, pingora connector) | full, fips | non-validated |
| T3 | AI subrequest callout TLS (web_search, file_search, guardrails, compact) | TLS 1.2/1.3 over `SubRequestConnector` | OpenSSL (installed provider) | full, fips | non-validated |
| T4 | Downstream listener TLS (client → proxy) | TLS 1.2/1.3 server handshake; optional WebPKI client-cert | OpenSSL (installed provider, `ServerConfig`) | full, fips | non-validated |
| T5 | `file_resolve` `file_url` download TLS (SSRF-guarded, redirect-free) | TLS 1.2/1.3 over `SubRequestConnector` | OpenSSL (installed provider) | full, fips | non-validated |
| T7 | Policy Valkey/Redis session-store TLS (`rediss://`) | TLS 1.2/1.3 via `redis` `tls-rustls` | OpenSSL (installed provider, `tokio-rustls`) | full | non-validated |
| T9 | TLS handshake CSPRNG | handshake randoms + key material | OpenSSL `RAND_bytes` (provider `SecureRandom`) | full, fips | non-validated |
| C1 | Upstream + sub-request server-cert chain & hostname verify | X.509 path + `CertificateVerify` + hostname match | `rustls-webpki` (chain) + OpenSSL (signatures) | full, fips | non-validated |
| C2 | Downstream mTLS client-cert verify | X.509 client-cert path + signature (optional CRL) | `rustls-webpki` + OpenSSL | full, fips | non-validated |
| C5 | OS native trust-store load | load system CA anchors (no primitive) | `rustls-native-certs` (+ `security-framework` macOS, `schannel` Windows) | full, fips | n/a |
| C6 | X.509 peer/chain structural parse | ASN.1 X.509 v3 DER parse (no verify) | `x509-parser` → `der-parser` → `asn1-rs` (rusticata) | full, fips | upstream-owned |

### Certificate / key-material handling

| ID | Operation | Algorithm | Provider | Profiles | Disposition |
| --- | --- | --- | --- | --- | --- |
| C3 | Listener cert + private-key PEM load | PEM→DER decode; PKCS8/PKCS1/SEC1 parse; zeroized on drop | `rustls-pki-types` + `zeroize` | full, fips | n/a |
| C4 | Upstream cert SHA-256 fingerprint (telemetry) | SHA-256 over DER cert (non-secret) | OpenSSL (installed provider) | full, fips | n/a |
| C7 | JWT verification-key ASN.1/PEM parse | PKCS1/PKCS8/SEC1 DER + PEM decode | `simple_asn1` + `pem` (via `jsonwebtoken`) | full | n/a |

### Application hashing & authentication

| ID | Operation | Algorithm | Provider | Profiles | Disposition |
| --- | --- | --- | --- | --- | --- |
| A1 | JWT signature verification (identity authn) | HS256/RS256/PS256/ES256/EdDSA verify (decode-only) | `aws-lc-rs` 1.18.1 (no `fips`) via `jsonwebtoken` | full | non-validated |
| A2 | OAuth delegated-token cache-key derivation | keyed HMAC-SHA256 (per-process secret) | RustCrypto `hmac` + `sha2` (upstream policy) | full | upstream-owned |
| A3 | AWS SigV4 signing-key + request signature | HMAC-SHA256 signing-key chain (RFC 2104) | OpenSSL EVP via `apis::hash` | full, fips | non-validated |
| A4 | AWS SigV4 payload & canonical-request hash | SHA-256 | OpenSSL EVP via `apis::hash` | full, fips | non-validated |
| A5 | PostgreSQL SCRAM-SHA-256 SASL auth | SHA-256 / HMAC / PBKDF2 (RFC 5802/7677) | `sqlx-postgres` (RustCrypto `sha2` + `hmac`) | full | non-validated |
| A6 | PostgreSQL SCRAM SASLprep normalization | RFC 4013 Unicode normalization (no primitive) | `stringprep` via `sqlx-postgres` | full | n/a |
| A7 | PostgreSQL SCRAM client-nonce RNG | ChaCha20 CSPRNG (OS `getrandom`-seeded) | `rand` → `chacha20` via `sqlx-postgres` | full | non-validated |
| A8 | PostgreSQL legacy MD5 password auth | `md5(md5(pw‖user)‖salt)` | `md-5` via `sqlx-postgres` | full | non-validated |
| A9 | PostgreSQL cleartext password auth | cleartext (confidentiality via A10 TLS) | `secrecy`-wrapped (no local primitive) | full | n/a |
| A10 | PostgreSQL store TLS handshake + cert verify | TLS 1.2/1.3, X.509 chain + hostname (`verify-full`) | `sqlx-postgres` → `native-tls` → OpenSSL / OS TLS | full, fips | non-validated (#1217) |
| A11 | PostgreSQL store client-cert mutual TLS | TLS mutual auth with client PEM cert/key | `sqlx-postgres` → `native-tls` → platform TLS | full, fips | non-validated (#1217) |
| A12 | HTTP Basic Auth credential verify | unsalted SHA-256, constant-time compare | OpenSSL EVP SHA-256 + `subtle` (core builtin) | full, fips | upstream-owned |

### Identity & change-detection digests

Non-secret SHA-256 digests via OpenSSL EVP (`apis::hash`) unless noted.

| ID | Operation | Algorithm | Provider | Profiles | Disposition |
| --- | --- | --- | --- | --- | --- |
| H1 | MCP approval target-identity fingerprint (binds approval → destination) | SHA-256, length-framed, key-sorted | OpenSSL EVP | full | non-validated |
| — | MCP approval owner / credential / forwarded-header sub-fingerprints (components of H1) | domain-separated SHA-256 | OpenSSL EVP | full | non-validated |
| H2 | Routing candidate `stable_id` fallback | SHA-256 (fallback only) | OpenSSL EVP | full, fips | n/a |
| H3 | Routing overlay change-detection hash | SHA-256 over file bytes | OpenSSL EVP | full, fips | n/a |
| H4 | Routing overlay semantic self-check | SHA-256 over JCS/RFC-8785 subset | OpenSSL EVP | full, fips | n/a |
| — | Anthropic `user_id` → `safety_identifier` | SHA-256 lowercase hex (pseudonymization) | OpenSSL EVP | full, fips | n/a |
| — | SSE item change-digest; MCP `tools/list` opaque id | 64-bit SipHash-1-3 (std `DefaultHasher`) | Rust std (non-crypto) | full/[full] | n/a |
| P1 | Policy session-identity binding | SHA-256 (anti-forgery) | RustCrypto `sha2` (upstream policy) | full | upstream-owned |
| P2 | Valkey session-store key-name derivation | SHA-256(`session_id`) (non-secret) | RustCrypto `sha2` (upstream policy) | full | n/a |

### Random number generation & identifiers

| ID | Operation | Algorithm | Provider | Profiles | Disposition |
| --- | --- | --- | --- | --- | --- |
| R-1 | Load-balancing candidate selection RNG | ChaCha thread-RNG uniform draw | `rand` → `chacha20` → `getrandom` | full, fips | n/a |
| R-2 | Correlation / idempotency IDs (`resp_`, `conv_`, …) | microsecond clock + one-time seed + counter | praxis-core `IdGenerator` | full, fips | upstream-owned |
| R-3 | MCP session id (`mcp-session-id`) | predictable `IdGenerator` id | praxis-core `IdGenerator` | full, fips | upstream-owned |
| R-4 | OAuth delegator per-process cache-key secret | 32-byte OS CSPRNG draw (keys A2) | `getrandom` (upstream policy) | full | upstream-owned |
| R-5 | Prometheus metrics-exporter sampling RNG | ChaCha CSPRNG (metrics sampling) | `rand` 0.9 → `rand_chacha` | full, fips | n/a |

## Reviewed non-primitives

Several crypto-family-named crates resolve into the graph but are **not**
standalone operations — support/trait layers (`digest`, `crypto-common`,
`const-oid`, `untrusted`), build-host code generators (`*-derive`,
`openssl-macros`, `tiny-keccak`), FFI companions, and naming coincidences
(`aws-smithy-*`, `openssl-probe`, `spin`, `blake2` as a non-secret Pingora cache
key, `rand_xoshiro` for metrics sampling). None executes a security primitive on
the request path. A crate on the FIPS denylist may never be classified here — a
denied name is a real primitive and is tracked as an operation or blocked by
`cargo xtask fips`.
