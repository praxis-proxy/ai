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
| Cargo features | `full` | `openai-responses`, `aws-sigv4-filter` |
| Responses API kernel (`openai_responses_*`, `responses_to_chat_completions`, agentic loop, file and web search dispatch) | yes | yes |
| `aws_sigv4_sign` filter (AWS request signing) | yes | yes: SHA-256 and HMAC-SHA256 through OpenSSL |
| `policy` filter (policy engine) | yes | no: its dependencies carry their own cryptography |
| Response store (`store-postgres`), Conversations API, context compaction, MCP tools | yes | no: `sqlx` pulls `sha2` for migration checksums, and PostgreSQL authentication is pure Rust |
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

The requirement also covers what the binary itself carries: a build that
registers a filter whose dependencies do their own cryptography outside the
system OpenSSL (the `policy` filter's plugins, the response store's sqlx)
refuses `PRAXIS_REQUIRE_FIPS` outright, naming the filters, whatever the
provider reports. Only the FIPS build, which compiles none of them, can
honor the variable; the standard image is refused by design.

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
- **AWS request signing** (`aws_sigv4_sign`) is one: the payload and
  canonical-request SHA-256, the four-step HMAC-SHA256 key derivation and
  the signature all run through OpenSSL's EVP digest and signing APIs, so on
  a FIPS host they execute inside the validated module. The `SigV4`
  canonicalization itself is protocol text assembly in praxis-ai, checked
  in its tests against the `aws-sigv4` crate, which the binary does not
  link.

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

On the FIPS host, what only it can prove (rootless podman required):

```console
make fips-host-check     # attest the host and the image's module build (target/fips/host-attestation.*)
make test-fips-host      # the test suites as the FIPS build, inside the UBI 9 toolchain image, fail-closed on FIPS mode
make fips-runtime-probe  # run the FIPS image under PRAXIS_REQUIRE_FIPS=1 and probe its listener from outside
```

`fips-host-check` states every fact the module's Security Policy requires of
the host (the kernel flag, `fips=1` on the command line, the `FIPS` crypto
policy, `fips-mode-setup --check`, the module the host's OpenSSL loads) and
of the image (the crypto policy podman propagates into it, the build of
`fips.so` it carries and whether that build is on a CMVP certificate), and
writes the attestation to `target/fips/` to keep with the deployment record.
`test-fips-host` runs the suites with `PRAXIS_FIPS_HOST=1`, so a green run
cannot have happened outside FIPS mode. `fips-runtime-probe` starts the
shipped image itself under `PRAXIS_REQUIRE_FIPS=1`, drives raw TLS probes
against its listener (approved algorithms negotiated, ChaCha20-only and
X25519-only clients refused), and checks the startup line. The CI `FIPS`
workflow runs all three on a RHEL 9 runner in FIPS mode for every change;
a release requires a recorded green `fips-host` run for the exact commit
being released, and the release workflow attests and probes the exact
pushed image, pulled back by digest, on the release run itself.

A hand check of the shipped image remains a two-liner:

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

## What is validated, and what is not

FIPS 140-3 validates a cryptographic module, not an application. What the
checks above prove is that praxis-ai routes its cryptography through the
OpenSSL provider the image carries and behaves as a FIPS deployment must.
Three facts stay outside what this repository can prove:

- **The module build.** A certificate names one exact build of
  `fips.so`. The pinned UBI 9 images carry a build rebuilt for a CVE fix
  that is still in validation with NIST; `fips-host-check` says so on every
  run (a warning, or a failure under `--require-certified`) rather than
  letting the package name imply a certificate. See
  `xtask/assets/fips/certified-modules.json` for the builds and sources.
- **The operating environment.** The certificate lists tested operating
  environments; running elsewhere relies on the CMVP porting rules.
- **The architecture.** rustls performing the TLS handshake over the
  validated provider is praxis's architecture, documented and tested here;
  whether a compliance program accepts it is that program's judgment, not a
  fact this repository can assert.

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
| `aws_sigv4_sign` filter | SHA-256 and HMAC-SHA256 for `SigV4` through OpenSSL (`praxis_ai_apis::hash`) | compliant; the `aws-sigv4` crate (RustCrypto `hmac`/`sha2`) is a test-only dependency |
| response stores, Conversations, compaction, MCP tools | sqlx's sha2 (migration checksums), sqlx-postgres' md-5/hmac/sha2/hkdf/rsa (SCRAM) | not in the FIPS build |
| `openai_file_resolve`, `azure_ad`, `gcp_adc`, MCP tool dispatch | reqwest over rustls with no bundled provider (TLS through the installed OpenSSL-backed provider); MCP tool dispatch additionally requires the store | not in the FIPS build |
| `basic_auth` filter (praxis core) | password hashing through OpenSSL's SHA-256 (EVP) | compliant; experimental in praxis-ai and off in every build unless enabled |
| sha2, hmac, aws-sigv4, rcgen (with ring) | test utilities, fixtures, xtask and the `SigV4` test oracle | development only, absent from the shipped binary and its manifest; aws-lc-rs itself is gone from every graph, the policy engine aside |

The report and Red Hat's scanner both confirm the last row on every build:
the embedded crate manifest lists none of the denied crates, and the binary
defines no symbol of a bundled crypto backend.
