# FIPS Tooling

Local, reproducible checks that a praxis-ai build is on the path to FIPS
140-3 compliance on Red Hat Enterprise Linux. The checks mirror what Red Hat's
release scanner (`openshift/check-payload`, Rust support in its PR #360) looks
at, so a clean local report is a strong predictor of a clean scan. The build
targets themselves (`make release-fips`, `make container-fips`) are described
in [Getting Started](getting-started.md#fips-build-and-compliance-check).

The Makefile installs the pinned [cargo-fips-check] tool on first use for
compliance reports, image verification, signature-store setup, and host
attestation. Make installs the binary under
`target/fips/tools/<revision>/bin/`. Set `FIPS_CHECK` to use a particular
binary, or add the installed binary's directory to `PATH` to invoke
`cargo fips-check` directly. The hosted `ubi-image` job and the UBI report
stage use the same pinned revision. `fips-check.toml` defines
the Praxis AI package, feature profile, source guards, and test graph.
`cargo xtask fips runtime-probe` drives the Praxis AI listener checks.

[cargo-fips-check]: https://github.com/praxis-proxy/cargo-fips-check

| Command | Makefile | Purpose |
|---|---|---|
| `cargo fips-check report [--deps-only] [--features LIST] [--offline] [--out FILE] [BINARY]` | `fips-deps`, `fips-report`, `fips-check` | The compliance report: environment, dependency graph, binary structure, source guards, with a reason and a pointer for every finding. Exit status 1 while findings remain. |
| `cargo fips-check verify-image REFERENCE` | `fips-verify-image` (run by `container-fips` and `fips-check` first) | Refuses any base image that is not digest-pinned, from `registry.access.redhat.com`, and signed by Red Hat's release key. |
| `cargo fips-check signature-store [--install]` | `fips-signature-store` | Whether podman's `registries.d` names Red Hat's signature store, without which every Red Hat image looks unsigned; `--install` adds the bundled entry for the current user on hosts whose podman packaging ships none (Debian, Ubuntu, GitHub's runners). CI runs it before `fips-verify-image`. |
| `check-payload scan image ...` | `fips-scanner`, `fips-oc`, `fips-scan` | Red Hat's own scanner at a pinned revision, run against the FIPS image with warnings fatal: the actual gate. |
| `cargo fips-check host-check [--image IMAGE] [--require-certified] [--out FILE] [--json FILE]` | `fips-host-check` | The FIPS-host attestation: the kernel flag, `fips=1` on the command line, the `FIPS` crypto policy, `fips-mode-setup --check`, the module the host's OpenSSL loads; with `--image`, the crypto policy podman propagates into the container and the build of `fips.so` inside it, graded against the tool’s certified-module inventory. Exit 1 on any unmet requirement; a module in validation warns unless `--require-certified`. |
| `cargo xtask fips runtime-probe IMAGE [--toolchain-image IMAGE] [--host-cargo] [--log FILE]` | `fips-runtime-probe` | Runs the shipped FIPS image under `PRAXIS_REQUIRE_FIPS=1` with a generated listener config, waits for a real TLS handshake, drives the `fips::listener_` probes of the integration suite against it (from the toolchain image, or with the host's cargo), and checks the startup status line reports FIPS on both signals. |
| `PRAXIS_TEST_FIPS_PROVIDER=1 cargo test ... --config 'target."cfg(all())".runner=["env","OPENSSL_CONF=$PWD/xtask/assets/fips/fips-provider.cnf"]'` | `test-fips-provider` | The FIPS feature set's unit tests with the RHEL FIPS provider active (`fips=yes` default properties) in every test process, so the digests and MACs the code computes, `aws_sigv4_sign`'s HMAC-SHA256 included, have to come from the provider. The runner sets `OPENSSL_CONF` on the test binaries only: cargo itself (its libgit2) cannot run under it, which is also why only lib and bin unit tests run. The path must be absolute (cargo runs each test binary from its package directory) and OpenSSL silently ignores a configuration file it cannot find, so `PRAXIS_TEST_FIPS_PROVIDER` makes the apis and filters test processes assert that the provider reports FIPS-approved default properties and refuses MD5; a run that never reached the provider fails instead of passing. Needs the host's `fips.so`. The UBI 9 report stage (`fips-check`) runs the crypto tests (`hash::`, `aws::`) the same way. |

## What the report checks

1. **Dependency graph**: no crate on the scanner's `rust_denied_crypto` list
   (`ring`, `aws-lc-rs`, `sha2`, `hmac`, ...) in the shipped binary's normal
   and build dependency graph, resolved for the assessed feature set. The
   report also checks test-only dependencies against the reasons in
   `fips-check.toml`. `make lint` runs these checks and the source guards
   via `--deps-only`. The report points to Praxis AI features that commonly
   pull in denied crates (the general-purpose SQLx profile, the
   reqwest-based filters, the policy engine), so the fix is usually a line in
   `FIPS_FEATURES`. A `sha2` or `hmac` finding that names `aws-sigv4` means
   the crate escaped `dev-dependencies`: `aws_sigv4_sign` signs through
   OpenSSL and only tests against it. The FIPS profile uses SQLx's
   certificate-only PostgreSQL feature set, which omits its direct
   cryptographic operations.
2. **Binary**: links the system `libcrypto.so.3` dynamically, defines no
   symbol of a bundled crypto backend (`ring_core_`, `aws_lc_`, `BORINGSSL_`,
   `OPENSSL_`), imports OpenSSL, carries the cargo-auditable manifest
   (`.dep-v0`, built from cargo's SBOM precursor, listing no denied crate)
   and the rustc producer string. A full report fails if the binary is
   missing, unreadable, or cannot be inspected as an ELF executable with
   `ldd`.
3. **Source guards**: the application never enables a FIPS provider itself,
   never uses OpenSSL's legacy (non-provider) digest API, never vendors or
   statically links OpenSSL.

Every finding comes with why the scanner cares, where in the tree to look and
what to do about it.

## Tool data

The shared tool carries the Red Hat signing key, the Podman signature-store
entry, and the certified-module inventory. See its
[documentation][cargo-fips-check] for provenance. This repository keeps
`xtask/assets/fips/fips-provider.cnf`, the `OPENSSL_CONF` used by tests
that exercise the RHEL FIPS provider.

## cargo-auditable

Red Hat's scanner finds pure-Rust cryptography through the crate list that
`cargo auditable build` embeds in the binary (the `.dep-v0` section); a binary
without it is graded inconclusive. The UBI builder installs `cargo-auditable`
from crates.io at the version pinned in `Containerfile.fips`
(`CARGO_AUDITABLE_VERSION`), and `make release-fips` uses it when it is
installed locally (`cargo install cargo-auditable --version 0.7.6 --locked`).
It is maintained by the Rust Secure Code Working Group and embeds data only,
never code. The report decodes the section itself (zlib-compressed JSON).

### Why the build uses cargo's SBOM precursor

The manifest has to list exactly the crates compiled into the binary: the
scanner fails on a denied crate's name alone, whether or not its code was
linked. On a stable toolchain cargo-auditable derives the list from
`cargo metadata`, and that resolve is not the build's:

- it unifies features across every workspace member, dev-dependencies
  included (the test utilities enable the policy engine, the stores and the
  reqwest-based filters, which pull `aws-lc-rs`, `sha2` and `hmac`), and
- it activates weak features (`dep?/feature`) that the real build never turns
  on: rustls always enables rustls-webpki's `alloc`, whose `ring?/alloc`
  entry puts `ring` in the resolve of a binary that never compiled it. That
  one cannot be avoided by any rustls user.

Cargo's SBOM precursor (`build.sbom`, still unstable behind `-Zsbom`) is
written by cargo's own unit graph and is exact; cargo-auditable 0.7 reads
it when present. `make release-fips` and `Containerfile.fips` therefore run

```console
RUSTC_BOOTSTRAP=1 CARGO_BUILD_SBOM=true cargo auditable -Zsbom \
    --config 'env.RUSTC_BOOTSTRAP.value="-1"' \
    --config 'env.RUSTC_BOOTSTRAP.force=true' \
    build --release -p praxis-ai-proxy ...
```

`RUSTC_BOOTSTRAP=1` lets a stable cargo accept the `-Z` flag (it also relaxes
cargo's guard against a build script setting `RUSTC_BOOTSTRAP` through
`cargo:rustc-env` from an error to a warning; the value is never forwarded
to rustc). The two `env` overrides replace it with `RUSTC_BOOTSTRAP=-1` for
rustc and every build script, and rustc treats `-1` as "no unstable
features", so the code compiled is exactly the stable code (the build scripts
of proc-macro2 and thiserror probe for unstable APIs when the variable is
set; with `-1` the probes fail, as they do without it). The report checks
the manifest's `format` field, which is 8 when it came from the precursor,
so a build that silently fell back to `cargo metadata` is a finding.

Cargo before 1.99 does not relink a binary when only the SBOM setting
changed (rust-lang/cargo#15695, fixed by #17216), so both recipes remove the
old binary first; drop that once the toolchains in use are 1.99 or newer.
Drop the whole workaround once `build.sbom` is stable
(rust-lang/cargo#13709).

## Red Hat's scanner

`make fips-scanner` fetches the pinned commit of `openshift/check-payload`
(`CHECK_PAYLOAD_REV` in the Makefile, the head of its PR #360, fetched by
commit so a rewrite of the PR cannot break the build) and builds it the way
upstream does (`CGO_ENABLED=0 go build`, vendored modules) into
`target/fips/check-payload/`; it needs Go 1.26 or newer. The scanner refuses
to run without the OpenShift CLI (`oc`) on `PATH`, even for a local image
scan: `make fips-oc` downloads Red Hat's pinned client release
(`OC_VERSION`) into `target/fips/bin/`, checks it against the sha256 Red Hat
publishes next to the tarball, and `make fips-scan` puts that directory on
`PATH` (an `oc` already on `PATH` works too). `make fips-scan` runs the
scanner against the FIPS image from podman's image store (under
`podman unshare` when podman is rootless) with `--fail-on-warnings`, so an
inconclusive verdict such as a missing manifest fails, as it does in Red
Hat's gated scans. All of this needs a Linux podman, rootless or root, not a
podman machine. Point `CHECK_PAYLOAD` at another build to use it instead.
Move `CHECK_PAYLOAD_REV` forward once the PR merges or a release carries
Rust support.

## The FIPS host run

`make test-fips-host` is the runtime proof: on a RHEL 9 host in FIPS mode it
runs `make test-fips test-integration-fips test-schema-fips` inside the
`toolchain` stage of `Containerfile.fips` (`make fips-toolchain`), so the
compiler and OpenSSL are the same Red Hat packages the FIPS image is built
with, while the kernel flag and the `FIPS` crypto policy are the host's,
which podman passes into the container. The run first prints
`make fips-host-facts`: the kernel flag, boot parameter, crypto policy,
OpenSSL packages and providers, and whether MD5 is refused, and fails there,
before anything compiles, unless they all agree that the container is in
FIPS mode.

Three variables drive the suites. `PRAXIS_FIPS_HOST=1` declares the host to
be in FIPS mode: the harness then fails closed the first time it installs
the crypto provider on a host that is not, and every FIPS behavior test
(`tests/integration/tests/suite/fips.rs`, `server/tests/fips_required.rs`)
insists on its approved-mode branch instead of keying on whatever the
provider reports. `PRAXIS_REQUIRE_FIPS=1` makes every proxy the suites start
enforce FIPS mode. `PRAXIS_TEST_FIPS_PROVIDER=1` arms the hash and
`aws_sigv4_sign` unit tests' assertion that the provider is in approved
mode. The cargo home and target directory live in the named volumes
`praxis-ai-fips-host-cargo` and `praxis-ai-fips-host-target`, so a second
run is incremental; the container runs as the invoking user
(`--userns=keep-id`), because praxis-ai refuses to start as root.

`make test-postgres-fips-host` complements those containerized suites with
the database boundary that needs access to the host's rootless podman. It
resolves the same FIPS feature set directly on the declared FIPS host, starts
a sibling PostgreSQL container with only `hostssl ... cert` TCP rules, and
performs a Responses write/read round trip over SQLx native TLS without a
password. A second TLS peer selects SCRAM; the test requires the
certificate-only SQLx build to refuse it as disabled before beginning a
password exchange. The composite action runs both immediately after
`test-fips-host`.

On a developer machine that is not in FIPS mode, `make test-integration-fips`
and `make test-schema-fips` still run the same suites as the FIPS build; the
FIPS behavior tests then assert their non-approved branch, so both sides of
every expectation stay exercised everywhere.

## The runner job

The `fips-host` job of the `FIPS` workflow runs on a self-hosted RHEL 9
runner in FIPS mode, selected by the labels `fips` and `rhel`. It tests the
exact image the hosted `ubi-image` job built and scanned, handed over as an
artifact and checked by image id, then runs `make fips-host-check`,
`make test-fips-host`, `make test-postgres-fips-host` and
`make fips-runtime-probe` through
`.github/actions/fips-host`; the release workflow requires a recorded green
`fips-host` run for the exact commit being released (its `fips-proof` gate
polls out a run still in flight) and runs the same composite action (without
the suites) against the pushed `-fips` image, pulled by digest. That digest
attestation runs alongside the release rather than gating it: an offline
runner leaves a self-hosted job queued, never skipped, so it reports red on
the release run instead of holding the release back. Releasing without the
recorded proof takes the release workflow's explicit `skip-fips-proof`
dispatch input, so a runner outage is a visible maintainer decision, never
a silent pass.

The runner needs `git`, `make`, `podman` (rootless), `gnupg2`, `gcc`,
`gcc-c++`, `cmake` and `openssl-devel`, checked up front by
`.github/actions/fips-runner-check`, and its runner group must grant this
repository access. The job never runs a fork's code (same-repository pull
requests only) and never runs in the merge queue, so queue throughput does
not depend on the single runner. The attestation and the probe log are
uploaded as artifacts on every run.

## Updating the pinned base image

The digests live in the `Makefile` (`FIPS_UBI9_DIGEST`,
`FIPS_UBI9_MINIMAL_DIGEST`) and, as defaults, in `Containerfile.fips`. To move
to a newer UBI 9:

```console
curl -sI -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' \
  https://registry.access.redhat.com/v2/ubi9/ubi/manifests/latest | grep -i docker-content-digest
cargo fips-check verify-image registry.access.redhat.com/ubi9/ubi@sha256:<new digest>
```

Only update both places once the verification passes.
