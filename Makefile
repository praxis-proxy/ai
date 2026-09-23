# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

VERSION          ?= $(shell perl -ne 'print $$1 if /^version\s*=\s*"(.+)"/' Cargo.toml)
IMAGE            ?= praxis-ai
CONTAINER_ENGINE ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
OPENAI_CONFORMANCE_ARGS ?=
RESPONSES_CONFORMANCE_ARGS ?=
V                ?=

# Experimental filter features are package-specific and off by default.
# Basic Auth is exposed by praxis-ai-proxy and forwarded by the integration-test
# crate; it is not a praxis-ai-filters feature.
FILTER_EXPERIMENTAL_FEATURES := azure-ad-filter,gcp-adc-filter,http-callout-filter,token-rate-limit-filter
INTEGRATION_EXPERIMENTAL_FEATURES := azure-ad-filter,basic-auth-filter,gcp-adc-filter,http-callout-filter,token-rate-limit-filter
# Features for `make release`; `full` matches the published container image.
PRAXIS_AI_FEATURES ?= full
# Crates that must never enter the default (standard) praxis-ai-proxy graph.
# openssl-sys is not on the list: praxis performs all cryptography through the
# system OpenSSL, so its bindings are part of every build by design.
DEFAULT_GRAPH_DENY := sqlx sqlx-core libsqlite3-sys native-tls rmcp sse-stream \
	jsonschema utoipa tiktoken-rs reqwest serde_json_path tonic prost
# Upper bound on crates (name@version, normal + build edges, host target) in the
# default graph. Linux hosts measure about 424, macOS about 428.
DEFAULT_GRAPH_BUDGET ?= 430
STORE_ALL_WORKSPACE_FEATURES := praxis-ai-proxy/store-all,praxis-tests-integration/store-all,praxis-tests-schema/store-all,praxis-tests-environment/store-all

ifneq ($(V),)
  _NOCAPTURE := -- --nocapture
endif

.PHONY: all build release check clean \
	test test-unit test-schema test-integration test-inference-fixtures \
	test-store-features \
	test-postgres-unit test-postgres-integration test-environment \
	test-token-rate-limit-valkey-unit test-token-rate-limit-valkey-integration \
	openai-conformance check-openai-conformance-reference test-openai-conformance \
	test-responses-conformance \
	lint lint-lean check-dep-budget fmt doc audit coverage-check \
	require-container-engine \
	container container-run \
	setup-hooks help \
	patch-praxis unpatch-praxis \
	require-podman require-go require-oc \
	build-fips release-fips check-fips lint-fips test-fips \
	container-fips container-fips-run \
	fips-check fips-check-ubi fips-deps fips-report fips-signature-store fips-verify-image \
	fips-oc fips-scan fips-scanner fips-smoke

# -------------------------------------------------------------------
# All
# -------------------------------------------------------------------

all: build fmt lint test audit

# -------------------------------------------------------------------
# Build
# -------------------------------------------------------------------

build:
	cargo build --workspace

release:
	cargo build --release -p praxis-ai-proxy --features $(PRAXIS_AI_FEATURES)

check:
	cargo check --workspace

clean:
	cargo clean

# -------------------------------------------------------------------
# Container
# -------------------------------------------------------------------

require-container-engine:
ifndef CONTAINER_ENGINE
	$(error No container engine found — install podman or docker)
endif

container: | require-container-engine
	$(CONTAINER_ENGINE) build -t $(IMAGE):$(VERSION) -f Containerfile .

container-run: | require-container-engine
	$(CONTAINER_ENGINE) run --rm --network=host $(IMAGE):$(VERSION) 2>&1

# -------------------------------------------------------------------
# Test
# -------------------------------------------------------------------

test:
	cargo test --workspace $(_NOCAPTURE)

test-unit:
	cargo test -p praxis-ai-apis $(_NOCAPTURE)
	cargo test -p praxis-ai-apis --features full $(_NOCAPTURE)
	cargo test -p praxis-ai-filters $(_NOCAPTURE)
	cargo test -p praxis-ai-filters --features full $(_NOCAPTURE)
	cargo test -p praxis-ai-filters --features full,$(FILTER_EXPERIMENTAL_FEATURES) $(_NOCAPTURE)
	cargo test -p praxis-ai-proxy $(_NOCAPTURE)
	cargo test -p praxis-ai-proxy --features full $(_NOCAPTURE)
	cargo test -p praxis-ai-proxy --features full,basic-auth-filter $(_NOCAPTURE)
	cargo test -p praxis-ai-build-support $(_NOCAPTURE)

test-store-features:
	cargo check -p praxis-ai-proxy
	cargo check -p praxis-ai-proxy --no-default-features --features standard,openai-all,store-sqlite
	cargo check -p praxis-ai-proxy --no-default-features --features standard,openai-all,store-all
	@# Lint each opt-in group on its own so a gate leak in a partial feature set
	@# cannot hide behind the lean and full builds that other targets cover.
	@for group in openai-responses openai-file-resolve-filter store store-sqlite \
		openai-conversations openai-compact openai-mcp-tools; do \
		echo "clippy: standard + $$group"; \
		cargo clippy -p praxis-ai-apis -p praxis-ai-filters -p praxis-ai-proxy --all-targets \
			--no-default-features --features praxis-ai-proxy/standard,praxis-ai-proxy/$$group \
			-- -D warnings || exit 1; \
	done
	cargo test -p praxis-ai-apis --no-default-features --features openai-all,store-sqlite $(_NOCAPTURE)
	cargo test -p praxis-ai-apis --no-default-features --features openai-all,store-all $(_NOCAPTURE)
	@if cargo tree -p praxis-ai-proxy --features full --edges normal | grep -q libsqlite3-sys; then \
		echo "ERROR: full proxy dependency graph contains libsqlite3-sys"; \
		exit 1; \
	fi
	@cargo tree -p praxis-ai-proxy --features full --edges features -i sqlx-core | grep -q '_tls-native-tls' || \
		(echo "ERROR: full proxy SQLx graph does not enable native TLS"; exit 1)
	@if cargo tree -p praxis-ai-proxy --features full --edges features -i sqlx-core | grep -q '_tls-rustls'; then \
		echo "ERROR: full proxy SQLx graph contains a rustls TLS backend"; \
		exit 1; \
	fi

test-schema:
	cargo test -p praxis-tests-schema --features store-all $(_NOCAPTURE)

test-integration:
	cargo test -p praxis-tests-integration --features store-all $(_NOCAPTURE)
	cargo test -p praxis-tests-integration --features store-all,$(INTEGRATION_EXPERIMENTAL_FEATURES) --test suite \
		-- examples::azure_ad examples::gcp_adc examples::lakera_guard examples::token_rate_limit \
		$(if $(V),--nocapture)

test-inference-fixtures:
	cargo test -p praxis-test-utils --features store-all $(_NOCAPTURE)
	cargo test -p xtask --features store-all inference_fixtures $(_NOCAPTURE)
	cargo test -p praxis-tests-integration --features store-all --test suite inference_fixtures $(_NOCAPTURE)

test-postgres-unit:
	cargo test -p praxis-ai-apis --no-default-features --features store-all store::tests::pg_ -- --ignored $(_NOCAPTURE)

# Every PostgreSQL integration test is #[ignore]d (each spawns its own
# container), so it runs only when named here. Enumerate every module explicitly:
# a bare substring filter such as `openai_response_store_postgres` incidentally
# matches the response-store mTLS variant (a prefix) but cannot select the
# Conversations certificate-auth module, silently dropping it from CI. Filters
# must follow `--` so libtest treats each as an OR filter. Add every new
# PostgreSQL integration module to this list.
test-postgres-integration:
	cargo test -p praxis-tests-integration --test suite -- --ignored \
		openai_response_store_postgres \
		openai_response_store_postgres_mtls \
		openai_conversations_postgres_mtls $(if $(V),--nocapture)

test-token-rate-limit-valkey-unit:
	cargo test -p praxis-ai-filters --features token-rate-limit-filter valkey $(_NOCAPTURE)

test-token-rate-limit-valkey-integration:
	cargo test -p praxis-tests-integration --features basic-auth-filter,token-rate-limit-filter --test suite \
		mixed_algorithm_rules_valkey_backend_isolates_budgets_across_gateway_replicas $(_NOCAPTURE)
	cargo test -p praxis-tests-integration --features basic-auth-filter,token-rate-limit-filter --test suite \
		authenticated_subject_valkey_backend_isolates_budgets_across_gateway_replicas $(_NOCAPTURE)

openai-conformance:
	cargo xtask openai-conformance $(OPENAI_CONFORMANCE_ARGS)

check-openai-conformance-reference:
	cargo xtask openai-conformance-reference --check

test-openai-conformance: openai-conformance

test-responses-conformance:
	uv run tests/integration/sdk/openai/test_responses_conformance.py -v $(RESPONSES_CONFORMANCE_ARGS)

test-environment:
	cargo test -p praxis-ai-llmd-ext-proc $(_NOCAPTURE)
	cargo test -p praxis-tests-integration --features llmd-ext-proc llmd_ext_proc $(_NOCAPTURE)
	cargo test -p praxis-tests-environment --features llmd-ext-proc $(_NOCAPTURE)

# -------------------------------------------------------------------
# Quality
# -------------------------------------------------------------------

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	cargo clippy --workspace --all-targets \
		--features praxis-ai-proxy/azure-ad-filter,praxis-ai-proxy/basic-auth-filter,praxis-ai-proxy/gcp-adc-filter,praxis-ai-proxy/http-callout-filter,praxis-ai-proxy/token-rate-limit-filter,praxis-tests-integration/azure-ad-filter,praxis-tests-integration/basic-auth-filter,praxis-tests-integration/gcp-adc-filter,praxis-tests-integration/http-callout-filter,praxis-tests-integration/token-rate-limit-filter \
		-- -D warnings
	$(MAKE) lint-lean
	$(MAKE) check-dep-budget
	cargo +nightly fmt --all -- --check
	cargo machete --with-metadata .
	cargo xtask lint-deps
	$(MAKE) fips-deps
	cargo xtask lint-separators
	cargo xtask lint-filter-docs
	cargo xtask lint-example-tests
	cargo xtask lint-markdown-links
	cargo xtask sync-example-readme
	cargo xtask sync-inference-readme
	cargo xtask sync-responses-readme
	cargo xtask check-inference
	cargo xtask check-responses-registry
	cargo xtask check-chat-completions-registry
	cargo xtask openresponses-coverage

# Lint the product crates with only the default-on gates. `-p` without
# `--workspace` keeps test-crate features from unifying in and hiding leaks.
lint-lean:
	cargo clippy -p praxis-ai-apis -p praxis-ai-filters -p praxis-ai-proxy --all-targets \
		--no-default-features --features praxis-ai-proxy/standard -- -D warnings
	RUSTDOCFLAGS="-D warnings" cargo doc -p praxis-ai-apis -p praxis-ai-filters -p praxis-ai-proxy \
		--no-deps --document-private-items --no-default-features --features praxis-ai-proxy/standard

# Fail if a heavy crate enters the default praxis-ai-proxy graph, or if the
# graph grows past DEFAULT_GRAPH_BUDGET crates.
check-dep-budget:
	@tree="$$(cargo tree --locked -p praxis-ai-proxy -e normal,build --target all \
		--prefix none --format '{p}')" || { echo "ERROR: cargo tree failed"; exit 1; }; \
	host="$$(cargo tree --locked -p praxis-ai-proxy -e normal,build \
		--prefix none --format '{p}')" || { echo "ERROR: cargo tree failed"; exit 1; }; \
	graph="$$(printf '%s\n' "$$tree" | awk '{print $$1"@"$$2}' | sort -u)"; \
	status=0; \
	for crate in $(DEFAULT_GRAPH_DENY); do \
		if printf '%s\n' "$$graph" | grep -q "^$$crate@"; then \
			echo "ERROR: $$crate is in the default praxis-ai-proxy graph:"; \
			cargo tree --locked -p praxis-ai-proxy -e normal,build --target all -i "$$crate" | head -n 15; \
			status=1; \
		fi; \
	done; \
	count=$$(printf '%s\n' "$$host" | awk '{print $$1"@"$$2}' | sort -u | wc -l); \
	[ "$$count" -gt 1 ] || { echo "ERROR: empty default dependency graph"; exit 1; }; \
	echo "default praxis-ai-proxy graph: $$count crates for the host target (budget $(DEFAULT_GRAPH_BUDGET))"; \
	[ "$$count" -le $(DEFAULT_GRAPH_BUDGET) ] || { echo "ERROR: over budget"; status=1; }; \
	exit $$status

fmt:
	cargo +nightly fmt --all

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items

audit:
	cargo audit
	cargo deny check

coverage-check:
	cargo llvm-cov --workspace --features $(STORE_ALL_WORKSPACE_FEATURES) --json \
		--exclude xtask \
		--ignore-filename-regex '(target/|tests/|store/postgres\.rs)' \
		--output-path coverage.json
	@LINE_PCT=$$(jq '.data[0].totals.lines.percent' coverage.json); \
	echo "Line coverage: $${LINE_PCT}%"; \
	if [ $$(echo "$${LINE_PCT} < 95" | bc -l) -eq 1 ]; then \
		echo "FAIL: coverage $${LINE_PCT}% is below 95% threshold"; \
		exit 1; \
	fi

# -------------------------------------------------------------------
# FIPS
# -------------------------------------------------------------------
#
# The published image (`full`) enables every non-experimental filter. The
# FIPS build turns off what is known not to be FIPS 140-3 compliant yet, so
# nobody has to know which features to pick:
#
#   aws-sigv4-filter     the aws-sigv4 crate signs with pure-Rust hmac/sha2
#   policy-engine        praxis-policy carries its own cryptography (sha2,
#                        hmac, jsonwebtoken on aws-lc-rs)
#   store, store-sqlite, store-postgres, openai-conversations, openai-compact
#                        sqlx enables sqlx-core's `migrate` feature with its
#                        tokio runtime, and that pulls sha2; store-postgres
#                        adds sqlx-postgres' md-5/hmac/sha2/rsa (SCRAM)
#   openai-file-resolve-filter, openai-mcp-tools, azure-ad-filter,
#   gcp-adc-filter       reqwest's `rustls` feature compiles aws-lc-rs in
#
# What remains of the opt-in groups is openai-responses (the Responses API
# kernel, which adds no crates). The experimental filters stay off for the
# same reasons they are off in the standard build. FIPS_FEATURES is the
# single place this is defined; Containerfile.fips (CARGO_FEATURES) mirrors
# it and must be kept in sync.
#
# The FIPS build goes to its own target directory so it never overwrites,
# or is mistaken for, the standard build.
#
#   make build-fips        FIPS build, debug profile
#   make release-fips      FIPS build, release profile
#   make lint-fips         clippy + rustfmt for the FIPS feature set
#   make test-fips         unit tests for the FIPS feature set
#   make container-fips    FIPS runtime image on UBI 9 (Red Hat toolchain,
#                          signature-verified base images)
#   make fips-check        build on UBI 9 and print the compliance report
#   make fips-report       the same report against the local FIPS build
#   make fips-deps         dependency graph only (seconds, no build; also
#                          runs under `make lint`, so a PR cannot reintroduce
#                          a denied crate into the FIPS build)
#   make fips-smoke        run the FIPS image once (validates its config)
#   make fips-scan         run Red Hat's scanner (check-payload) on the
#                          FIPS image, warnings fatal: the actual gate
#   make fips-scanner      build check-payload at the pinned revision
#   make fips-oc           download the OpenShift CLI the scanner insists on
#   make fips-signature-store
#                          point podman at Red Hat's signature store; needed
#                          once on Debian/Ubuntu hosts, a no-op elsewhere
#
# The report, the image verification and the signature-store setup are
# `cargo xtask fips` commands (xtask/src/fips/). XTASK_FIPS builds xtask
# without its default features, so these targets never compile the standard
# proxy build to run.
#
# See docs/developing/fips.md and docs/developing/getting-started.md.

FIPS_FEATURES           := openai-responses
# The same list qualified for a multi-package cargo invocation.
_COMMA                  := ,
FIPS_FEATURES_QUALIFIED := $(subst $(_COMMA),$(_COMMA)praxis-ai-proxy/,praxis-ai-proxy/$(FIPS_FEATURES))
FIPS_TARGET_DIR         := target/fips
FIPS_BIN                ?= $(FIPS_TARGET_DIR)/release/praxis-ai
FIPS_CARGO_ARGS         := -p praxis-ai-proxy --no-default-features --features $(FIPS_FEATURES) --target-dir $(FIPS_TARGET_DIR)
# Red Hat's scanner reads the crate list that `cargo auditable` embeds in the
# binary (the .dep-v0 section); without it a binary is graded inconclusive.
# `make release-fips` embeds it when cargo-auditable is installed (`cargo
# install cargo-auditable --version 0.7.6 --locked`); the report says so when
# it was not.
#
# The list must be exactly the crates compiled in. On a stable toolchain
# cargo-auditable derives it from `cargo metadata`, which unifies features
# across the whole workspace and activates weak features (`dep?/feature`)
# the real build never turns on; with rustls that puts `ring` in the manifest
# of a binary that never compiled it, and the scanner fails on the name alone.
# Cargo's SBOM precursor (`-Zsbom`, unstable) is the exact list, so the
# release build enables it: RUSTC_BOOTSTRAP=1 lets stable cargo accept the
# flag, and the env overrides hand rustc and every build script
# RUSTC_BOOTSTRAP=-1, which forbids unstable features, so the code compiled is
# the stable code. Drop this once cargo's `build.sbom` is stable
# (rust-lang/cargo#13709). Same recipe in Containerfile.fips.
CARGO_AUDITABLE         := $(shell command -v cargo-auditable >/dev/null 2>&1 && echo "cargo auditable" || echo "cargo")
FIPS_SBOM_ENV           := RUSTC_BOOTSTRAP=1 CARGO_BUILD_SBOM=true
FIPS_SBOM_ARGS          := -Zsbom --config 'env.RUSTC_BOOTSTRAP.value="-1"' --config 'env.RUSTC_BOOTSTRAP.force=true'
FIPS_UBI9_DIGEST        := sha256:a4b9ec09b1e790a53ef25b7777c539976abe519248264298e5194dcbceac8c31
FIPS_UBI9_MINIMAL_DIGEST := sha256:8ebe2ad8fdf3cab3e5a53c1edc69194c98209cfadab24b884f4ad9ebcf7bbbfc
FIPS_UBI9_IMAGE         := registry.access.redhat.com/ubi9/ubi@$(FIPS_UBI9_DIGEST)
FIPS_UBI9_MINIMAL_IMAGE := registry.access.redhat.com/ubi9/ubi-minimal@$(FIPS_UBI9_MINIMAL_DIGEST)
FIPS_CHECK_IMAGE        ?= praxis-ai-fips-check
FIPS_BUILD_ARGS         := --build-arg UBI9_DIGEST=$(FIPS_UBI9_DIGEST) \
	--build-arg UBI9_MINIMAL_DIGEST=$(FIPS_UBI9_MINIMAL_DIGEST) \
	--build-arg CARGO_FEATURES=$(FIPS_FEATURES)
XTASK_FIPS              := cargo run -q -p xtask --no-default-features --
# Red Hat's scanner, openshift/check-payload, at the revision that added Rust
# support (the head of its PR #360, fetched by commit so a rewrite of the PR
# cannot break the build). `make fips-scanner` builds it into target/fips;
# point CHECK_PAYLOAD at another build to use it instead.
CHECK_PAYLOAD_REPO      := https://github.com/openshift/check-payload
CHECK_PAYLOAD_REV       := 1ce4e04ed214b98997797ce19a2442f794632e65
CHECK_PAYLOAD_DIR       := $(FIPS_TARGET_DIR)/check-payload
CHECK_PAYLOAD           ?= $(CHECK_PAYLOAD_DIR)/check-payload
# The FIPS image as podman's storage names it: a bare name gets podman's
# implicit localhost/ prefix, a registry-qualified IMAGE does not.
_IMAGE_HEAD             := $(firstword $(subst /, ,$(IMAGE)))
FIPS_IMAGE_REF          := $(if $(or $(findstring .,$(_IMAGE_HEAD)),$(findstring :,$(_IMAGE_HEAD)),$(filter localhost,$(_IMAGE_HEAD))),$(IMAGE),localhost/$(IMAGE)):$(VERSION)-fips
# The scanner mounts the image from podman's store, which needs the user
# namespace only for rootless podman.
PODMAN_UNSHARE          := $(if $(filter 0,$(shell id -u)),,podman unshare)
# check-payload refuses to run without the OpenShift CLI (oc) on PATH, even
# for a local image scan. `make fips-oc` downloads Red Hat's pinned client
# release into target/fips/bin and checks its published sha256 (the checksum
# Red Hat lists next to the tarball on mirror.openshift.com); fips-scan puts
# that directory on PATH. Pinned for Linux x86_64, which is what CI runs; on
# any other platform install oc yourself and it is picked up from PATH.
UNAME_S                 := $(shell uname -s | tr '[:upper:]' '[:lower:]')
UNAME_M                 := $(shell uname -m)
SHA256SUM               := $(if $(filter darwin,$(UNAME_S)),gsha256sum,sha256sum)
OC_VERSION              := 4.22.14
OC_DIR                  := $(FIPS_TARGET_DIR)/bin
OC                      := $(OC_DIR)/oc
OC_SHA256_linux_x86_64  := 7dbe8c2813bc09e18a666155eb4fa88dc3260c3832c553aa16d86c7c4277ba03
OC_SHA256               := $(OC_SHA256_$(UNAME_S)_$(UNAME_M))

require-podman:
	@command -v podman >/dev/null || { echo "podman is required: Red Hat image signatures can only be verified with podman"; exit 1; }

require-go:
	@command -v go >/dev/null || { echo "go is required to build check-payload"; exit 1; }

require-oc:
	@command -v oc >/dev/null || [ -x $(OC) ] || { echo "oc (the OpenShift CLI) is required: check-payload refuses to scan without it on PATH; run 'make fips-oc'"; exit 1; }

$(OC):
	@mkdir -p $(OC_DIR)
	curl -sSfL -o $(OC_DIR)/oc.tar.gz \
		https://mirror.openshift.com/pub/openshift-v4/$(UNAME_M)/clients/ocp/$(OC_VERSION)/openshift-client-linux-$(OC_VERSION).tar.gz
	$(if $(OC_SHA256),echo "$(OC_SHA256)  $(OC_DIR)/oc.tar.gz" | $(SHA256SUM) -c,$(error no pinned SHA256 for oc on $(UNAME_S)/$(UNAME_M); refusing to use an unverified download))
	tar xz -C $(OC_DIR) -f $(OC_DIR)/oc.tar.gz oc
	rm -f $(OC_DIR)/oc.tar.gz

fips-oc: $(OC)

# The debug build is the edit-compile loop; only the release build carries
# the manifest.
build-fips:
	cargo build $(FIPS_CARGO_ARGS)

# cargo before 1.99 does not relink a binary when only the SBOM setting
# changed (rust-lang/cargo#15695, fixed by #17216), so the old binary goes
# first; everything else stays cached. Drop the clean once the toolchains in
# use (here and the UBI rust-toolset) are 1.99 or newer.
release-fips:
ifeq ($(CARGO_AUDITABLE),cargo auditable)
	cargo clean --release -p praxis-ai-proxy --target-dir $(FIPS_TARGET_DIR)
	$(FIPS_SBOM_ENV) cargo auditable $(FIPS_SBOM_ARGS) build --release $(FIPS_CARGO_ARGS)
else
	@echo "warning: cargo-auditable is not installed; no crate manifest will be embedded (cargo install cargo-auditable --version 0.7.6 --locked)"
	cargo build --release $(FIPS_CARGO_ARGS)
endif

check-fips:
	cargo check $(FIPS_CARGO_ARGS)

# Clippy over every target of the FIPS build, plus the rustfmt check (which
# is feature-independent but belongs in "is the FIPS version clean").
lint-fips:
	cargo clippy $(FIPS_CARGO_ARGS) --all-targets -- -D warnings
	cargo +nightly fmt --all -- --check

# Unit tests of the crates that make up the FIPS binary, resolved exactly as
# the FIPS build resolves them: no default features anywhere, only
# FIPS_FEATURES on the binary. The integration suites run the standard build
# through the test harness and are covered by `make test-integration`.
test-fips:
	cargo test --target-dir $(FIPS_TARGET_DIR) --no-default-features \
		-p praxis-ai-proxy -p praxis-ai-filters -p praxis-ai-apis \
		--features $(FIPS_FEATURES_QUALIFIED) $(_NOCAPTURE)

# podman finds Red Hat's detached image signatures through its registries.d
# (containers-registries.d(5)). Fedora and RHEL ship the entry; Debian and
# Ubuntu, GitHub's runners included, ship no registries.d at all, and then
# every Red Hat image looks unsigned. This installs the bundled entry for the
# current user when the registries.d podman reads names none, and does
# nothing otherwise. CI runs it before fips-verify-image.
fips-signature-store:
	$(XTASK_FIPS) fips signature-store --install

fips-verify-image: | require-podman
	$(XTASK_FIPS) fips verify-image --pinned-in Containerfile.fips $(FIPS_UBI9_IMAGE)
	$(XTASK_FIPS) fips verify-image --pinned-in Containerfile.fips $(FIPS_UBI9_MINIMAL_IMAGE)

container-fips: fips-verify-image
	podman build -f Containerfile.fips --target runtime $(FIPS_BUILD_ARGS) \
		-t $(IMAGE):$(VERSION)-fips .

container-fips-run: | require-podman
	podman run --rm --network=host $(IMAGE):$(VERSION)-fips 2>&1

# The binary starts on ubi-minimal, loads the system OpenSSL and accepts its
# built-in default config; a cheap proof that the image runs before the scan.
fips-smoke: | require-podman
	podman run --rm --entrypoint praxis-ai $(IMAGE):$(VERSION)-fips --validate

fips-check: fips-check-ubi

fips-check-ubi: fips-verify-image
	podman build -f Containerfile.fips --target report $(FIPS_BUILD_ARGS) \
		-t $(FIPS_CHECK_IMAGE) .
	podman run --rm $(FIPS_CHECK_IMAGE)

fips-report:
	$(XTASK_FIPS) fips report --features $(FIPS_FEATURES) $(FIPS_BIN)

# The graph check is `cargo xtask fips report` (cargo tree scoped to the
# binary and its feature set) rather than cargo-deny: cargo-deny resolves
# features workspace-wide, and the test crates enable the policy engine and
# the stores on the binary, so it cannot see the FIPS build's real graph.
fips-deps:
	$(XTASK_FIPS) fips report --deps-only --features $(FIPS_FEATURES)

# --fail-on-warnings makes an inconclusive verdict (for example a binary
# without a crate manifest) fail, as Red Hat's gated scans do. Needs a Linux
# podman (rootless or root), not a podman machine.
fips-scan: | require-podman require-oc
	@[ -x "$(CHECK_PAYLOAD)" ] || { echo "check-payload not found at $(CHECK_PAYLOAD): run 'make fips-scanner' (needs go) or set CHECK_PAYLOAD"; exit 1; }
	PATH="$(abspath $(OC_DIR)):$$PATH" $(PODMAN_UNSHARE) $(CHECK_PAYLOAD) scan image \
		--spec containers-storage:$(FIPS_IMAGE_REF) --fail-on-warnings

# Built as upstream builds it (CGO_ENABLED=0, vendored modules).
fips-scanner: | require-go
	@mkdir -p $(CHECK_PAYLOAD_DIR)
	@[ -d $(CHECK_PAYLOAD_DIR)/.git ] || git -C $(CHECK_PAYLOAD_DIR) init --quiet
	git -C $(CHECK_PAYLOAD_DIR) fetch --quiet --depth 1 $(CHECK_PAYLOAD_REPO) $(CHECK_PAYLOAD_REV)
	git -C $(CHECK_PAYLOAD_DIR) checkout --quiet FETCH_HEAD
	cd $(CHECK_PAYLOAD_DIR) && CGO_ENABLED=0 go build -o check-payload .

# -------------------------------------------------------------------
# Praxis path override (test against local ../praxis)
# -------------------------------------------------------------------

patch-praxis:
	@if [ ! -d "../praxis" ]; then \
		echo "ERROR: ../praxis not found — clone praxis core as a sibling directory first"; \
		exit 1; \
	fi
	@if grep -q '\[patch\.crates-io\]' Cargo.toml; then \
		echo "Already patched — run 'make unpatch-praxis' first"; \
		exit 1; \
	fi
	@printf '\n[patch.crates-io]\n\
	praxis-proxy-core = { path = "../praxis/core" }\n\
	praxis-proxy-filter = { path = "../praxis/filter" }\n\
	praxis-proxy-protocol = { path = "../praxis/protocol" }\n\
	praxis-proxy-tls = { path = "../praxis/tls" }\n\
	praxis-proxy = { path = "../praxis/server" }\n' >> Cargo.toml
	@echo "Patched Cargo.toml to use ../praxis path dependencies"

unpatch-praxis:
	@if ! grep -q '\[patch\.crates-io\]' Cargo.toml; then \
		echo "Nothing to unpatch"; \
		exit 0; \
	fi
	@sed -i.bak '/^\[patch\.crates-io\]/,$$d' Cargo.toml && rm -f Cargo.toml.bak
	@echo "Removed [patch.crates-io] from Cargo.toml"

# -------------------------------------------------------------------
# Dev Setup
# -------------------------------------------------------------------

setup-hooks:
	ln -sf ../../.hooks/pre-commit .git/hooks/pre-commit
	@echo "Git hooks installed."

# -------------------------------------------------------------------
# Help
# -------------------------------------------------------------------

help:
	@echo "Variables:"
	@echo "  V=1                  show test output (--nocapture)"
	@echo ""
	@echo "Top-level:"
	@echo "  all                  build + lint + test + audit"
	@echo ""
	@echo "Build:"
	@echo "  build                cargo build --workspace"
	@echo "  release              cargo build --release -p praxis-ai-proxy --features $(PRAXIS_AI_FEATURES)"
	@echo "  check                cargo check --workspace"
	@echo "  clean                cargo clean"
	@echo ""
	@echo "Test:"
	@echo "  test                 run all tests"
	@echo "  test-unit            unit tests (providers, filters, server)"
	@echo "  test-store-features   check the lean and store builds and lint each feature group alone"
	@echo "  test-schema          schema validation tests"
	@echo "  test-integration     integration tests"
	@echo "  test-inference-fixtures  inference fixture and replay tests"
	@echo "  test-postgres-unit       postgres store unit tests (needs DATABASE_URL)"
	@echo "  test-postgres-integration postgres store integration tests (needs container engine)"
	@echo "  test-token-rate-limit-valkey-unit        token_rate_limit Valkey unit tests (needs TOKEN_RATE_LIMIT_VALKEY_URL)"
	@echo "  test-token-rate-limit-valkey-integration token_rate_limit Valkey integration test (needs TOKEN_RATE_LIMIT_VALKEY_URL)"
	@echo "  test-environment     llm-d ext_proc environment tests"
	@echo "  openai-conformance   compare registered API areas with OpenAI's OpenAPI spec"
	@echo "  check-openai-conformance-reference  verify the pinned complete OpenAI reference"
	@echo "  test-responses-conformance  Run OpenResponses suite against the translation filter (needs bun, uv, vLLM)"
	@echo ""
	@echo "Quality:"
	@echo "  lint                 clippy + rustfmt + dependency, docs, and example checks"
	@echo "  fmt                  format with nightly rustfmt"
	@echo "  doc                  rustdoc with warnings"
	@echo "  audit                cargo audit + cargo deny"
	@echo ""
	@echo "Container:"
	@echo "  container            build praxis-ai container image"
	@echo "  container-run        run container in foreground (host network)"
	@echo ""
	@echo "FIPS (feature set: $(FIPS_FEATURES), see docs/developing/fips.md):"
	@echo "  build-fips           FIPS build, debug profile, into target/fips"
	@echo "  release-fips         FIPS build, release profile, with the cargo-auditable crate manifest"
	@echo "  check-fips           cargo check of the FIPS build"
	@echo "  lint-fips            clippy (all targets) + rustfmt check for the FIPS feature set"
	@echo "  test-fips            unit tests resolved as the FIPS build (no defaults, FIPS_FEATURES on the binary)"
	@echo "  container-fips       FIPS runtime image on UBI 9 (Red Hat toolchain, signature-verified bases)"
	@echo "  container-fips-run   run the FIPS image in foreground (host network)"
	@echo "  fips-check           build on UBI 9 and print the compliance report (fails while findings remain)"
	@echo "  fips-report          compliance report against the local FIPS build (FIPS_BIN=target/fips/release/praxis-ai)"
	@echo "  fips-smoke           run the FIPS image once to validate its config"
	@echo "  fips-scan            run Red Hat's scanner (check-payload) on the FIPS image, warnings fatal"
	@echo "  fips-scanner         build check-payload at the pinned revision into target/fips (needs go)"
	@echo "  fips-oc              download the pinned OpenShift CLI (oc) the scanner insists on, checksum-verified"
	@echo "  fips-deps            dependency graph vs Red Hat's crypto denylist (seconds, no build)"
	@echo "  fips-verify-image    verify the pinned UBI 9 base images are Red Hat's (digest + signature)"
	@echo "  fips-signature-store point podman at Red Hat's signature store (once, on Debian/Ubuntu hosts)"
	@echo ""
	@echo "Praxis override:"
	@echo "  patch-praxis         use ../praxis path deps instead of crates.io"
	@echo "  unpatch-praxis       revert to crates.io praxis deps"
