# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

VERSION          ?= $(shell perl -ne 'print $$1 if /^version\s*=\s*"(.+)"/' Cargo.toml)
IMAGE            ?= praxis-ai
CONTAINER_ENGINE ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
OPENAI_CONFORMANCE_ARGS ?=
V                ?=

# Experimental filter features are off by default in builds, so lint and
# test explicitly enable them — otherwise the gated filter code is never
# compiled, linted, or tested by CI.
EXPERIMENTAL_FEATURES := azure-ad-filter,gcp-adc-filter,http-callout-filter,token-rate-limit-filter

ifneq ($(V),)
  _NOCAPTURE := -- --nocapture
endif

.PHONY: all build release check clean \
	test test-unit test-schema test-integration test-inference-fixtures \
	test-postgres-unit test-postgres-integration test-environment \
	test-token-rate-limit-valkey-unit test-token-rate-limit-valkey-integration \
	openai-conformance check-openai-conformance-reference test-openai-conformance \
	lint fmt doc audit coverage-check \
	require-container-engine \
	container container-run \
	setup-hooks help \
	patch-praxis unpatch-praxis

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
	cargo build --workspace --release

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
	cargo test -p praxis-ai-filters $(_NOCAPTURE)
	cargo test -p praxis-ai-filters --features $(EXPERIMENTAL_FEATURES) $(_NOCAPTURE)
	cargo test -p praxis-ai-proxy $(_NOCAPTURE)
	cargo test -p praxis-ai-build-support $(_NOCAPTURE)

test-schema:
	cargo test -p praxis-tests-schema $(_NOCAPTURE)

test-integration:
	cargo test -p praxis-tests-integration $(_NOCAPTURE)
	cargo test -p praxis-tests-integration --features $(EXPERIMENTAL_FEATURES) --test suite \
		-- examples::azure_ad examples::gcp_adc examples::lakera_guard examples::token_rate_limit \
		$(if $(V),--nocapture)

test-inference-fixtures:
	cargo test -p praxis-test-utils $(_NOCAPTURE)
	cargo test -p xtask inference_fixtures $(_NOCAPTURE)
	cargo test -p praxis-tests-integration --test suite inference_fixtures $(_NOCAPTURE)

test-postgres-unit:
	cargo test -p praxis-ai-apis store::tests::pg_ -- --ignored $(_NOCAPTURE)

test-postgres-integration:
	cargo test -p praxis-tests-integration --test suite openai_response_store_postgres -- --ignored $(_NOCAPTURE)

test-token-rate-limit-valkey-unit:
	cargo test -p praxis-ai-filters --features token-rate-limit-filter valkey $(_NOCAPTURE)

test-token-rate-limit-valkey-integration:
	cargo test -p praxis-tests-integration --features token-rate-limit-filter --test suite \
		mixed_algorithm_rules_valkey_backend_isolates_budgets_across_gateway_replicas $(_NOCAPTURE)

openai-conformance:
	cargo xtask openai-conformance $(OPENAI_CONFORMANCE_ARGS)

check-openai-conformance-reference:
	cargo xtask openai-conformance-reference --check

test-openai-conformance: openai-conformance


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
		--features praxis-ai-proxy/azure-ad-filter,praxis-ai-proxy/gcp-adc-filter,praxis-ai-proxy/http-callout-filter,praxis-ai-proxy/token-rate-limit-filter,praxis-tests-integration/azure-ad-filter,praxis-tests-integration/gcp-adc-filter,praxis-tests-integration/http-callout-filter,praxis-tests-integration/token-rate-limit-filter \
		-- -D warnings
	cargo +nightly fmt --all -- --check
	cargo machete --with-metadata .
	cargo xtask lint-deps
	cargo xtask lint-separators
	cargo xtask lint-filter-docs
	cargo xtask lint-example-tests
	cargo xtask lint-markdown-links
	cargo xtask sync-example-readme
	cargo xtask sync-inference-readme
	cargo xtask sync-responses-readme
	cargo xtask check-inference

fmt:
	cargo +nightly fmt --all

doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items

audit:
	cargo audit
	cargo deny check

coverage-check:
	cargo llvm-cov --workspace --json \
		--exclude xtask \
		--ignore-filename-regex '(target/|tests/)' \
		--output-path coverage.json
	@LINE_PCT=$$(jq '.data[0].totals.lines.percent' coverage.json); \
	echo "Line coverage: $${LINE_PCT}%"; \
	if [ $$(echo "$${LINE_PCT} < 95" | bc -l) -eq 1 ]; then \
		echo "FAIL: coverage $${LINE_PCT}% is below 95% threshold"; \
		exit 1; \
	fi

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
	@echo "  release              cargo build --workspace --release"
	@echo "  check                cargo check --workspace"
	@echo "  clean                cargo clean"
	@echo ""
	@echo "Test:"
	@echo "  test                 run all tests"
	@echo "  test-unit            unit tests (providers, filters, server)"
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
	@echo "Praxis override:"
	@echo "  patch-praxis         use ../praxis path deps instead of crates.io"
	@echo "  unpatch-praxis       revert to crates.io praxis deps"
