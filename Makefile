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
	@echo "Praxis override:"
	@echo "  patch-praxis         use ../praxis path deps instead of crates.io"
	@echo "  unpatch-praxis       revert to crates.io praxis deps"
