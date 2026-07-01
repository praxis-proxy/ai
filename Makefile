MSRV := 1.96.0

.PHONY: fmt fmt-check clippy machete lint check test msrv audit deny supply-chain coverage integration praxis-compat

## Formatting

fmt:
	cargo +nightly fmt --all

fmt-check:
	cargo +nightly fmt --all -- --check

## Linting

clippy:
	cargo +$(MSRV) clippy --workspace --all-targets --all-features -- -D warnings

machete:
	cargo machete

lint: fmt-check clippy machete

## Build

check:
	cargo check --workspace --all-targets --all-features

## Testing

test:
	cargo test --workspace --all-features

integration:
	cargo test --workspace --all-features -- --ignored integration

msrv:
	cargo +$(MSRV) check --workspace --all-targets --all-features

## Security

audit:
	cargo audit

deny:
	cargo deny check

supply-chain: audit deny

## Coverage

coverage:
	cargo llvm-cov --workspace --all-features --fail-under-lines 95
