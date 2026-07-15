# syntax=docker/dockerfile:1

# ------------------------------------------------------------------------------
# Stage 1: Build
# ------------------------------------------------------------------------------

FROM rust:1.96-alpine AS builder

ENV OPENSSL_STATIC=1

RUN apk add --no-cache musl-dev openssl-dev openssl-libs-static pkgconf cmake make g++

WORKDIR /src

# ------------------------------------------------------------------------------
# Cache Build
# ------------------------------------------------------------------------------

# Cache dependency builds: copy only manifests first, then
# create stub source files so `cargo build` resolves and
# compiles all dependencies without the real source code.

# Workspace manifests
COPY Cargo.toml Cargo.lock ./
COPY apis/Cargo.toml ./apis/Cargo.toml
COPY filters/Cargo.toml ./filters/Cargo.toml
COPY server/Cargo.toml ./server/Cargo.toml

# The server crate has a build.rs that discovers external filter
# crates via cargo metadata for build-time auto-registration.
COPY server/build.rs ./server/build.rs

# Strip workspace members not needed for the binary.
RUN sed -i '/xtask/d; /tests\//d' Cargo.toml

# Create stub source files for all crates.
RUN mkdir -p apis/src filters/src server/src \
    && echo '//! stub' > apis/src/lib.rs \
    && echo '//! stub' > filters/src/lib.rs \
    && echo '//! stub' > server/src/lib.rs \
    && printf '//! stub\nfn main() {}\n' > server/src/main.rs

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p praxis-ai-proxy

# ------------------------------------------------------------------------------
# Cache Tricks
# ------------------------------------------------------------------------------

# Replace stubs with real source, then rebuild. Only the project
# crates recompile; all external dependencies are cached.
COPY apis/src ./apis/src
COPY filters/src ./filters/src
COPY server/src ./server/src

RUN find apis/src filters/src server/src \
    -name '*.rs' -exec touch {} +

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release -p praxis-ai-proxy \
    && cp target/release/praxis-ai /usr/local/bin/praxis-ai

# ------------------------------------------------------------------------------
# Stage 2: Runtime
# ------------------------------------------------------------------------------

FROM alpine:3.23

LABEL org.opencontainers.image.source="https://github.com/praxis-proxy/ai" \
    org.opencontainers.image.description="Praxis AI proxy server" \
    org.opencontainers.image.licenses="MIT"

RUN apk add --no-cache ca-certificates \
    && addgroup -S praxis \
    && adduser -S -G praxis -h /nonexistent -s /sbin/nologin praxis \
    && mkdir -p /etc/praxis

COPY --from=builder --chown=root:root --chmod=0555 \
    /usr/local/bin/praxis-ai /usr/local/bin/praxis-ai

USER praxis:praxis

WORKDIR /etc/praxis

EXPOSE 8080 9901

HEALTHCHECK --interval=5s --timeout=3s --start-period=2s \
    CMD wget -qO- http://127.0.0.1:9901/healthy || exit 1

ENTRYPOINT ["praxis-ai"]
