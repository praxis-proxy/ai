// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Build script for the Praxis server.
//!
//! Discovers external filter crates via `cargo metadata` and generates
//! a registration function that calls each crate's `register_filters()`
//! at startup.
//!
//! This file is a thin orchestrator over the build-script environment
//! (`OUT_DIR`, `CARGO_MANIFEST_DIR`, `CARGO_FEATURE_*`, `TARGET`). The
//! dependency-graph scanning and marker-matching logic lives in
//! `praxis-ai-build-support` instead, because build scripts are not
//! compiled as ordinary `cargo test` targets and cannot carry their
//! own unit tests.

#![allow(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "build script: expect covers unrecoverable env errors; println is cargo directives"
)]

use build_support::ActiveFeatures;
use cargo_metadata::{CargoOpt, Metadata};

/// Manifest path of this package, resolved at compile time so that
/// `cargo metadata` always targets `praxis-ai-proxy` regardless of the
/// build script's current working directory.
const MANIFEST_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

fn main() {
    let Some(metadata) = load_metadata() else {
        // `cargo metadata` could not resolve this package in isolation. That
        // happens whenever the manifest is read away from its workspace: notably a
        // vendored build, where `cargo vendor` flattens the tree so the sibling
        // `path` dependencies are no longer beside it, as in a hermetic container
        // build.
        //
        // Only discovery of *external* filter crates is lost. Praxis AI's own
        // filters are registered explicitly by `build_full_registry`, so a server
        // built this way still has its full built-in pipeline. `load_metadata`
        // has already reported the cause.
        write_generated_file(&build_support::generate_registration_code(&[]));
        // Still emit the static directives: without them Cargo falls back to
        // watching only this package's directory, so fixing the workspace root
        // afterwards would not re-run discovery and the empty registration would
        // be silently linked in.
        emit_static_rerun_directives();
        return;
    };

    let crates = build_support::discover_external_filter_crate_names(&metadata);
    let code = build_support::generate_registration_code(&crates);
    write_generated_file(&code);
    emit_rerun_directives(&metadata);
}

/// Report that external filter discovery was skipped, naming the cause.
fn warn_discovery_skipped(cause: &dyn std::fmt::Display) {
    println!("cargo::warning=praxis-ai-proxy: external filter discovery skipped: {cause}");
}

/// Load cargo metadata, narrowed to dependencies available for the current
/// target when Cargo provides one.
///
/// Returns `None` when metadata resolution fails, which is not an error: see the
/// fallback in [`main`]. This mirrors the equivalent build script in Praxis core.
fn load_metadata() -> Option<Metadata> {
    let active_features = active_features()?;
    let mut command = cargo_metadata::MetadataCommand::new();
    command.manifest_path(MANIFEST_PATH);
    apply_active_features(&mut command, active_features);
    if let Ok(target) = std::env::var("TARGET") {
        command.other_options(vec!["--filter-platform".to_owned(), target]);
    }

    match command.exec() {
        Ok(metadata) => Some(metadata),
        Err(error) => {
            warn_discovery_skipped(&error);
            None
        },
    }
}

/// Resolve active package features from Cargo's build-script environment.
///
/// Returns `None` when this package's own manifest cannot be read, for the same
/// reasons described in [`main`].
fn active_features() -> Option<ActiveFeatures> {
    let metadata = match cargo_metadata::MetadataCommand::new()
        .manifest_path(MANIFEST_PATH)
        .no_deps()
        .exec()
    {
        Ok(metadata) => metadata,
        Err(error) => {
            warn_discovery_skipped(&error);
            return None;
        },
    };
    let Some(package) = metadata.packages.iter().find(|pkg| pkg.name == env!("CARGO_PKG_NAME")) else {
        warn_discovery_skipped(&concat!(
            env!("CARGO_PKG_NAME"),
            " is absent from its own cargo metadata"
        ));
        return None;
    };

    Some(build_support::resolve_active_features(
        &package.features,
        std::env::vars(),
    ))
}

/// Apply the current build's active feature set to a `cargo metadata` command.
fn apply_active_features(command: &mut cargo_metadata::MetadataCommand, active_features: ActiveFeatures) {
    if !active_features.default_enabled {
        command.features(CargoOpt::NoDefaultFeatures);
    }
    if !active_features.names.is_empty() {
        command.features(CargoOpt::SomeFeatures(active_features.names));
    }
}

/// Write the generated registration code to `$OUT_DIR/external_filters.rs`.
fn write_generated_file(code: &str) {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest = std::path::Path::new(&out_dir).join("external_filters.rs");
    std::fs::write(&dest, code).expect("failed to write external_filters.rs");
}

/// Tell Cargo to re-run this build script when this crate's own inputs change.
///
/// Emitted on every path, including the fallback, because any `rerun-if-changed`
/// replaces Cargo's default of watching the whole package directory.
fn emit_static_rerun_directives() {
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../Cargo.toml");
    println!("cargo:rerun-if-changed=../Cargo.lock");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Tell Cargo when to re-run this build script.
fn emit_rerun_directives(metadata: &Metadata) {
    emit_static_rerun_directives();

    for manifest_path in build_support::direct_runtime_dependency_manifest_paths(metadata) {
        println!("cargo:rerun-if-changed={manifest_path}");
    }
}
