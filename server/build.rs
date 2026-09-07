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
    reason = "build script: panics are the only error path; println is cargo directives"
)]

use build_support::ActiveFeatures;
use cargo_metadata::{CargoOpt, Metadata};

/// Manifest path of this package, resolved at compile time so that
/// `cargo metadata` always targets `praxis-ai-proxy` regardless of the
/// build script's current working directory.
const MANIFEST_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

fn main() {
    let metadata = load_metadata();
    let crates = build_support::discover_external_filter_crate_names(&metadata);
    let code = build_support::generate_registration_code(&crates);
    write_generated_file(&code);
    emit_rerun_directives(&metadata);
}

/// Load cargo metadata, narrowed to dependencies available for the current
/// target when Cargo provides one.
fn load_metadata() -> Metadata {
    let active_features = active_features();
    let mut command = cargo_metadata::MetadataCommand::new();
    command.manifest_path(MANIFEST_PATH);
    apply_active_features(&mut command, active_features);
    if let Ok(target) = std::env::var("TARGET") {
        command.other_options(vec!["--filter-platform".to_owned(), target]);
    }

    command.exec().expect("failed to run cargo metadata")
}

/// Resolve active package features from Cargo's build-script environment.
fn active_features() -> ActiveFeatures {
    let metadata = cargo_metadata::MetadataCommand::new()
        .manifest_path(MANIFEST_PATH)
        .no_deps()
        .exec()
        .expect("failed to read package feature metadata");
    let package = metadata
        .packages
        .iter()
        .find(|pkg| pkg.name == env!("CARGO_PKG_NAME"))
        .expect("praxis-ai package not found in metadata");

    build_support::resolve_active_features(&package.features, std::env::vars())
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

/// Tell Cargo when to re-run this build script.
fn emit_rerun_directives(metadata: &Metadata) {
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../Cargo.toml");
    println!("cargo:rerun-if-changed=../Cargo.lock");
    println!("cargo:rerun-if-changed=build.rs");

    for manifest_path in build_support::direct_runtime_dependency_manifest_paths(metadata) {
        println!("cargo:rerun-if-changed={manifest_path}");
    }
}
