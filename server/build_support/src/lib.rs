// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Dependency-graph and marker-scanning logic for `praxis-ai-proxy`'s
//! external-filter auto-discovery build script.
//!
//! Isolated from `server/build.rs` because build scripts are never
//! compiled as `cargo test` targets, so this logic must live in an
//! ordinary library crate to be unit tested (see `tests.rs`) against
//! real `cargo metadata` output. `build.rs` stays a thin orchestrator
//! over the build-script environment.

#![deny(unsafe_code)]

use std::{
    collections::{BTreeMap, HashMap},
    fmt::Write as _,
};

use cargo_metadata::{DependencyKind, Metadata, NodeDep, Package, PackageId, Resolve};

// -----------------------------------------------------------------------------
// Public Types
// -----------------------------------------------------------------------------

/// Active Cargo feature selection for a build script invocation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ActiveFeatures {
    /// Whether Cargo enabled the package's default feature set.
    pub default_enabled: bool,

    /// Explicit active non-default feature names.
    pub names: Vec<String>,
}

// -----------------------------------------------------------------------------
// Public Functions
// -----------------------------------------------------------------------------

/// Convert a Cargo feature name to the corresponding build-script env
/// suffix, e.g. `my-feature` becomes `MY_FEATURE`.
#[must_use]
pub fn feature_env_name(name: &str) -> String {
    name.replace('-', "_").to_ascii_uppercase()
}

/// Resolve active package features from build-script environment
/// variables.
///
/// Pure over its inputs so it can be unit tested without a real
/// build-script environment: pass `std::env::vars()` in production
/// and a fixed list in tests.
#[must_use]
pub fn resolve_active_features(
    feature_names: &BTreeMap<String, Vec<String>>,
    env_vars: impl IntoIterator<Item = (String, String)>,
) -> ActiveFeatures {
    let feature_names_by_env: HashMap<String, String> = feature_names
        .keys()
        .map(|name| (feature_env_name(name), name.clone()))
        .collect();

    let mut default_enabled = false;
    let mut names: Vec<String> = Vec::new();
    for (key, _value) in env_vars {
        let Some(suffix) = key.strip_prefix("CARGO_FEATURE_") else {
            continue;
        };
        if suffix == "DEFAULT" {
            default_enabled = true;
        } else if let Some(name) = feature_names_by_env.get(suffix) {
            names.push(name.clone());
        }
    }
    names.sort();
    names.dedup();

    ActiveFeatures { default_enabled, names }
}

/// Collect the dependency edges of the resolved root package.
///
/// Anchored on [`Resolve::root`] rather than a hardcoded package
/// name: matching packages by the literal name `"praxis-proxy"`
/// previously collided with the core alias `praxis = { package =
/// "praxis-proxy" }`, silently scanning the wrong node's edges and
/// discovering nothing.
///
/// # Panics
///
/// Panics if `resolve.root` is `None`, or if no node in
/// `resolve.nodes` matches it; verify `MetadataCommand` has an
/// explicit `manifest_path` and does not pass `--no-deps`.
#[must_use]
#[expect(clippy::expect_used, reason = "build-time discovery: panics are the only error path")]
pub fn collect_root_deps(resolve: &Resolve) -> Vec<&NodeDep> {
    let root = resolve.root.as_ref().expect(
        "cargo metadata returned no root package; verify MetadataCommand is invoked with an \
         explicit manifest_path and without --no-deps",
    );
    resolve
        .nodes
        .iter()
        .find(|node| &node.id == root)
        .expect(
            "cargo metadata resolve graph has no node for the root package; verify \
             MetadataCommand is invoked with an explicit manifest_path and without --no-deps",
        )
        .deps
        .iter()
        .collect()
}

/// Check whether a dependency edge is available to normal runtime
/// code (as opposed to `dev-dependencies` or `build-dependencies`).
#[must_use]
pub fn is_runtime_dependency(dep: &NodeDep) -> bool {
    dep.dep_kinds.iter().any(|kind| kind.kind == DependencyKind::Normal)
}

/// Check whether a package carries `[package.metadata.praxis-filters]`.
#[must_use]
pub fn has_praxis_filter_marker(pkg: &Package) -> bool {
    pkg.metadata
        .as_object()
        .is_some_and(|obj| obj.contains_key("praxis-filters"))
}

/// Scan the resolved root package's direct runtime dependencies for
/// `[package.metadata.praxis-filters]` and return their Rust import
/// names (post Cargo alias/rename).
///
/// Only scans [`Resolve::root`]'s direct edges, never the full
/// `metadata.packages` list — otherwise a marked crate reachable only
/// transitively, or gated behind a disabled feature, would be
/// discovered even though it isn't an active dependency.
///
/// # Panics
///
/// Panics if `metadata.resolve` is `None` (see [`collect_root_deps`]).
#[must_use]
pub fn discover_external_filter_crate_names(metadata: &Metadata) -> Vec<String> {
    let packages = packages_by_id(&metadata.packages);
    let resolve = resolve_or_panic(metadata);

    let mut crates: Vec<String> = collect_root_deps(resolve)
        .into_iter()
        .filter(|dep| is_runtime_dependency(dep))
        .filter_map(|dep| packages.get(&dep.pkg).map(|pkg| (dep, *pkg)))
        .filter(|(_, pkg)| has_praxis_filter_marker(pkg))
        // dep.name is the post-alias import path; the marker lookup above keys on
        // dep.pkg, so renaming a dependency cannot hide or spoof its marker status.
        .map(|(dep, _)| dep.name.clone())
        .collect();

    crates.sort();
    crates.dedup();
    crates
}

/// Return manifest paths for the resolved root package's direct
/// runtime dependencies, for `cargo:rerun-if-changed` directives.
///
/// # Panics
///
/// Panics if `metadata.resolve` is `None` (see [`collect_root_deps`]).
#[must_use]
pub fn direct_runtime_dependency_manifest_paths(metadata: &Metadata) -> Vec<String> {
    let packages = packages_by_id(&metadata.packages);
    let resolve = resolve_or_panic(metadata);

    let mut paths: Vec<String> = collect_root_deps(resolve)
        .into_iter()
        .filter(|dep| is_runtime_dependency(dep))
        .filter_map(|dep| packages.get(&dep.pkg).map(|pkg| pkg.manifest_path.to_string()))
        .collect();

    paths.sort();
    paths.dedup();
    paths
}

/// Generate the `register_external_filters` function body.
#[must_use]
#[expect(clippy::expect_used, reason = "writing to a String cannot fail")]
pub fn generate_registration_code(crates: &[String]) -> String {
    let mut code = String::from(
        "/// Register all auto-discovered external filter crates.\n\
         ///\n\
         /// Generated by `build.rs` from dependencies carrying\n\
         /// `[package.metadata.praxis-filters]` in their `Cargo.toml`.\n\
         ///\n\
         /// # Panics\n\
         ///\n\
         /// Panics if any external filter name conflicts with a\n\
         /// built-in or previously registered filter.\n",
    );

    if crates.is_empty() {
        code.push_str(
            "#[expect(\n    \
             unused_variables,\n    \
             clippy::needless_pass_by_ref_mut,\n    \
             reason = \"generated: no external filters discovered\"\n\
             )]\n",
        );
    }

    code.push_str("fn register_external_filters(registry: &mut praxis_filter::FilterRegistry) {\n");

    for crate_name in crates {
        writeln!(code, "    {crate_name}::register_filters(registry);").expect("writing to String should not fail");
    }

    code.push_str("}\n");
    code
}

// -----------------------------------------------------------------------------
// Private Utility Functions
// -----------------------------------------------------------------------------

/// Build a package lookup by package ID.
fn packages_by_id(packages: &[Package]) -> HashMap<&PackageId, &Package> {
    packages.iter().map(|pkg| (&pkg.id, pkg)).collect()
}

/// Borrow the resolve graph, panicking with an actionable message if
/// `cargo metadata` did not produce one.
#[expect(clippy::expect_used, reason = "build-time discovery: panics are the only error path")]
fn resolve_or_panic(metadata: &Metadata) -> &Resolve {
    metadata
        .resolve
        .as_ref()
        .expect("cargo metadata returned no resolve graph; verify MetadataCommand does not pass --no-deps")
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests"
)]
mod tests;
