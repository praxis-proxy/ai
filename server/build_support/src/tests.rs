// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Unit tests for build-script dependency discovery.
//!
//! The fixture-based tests spawn real `cargo metadata` subprocesses
//! against small on-disk crates, exercising actual Cargo dependency
//! resolution (feature gating, renamed dependencies) rather than
//! hand-built `cargo_metadata` structs.

use std::{collections::BTreeMap, fs, path::PathBuf};

use cargo_metadata::{CargoOpt, Metadata, MetadataCommand};
use tempfile::TempDir;

use super::{
    direct_runtime_dependency_manifest_paths, discover_external_filter_crate_names, feature_env_name,
    generate_registration_code, resolve_active_features,
};

#[test]
fn feature_env_name_replaces_hyphens_and_uppercases() {
    assert_eq!(feature_env_name("my-cool-feature"), "MY_COOL_FEATURE");
}

#[test]
fn resolve_active_features_detects_named_features_and_ignores_unmatched() {
    let mut feature_names = BTreeMap::new();
    feature_names.insert("my-feature".to_owned(), Vec::new());
    feature_names.insert("other-feature".to_owned(), Vec::new());

    let env_vars = vec![
        ("CARGO_FEATURE_DEFAULT".to_owned(), "1".to_owned()),
        ("CARGO_FEATURE_MY_FEATURE".to_owned(), "1".to_owned()),
        ("CARGO_FEATURE_SOME_UNKNOWN".to_owned(), "1".to_owned()),
        ("PATH".to_owned(), "/usr/bin".to_owned()),
    ];

    let active = resolve_active_features(&feature_names, env_vars);

    assert!(
        active.default_enabled,
        "CARGO_FEATURE_DEFAULT should mark default features enabled"
    );
    assert_eq!(
        active.names,
        vec!["my-feature".to_owned()],
        "only the matched feature should be reported, and an unmatched env var ignored"
    );
}

#[test]
fn generate_registration_code_emits_calls_and_empty_case_suppression() {
    let code = generate_registration_code(&["alpha".to_owned(), "beta".to_owned()]);
    assert!(
        code.contains("alpha::register_filters(registry);"),
        "missing call for alpha: {code}"
    );
    assert!(
        code.contains("beta::register_filters(registry);"),
        "missing call for beta: {code}"
    );
    assert!(
        !code.contains("no external filters discovered"),
        "suppression should not appear when crates exist: {code}"
    );

    let empty_code = generate_registration_code(&[]);
    assert!(
        empty_code.contains("no external filters discovered"),
        "empty case must document the unused parameter: {empty_code}"
    );
}

#[test]
#[should_panic(expected = "cargo metadata returned no root package")]
fn discover_external_filter_crate_names_panics_when_resolve_has_no_root() {
    let metadata = minimal_metadata_json(r#"{"nodes":[],"root":null}"#);
    let _unused = discover_external_filter_crate_names(&metadata);
}

#[test]
#[should_panic(expected = "cargo metadata returned no resolve graph")]
fn discover_external_filter_crate_names_panics_when_resolve_is_missing() {
    let metadata = minimal_metadata_json("null");
    let _unused = discover_external_filter_crate_names(&metadata);
}

#[test]
#[should_panic(expected = "resolve graph has no node for the root package")]
fn discover_external_filter_crate_names_panics_when_root_has_no_matching_node() {
    let metadata = minimal_metadata_json(r#"{"nodes":[],"root":"praxis-ai-proxy 0.1.0 (path+file:///tmp/fixture)"}"#);
    let _unused = discover_external_filter_crate_names(&metadata);
}

#[test]
fn discovers_normal_and_renamed_marked_dependencies_only() {
    let fixture = Fixture::build();
    let discovered = discover_external_filter_crate_names(&fixture.metadata(&[]));

    assert_eq!(
        discovered,
        vec!["normal_marked".to_owned(), "renamed_marked".to_owned()],
        "should discover the normal marked dep (hyphenated crate name normalized to its Rust \
         import form) and the renamed marked dep under its aliased import name — never its \
         original crate name `renamed-target` — and nothing else, with default features: \
         {discovered:?}"
    );
}

#[test]
fn optional_dependency_discovery_follows_its_gating_feature() {
    let fixture = Fixture::build();

    let without_feature = discover_external_filter_crate_names(&fixture.metadata(&[]));
    assert!(
        !without_feature.contains(&"optional_marked".to_owned()),
        "must be absent when its gating feature is disabled: {without_feature:?}"
    );

    let with_feature = discover_external_filter_crate_names(&fixture.metadata(&["opt-in".to_owned()]));
    assert!(
        with_feature.contains(&"optional_marked".to_owned()),
        "must be discovered once its gating feature is enabled: {with_feature:?}"
    );
}

#[test]
fn unmarked_dev_and_transitive_dependencies_are_excluded() {
    let fixture = Fixture::build();
    let discovered = discover_external_filter_crate_names(&fixture.metadata(&["opt-in".to_owned()]));

    assert!(
        !discovered.contains(&"plain_dep".to_owned()),
        "a dependency without the praxis-filters marker must never be discovered: {discovered:?}"
    );
    assert!(
        !discovered.contains(&"dev_marked".to_owned()),
        "a marked dev-dependency must be excluded: only normal runtime deps are discovered: {discovered:?}"
    );
    // Regression test for #478: `nested-marked` is only a transitive dependency (via the
    // unmarked `praxis-proxy` decoy), so it must never surface from `collect_root_deps`'s
    // direct-edge scan.
    assert!(
        !discovered.contains(&"nested_marked".to_owned()),
        "a marked crate reachable only through a transitive dependency must not be discovered: {discovered:?}"
    );
}

#[test]
fn direct_runtime_dependency_manifest_paths_excludes_dev_and_disabled_optional_deps() {
    let fixture = Fixture::build();
    let metadata = fixture.metadata(&[]);

    let paths = direct_runtime_dependency_manifest_paths(&metadata);

    assert!(
        paths.iter().any(|path| path.contains("normal-marked")),
        "should include the normal runtime dependency's manifest: {paths:?}"
    );
    assert!(
        !paths.iter().any(|path| path.contains("dev-marked")),
        "must exclude dev-dependency manifests: {paths:?}"
    );
    assert!(
        !paths.iter().any(|path| path.contains("optional-marked")),
        "must exclude a disabled optional dependency's manifest: {paths:?}"
    );
}

// -----------------------------------------------------------------------------
// Test Utilities
// -----------------------------------------------------------------------------

/// Build a minimal but structurally-valid `cargo_metadata::Metadata`,
/// substituting a caller-supplied `resolve` field body (or `null`).
fn minimal_metadata_json(resolve_body: &str) -> Metadata {
    let json = format!(
        r#"{{"packages":[],"workspace_members":[],"resolve":{resolve_body},"workspace_root":"/tmp/fixture","target_directory":"/tmp/fixture/target","build_directory":null,"version":1}}"#
    );
    MetadataCommand::parse(json).expect("fixture metadata JSON should deserialize")
}

/// Fixture workspace covering discovery's dependency shapes: a
/// normal marked dep, a renamed (`package = "..."`) marked dep, a
/// feature-gated optional marked dep, an unmarked dep, a marked
/// dev-dependency, and a `praxis-proxy` decoy whose only marked
/// dependency is purely transitive.
struct Fixture {
    _dir: TempDir,
    manifest_path: PathBuf,
}

impl Fixture {
    /// Write the fixture crates to a fresh temporary directory.
    fn build() -> Self {
        let dir = TempDir::new().expect("create fixture temp dir");
        let root = dir.path();

        Self::write_crate(root, "normal-marked", "", true);
        Self::write_crate(root, "renamed-target", "", true);
        Self::write_crate(root, "optional-marked", "", true);
        Self::write_crate(root, "plain-dep", "", false);
        Self::write_crate(root, "dev-marked", "", true);
        Self::write_crate(root, "nested-marked", "", true);
        Self::write_crate(
            root,
            "praxis-proxy",
            "nested-marked = { path = \"../nested-marked\" }\n",
            false,
        );
        let manifest_path = Self::write_root_manifest(root);

        Self {
            _dir: dir,
            manifest_path,
        }
    }

    /// Write the fixture root crate and return its manifest path.
    fn write_root_manifest(root: &std::path::Path) -> PathBuf {
        let manifest_path = root.join("fixture-server").join("Cargo.toml");
        fs::create_dir_all(root.join("fixture-server").join("src")).expect("create fixture root crate dir");
        fs::write(
            &manifest_path,
            "[workspace]\n\n[package]\nname = \"fixture-server\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\nnormal-marked = { path = \"../normal-marked\" }\nrenamed_marked = { package = \"renamed-target\", path = \"../renamed-target\" }\noptional-marked = { path = \"../optional-marked\", optional = true }\nplain-dep = { path = \"../plain-dep\" }\npraxis-proxy = { path = \"../praxis-proxy\" }\n\n[dev-dependencies]\ndev-marked = { path = \"../dev-marked\" }\n\n[features]\nopt-in = [\"dep:optional-marked\"]\n",
        )
        .expect("write fixture root Cargo.toml");
        fs::write(
            root.join("fixture-server").join("src").join("lib.rs"),
            "//! fixture root crate\n",
        )
        .expect("write fixture root lib.rs");

        manifest_path
    }

    /// Run real `cargo metadata` against the fixture with the given
    /// non-default features enabled.
    fn metadata(&self, features: &[String]) -> Metadata {
        let mut command = MetadataCommand::new();
        command.manifest_path(self.manifest_path.clone());
        command.features(CargoOpt::NoDefaultFeatures);
        if !features.is_empty() {
            command.features(CargoOpt::SomeFeatures(features.to_vec()));
        }
        command
            .exec()
            .expect("cargo metadata should succeed against the fixture workspace")
    }

    /// Write one fixture crate under `root/<name>/`.
    fn write_crate(root: &std::path::Path, name: &str, extra_deps: &str, marked: bool) {
        let crate_dir = root.join(name);
        fs::create_dir_all(crate_dir.join("src")).expect("create fixture crate dir");

        let marker = if marked {
            "\n[package.metadata.praxis-filters]\n"
        } else {
            ""
        };
        let dependencies = if extra_deps.is_empty() {
            String::new()
        } else {
            format!("\n[dependencies]\n{extra_deps}")
        };
        fs::write(
            crate_dir.join("Cargo.toml"),
            format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n{marker}{dependencies}"),
        )
        .expect("write fixture crate Cargo.toml");
        fs::write(crate_dir.join("src").join("lib.rs"), "//! fixture crate\n").expect("write fixture crate lib.rs");
    }
}
