// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! End-to-end regression test for external-filter auto-discovery
//! (issue #478).

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    reason = "integration test"
)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        process::Command,
    };

    use tempfile::TempDir;

    /// Name of the filter exported by the fixture external filter crate.
    /// Must match the name passed to `export_filters!` in
    /// [`Fixture::write_filter_crate`].
    const FIXTURE_FILTER_NAME: &str = "e2e_fixture_filter";

    #[test]
    fn external_filter_dependency_is_discovered_and_registered_at_runtime() {
        if !Fixture::praxis_filter_dir().join("Cargo.toml").is_file() {
            eprintln!(
                "skipping external_filter_discovery_e2e; sibling ../praxis checkout not found \
                 (clone https://github.com/praxis-proxy/praxis as ../praxis to run this test)"
            );
            return;
        }

        let workspace = Fixture::write();

        let output = Command::new(env!("CARGO"))
            .arg("run")
            .current_dir(workspace.fixture_server_dir())
            .output()
            .expect("failed to spawn `cargo run` for the fixture server");

        assert!(
            output.status.success(),
            "fixture server build/run failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        let expected_line = format!("FILTER:{FIXTURE_FILTER_NAME}");
        assert!(
            stdout.lines().any(|line| line == expected_line),
            "expected the auto-discovered external filter to be registered at runtime; got \
             stdout:\n{stdout}"
        );
    }

    /// An isolated, on-disk Cargo workspace: `filter-crate` is an
    /// external filter crate carrying `[package.metadata.praxis-filters]`;
    /// `fixture-server` is a `praxis-ai-proxy` stand-in whose `build.rs`
    /// calls the real `praxis-ai-build-support` discovery functions and
    /// whose `main` prints every filter registered into a fresh
    /// `praxis_filter::FilterRegistry`.
    struct Fixture {
        dir: TempDir,
    }

    impl Fixture {
        /// Write the fixture workspace to a fresh temporary directory.
        fn write() -> Self {
            let dir = TempDir::new().expect("create fixture temp dir");
            let root = dir.path();

            fs::write(
                root.join("Cargo.toml"),
                "[workspace]\nmembers = [\"filter-crate\", \"fixture-server\"]\n",
            )
            .expect("write fixture workspace Cargo.toml");
            Self::write_toolchain_pin(root);
            Self::write_filter_crate(root);
            Self::write_fixture_server(root);

            Self { dir }
        }

        /// Directory containing the fixture server crate.
        fn fixture_server_dir(&self) -> PathBuf {
            self.dir.path().join("fixture-server")
        }

        /// Absolute path to the sibling Praxis core repository's filter
        /// crate, resolved from `CARGO_MANIFEST_DIR` so this test does
        /// not depend on the current working directory.
        fn praxis_filter_dir() -> PathBuf {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .expect("server/ should have a workspace root two levels up")
                .join("praxis/filter")
        }

        /// Absolute path to this repo's `praxis-ai-build-support` crate.
        fn build_support_dir() -> PathBuf {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("build_support")
        }

        /// Pin the same toolchain as the enclosing repo: the Praxis core
        /// crates this fixture depends on require it, and the fixture
        /// otherwise has no `rust-toolchain.toml` of its own to inherit.
        fn write_toolchain_pin(root: &Path) {
            let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("server/ should have a parent directory");
            let toolchain = fs::read_to_string(workspace_root.join("rust-toolchain.toml"))
                .expect("read repo's rust-toolchain.toml");
            fs::write(root.join("rust-toolchain.toml"), toolchain).expect("write fixture rust-toolchain.toml");
        }

        /// Write the marked external filter crate.
        fn write_filter_crate(root: &Path) {
            let crate_dir = root.join("filter-crate");
            fs::create_dir_all(crate_dir.join("src")).expect("create filter-crate dir");

            fs::write(crate_dir.join("Cargo.toml"), Self::filter_crate_manifest())
                .expect("write filter-crate Cargo.toml");
            fs::write(crate_dir.join("src/lib.rs"), Self::filter_crate_lib_rs())
                .expect("write filter-crate src/lib.rs");
        }

        /// `Cargo.toml` contents for the fixture external filter crate.
        fn filter_crate_manifest() -> String {
            let praxis_filter_dir = Self::praxis_filter_dir();
            format!(
                "[package]\nname = \"e2e-fixture-filter\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[package.metadata.praxis-filters]\n\n[dependencies]\nasync-trait = \"0.1\"\npraxis-proxy-filter = {{ path = {praxis_filter_dir:?} }}\nserde_yaml = {{ package = \"yaml_serde\", version = \"0.10.4\" }}\n"
            )
        }

        /// `src/lib.rs` contents for the fixture external filter crate:
        /// a trivial `HttpFilter` exported via `export_filters!`.
        fn filter_crate_lib_rs() -> String {
            format!(
                "use async_trait::async_trait;\nuse praxis_filter::{{FilterAction, FilterError, HttpFilter, HttpFilterContext, export_filters}};\n\nstruct NoopFilter;\n\n#[async_trait]\nimpl HttpFilter for NoopFilter {{\n    fn name(&self) -> &'static str {{\n        \"{FIXTURE_FILTER_NAME}\"\n    }}\n\n    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {{\n        Ok(FilterAction::Continue)\n    }}\n}}\n\nfn from_config(_config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {{\n    Ok(Box::new(NoopFilter))\n}}\n\nexport_filters! {{\n    http \"{FIXTURE_FILTER_NAME}\" => from_config,\n}}\n"
            )
        }

        /// Write the `praxis-ai-proxy` stand-in that exercises the real
        /// discovery build script logic.
        fn write_fixture_server(root: &Path) {
            let crate_dir = root.join("fixture-server");
            fs::create_dir_all(crate_dir.join("src")).expect("create fixture-server dir");

            fs::write(crate_dir.join("Cargo.toml"), Self::fixture_server_manifest())
                .expect("write fixture-server Cargo.toml");
            fs::write(crate_dir.join("build.rs"), Self::fixture_server_build_rs())
                .expect("write fixture-server build.rs");
            fs::write(crate_dir.join("src/main.rs"), Self::fixture_server_main_rs())
                .expect("write fixture-server src/main.rs");
        }

        /// `Cargo.toml` contents for the fixture server crate.
        fn fixture_server_manifest() -> String {
            let praxis_filter_dir = Self::praxis_filter_dir();
            let build_support_dir = Self::build_support_dir();
            format!(
                "[package]\nname = \"e2e-fixture-server\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\npraxis-proxy-filter = {{ path = {praxis_filter_dir:?} }}\ne2e-fixture-filter = {{ path = \"../filter-crate\" }}\nserde_yaml = {{ package = \"yaml_serde\", version = \"0.10.4\" }}\n\n[build-dependencies]\ncargo_metadata = \"0.23.1\"\npraxis-ai-build-support = {{ path = {build_support_dir:?} }}\n"
            )
        }

        /// `build.rs` contents for the fixture server: a minimal
        /// version of `server/build.rs` calling the real
        /// `praxis-ai-build-support` discovery functions.
        fn fixture_server_build_rs() -> &'static str {
            "fn main() {\n    let mut command = cargo_metadata::MetadataCommand::new();\n    command.manifest_path(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/Cargo.toml\"));\n    let metadata = command.exec().expect(\"cargo metadata should succeed\");\n\n    let crates = build_support::discover_external_filter_crate_names(&metadata);\n    let code = build_support::generate_registration_code(&crates);\n\n    let out_dir = std::env::var(\"OUT_DIR\").expect(\"OUT_DIR not set\");\n    let dest = std::path::Path::new(&out_dir).join(\"external_filters.rs\");\n    std::fs::write(&dest, code).expect(\"failed to write external_filters.rs\");\n}\n"
        }

        /// `src/main.rs` contents for the fixture server: builds a
        /// registry with the auto-discovered filters and prints every
        /// registered filter name.
        fn fixture_server_main_rs() -> &'static str {
            "include!(concat!(env!(\"OUT_DIR\"), \"/external_filters.rs\"));\n\nfn main() {\n    let mut registry = praxis_filter::FilterRegistry::with_builtins();\n    register_external_filters(&mut registry);\n    let mut names = registry.available_filters();\n    names.sort();\n    for name in names {\n        println!(\"FILTER:{name}\");\n    }\n}\n"
        }
    }
}
