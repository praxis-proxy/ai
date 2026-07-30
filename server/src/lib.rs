// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Server bootstrap for Praxis AI.

pub(crate) mod pipelines;
pub(crate) mod reload;
mod server;
pub(crate) mod watcher;
pub use pipelines::resolve_pipelines;
pub use praxis_core::logging::init_tracing;
pub use server::{check_root_privilege, fatal, resolve_config_path, run_server, run_server_with_registry};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Built-in fallback configuration branded for praxis-ai.
const DEFAULT_CONFIG: &str = include_str!("default.yaml");

// -----------------------------------------------------------------------------
// Configuration Loading
// -----------------------------------------------------------------------------

/// Load configuration from an explicit path, falling back to
/// `praxis.yaml` in the working directory, then the praxis-ai
/// built-in default.
///
/// # Errors
///
/// Returns [`ProxyError::Config`] if the resolved config source
/// cannot be loaded or is invalid.
///
/// [`ProxyError::Config`]: praxis_core::errors::ProxyError::Config
pub fn load_config(
    explicit_path: Option<&str>,
) -> Result<praxis_core::config::Config, praxis_core::errors::ProxyError> {
    praxis_core::config::Config::load(explicit_path, DEFAULT_CONFIG)
}

// -----------------------------------------------------------------------------
// External Filter Discovery
// -----------------------------------------------------------------------------

// Provides: fn register_external_filters(&mut FilterRegistry)
include!(concat!(env!("OUT_DIR"), "/external_filters.rs"));

/// Build a [`FilterRegistry`] with core builtins, AI filters, and
/// auto-discovered external filters.
///
/// [`FilterRegistry`]: praxis_filter::FilterRegistry
#[must_use]
pub fn build_full_registry() -> praxis_filter::FilterRegistry {
    let mut registry = praxis_filter::FilterRegistry::with_builtins();
    praxis_ai_filters::register_ai_filters(&mut registry);
    register_external_filters(&mut registry);
    registry
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use praxis_core::config::Config;

    use super::*;

    #[test]
    fn default_config_parses_successfully() {
        let config = Config::from_yaml(DEFAULT_CONFIG).expect("DEFAULT_CONFIG should parse");
        assert!(
            !config.listeners.is_empty(),
            "default config should define at least one listener"
        );
    }

    #[test]
    fn default_config_brands_praxis_ai() {
        let config = Config::from_yaml(DEFAULT_CONFIG).expect("DEFAULT_CONFIG should parse");
        let body = config.filter_chains[0].filters[0].config["body"]
            .as_str()
            .expect("AI default static response should define a string body");
        assert_eq!(
            body, r#"{"status": "ok", "server": "praxis-ai"}"#,
            "default config should use the AI-branded response body"
        );
    }

    #[test]
    fn load_config_none_succeeds() {
        let config = load_config(None).expect("load_config(None) should succeed");
        assert!(
            !config.listeners.is_empty(),
            "loaded config should define at least one listener"
        );
    }
}
