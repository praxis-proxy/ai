// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! Test utilities for loading and patching example configuration files.

use std::collections::HashMap;

use praxis_core::config::Config;

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

/// Load an example config YAML, patch the listener and endpoint
/// addresses with free ports, and return the parsed [`Config`].
///
/// `port_map` maps original `"host:port"` strings to replacement
/// ports on `127.0.0.1`.
///
/// # Panics
///
/// Panics if the config file cannot be read or parsed.
///
/// # Examples
///
/// ```no_run
/// use std::collections::HashMap;
///
/// let config = praxis_test_utils::load_example_config(
///     "model-to-header-routing.yaml",
///     9090,
///     HashMap::from([("10.0.1.1:8080", 19998_u16)]),
/// );
/// assert!(!config.listeners.is_empty());
/// ```
///
/// [`Config`]: praxis_core::config::Config
#[expect(clippy::needless_pass_by_value, reason = "callers construct inline")]
pub fn load_example_config(filename: &str, listener_port: u16, port_map: HashMap<&str, u16>) -> Config {
    let path = example_config_path(filename);
    let yaml = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let patched = allow_loopback_endpoints(&patch_yaml(&yaml, listener_port, &port_map));
    Config::from_yaml(&patched).unwrap_or_else(|e| panic!("parse {filename}: {e}"))
}

/// Enable `allow_private_endpoints` on a patched example config.
///
/// The port map rewrites example endpoints to loopback test backends, which
/// the endpoint SSRF validation would otherwise reject; the flag is a
/// property of the test harness rewrite, not of the example itself.
///
/// # Examples
///
/// ```
/// let yaml = praxis_test_utils::allow_loopback_endpoints("listeners: []\n");
/// assert!(yaml.contains("allow_private_endpoints: true"));
/// ```
pub fn allow_loopback_endpoints(yaml: &str) -> String {
    ensure_insecure_option_bool(yaml, "allow_private_endpoints", true)
}

/// Set `insecure_options.<key>` when absent, via the parsed mapping.
///
/// Textual `replacen` misses layouts such as `insecure_options: # comment`
/// or inline mappings; parsing preserves existing entries and explicit values.
///
/// # Panics
///
/// Panics if `yaml` is not valid YAML or its document root is not a mapping,
/// so callers see the original parse/diagnostic instead of a later
/// `Config::from_yaml` failure after a silent text append.
fn ensure_insecure_option_bool(yaml: &str, key: &str, value: bool) -> String {
    if insecure_options_has_key(yaml, key) {
        return yaml.to_owned();
    }
    let mut root: serde_yaml::Value =
        serde_yaml::from_str(yaml).unwrap_or_else(|e| panic!("example config is not valid YAML: {e}"));
    let root_map = root
        .as_mapping_mut()
        .unwrap_or_else(|| panic!("example config root must be a mapping"));
    let opts_key = serde_yaml::Value::String("insecure_options".into());
    let opts = root_map
        .entry(opts_key)
        .or_insert_with(|| serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
    if let serde_yaml::Value::Mapping(m) = opts {
        m.insert(serde_yaml::Value::String(key.into()), serde_yaml::Value::Bool(value));
    } else {
        let mut m = serde_yaml::Mapping::new();
        m.insert(serde_yaml::Value::String(key.into()), serde_yaml::Value::Bool(value));
        *opts = serde_yaml::Value::Mapping(m);
    }
    serde_yaml::to_string(&root).unwrap_or_else(|e| panic!("serialize example config YAML: {e}"))
}

/// True when the parsed `insecure_options` mapping sets `key`.
///
/// String search is not used: a comment mentioning the key must not
/// suppress the harness override.
fn insecure_options_has_key(yaml: &str, key: &str) -> bool {
    let Ok(serde_yaml::Value::Mapping(root)) = serde_yaml::from_str::<serde_yaml::Value>(yaml) else {
        return false;
    };
    let Some(serde_yaml::Value::Mapping(opts)) = root.get(serde_yaml::Value::String("insecure_options".into())) else {
        return false;
    };
    opts.contains_key(serde_yaml::Value::String(key.into()))
}

/// Resolve the absolute path to an example config file.
///
/// # Examples
///
/// ```
/// let path = praxis_test_utils::example_config_path("model-to-header-routing.yaml");
/// assert!(path.contains("examples/configs/"));
/// ```
pub fn example_config_path(filename: &str) -> String {
    format!("{}/../../examples/configs/{filename}", env!("CARGO_MANIFEST_DIR"),)
}

/// Replace the default listener address and all endpoint addresses
/// in a YAML string.
///
/// Rewrites both `0.0.0.0:8080` and `127.0.0.1:8080` to the given
/// `listener_port`, and applies every entry in `port_map`.
///
/// Rewriting is a single left-to-right pass, and an address only matches when
/// its port is not followed by another digit. Both rules keep one replacement
/// from corrupting another: a chain of plain `str::replace` calls rewrites
/// text it has already produced, so patching `127.0.0.1:3001` over a listener
/// already moved to `127.0.0.1:30011` would leave a trailing `1` welded to the
/// substituted port.
///
/// # Examples
///
/// ```
/// use std::collections::HashMap;
///
/// let yaml = "address: \"0.0.0.0:8080\"";
/// let result = praxis_test_utils::patch_yaml(yaml, 9999, &HashMap::new());
/// assert_eq!(result, "address: \"127.0.0.1:9999\"");
/// ```
///
/// # Panics
///
/// Panics if `port_map` contains an empty address, which would match at every
/// position without consuming any input.
pub fn patch_yaml(yaml: &str, listener_port: u16, port_map: &HashMap<&str, u16>) -> String {
    assert!(
        port_map.keys().all(|address| !address.is_empty()),
        "port_map addresses must not be empty"
    );

    // Longest needle first so a shorter address that prefixes another cannot
    // claim the match; `port_map` wins ties, since it is caller-supplied.
    let mut rules: Vec<(&str, u16)> = port_map.iter().map(|(address, port)| (*address, *port)).collect();
    rules.sort_unstable_by_key(|(address, _)| std::cmp::Reverse(address.len()));
    for listener in ["0.0.0.0:8080", "127.0.0.1:8080"] {
        if !port_map.contains_key(listener) {
            rules.push((listener, listener_port));
        }
    }

    let mut result = String::with_capacity(yaml.len());
    let mut rest = yaml;
    'next: while !rest.is_empty() {
        for (address, port) in &rules {
            // A trailing digit means the needle matched a longer port, not
            // this address.
            if let Some(tail) = rest.strip_prefix(*address)
                && !tail.starts_with(|next: char| next.is_ascii_digit())
            {
                result.push_str("127.0.0.1:");
                result.push_str(&port.to_string());
                rest = tail;
                continue 'next;
            }
        }
        let mut chars = rest.chars();
        if let Some(head) = chars.next() {
            result.push(head);
        }
        rest = chars.as_str();
    }
    result
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_yaml_replaces_listener_0000() {
        let yaml = "address: \"0.0.0.0:8080\"";
        let result = patch_yaml(yaml, 9999, &HashMap::new());
        assert_eq!(result, "address: \"127.0.0.1:9999\"", "0.0.0.0 should be replaced");
    }

    #[test]
    fn patch_yaml_replaces_listener_localhost() {
        let yaml = "address: \"127.0.0.1:8080\"";
        let result = patch_yaml(yaml, 9999, &HashMap::new());
        assert_eq!(result, "address: \"127.0.0.1:9999\"", "localhost should be replaced");
    }

    #[test]
    fn patch_yaml_replaces_endpoints() {
        let map = HashMap::from([("127.0.0.1:3000", 5555_u16), ("127.0.0.1:4000", 6666_u16)]);
        let yaml = "- \"127.0.0.1:3000\"\n- \"127.0.0.1:4000\"";
        let result = patch_yaml(yaml, 8080, &map);
        assert!(
            result.contains("127.0.0.1:5555"),
            "first endpoint should be patched to port 5555"
        );
        assert!(
            result.contains("127.0.0.1:6666"),
            "second endpoint should be patched to port 6666"
        );
    }

    /// Regression: a listener port that begins with an endpoint port used to
    /// be rewritten a second time by that endpoint's replacement.
    #[test]
    fn patch_yaml_does_not_rewrite_a_listener_that_extends_an_endpoint() {
        let map = HashMap::from([("127.0.0.1:3001", 26000_u16)]);
        let yaml = "address: \"0.0.0.0:8080\"\nendpoint: \"127.0.0.1:3001\"";

        let result = patch_yaml(yaml, 30011, &map);

        assert_eq!(
            result, "address: \"127.0.0.1:30011\"\nendpoint: \"127.0.0.1:26000\"",
            "the listener keeps its own port and the endpoint is patched once"
        );
    }

    /// The pass must not re-examine text it just produced.
    #[test]
    fn patch_yaml_does_not_rewrite_its_own_output() {
        let map = HashMap::from([("127.0.0.1:3000", 3001_u16), ("127.0.0.1:3001", 4000_u16)]);
        let yaml = "a: \"127.0.0.1:3000\"\nb: \"127.0.0.1:3001\"";

        let result = patch_yaml(yaml, 8080, &map);

        assert_eq!(
            result, "a: \"127.0.0.1:3001\"\nb: \"127.0.0.1:4000\"",
            "the port substituted for the first address must not be patched again"
        );
    }

    #[test]
    fn patch_yaml_leaves_unmatched_unchanged() {
        let yaml = "upstream: \"10.0.0.1:443\"";
        let result = patch_yaml(yaml, 8080, &HashMap::new());
        assert_eq!(result, yaml, "unmatched addresses should stay unchanged");
    }

    /// An empty address matches everywhere without consuming input, so the
    /// scan would append replacements forever; reject it up front.
    #[test]
    #[should_panic(expected = "port_map addresses must not be empty")]
    fn patch_yaml_rejects_an_empty_address() {
        patch_yaml("address: \"0.0.0.0:8080\"", 8080, &HashMap::from([("", 5555_u16)]));
    }

    #[test]
    fn example_config_path_resolves() {
        let path = example_config_path("model-to-header-routing.yaml");
        assert!(std::path::Path::new(&path).exists(), "expected {path} to exist");
    }

    #[test]
    fn load_example_config_parses() {
        let config = load_example_config(
            "model-to-header-routing.yaml",
            19999,
            HashMap::from([("10.0.1.1:8080", 19998_u16)]),
        );
        assert_eq!(
            config.listeners[0].address, "127.0.0.1:19999",
            "listener address should be patched"
        );
    }

    #[test]
    fn allow_loopback_inserts_private_endpoints_when_insecure_options_has_inline_comment() {
        let yaml = "listeners: []\ninsecure_options: # test settings\n  allow_private_upstreams: true\n";
        let patched = allow_loopback_endpoints(yaml);
        assert!(
            insecure_options_has_key(&patched, "allow_private_endpoints"),
            "inline comment on insecure_options must not block allow_private_endpoints insert"
        );
        assert!(
            insecure_options_has_key(&patched, "allow_private_upstreams"),
            "existing insecure_options entries must be preserved"
        );
    }

    #[test]
    #[should_panic(expected = "example config is not valid YAML")]
    fn allow_loopback_panics_on_invalid_yaml_with_parse_diagnostic() {
        let _ = allow_loopback_endpoints("listeners: [\n");
    }

    #[test]
    #[should_panic(expected = "example config root must be a mapping")]
    fn allow_loopback_panics_when_root_is_not_a_mapping() {
        let _ = allow_loopback_endpoints("- just\n- a\n- list\n");
    }
}
