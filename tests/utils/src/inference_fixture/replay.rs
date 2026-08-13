//! Data-driven materialization and replay of two-sided inference fixtures.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    net::{IpAddr, SocketAddr},
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use futures::StreamExt as _;
use http::{HeaderName, HeaderValue};
use praxis_core::config::{ChainRef, Config, FilterEntry, ProtocolKind};
use serde_json::Value;

use super::{
    FixtureError, FixtureProvenance, ImportedUpstream, InferenceScenario, NormalizationMetadata, RecordedBody,
    RecordedExchange, RecordedRequest, RecordedResponse, RedactionRules, ScenarioExpectation, ScenarioTurn,
    WIRE_FIXTURE_VERSION, WireFixture, WireTurn,
    bounds::{MAX_SCRIPTED_RESPONSE_BODY_BYTES, body_has_rendered_content, parse_response_body, validate_request_body},
    header_policy::{http_fixture_headers, recorded_transport_headers},
    http_server::ScriptedHttpServer,
    sanitize::sanitize_fixture_preserving_structure,
};
use crate::{ProxyGuard, example_config_path, free_port_guard, patch_yaml, start_proxy};

/// Maximum time allowed for one backend capture to become observable.
const EXCHANGE_TIMEOUT: Duration = Duration::from_secs(5);

/// Result of replaying a scenario against an expected two-sided fixture.
#[derive(Debug, Eq, PartialEq)]
pub struct ReplayReport {
    /// Fully normalized wire behavior observed during replay.
    pub actual: WireFixture,
}

/// Runs inference scenarios through a named Praxis example configuration.
pub struct ScenarioRunner;

impl ScenarioRunner {
    /// Materializes a scenario using imported ordered upstream exchanges.
    ///
    /// One scripted backend and one Praxis pipeline are used for the complete
    /// scenario, and client turns are sent sequentially to preserve connection,
    /// script, and multi-turn ordering state.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible provenance, missing or extra upstream
    /// entries, HTTP/runtime failures, expectation failures, or a normalized
    /// imported request that differs from the request Praxis emitted.
    pub async fn materialize(
        scenario: &InferenceScenario,
        provenance: FixtureProvenance,
        upstream: Vec<ImportedUpstream>,
    ) -> Result<WireFixture, FixtureError> {
        let rules = RedactionRules::default();
        Self::materialize_with_rules(scenario, provenance, upstream, &rules).await
    }

    /// Materializes a scenario with caller-provided literal redactions.
    ///
    /// The rules participate in request comparison and in the runner's single
    /// final fixture sanitization pass.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::materialize`], plus errors caused by
    /// invalid custom redaction rules.
    #[expect(
        clippy::too_many_lines,
        clippy::large_stack_frames,
        reason = "the scenario lifecycle keeps ownership and one-time sanitization order explicit"
    )]
    pub async fn materialize_with_rules(
        scenario: &InferenceScenario,
        provenance: FixtureProvenance,
        upstream: Vec<ImportedUpstream>,
        rules: &RedactionRules,
    ) -> Result<WireFixture, FixtureError> {
        scenario.validate_version()?;
        validate_upstream_inputs(scenario, &provenance, &upstream)?;
        validate_scenario_requests(scenario)?;
        let bound = scenario.bind_model(&provenance.model);
        validate_scenario_requests(&bound)?;
        let config_source = load_and_validate_config_source(&bound)?;

        let mut expected_requests = Vec::with_capacity(upstream.len());
        let mut response_owners = Vec::with_capacity(upstream.len());
        for imported in upstream {
            expected_requests.push(imported.exchange.request);
            response_owners.push(Arc::new(imported.exchange.response));
        }

        let scripts = response_owners.iter().map(Arc::clone).collect();
        let backend = ScriptedHttpServer::start_for_proxy(scripts)?;

        let listener = free_port_guard();
        let listener_port = listener.port();
        let config = build_replay_config(
            &config_source,
            listener_port,
            &bound.upstream_authority,
            backend.addr().port(),
        )?;
        let released_port = listener.release();
        debug_assert_eq!(released_port, listener_port);
        let mut proxy = start_proxy(&config);
        backend.finish_proxy_readiness();

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .build()
            .map_err(|_source| FixtureError::ReplayHttp)?;
        let mut pending = Vec::with_capacity(bound.turns.len());
        let mut previous_response_id = None;
        for (turn_index, mut turn) in bound.turns.into_iter().enumerate() {
            turn.bind_previous_response_id(previous_response_id.as_deref())?;
            let response = send_recorded_request(&client, proxy.addr(), &turn.request).await?;
            previous_response_id = response.response_id().map(str::to_owned);
            pending.push(PendingTurn {
                scenario: turn,
                client_response: response,
            });
            backend.wait_for_exchanges(turn_index + 1, EXCHANGE_TIMEOUT).await?;
        }

        let captured_requests = finish_backend_after_proxy_shutdown(&mut proxy, backend, pending.len())?;

        let upstream_responses = response_owners
            .into_iter()
            .map(|response| {
                Arc::try_unwrap(response)
                    .map_err(|_response| runtime_error("scripted response remained shared after backend shutdown"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        compare_imported_requests(&bound.id, &provenance, &captured_requests, expected_requests, rules)?;

        let mut turns = Vec::with_capacity(pending.len());
        for (turn_index, ((pending, upstream_request), upstream_response)) in pending
            .into_iter()
            .zip(captured_requests)
            .zip(upstream_responses)
            .enumerate()
        {
            let PendingTurn {
                scenario,
                client_response,
            } = pending;
            let turn = WireTurn {
                name: scenario.name,
                client: RecordedExchange {
                    request: scenario.request,
                    response: client_response,
                },
                upstream: RecordedExchange {
                    request: upstream_request,
                    response: upstream_response,
                },
            };
            validate_expectation(turn_index, &turn, &scenario.expect)?;
            turns.push(turn);
        }

        let mut fixture = WireFixture {
            version: WIRE_FIXTURE_VERSION,
            scenario_id: bound.id,
            protocol: bound.protocol,
            provenance,
            normalization: NormalizationMetadata {
                version: 1,
                linked_ids: BTreeMap::new(),
            },
            turns,
        };
        sanitize_fixture_preserving_structure(&mut fixture, rules)?;
        Ok(fixture)
    }

    /// Records all ordered turns of a scenario against one live provider target.
    ///
    /// Exactly one recorder, one Praxis pipeline, and one no-redirect client are
    /// used for the complete scenario. The completed two-sided fixture is
    /// sanitized once and commit-safety validated before it is returned.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid target/scenario/config state, provider or
    /// client transport failure, incomplete recorder accounting, expectation
    /// mismatch, sanitization failure, or unsafe fixture content.
    pub async fn record_live(
        scenario: &InferenceScenario,
        target: super::ProviderTarget,
    ) -> Result<WireFixture, FixtureError> {
        let rules = RedactionRules::default();
        Self::record_live_with_rules(scenario, target, &rules).await
    }

    /// Records all ordered turns using caller-provided literal redactions.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::record_live`], plus errors caused by
    /// invalid custom redaction rules.
    pub async fn record_live_with_rules(
        scenario: &InferenceScenario,
        target: super::ProviderTarget,
        rules: &RedactionRules,
    ) -> Result<WireFixture, FixtureError> {
        super::record::record_live(scenario, target, rules).await
    }

    /// Replays a scenario using the expected fixture's upstream responses.
    ///
    /// # Errors
    ///
    /// Returns an error if the expected fixture is incompatible with the
    /// scenario, materialization fails, or the first normalized path differs.
    pub async fn replay(scenario: &InferenceScenario, expected: &WireFixture) -> Result<ReplayReport, FixtureError> {
        scenario.validate_version()?;
        expected.validate_version()?;
        if expected.scenario_id != scenario.id {
            return Err(mismatch("scenario_id", "scenario identity mismatch"));
        }
        if expected.protocol != scenario.protocol {
            return Err(mismatch("protocol", "protocol mismatch"));
        }
        if expected.turns.len() != scenario.turns.len() {
            return Err(mismatch("turns", "turn count mismatch"));
        }

        // Materialization owns its scripts. Cloning at this API boundary is
        // necessary because replay must retain the caller's expected fixture.
        let upstream = expected
            .turns
            .iter()
            .map(|turn| ImportedUpstream {
                source_id: expected.provenance.source_id.clone(),
                provider: Some(expected.provenance.provider.clone()),
                model: Some(expected.provenance.model.clone()),
                exchange: turn.upstream.clone(),
            })
            .collect();
        let actual = Self::materialize(scenario, expected.provenance.clone(), upstream).await?;

        let actual_value = fixture_comparison_value(&actual)?;
        let expected_value = fixture_comparison_value(expected)?;
        compare_values(&actual_value, &expected_value, "fixture")?;
        Ok(ReplayReport { actual })
    }
}

/// Validates scenario-controlled requests before any replay networking.
pub(super) fn validate_scenario_requests(scenario: &InferenceScenario) -> Result<(), FixtureError> {
    for turn in &scenario.turns {
        validate_request_method(&turn.request.method)?;
        validate_origin_form_path(&turn.request.path)?;
        super::header_policy::validate_recorded_headers(&turn.request.headers)?;
        validate_request_body(&turn.request.body)?;
    }
    Ok(())
}

/// Builds a comparison projection that hides nondeterministic raw identifier keys.
fn fixture_comparison_value(fixture: &WireFixture) -> Result<Value, FixtureError> {
    let mut value = serde_json::to_value(fixture).map_err(FixtureError::JsonBodyRender)?;
    let Some(linked_ids) = value
        .get_mut("normalization")
        .and_then(|normalization| normalization.get_mut("linked_ids"))
    else {
        return Err(runtime_error("fixture normalization metadata could not be inspected"));
    };
    let Value::Object(mapping) = linked_ids else {
        return Err(runtime_error("fixture normalization mapping is malformed"));
    };
    let mut targets = std::mem::take(mapping).into_values().collect::<Vec<_>>();
    targets.sort_unstable_by(|left, right| left.as_str().cmp(&right.as_str()));
    *linked_ids = Value::Array(targets);
    Ok(value)
}

/// Loads a scenario config and proves its named upstream exists before networking.
pub(super) fn load_and_validate_config_source(scenario: &InferenceScenario) -> Result<String, FixtureError> {
    let config_root = example_config_path("");
    let path = resolve_example_config_path(Path::new(&config_root), Path::new(&scenario.example_config))?;
    let source =
        std::fs::read_to_string(path).map_err(|_source| runtime_error("scenario example config could not be read"))?;
    let contained = contain_replay_external_resources(&source)?;
    let config =
        Config::from_yaml(&contained).map_err(|_source| runtime_error("scenario example config is invalid"))?;
    validate_replay_filters(&config)?;
    let value =
        serde_json::to_value(config).map_err(|_source| runtime_error("scenario config could not be inspected"))?;
    if !config_contains_endpoint_address(&value, &scenario.upstream_authority)? {
        return Err(runtime_error(
            "scenario upstream authority was not found in example config",
        ));
    }
    Ok(source)
}

/// Resolves one normal relative config path without following symlinks.
fn resolve_example_config_path(root: &Path, relative: &Path) -> Result<PathBuf, FixtureError> {
    if !has_only_normal_relative_components(relative) {
        return Err(runtime_error("scenario example config path is not contained"));
    }

    let canonical_root = root
        .canonicalize()
        .map_err(|_source| runtime_error("scenario example config path is not contained"))?;
    let mut candidate = canonical_root.clone();
    for component in relative.components() {
        candidate.push(component.as_os_str());
        let metadata = std::fs::symlink_metadata(&candidate)
            .map_err(|_source| runtime_error("scenario example config path is not contained"))?;
        if metadata.file_type().is_symlink() {
            return Err(runtime_error("scenario example config path is not contained"));
        }
    }

    let canonical = candidate
        .canonicalize()
        .map_err(|_source| runtime_error("scenario example config path is not contained"))?;
    if !canonical.starts_with(&canonical_root) || !canonical.is_file() {
        return Err(runtime_error("scenario example config path is not contained"));
    }
    Ok(canonical)
}

/// Checks the portable lexical grammar before filesystem resolution.
fn has_only_normal_relative_components(path: &Path) -> bool {
    let Some(path_text) = path.to_str() else {
        return false;
    };
    !path_text.contains('\\')
        && path_text.split('/').all(|component| {
            !component.is_empty()
                && !matches!(component, "." | "..")
                && !matches!(component.as_bytes(), [drive, b':', ..] if drive.is_ascii_alphabetic())
        })
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

/// Applies the exact known authority patch and proves every network surface is local.
pub(super) fn build_replay_config(
    source: &str,
    listener_port: u16,
    upstream_authority: &str,
    backend_port: u16,
) -> Result<Config, FixtureError> {
    let patched = patch_yaml(
        source,
        listener_port,
        &HashMap::from([(upstream_authority, backend_port)]),
    );
    let contained = contain_replay_external_resources(&patched)?;
    let config =
        Config::from_yaml(&contained).map_err(|_source| runtime_error("patched scenario config is invalid"))?;
    let expected_backend = format!("127.0.0.1:{backend_port}");
    let expected_listener = SocketAddr::from(([127, 0, 0, 1], listener_port));
    validate_replay_network_surfaces(&config, expected_listener, &expected_backend)?;
    validate_replay_filters(&config)?;
    Ok(config)
}

/// Rejects filter entries that can create independent live callouts.
fn validate_replay_filters(config: &Config) -> Result<(), FixtureError> {
    if config.clusters.iter().any(|cluster| cluster.health_check.is_some()) {
        return Err(runtime_error("scenario health check is not replay-contained"));
    }
    for chain in &config.filter_chains {
        validate_replay_filter_entries(&chain.filters)?;
    }
    Ok(())
}

/// Traverses top-level and inline branch filters without scanning inert config data.
fn validate_replay_filter_entries(filters: &[FilterEntry]) -> Result<(), FixtureError> {
    for filter in filters {
        if !matches!(
            filter.filter_type.as_str(),
            "agentic_loop"
                | "anthropic_messages_format"
                | "anthropic_messages_protocol"
                | "anthropic_to_openai"
                | "anthropic_stream_events"
                | "iterative_request_router"
                | "openai_responses_proxy"
                | "path_rewrite"
                | "openai_responses_format"
                | "openai_responses_validate"
                | "openai_response_store"
                | "openai_responses_rehydrate"
                | "responses_to_chat_completions"
                | "router"
                | "load_balancer"
        ) {
            return Err(runtime_error("scenario filter is not replay-contained"));
        }
        for branch in filter.branch_chains.iter().flatten() {
            for chain in &branch.chains {
                if let ChainRef::Inline { filters, .. } = chain {
                    validate_replay_filter_entries(filters)?;
                }
            }
        }
    }
    Ok(())
}

/// Rewrites accepted SQLite targets to an in-memory store and rejects every
/// configured local file or database resource that replay does not own.
fn contain_replay_external_resources(source: &str) -> Result<String, FixtureError> {
    let mut document: serde_yaml::Value =
        serde_yaml::from_str(source).map_err(|_source| runtime_error("patched scenario config is invalid"))?;
    contain_replay_external_value(&mut document)?;
    serde_yaml::to_string(&document).map_err(|_source| runtime_error("patched scenario config is invalid"))
}

/// Applies replay resource containment recursively before config construction.
fn contain_replay_external_value(value: &mut serde_yaml::Value) -> Result<(), FixtureError> {
    match value {
        serde_yaml::Value::Mapping(mapping) => {
            contain_replay_external_mapping(mapping)?;
            for value in mapping.values_mut() {
                contain_replay_external_value(value)?;
            }
        },
        serde_yaml::Value::Sequence(values) => {
            for value in values {
                contain_replay_external_value(value)?;
            }
        },
        serde_yaml::Value::Tagged(tagged) => contain_replay_external_value(&mut tagged.value)?,
        serde_yaml::Value::Null
        | serde_yaml::Value::Bool(_)
        | serde_yaml::Value::Number(_)
        | serde_yaml::Value::String(_) => {},
    }
    Ok(())
}

/// Contains file-backed fields and one dynamically nested database target.
fn contain_replay_external_mapping(mapping: &mut serde_yaml::Mapping) -> Result<(), FixtureError> {
    if mapping
        .iter()
        .any(|(key, value)| key.as_str().is_some_and(is_file_resource_key) && !matches!(value, serde_yaml::Value::Null))
    {
        return Err(runtime_error("scenario local resource is not replay-contained"));
    }
    let backend = mapping.iter().find_map(|(key, value)| {
        key.as_str()
            .filter(|key| key.eq_ignore_ascii_case("backend"))
            .and_then(|_key| value.as_str())
            .map(str::to_owned)
    });
    let database_key = mapping
        .keys()
        .find(|key| key.as_str().is_some_and(|key| key.eq_ignore_ascii_case("database_url")))
        .cloned();
    let Some(database_key) = database_key else {
        return Ok(());
    };
    let database_url = mapping
        .get(&database_key)
        .and_then(serde_yaml::Value::as_str)
        .ok_or_else(|| runtime_error("scenario outbound URL is malformed"))?;
    validate_replay_database_target(backend.as_deref(), database_url)?;
    mapping.insert(database_key, serde_yaml::Value::String("sqlite::memory:".to_owned()));
    Ok(())
}

/// Recognizes config fields whose values are loaded from the local filesystem.
fn is_file_resource_key(key: &str) -> bool {
    let key = key.to_ascii_lowercase();
    key.ends_with("_path")
        || key.ends_with("-path")
        || key.ends_with("_paths")
        || key.ends_with("-paths")
        || key.ends_with("_file")
        || key.ends_with("-file")
        || key.ends_with("_files")
        || key.ends_with("-files")
        || matches!(
            key.as_str(),
            "ssl_root_cert" | "ssl_cert" | "ssl_key" | "certificate_file" | "private_key_file" | "ca_file"
        )
}

/// Accepts SQLite URLs and scheme-less paths that replay replaces in memory.
fn validate_replay_database_target(backend: Option<&str>, database_url: &str) -> Result<(), FixtureError> {
    if backend.is_some_and(|backend| backend.eq_ignore_ascii_case("postgres"))
        || database_url
            .split_once(':')
            .is_some_and(|(scheme, _rest)| matches!(scheme.to_ascii_lowercase().as_str(), "postgres" | "postgresql"))
    {
        return Err(runtime_error("scenario database backend is not replay-contained"));
    }
    if backend.is_some_and(|backend| !backend.eq_ignore_ascii_case("sqlite")) {
        return Err(runtime_error("scenario database backend is not replay-contained"));
    }
    if database_url.trim().is_empty()
        || database_url.contains("://")
            && !database_url
                .split_once(':')
                .is_some_and(|(scheme, _rest)| scheme.eq_ignore_ascii_case("sqlite"))
    {
        return Err(runtime_error("scenario outbound URL is malformed"));
    }
    Ok(())
}

/// Returns whether any recursively nested endpoint equals the named authority.
#[expect(
    clippy::too_many_lines,
    reason = "recursive config traversal handles every JSON shape and both endpoint representations"
)]
fn config_contains_endpoint_address(value: &Value, expected: &str) -> Result<bool, FixtureError> {
    match value {
        Value::Object(object) => {
            for (key, value) in object {
                if key == "endpoints" {
                    let Value::Array(configured) = value else {
                        return Err(runtime_error("scenario provider endpoints are malformed"));
                    };
                    for endpoint in configured {
                        let address = match endpoint {
                            Value::String(address) => address,
                            Value::Object(weighted) => {
                                let Some(Value::String(address)) = weighted.get("address") else {
                                    return Err(runtime_error("scenario provider endpoint is malformed"));
                                };
                                address
                            },
                            Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) => {
                                return Err(runtime_error("scenario provider endpoint is malformed"));
                            },
                        };
                        if address == expected {
                            return Ok(true);
                        }
                    }
                } else if config_contains_endpoint_address(value, expected)? {
                    return Ok(true);
                }
            }
        },
        Value::Array(values) => {
            for value in values {
                if config_contains_endpoint_address(value, expected)? {
                    return Ok(true);
                }
            }
        },
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {},
    }
    Ok(false)
}

/// Proves all listener, admin, socket, and HTTP(S) destinations are loopback.
fn validate_replay_network_surfaces(
    config: &Config,
    expected_listener: SocketAddr,
    expected_backend: &str,
) -> Result<(), FixtureError> {
    let [listener] = config.listeners.as_slice() else {
        return Err(runtime_error("scenario replay requires exactly one listener"));
    };
    let address = listener
        .address
        .parse::<SocketAddr>()
        .map_err(|_source| runtime_error("scenario listener bind is malformed"))?;
    if address != expected_listener
        || listener.protocol != ProtocolKind::Http
        || listener.upstream.is_some()
        || listener.cluster.is_some()
    {
        return Err(runtime_error("scenario listener is not replay-owned"));
    }
    if config.admin.address.is_some() {
        return Err(runtime_error("scenario admin bind is not replay-owned"));
    }

    let mut value =
        serde_json::to_value(config).map_err(|_source| runtime_error("scenario config could not be inspected"))?;
    let Value::Object(root) = &mut value else {
        return Err(runtime_error("scenario config could not be inspected"));
    };
    root.remove("listeners");
    root.remove("admin");
    let mut found_expected_backend = false;
    validate_replay_network_value(&value, expected_backend, &mut found_expected_backend)?;
    if !found_expected_backend {
        return Err(runtime_error("scenario upstream authority patch was not applied"));
    }
    Ok(())
}

/// Recursively allowlists every config value that can name a network destination.
fn validate_replay_network_value(
    value: &Value,
    expected_backend: &str,
    found_expected_backend: &mut bool,
) -> Result<(), FixtureError> {
    validate_replay_network_value_with_context(value, expected_backend, found_expected_backend, false)
}

/// Recursively validates config strings while retaining whether an ancestor key names a network value.
fn validate_replay_network_value_with_context(
    value: &Value,
    expected_backend: &str,
    found_expected_backend: &mut bool,
    network_context: bool,
) -> Result<(), FixtureError> {
    match value {
        Value::Object(object) => {
            validate_replay_network_object(object, expected_backend, found_expected_backend, network_context)?;
        },
        Value::Array(values) => {
            for value in values {
                validate_replay_network_value_with_context(
                    value,
                    expected_backend,
                    found_expected_backend,
                    network_context,
                )?;
            }
        },
        Value::String(value) => {
            if validate_replay_owned_network_string(value, network_context, expected_backend)? {
                *found_expected_backend = true;
            }
        },
        Value::Null | Value::Bool(_) | Value::Number(_) => {},
    }
    Ok(())
}

/// Validates every member of one config object with its key context.
fn validate_replay_network_object(
    object: &serde_json::Map<String, Value>,
    expected_backend: &str,
    found_expected_backend: &mut bool,
    network_context: bool,
) -> Result<(), FixtureError> {
    for (key, value) in object {
        if key == "endpoints" {
            validate_endpoint_collection(value, expected_backend, found_expected_backend)?;
        } else if key == "endpoint" {
            let Value::String(endpoint) = value else {
                return Err(runtime_error("scenario outbound endpoint is malformed"));
            };
            validate_singular_endpoint(endpoint, expected_backend)?;
            *found_expected_backend = true;
        } else if key.eq_ignore_ascii_case("database_url") {
            let Value::String(database_url) = value else {
                return Err(runtime_error("scenario outbound URL is malformed"));
            };
            validate_database_url(database_url)?;
        } else {
            validate_replay_network_value_with_context(
                value,
                expected_backend,
                found_expected_backend,
                network_context || is_network_value_key(key),
            )?;
        }
    }
    Ok(())
}

/// Returns whether a config key conventionally contains a network destination.
fn is_network_value_key(key: &str) -> bool {
    const NAMES: [&str; 10] = [
        "endpoint",
        "address",
        "url",
        "uri",
        "host",
        "authority",
        "bind",
        "listen",
        "listener",
        "socket",
    ];
    if key.eq_ignore_ascii_case("file_url") {
        // `file_url` is an existing resolve/passthrough enum, while compound
        // names such as `files_api_url` are actual outbound destinations.
        return false;
    }
    NAMES.into_iter().any(|name| {
        key.eq_ignore_ascii_case(name)
            || key
                .get(key.len().saturating_sub(name.len() + 1)..)
                .is_some_and(|suffix| {
                    suffix.get(..1).is_some_and(|separator| matches!(separator, "_" | "-"))
                        && suffix.get(1..).is_some_and(|suffix| suffix.eq_ignore_ascii_case(name))
                })
    })
}

/// Validates plural endpoint shapes without copying address strings.
fn validate_endpoint_collection(
    value: &Value,
    expected_backend: &str,
    found_expected_backend: &mut bool,
) -> Result<(), FixtureError> {
    let Value::Array(configured) = value else {
        return Err(runtime_error("scenario provider endpoints are malformed"));
    };
    for endpoint in configured {
        validate_endpoint_entry(endpoint, expected_backend, found_expected_backend)?;
    }
    Ok(())
}

/// Validates one endpoint address plus every sibling member of a weighted entry.
fn validate_endpoint_entry(
    endpoint: &Value,
    expected_backend: &str,
    found_expected_backend: &mut bool,
) -> Result<(), FixtureError> {
    let (address, weighted) = match endpoint {
        Value::String(address) => (address, None),
        Value::Object(weighted) => {
            let Some(Value::String(address)) = weighted.get("address") else {
                return Err(runtime_error("scenario provider endpoint is malformed"));
            };
            (address, Some(weighted))
        },
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_) => {
            return Err(runtime_error("scenario provider endpoint is malformed"));
        },
    };
    address
        .parse::<SocketAddr>()
        .map_err(|_source| runtime_error("scenario provider endpoint is not a socket address"))?;
    if address != expected_backend {
        return Err(runtime_error("scenario outbound target is not replay-owned"));
    }
    *found_expected_backend = true;
    if let Some(weighted) = weighted {
        validate_weighted_endpoint_siblings(weighted, expected_backend, found_expected_backend)?;
    }
    Ok(())
}

/// Validates every non-address member of one weighted endpoint in place.
fn validate_weighted_endpoint_siblings(
    weighted: &serde_json::Map<String, Value>,
    expected_backend: &str,
    found_expected_backend: &mut bool,
) -> Result<(), FixtureError> {
    for (key, value) in weighted {
        if key != "address" {
            validate_replay_network_value_with_context(
                value,
                expected_backend,
                found_expected_backend,
                is_network_value_key(key),
            )?;
        }
    }
    Ok(())
}

/// Validates a singular endpoint expressed as a socket, authority, or network URI.
fn validate_singular_endpoint(endpoint: &str, expected_backend: &str) -> Result<(), FixtureError> {
    if validate_replay_owned_network_string(endpoint, true, expected_backend)? {
        Ok(())
    } else {
        Err(runtime_error("scenario outbound endpoint is malformed"))
    }
}

/// Requires one recognized outbound target to select the replay-owned backend.
fn validate_replay_owned_network_string(
    value: &str,
    network_context: bool,
    expected_backend: &str,
) -> Result<bool, FixtureError> {
    let is_network = validate_network_string_with_context(value, network_context)?;
    if is_network && !network_string_targets_expected_socket(value, expected_backend) {
        return Err(runtime_error("scenario outbound target is not replay-owned"));
    }
    Ok(is_network)
}

/// Compares a validated URL/socket/authority with the exact replay backend.
fn network_string_targets_expected_socket(value: &str, expected_backend: &str) -> bool {
    let Ok(expected) = expected_backend.parse::<SocketAddr>() else {
        return false;
    };
    if let Ok(socket) = value.parse::<SocketAddr>() {
        return socket == expected;
    }
    if value.contains("://") {
        return reqwest::Url::parse(value).is_ok_and(|url| {
            url.host_str().and_then(|host| host.parse::<IpAddr>().ok()) == Some(expected.ip())
                && url.port_or_known_default() == Some(expected.port())
        });
    }
    let Some(authority) = authority_prefix(value)
        .map(authority_without_userinfo)
        .and_then(|authority| authority.parse::<http::uri::Authority>().ok())
    else {
        return false;
    };
    authority.host().parse::<IpAddr>().ok() == Some(expected.ip()) && authority.port_u16() == Some(expected.port())
}

/// Preserves API-owned scheme-less SQLite paths while containing strong network syntax.
fn validate_database_url(database_url: &str) -> Result<(), FixtureError> {
    validate_network_string_with_context(database_url, false).map(|_network_target| ())
}

/// Applies the generic fail-closed network check to every config string.
#[cfg(test)]
fn validate_generic_network_string(value: &str) -> Result<(), FixtureError> {
    validate_network_string_with_context(value, false).map(|_network_target| ())
}

/// Validates recognized network forms, including key-dependent bare hosts.
fn validate_network_string_with_context(value: &str, network_context: bool) -> Result<bool, FixtureError> {
    if let Some(result) = validate_local_resource_string(value) {
        result?;
        return Ok(false);
    }
    if value.starts_with("//") {
        return Err(runtime_error("scenario outbound URL is malformed"));
    }
    if let Some(result) = validate_ip_or_socket(value) {
        return result;
    }
    if value.contains("://") {
        validate_absolute_network_url(value)?;
        return Ok(true);
    }
    if has_known_network_scheme_marker(value) {
        return Err(runtime_error("scenario outbound URL is malformed"));
    }
    if let Some(result) = validate_numeric_port_authority(value) {
        return result;
    }
    if network_context {
        validate_bare_network_value(value)?;
        return Ok(true);
    }
    Ok(false)
}

/// Validates an exact IP address or socket without mistaking ordinary colon text for one.
fn validate_ip_or_socket(value: &str) -> Option<Result<bool, FixtureError>> {
    let ip = value
        .parse::<SocketAddr>()
        .map(|socket| socket.ip())
        .or_else(|_source| value.parse::<IpAddr>());
    ip.ok().map(|ip| {
        if is_loopback_ip(ip) {
            Ok(true)
        } else {
            Err(runtime_error("scenario outbound socket is not loopback"))
        }
    })
}

/// Validates a numeric-port authority or rejects a malformed numeric attempt.
fn validate_numeric_port_authority(value: &str) -> Option<Result<bool, FixtureError>> {
    let authority_with_userinfo = authority_prefix(value)?;
    let authority_value = authority_without_userinfo(authority_with_userinfo);
    match bracketed_authority_has_port(authority_value) {
        Err(error) => return Some(Err(error)),
        Ok(Some(_has_port)) => {
            let authority = authority_with_userinfo
                .parse::<http::uri::Authority>()
                .map_err(|_source| runtime_error("scenario outbound URL is malformed"));
            return Some(authority.and_then(|authority| validate_authority_host(&authority).map(|()| true)));
        },
        Ok(None) => {},
    }
    let (_host, port) = authority_value.rsplit_once(':')?;
    if port.is_empty() || !port.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    if port.parse::<u16>().is_err() {
        return Some(Err(runtime_error("scenario outbound URL is malformed")));
    }
    let authority = authority_with_userinfo
        .parse::<http::uri::Authority>()
        .map_err(|_source| runtime_error("scenario outbound URL is malformed"));
    Some(authority.and_then(|authority| validate_authority_host(&authority).map(|()| true)))
}

/// Classifies explicit local resource schemes before generic URI handling.
fn validate_local_resource_string(value: &str) -> Option<Result<(), FixtureError>> {
    let (scheme, _remainder) = value.split_once(':')?;
    if scheme.eq_ignore_ascii_case("file") || scheme.eq_ignore_ascii_case("unix") {
        return Some(Err(runtime_error("scenario local resource is not replay-contained")));
    }
    if !scheme.eq_ignore_ascii_case("sqlite") {
        return None;
    }
    // The response store accepts empty paths as temporary databases, including
    // bare and query-delimited `sqlite:`/`sqlite://` forms. Replay owns only
    // network containment; detailed path and option validation stays with the backend.
    Some(Ok(()))
}

/// Requires every URI with `://` syntax to select a loopback target.
fn validate_absolute_network_url(value: &str) -> Result<(), FixtureError> {
    let url = reqwest::Url::parse(value).map_err(|_source| runtime_error("scenario outbound URL is malformed"))?;
    if matches!(url.scheme(), "postgres" | "postgresql") {
        return validate_postgres_network_url(&url);
    }
    let Some(host) = url.host_str() else {
        return Err(runtime_error("scenario outbound URL is malformed"));
    };
    let host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    validate_network_host(host)
}

/// Requires every target-bearing `PostgreSQL` URL field to remain loopback.
fn validate_postgres_network_url(url: &reqwest::Url) -> Result<(), FixtureError> {
    let host = url
        .host_str()
        .ok_or_else(|| runtime_error("scenario outbound URL is malformed"))?;
    let host = host
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(host);
    validate_network_host(host)?;

    for (key, value) in url.query_pairs() {
        match key.as_ref() {
            "host" => validate_network_host(&value)?,
            "hostaddr" => {
                let address = value
                    .parse::<IpAddr>()
                    .map_err(|_source| runtime_error("scenario outbound URL is malformed"))?;
                if !is_loopback_ip(address) {
                    return Err(runtime_error("scenario outbound URL is not loopback"));
                }
            },
            "port" => {
                value
                    .parse::<u16>()
                    .map_err(|_source| runtime_error("scenario outbound URL is malformed"))?;
            },
            _ => {},
        }
    }
    Ok(())
}

/// Returns whether a known network scheme was written without a valid authority delimiter.
fn has_known_network_scheme_marker(value: &str) -> bool {
    let Some((scheme, _remainder)) = value.split_once(':') else {
        return false;
    };
    [
        "http",
        "https",
        "grpc",
        "grpcs",
        "ws",
        "wss",
        "postgres",
        "postgresql",
        "mysql",
        "redis",
        "rediss",
        "mongodb",
        "amqp",
        "amqps",
        "tcp",
        "udp",
        "ftp",
        "sftp",
        "ssh",
        "nats",
        "kafka",
    ]
    .into_iter()
    .any(|known| scheme.eq_ignore_ascii_case(known))
}

/// Validates a bare hostname, host/path, or userinfo host under a network key.
fn validate_bare_network_value(value: &str) -> Result<(), FixtureError> {
    let authority_with_userinfo =
        authority_prefix(value).ok_or_else(|| runtime_error("scenario outbound URL is malformed"))?;
    let authority_value = authority_without_userinfo(authority_with_userinfo);
    let bracketed_has_port = bracketed_authority_has_port(authority_value)?;
    let authority = authority_value
        .parse::<http::uri::Authority>()
        .map_err(|_source| runtime_error("scenario outbound URL is malformed"))?;
    let has_port_separator = bracketed_has_port.unwrap_or_else(|| authority_value.contains(':'));
    if has_port_separator && authority.port_u16().is_none() {
        return Err(runtime_error("scenario outbound URL is malformed"));
    }
    validate_authority_host(&authority)
}

/// Borrows only the authority component before any path, query, or fragment.
fn authority_prefix(value: &str) -> Option<&str> {
    value
        .split(['/', '?', '#'])
        .next()
        .filter(|authority| !authority.is_empty())
}

/// Removes userinfo without treating an `@` in a suffix as authority syntax.
fn authority_without_userinfo(authority: &str) -> &str {
    authority
        .rsplit_once('@')
        .map_or(authority, |(_userinfo, authority)| authority)
}

/// Validates that bracketed host text ends at `]` or has exactly one valid u16 port.
fn bracketed_authority_has_port(authority: &str) -> Result<Option<bool>, FixtureError> {
    let Some(bracketed) = authority.strip_prefix('[') else {
        if authority.contains(['[', ']']) {
            return Err(runtime_error("scenario outbound URL is malformed"));
        }
        return Ok(None);
    };
    let Some(close_bracket) = bracketed.find(']') else {
        return Err(runtime_error("scenario outbound URL is malformed"));
    };
    let suffix = bracketed
        .get(close_bracket + 1..)
        .ok_or_else(|| runtime_error("scenario outbound URL is malformed"))?;
    if suffix.is_empty() {
        return Ok(Some(false));
    }
    let Some(port) = suffix.strip_prefix(':') else {
        return Err(runtime_error("scenario outbound URL is malformed"));
    };
    if port.is_empty() || port.parse::<u16>().is_err() {
        return Err(runtime_error("scenario outbound URL is malformed"));
    }
    Ok(Some(true))
}

/// Requires one host string to be localhost or a literal loopback address.
fn validate_network_host(host: &str) -> Result<(), FixtureError> {
    if host.eq_ignore_ascii_case("localhost") || host.parse::<IpAddr>().is_ok_and(is_loopback_ip) {
        return Ok(());
    }
    Err(runtime_error("scenario outbound URL is not loopback"))
}

/// Requires one parsed URI authority to name a literal loopback host.
fn validate_authority_host(authority: &http::uri::Authority) -> Result<(), FixtureError> {
    let host = authority
        .host()
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or_else(|| authority.host());
    validate_network_host(host)
}

/// Treats IPv4-mapped loopback addresses as loopback too.
fn is_loopback_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ip.is_loopback(),
        IpAddr::V6(ip) => ip
            .to_ipv4_mapped()
            .map_or_else(|| ip.is_loopback(), |ip| ip.is_loopback()),
    }
}

/// Client result retained until captured upstream requests are available.
struct PendingTurn {
    /// Bound scenario turn that supplied the exact client request.
    scenario: ScenarioTurn,
    /// Final client response observed after all response frames were read.
    client_response: RecordedResponse,
}

/// Proves the proxy producer has stopped before taking final backend accounting.
fn finish_backend_after_proxy_shutdown(
    proxy: &mut ProxyGuard,
    backend: ScriptedHttpServer,
    expected_exchanges: usize,
) -> Result<Vec<RecordedRequest>, FixtureError> {
    proxy
        .shutdown()
        .map_err(|_error| runtime_error("scenario proxy shutdown did not complete"))?;
    backend.finish(expected_exchanges)
}

/// Validates model/provider coherence and exact turn/script counts.
fn validate_upstream_inputs(
    scenario: &InferenceScenario,
    provenance: &FixtureProvenance,
    upstream: &[ImportedUpstream],
) -> Result<(), FixtureError> {
    if scenario.turns.len() != upstream.len() {
        return Err(mismatch("turns", "imported upstream count mismatch"));
    }
    if provenance.model.is_empty() {
        return Err(mismatch("provenance.model", "fixture model is empty"));
    }
    for (index, imported) in upstream.iter().enumerate() {
        if imported.model.as_deref().is_some_and(|model| model != provenance.model) {
            return Err(mismatch(
                &format!("upstream[{index}].model"),
                "model provenance mismatch",
            ));
        }
        if imported
            .provider
            .as_deref()
            .is_some_and(|provider| provider != provenance.provider)
        {
            return Err(mismatch(
                &format!("upstream[{index}].provider"),
                "provider provenance mismatch",
            ));
        }
        if imported.exchange.request.method.eq_ignore_ascii_case("HEAD")
            && body_has_rendered_content(&imported.exchange.response.body)
        {
            return Err(runtime_error("scripted HEAD response forbids content"));
        }
    }
    Ok(())
}

/// Sends one exact bound scenario request and captures the final client response.
pub(super) async fn send_recorded_request(
    client: &reqwest::Client,
    proxy_addr: &str,
    request: &RecordedRequest,
) -> Result<RecordedResponse, FixtureError> {
    send_recorded_request_with_optional_header(client, proxy_addr, request, None).await
}

/// Sends one scenario request with one recorder-only capability header.
pub(super) async fn send_recorded_request_with_header(
    client: &reqwest::Client,
    proxy_addr: &str,
    request: &RecordedRequest,
    name: &HeaderName,
    value: &HeaderValue,
) -> Result<RecordedResponse, FixtureError> {
    send_recorded_request_with_optional_header(client, proxy_addr, request, Some((name, value))).await
}

/// Shared exact request sender with an optional trusted-hop header.
async fn send_recorded_request_with_optional_header(
    client: &reqwest::Client,
    proxy_addr: &str,
    request: &RecordedRequest,
    extra_header: Option<(&HeaderName, &HeaderValue)>,
) -> Result<RecordedResponse, FixtureError> {
    let path = validate_origin_form_path(&request.path)?;
    let url = loopback_request_url(proxy_addr, path)?;
    let method = validate_request_method(&request.method)?;
    let mut headers = recorded_transport_headers(&request.headers)?;
    if let Some((name, value)) = extra_header {
        headers.insert(name.clone(), value.clone());
    }
    let body = request.body.render()?;
    let response = client
        .request(method, url)
        .headers(headers)
        .body(body)
        .send()
        .await
        .map_err(|_source| FixtureError::ReplayHttp)?;
    capture_client_response(response).await
}

/// Parses one recorded method before any replay networking starts.
fn validate_request_method(method: &str) -> Result<reqwest::Method, FixtureError> {
    reqwest::Method::from_bytes(method.as_bytes())
        .map_err(|_invalid_method| mismatch("client.request.method", "invalid HTTP method"))
}

/// Parses and preserves one exact HTTP origin-form path and query.
fn validate_origin_form_path(path: &str) -> Result<&str, FixtureError> {
    let invalid = || mismatch("client.request.path", "request path must be origin-form");
    if path.bytes().any(|byte| byte.is_ascii_control()) || path.contains('#') || !valid_percent_encoding(path) {
        return Err(invalid());
    }
    let uri = path.parse::<http::Uri>().map_err(|_source| invalid())?;
    if uri.scheme().is_some()
        || uri.authority().is_some()
        || !uri.path().starts_with('/')
        || uri.path().starts_with("//")
        || uri.path().split('/').any(decodes_to_dot_segment)
    {
        return Err(invalid());
    }
    let Some(path_and_query) = uri.path_and_query() else {
        return Err(invalid());
    };
    if path_and_query.as_str() != path {
        return Err(invalid());
    }
    let _normalized = loopback_request_url("127.0.0.1:1", path)?;
    Ok(path)
}

/// Constructs the exact loopback URL consumed by reqwest and rejects serialization changes.
fn loopback_request_url(authority: &str, path: &str) -> Result<reqwest::Url, FixtureError> {
    let invalid = || mismatch("client.request.path", "request path must be origin-form");
    let absolute = format!("http://{authority}{path}");
    let normalized = reqwest::Url::parse(&absolute).map_err(|_source| invalid())?;
    let (raw_path, raw_query) = path
        .split_once('?')
        .map_or((path, None), |(path, query)| (path, Some(query)));
    if normalized.path() != raw_path || normalized.query() != raw_query {
        return Err(invalid());
    }
    Ok(normalized)
}

/// Returns whether one raw path segment percent-decodes exactly to `.` or `..`.
fn decodes_to_dot_segment(segment: &str) -> bool {
    let mut bytes = segment.bytes();
    let mut dot_count = 0;
    while let Some(byte) = bytes.next() {
        let decoded = if byte == b'%' {
            let Some(high) = bytes.next().and_then(hex_value) else {
                return false;
            };
            let Some(low) = bytes.next().and_then(hex_value) else {
                return false;
            };
            (high << 4) | low
        } else {
            byte
        };
        if decoded != b'.' || dot_count == 2 {
            return false;
        }
        dot_count += 1;
    }
    matches!(dot_count, 1 | 2)
}

/// Converts one ASCII hexadecimal digit to its numeric value.
const fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Requires every percent marker to introduce two hexadecimal digits.
fn valid_percent_encoding(value: &str) -> bool {
    let mut bytes = value.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'%'
            && !matches!(
                (bytes.next(), bytes.next()),
                (Some(high), Some(low)) if high.is_ascii_hexdigit() && low.is_ascii_hexdigit()
            )
        {
            return false;
        }
    }
    true
}

/// Collects one final client response with a deterministic size ceiling.
async fn capture_client_response(response: reqwest::Response) -> Result<RecordedResponse, FixtureError> {
    let status = response.status().as_u16();
    let headers = http_fixture_headers(response.headers())?;
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_source| FixtureError::ReplayHttp)?;
        if body.len().saturating_add(chunk.len()) > MAX_SCRIPTED_RESPONSE_BODY_BYTES {
            return Err(runtime_error("scenario client response exceeded replay limit"));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(RecordedResponse {
        status,
        headers,
        body: parse_response_body(content_type.as_deref(), &body)?,
    })
}

/// Normalizes both observed and imported requests before comparing each path.
fn compare_imported_requests(
    scenario_id: &str,
    provenance: &FixtureProvenance,
    captured: &[RecordedRequest],
    expected: Vec<RecordedRequest>,
    rules: &RedactionRules,
) -> Result<(), FixtureError> {
    // The observed request must remain in the returned fixture, so a clone is
    // required at this comparison boundary; response/SSE payloads are not copied.
    let actual = normalize_request_projection(scenario_id, provenance, captured.to_vec(), rules)?;
    let expected = normalize_request_projection(scenario_id, provenance, expected, rules)?;
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        let actual = serde_json::to_value(actual).map_err(FixtureError::JsonBodyRender)?;
        let expected = serde_json::to_value(expected).map_err(FixtureError::JsonBodyRender)?;
        compare_values(&actual, &expected, &format!("turns[{index}].upstream.request"))?;
    }
    Ok(())
}

/// Applies fixture sanitization semantics to an ordered request projection.
fn normalize_request_projection(
    scenario_id: &str,
    provenance: &FixtureProvenance,
    requests: Vec<RecordedRequest>,
    rules: &RedactionRules,
) -> Result<Vec<RecordedRequest>, FixtureError> {
    let mut fixture = WireFixture {
        version: WIRE_FIXTURE_VERSION,
        scenario_id: scenario_id.to_owned(),
        protocol: super::InferenceProtocol::OpenaiChatCompletions,
        provenance: provenance.clone(),
        normalization: NormalizationMetadata {
            version: 1,
            linked_ids: BTreeMap::new(),
        },
        turns: requests
            .into_iter()
            .enumerate()
            .map(|(index, request)| WireTurn {
                name: format!("request-{index}"),
                client: empty_exchange(),
                upstream: RecordedExchange {
                    request,
                    response: empty_response(),
                },
            })
            .collect(),
    };
    sanitize_fixture_preserving_structure(&mut fixture, rules)?;
    Ok(fixture.turns.into_iter().map(|turn| turn.upstream.request).collect())
}

/// Validates all data-driven expectations for one completed turn.
#[expect(
    clippy::too_many_lines,
    reason = "all scenario expectation fields are checked in schema order"
)]
pub(super) fn validate_expectation(
    index: usize,
    turn: &WireTurn,
    expected: &ScenarioExpectation,
) -> Result<(), FixtureError> {
    let base = format!("turns[{index}]");
    if turn.client.response.status != expected.client_status {
        return Err(mismatch(
            &format!("{base}.client.response.status"),
            "status expectation mismatch",
        ));
    }
    if turn.client.response.body.kind() != expected.client_body_kind {
        return Err(mismatch(
            &format!("{base}.client.response.body.kind"),
            "body kind expectation mismatch",
        ));
    }
    if turn.upstream.request.path != expected.upstream_path {
        return Err(mismatch(
            &format!("{base}.upstream.request.path"),
            "upstream path expectation mismatch",
        ));
    }
    if turn.upstream.request.body.kind() != expected.upstream_body_kind {
        return Err(mismatch(
            &format!("{base}.upstream.request.body.kind"),
            "body kind expectation mismatch",
        ));
    }
    validate_sse_events(
        &turn.client.response.body,
        &expected.client_sse_events,
        &expected.client_sse_repeatable_events,
        &expected.client_sse_interleaved_events,
        &format!("{base}.client.response.body.frames"),
    )?;
    validate_sse_events(
        &turn.upstream.response.body,
        &expected.upstream_sse_events,
        &expected.upstream_sse_repeatable_events,
        &expected.upstream_sse_interleaved_events,
        &format!("{base}.upstream.response.body.frames"),
    )
}

/// Requires an ordered named-event pattern with explicit repetitions and interleaving.
fn validate_sse_events(
    body: &RecordedBody,
    expected: &[String],
    repeatable: &[String],
    interleaved: &[String],
    path: &str,
) -> Result<(), FixtureError> {
    let frames = match body {
        RecordedBody::Sse { frames, .. } => frames.as_slice(),
        RecordedBody::Empty | RecordedBody::Json { .. } | RecordedBody::Base64 { .. } => &[],
    };
    let membership = SseEventMembership::new(repeatable, interleaved);
    if matches_sse_event_pattern(frames, expected, &membership) {
        Ok(())
    } else {
        Err(mismatch(path, "SSE event order mismatch"))
    }
}

/// Borrowed membership indexes used throughout one SSE pattern match.
struct SseEventMembership<'a> {
    /// Event names allowed to repeat contiguously within their ordered stage.
    repeatable: BTreeSet<&'a str>,
    /// Event names allowed between any ordered stages.
    interleaved: BTreeSet<&'a str>,
}

impl<'a> SseEventMembership<'a> {
    /// Indexes repeatable and interleaved event membership without cloning names.
    fn new(repeatable: &'a [String], interleaved: &'a [String]) -> Self {
        Self {
            repeatable: repeatable.iter().map(String::as_str).collect(),
            interleaved: interleaved.iter().map(String::as_str).collect(),
        }
    }
}

/// Matches ordered stages after removing only explicitly interleaved named frames.
fn matches_sse_event_pattern(
    frames: &[super::SseFrame],
    expected: &[String],
    membership: &SseEventMembership<'_>,
) -> bool {
    if expected.is_empty() {
        return frames.iter().all(|frame| {
            frame
                .event
                .as_deref()
                .is_none_or(|event| membership.interleaved.contains(event))
        });
    }

    let mut actual_index = 0;
    for expected_event in expected {
        skip_interleaved_sse_events(frames, &membership.interleaved, &mut actual_index);
        if frames.get(actual_index).and_then(|frame| frame.event.as_deref()) != Some(expected_event.as_str()) {
            return false;
        }
        actual_index += 1;
        skip_interleaved_sse_events(frames, &membership.interleaved, &mut actual_index);
        if membership.repeatable.contains(expected_event.as_str()) {
            while frames.get(actual_index).and_then(|frame| frame.event.as_deref()) == Some(expected_event.as_str()) {
                actual_index += 1;
                skip_interleaved_sse_events(frames, &membership.interleaved, &mut actual_index);
            }
        }
    }
    skip_interleaved_sse_events(frames, &membership.interleaved, &mut actual_index);
    actual_index == frames.len()
}

/// Advances past only explicitly declared named interleaved frames.
fn skip_interleaved_sse_events(frames: &[super::SseFrame], interleaved: &BTreeSet<&str>, actual_index: &mut usize) {
    while frames
        .get(*actual_index)
        .and_then(|frame| frame.event.as_deref())
        .is_some_and(|event| interleaved.contains(event))
    {
        *actual_index += 1;
    }
}

/// Recursively compares normalized JSON and reports only path and rule.
fn compare_values(actual: &Value, expected: &Value, path: &str) -> Result<(), FixtureError> {
    compare_values_inner(actual, expected, path, false)
}

/// Recursively compares normalized JSON while hiding data-controlled map keys.
#[expect(
    clippy::too_many_lines,
    reason = "all JSON shapes produce one path-only comparison policy"
)]
fn compare_values_inner(actual: &Value, expected: &Value, path: &str, opaque_keys: bool) -> Result<(), FixtureError> {
    match (actual, expected) {
        (Value::Object(actual), Value::Object(expected)) => {
            let keys = actual.keys().chain(expected.keys()).collect::<BTreeSet<_>>();
            for (index, key) in keys.into_iter().enumerate() {
                let child_path = if opaque_keys {
                    format!("{path}.<key[{index}]>")
                } else {
                    format!("{path}.{key}")
                };
                let (Some(actual), Some(expected)) = (actual.get(key), expected.get(key)) else {
                    return Err(mismatch(&child_path, "field presence mismatch"));
                };
                let child_has_opaque_keys = opaque_keys || matches!(key.as_str(), "headers" | "linked_ids" | "value");
                compare_values_inner(actual, expected, &child_path, child_has_opaque_keys)?;
            }
            Ok(())
        },
        (Value::Array(actual), Value::Array(expected)) => {
            if actual.len() != expected.len() {
                return Err(mismatch(path, "array length mismatch"));
            }
            for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                compare_values_inner(actual, expected, &format!("{path}[{index}]"), opaque_keys)?;
            }
            Ok(())
        },
        (Value::Null, Value::Null)
        | (Value::Bool(_), Value::Bool(_))
        | (Value::Number(_), Value::Number(_))
        | (Value::String(_), Value::String(_)) => {
            if actual == expected {
                Ok(())
            } else {
                Err(mismatch(path, "value mismatch"))
            }
        },
        _ => Err(mismatch(path, "value type mismatch")),
    }
}

/// Creates an empty exchange used only for deterministic request normalization.
fn empty_exchange() -> RecordedExchange {
    RecordedExchange {
        request: RecordedRequest {
            method: "GET".to_owned(),
            path: "/".to_owned(),
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        },
        response: empty_response(),
    }
}

/// Creates an empty response used only for deterministic request normalization.
fn empty_response() -> RecordedResponse {
    RecordedResponse {
        status: 204,
        headers: BTreeMap::new(),
        body: RecordedBody::Empty,
    }
}

/// Creates a path-aware mismatch error without serializing fixture values.
fn mismatch(path: &str, rule: &'static str) -> FixtureError {
    FixtureError::ReplayMismatch {
        path: path.to_owned(),
        rule,
    }
}

/// Creates an opaque replay runtime error without fixture values.
fn runtime_error(message: &'static str) -> FixtureError {
    FixtureError::ReplayRuntime { message }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests {
    use std::{
        collections::BTreeMap, error::Error as _, fmt::Write as _, net::TcpListener, path::Path, time::Duration,
    };

    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use praxis_ai_apis::openai::ResponseStoreFilter;
    use praxis_core::config::Config;
    use serde_json::{Value, json};

    use super::{
        super::{
            BodyKind, FixtureError, FixtureProvenance, ImportedUpstream, InferenceProtocol, InferenceScenario,
            NormalizationMetadata, ProvenanceKind, RecordedBody, RecordedExchange, RecordedRequest, RecordedResponse,
            ScenarioExpectation, ScenarioTurn, SseFrame, WireFixture, bounds::MAX_SCENARIO_REQUEST_BODY_BYTES,
            sanitize::RedactionRules,
        },
        ScenarioRunner, ScriptedHttpServer, SseEventMembership, build_replay_config, compare_imported_requests,
        finish_backend_after_proxy_shutdown, fixture_comparison_value, has_only_normal_relative_components,
        is_network_value_key, load_and_validate_config_source, resolve_example_config_path, send_recorded_request,
        validate_endpoint_collection, validate_generic_network_string, validate_network_string_with_context,
        validate_replay_filters, validate_replay_network_value, validate_scenario_requests, validate_singular_endpoint,
        validate_sse_events, validate_upstream_inputs,
    };
    use crate::{example_config_path, free_port_guard, proxy::blocked_proxy_guard_for_test};

    #[tokio::test]
    async fn runner_rejects_vacuous_turns_before_starting_a_backend() {
        let scenario = scenario_with_turns(Vec::new());
        let materialize_error = ScenarioRunner::materialize(&scenario, provenance(), Vec::new())
            .await
            .unwrap_err();
        let expected = WireFixture {
            version: 1,
            scenario_id: scenario.id.clone(),
            protocol: scenario.protocol,
            provenance: provenance(),
            normalization: NormalizationMetadata {
                version: 1,
                linked_ids: BTreeMap::new(),
            },
            turns: Vec::new(),
        };

        let replay_error = ScenarioRunner::replay(&scenario, &expected).await.unwrap_err();

        assert!(matches!(
            materialize_error,
            FixtureError::InvalidInferenceTurnCount {
                document: "inference scenario",
                count: 0
            }
        ));
        assert!(matches!(
            replay_error,
            FixtureError::InvalidInferenceTurnCount {
                document: "inference scenario",
                count: 0
            }
        ));
    }

    #[tokio::test]
    async fn materialize_uses_bound_model_in_transformed_upstream_request() {
        let scenario = scenario_with_turns(vec![scenario_turn("initial", "What is 2+2?", false)]);
        let imported = imported_turn("fixture-model", "What is 2+2?", chat_response("4", "chatcmpl-first"));

        let fixture = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect("materialization should complete through the named example config");

        assert_eq!(fixture.turns.len(), 1);
        assert_eq!(
            fixture.turns[0].client.request.body.json_value()["model"],
            "fixture-model"
        );
        assert_eq!(fixture.turns[0].upstream.request.path, "/v1/chat/completions");
        assert_eq!(
            fixture.turns[0].upstream.request.body.json_value()["model"],
            "fixture-model"
        );
        assert_eq!(
            fixture.turns[0].upstream.request.body.json_value()["max_completion_tokens"],
            64
        );
        assert_eq!(fixture.turns[0].client.response.status, 200);
        assert_eq!(
            fixture.turns[0].client.response.body.json_value()["content"][0]["text"],
            "4"
        );
    }

    #[tokio::test]
    async fn materialize_with_rules_uses_custom_literals_in_its_single_final_sanitization() {
        let source = "custom-runner-source";
        let replacement = "<custom-runner-redacted>";
        let scenario = scenario_with_turns(vec![scenario_turn("initial", source, false)]);
        let imported = imported_turn("fixture-model", source, chat_response("safe answer", "chatcmpl-custom"));
        let rules = RedactionRules {
            literals: BTreeMap::from([(source.to_owned(), replacement.to_owned())]),
        };

        let fixture = ScenarioRunner::materialize_with_rules(&scenario, provenance(), vec![imported], &rules)
            .await
            .expect("custom literals should be applied by materialization");
        let serialized = serde_json::to_string(&fixture).unwrap();

        assert!(!serialized.contains(source));
        assert!(serialized.contains(replacement));
    }

    #[test]
    fn imported_request_comparison_allows_payload_and_header_value_redaction() {
        let payload_source = "imported-payload-source";
        let payload_replacement = "<payload-redacted>";
        let header_source = "imported-header-source";
        let header_replacement = "<header-redacted>";
        let mut expected = transformed_request("fixture-model", payload_source, false);
        expected
            .headers
            .insert("request-id".to_owned(), vec![header_source.to_owned()]);
        let mut actual = transformed_request("fixture-model", payload_replacement, false);
        actual
            .headers
            .insert("request-id".to_owned(), vec![header_replacement.to_owned()]);
        let rules = RedactionRules {
            literals: BTreeMap::from([
                (payload_source.to_owned(), payload_replacement.to_owned()),
                (header_source.to_owned(), header_replacement.to_owned()),
            ]),
        };

        compare_imported_requests("messages/runner-test", &provenance(), &[actual], vec![expected], &rules)
            .expect("equivalent imported value redactions should compare after normalization");
    }

    #[tokio::test]
    async fn imported_request_mismatch_is_normalized_and_reported_without_values() {
        let scenario = scenario_with_turns(vec![scenario_turn("initial", "do-not-dump-this-prompt", false)]);
        let mut imported = imported_turn(
            "fixture-model",
            "do-not-dump-this-prompt",
            chat_response("private-response", "chatcmpl-mismatch"),
        );
        imported.exchange.request.body.json_value_mut()["max_completion_tokens"] = json!(999);

        let error = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect_err("different normalized upstream requests must be rejected");
        let rendered = error.to_string();

        assert!(rendered.contains("turns[0].upstream.request.body.value.<key["));
        assert!(rendered.contains("value mismatch"));
        assert!(!rendered.contains("max_completion_tokens"));
        assert!(!rendered.contains("999"));
        assert!(!rendered.contains("64"));
        assert!(!rendered.contains("do-not-dump-this-prompt"));
        assert!(!rendered.contains("private-response"));
    }

    #[tokio::test]
    async fn materialize_preserves_two_turn_order_on_one_scenario_backend() {
        let scenario = scenario_with_turns(vec![
            scenario_turn("first", "first prompt", false),
            scenario_turn("second", "second prompt", false),
        ]);
        let upstream = vec![
            imported_turn(
                "fixture-model",
                "first prompt",
                chat_response("first answer", "chatcmpl-first"),
            ),
            imported_turn(
                "fixture-model",
                "second prompt",
                chat_response("second answer", "chatcmpl-second"),
            ),
        ];

        let fixture = ScenarioRunner::materialize(&scenario, provenance(), upstream)
            .await
            .expect("two ordered turns should materialize");

        assert_eq!(
            fixture.turns.iter().map(|turn| turn.name.as_str()).collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert_eq!(
            fixture.turns[0].upstream.request.body.json_value()["messages"][0]["content"],
            "first prompt"
        );
        assert_eq!(
            fixture.turns[1].upstream.request.body.json_value()["messages"][0]["content"],
            "second prompt"
        );
        assert_eq!(
            fixture.turns[0].client.response.body.json_value()["content"][0]["text"],
            "first answer"
        );
        assert_eq!(
            fixture.turns[1].client.response.body.json_value()["content"][0]["text"],
            "second answer"
        );
    }

    #[tokio::test]
    async fn real_proxy_readiness_does_not_consume_a_legitimate_first_get_root() {
        let scenario = root_get_scenario();
        let imported = ImportedUpstream {
            source_id: Some("root-readiness".to_owned()),
            provider: Some("test-provider".to_owned()),
            model: Some("fixture-model".to_owned()),
            exchange: RecordedExchange {
                request: scenario.turns[0].request.clone(),
                response: RecordedResponse {
                    status: 200,
                    headers: BTreeMap::from([("content-type".to_owned(), vec!["application/json".to_owned()])]),
                    body: RecordedBody::Json {
                        value: json!({"scenario": true}),
                    },
                },
            },
        };

        let fixture = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect("real proxy readiness and first scenario GET / must remain distinct");

        assert_eq!(fixture.turns[0].upstream.request.method, "GET");
        assert_eq!(fixture.turns[0].upstream.request.path, "/");
        assert!(matches!(
            fixture.turns[0].client.response.body,
            RecordedBody::Json { .. }
        ));
    }

    #[tokio::test]
    async fn materialize_validates_exact_client_sse_event_order() {
        let expected_events = vec![
            "message_start".to_owned(),
            "content_block_start".to_owned(),
            "content_block_delta".to_owned(),
            "content_block_stop".to_owned(),
            "message_delta".to_owned(),
            "message_stop".to_owned(),
        ];
        let mut turn = scenario_turn("stream", "Hello", true);
        turn.expect.client_body_kind = BodyKind::Sse;
        turn.expect.client_sse_events = expected_events;
        let scenario = scenario_with_turns(vec![turn]);
        let imported = imported_streaming_turn("fixture-model", "Hello");

        let fixture = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect("canonical streaming translation should satisfy exact event order");

        let RecordedBody::Sse { frames, done } = &fixture.turns[0].client.response.body else {
            panic!("client response should be recorded as SSE");
        };
        assert!(!done, "Anthropic event streams do not use the OpenAI DONE marker");
        assert_eq!(
            frames
                .iter()
                .filter_map(|frame| frame.event.as_deref())
                .collect::<Vec<_>>(),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
    }

    #[test]
    fn repeatable_sse_pattern_accepts_one_or_more_contiguous_named_events() {
        // Catches tying a provider-neutral semantic stage to one provider's chunk count.
        let expected = [
            "message_start".to_owned(),
            "content_block_delta".to_owned(),
            "message_stop".to_owned(),
        ];
        let repeatable = ["content_block_delta".to_owned()];

        for delta_count in [1_usize, 2, 7] {
            let mut names = vec![Some("message_start")];
            names.extend(std::iter::repeat_n(Some("content_block_delta"), delta_count));
            names.push(Some("message_stop"));
            let body = named_sse_body(&names);

            validate_sse_events(&body, &expected, &repeatable, &[], "$.frames")
                .expect("positive contiguous repetition count must satisfy the pattern");
        }
    }

    #[test]
    fn sse_event_membership_indexes_borrowed_repeatable_and_interleaved_names() {
        let repeatable = ["content_block_delta".to_owned()];
        let interleaved = ["ping".to_owned()];

        let membership = SseEventMembership::new(&repeatable, &interleaved);

        let indexed_repeatable = membership.repeatable.get("content_block_delta").unwrap();
        let indexed_interleaved = membership.interleaved.get("ping").unwrap();
        assert_eq!(indexed_repeatable.as_ptr(), repeatable[0].as_ptr());
        assert_eq!(indexed_interleaved.as_ptr(), interleaved[0].as_ptr());
    }

    #[test]
    fn repeatable_sse_pattern_rejects_missing_anonymous_extra_interleaved_and_misordered_events() {
        // Catches treating event declarations as a lossy deduplicated set instead of an ordered pattern.
        let expected = [
            "message_start".to_owned(),
            "content_block_delta".to_owned(),
            "content_block_stop".to_owned(),
            "message_stop".to_owned(),
        ];
        let repeatable = ["content_block_delta".to_owned()];
        let invalid = [
            (
                "zero repetitions",
                vec![Some("message_start"), Some("content_block_stop"), Some("message_stop")],
            ),
            (
                "anonymous frame",
                vec![
                    Some("message_start"),
                    Some("content_block_delta"),
                    None,
                    Some("content_block_stop"),
                    Some("message_stop"),
                ],
            ),
            (
                "extra nonrepeatable event",
                vec![
                    Some("message_start"),
                    Some("message_start"),
                    Some("content_block_delta"),
                    Some("content_block_stop"),
                    Some("message_stop"),
                ],
            ),
            (
                "interleaved repetition",
                vec![
                    Some("message_start"),
                    Some("content_block_delta"),
                    Some("content_block_stop"),
                    Some("content_block_delta"),
                    Some("message_stop"),
                ],
            ),
            (
                "misordered event",
                vec![
                    Some("message_start"),
                    Some("content_block_stop"),
                    Some("content_block_delta"),
                    Some("message_stop"),
                ],
            ),
        ];

        for (case, names) in invalid {
            let error = validate_sse_events(&named_sse_body(&names), &expected, &repeatable, &[], "$.frames")
                .expect_err("invalid SSE pattern must fail");

            assert!(
                error.to_string().contains("SSE event order mismatch"),
                "{case}: {error}"
            );
        }
    }

    #[test]
    fn interleaved_sse_pattern_accepts_declared_events_zero_once_or_many_anywhere() {
        let expected = [
            "message_start".to_owned(),
            "content_block_delta".to_owned(),
            "message_stop".to_owned(),
        ];
        let repeatable = ["content_block_delta".to_owned()];
        let interleaved = ["ping".to_owned()];
        let cases = [
            vec![Some("message_start"), Some("content_block_delta"), Some("message_stop")],
            vec![
                Some("ping"),
                Some("message_start"),
                Some("ping"),
                Some("content_block_delta"),
                Some("ping"),
                Some("message_stop"),
                Some("ping"),
            ],
            vec![
                Some("ping"),
                Some("ping"),
                Some("message_start"),
                Some("ping"),
                Some("content_block_delta"),
                Some("ping"),
                Some("content_block_delta"),
                Some("ping"),
                Some("ping"),
                Some("message_stop"),
                Some("ping"),
                Some("ping"),
            ],
        ];

        for names in cases {
            validate_sse_events(
                &named_sse_body(&names),
                &expected,
                &repeatable,
                &interleaved,
                "$.frames",
            )
            .expect("declared interleaved events may appear any number of times at any position");
        }
    }

    #[test]
    fn interleaved_sse_pattern_preserves_order_and_rejects_undeclared_or_anonymous_extras() {
        let expected = ["message_start".to_owned(), "message_stop".to_owned()];
        let interleaved = ["ping".to_owned()];
        let invalid = [
            vec![Some("message_start"), Some("unknown"), Some("message_stop")],
            vec![Some("message_start"), None, Some("message_stop")],
            vec![Some("message_stop"), Some("ping"), Some("message_start")],
        ];

        for names in invalid {
            let error = validate_sse_events(&named_sse_body(&names), &expected, &[], &interleaved, "$.frames")
                .expect_err("interleaving must not weaken ordered-event matching");
            assert!(error.to_string().contains("SSE event order mismatch"));
        }
    }

    #[test]
    fn empty_named_upstream_pattern_allows_data_only_sse_frames() {
        // Catches rejecting OpenAI-compatible data-only provider streams when no named pattern is declared.
        let body = named_sse_body(&[None, None]);

        validate_sse_events(&body, &[], &[], &[], "$.frames")
            .expect("data-only upstream frames remain valid for an empty named-event pattern");
    }

    fn named_sse_body(names: &[Option<&str>]) -> RecordedBody {
        RecordedBody::Sse {
            frames: names
                .iter()
                .map(|event| SseFrame {
                    event: event.map(str::to_owned),
                    data: "{}".to_owned(),
                    id: None,
                    retry: None,
                })
                .collect(),
            done: false,
        }
    }

    #[tokio::test]
    async fn materialize_validates_named_upstream_sse_event_order() {
        let mut scenario = streaming_responses_scenario();
        let imported = imported_named_responses_stream(&scenario);
        let fixture = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect("matching named upstream events should materialize");
        assert_eq!(
            fixture.turns[0].upstream.response.body.sse_event_names(),
            ["response.created", "response.output_text.delta", "response.completed"]
        );

        scenario.turns[0].expect.upstream_sse_events.swap(0, 1);
        let imported = imported_named_responses_stream(&scenario);
        let error = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect_err("wrong named upstream order must fail its upstream expectation");

        assert!(error.to_string().contains("turns[0].upstream.response.body.frames"));
        assert!(error.to_string().contains("SSE event order mismatch"));
    }

    #[tokio::test]
    async fn replay_returns_actual_fixture_and_reports_path_only_mismatch() {
        let scenario = scenario_with_turns(vec![scenario_turn("initial", "What is 2+2?", false)]);
        let imported = imported_turn("fixture-model", "What is 2+2?", chat_response("4", "chatcmpl-replay"));
        let expected = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect("expected fixture should materialize");

        let report = ScenarioRunner::replay(&scenario, &expected)
            .await
            .expect("replaying the same fixture should match");
        assert_ne!(
            report.actual.normalization.linked_ids.keys().collect::<Vec<_>>(),
            expected.normalization.linked_ids.keys().collect::<Vec<_>>(),
            "the replay report must retain the identifiers actually observed during replay"
        );
        assert_eq!(
            fixture_comparison_value(&report.actual).unwrap(),
            fixture_comparison_value(&expected).unwrap(),
            "nondeterministic raw identifier keys must not change fixture equivalence"
        );

        let mut mismatched = expected;
        mismatched.turns[0].client.response.body.json_value_mut()["content"][0]["text"] = json!("secret mismatch");
        let error = ScenarioRunner::replay(&scenario, &mismatched)
            .await
            .expect_err("changed client response should fail path-aware comparison");
        let rendered = error.to_string();
        assert!(rendered.contains("turns[0].client.response.body.value.<key["));
        assert!(rendered.contains("value mismatch"));
        assert!(!rendered.contains("content"));
        assert!(!rendered.contains("text"));
        assert!(!rendered.contains("secret mismatch"));
        assert!(!rendered.contains("first answer"));
    }

    #[tokio::test]
    async fn replay_mismatch_never_exposes_an_arbitrary_json_key() {
        let scenario = scenario_with_turns(vec![scenario_turn("initial", "hello", false)]);
        let imported = imported_turn("fixture-model", "hello", chat_response("world", "chatcmpl-key"));
        let mut expected = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect("expected fixture should materialize");
        expected.turns[0].client.response.body.json_value_mut()["secret-object-key-never-log"] = json!(true);

        let error = ScenarioRunner::replay(&scenario, &expected)
            .await
            .expect_err("an added body key must mismatch");
        let surfaces = error_surfaces(&error);

        assert!(surfaces.contains("field presence mismatch"));
        assert!(!surfaces.contains("secret-object-key-never-log"));
    }

    #[tokio::test]
    async fn replay_rejects_an_altered_expected_normalization_version() {
        let scenario = scenario_with_turns(vec![scenario_turn("initial", "hello", false)]);
        let imported = imported_turn("fixture-model", "hello", chat_response("world", "chatcmpl-version"));
        let mut expected = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect("expected fixture should materialize");
        expected.normalization.version = 99;

        let error = ScenarioRunner::replay(&scenario, &expected)
            .await
            .expect_err("altered normalization version must fail fixture validation");

        assert!(matches!(
            error,
            FixtureError::UnsupportedNormalizationVersion { version: 99 }
        ));
        assert!(!error.to_string().contains("99"));
    }

    #[tokio::test]
    async fn replay_rejects_an_altered_expected_normalization_mapping() {
        let scenario = scenario_with_turns(vec![scenario_turn("initial", "hello", false)]);
        let imported = imported_turn("fixture-model", "hello", chat_response("world", "chatcmpl-mapping"));
        let mut expected = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect("expected fixture should materialize");
        let target = expected
            .normalization
            .linked_ids
            .values_mut()
            .next()
            .expect("fixture should normalize the response identifier");
        *target = "chatcmpl-recorded-9999".to_owned();

        let error = ScenarioRunner::replay(&scenario, &expected)
            .await
            .expect_err("altered normalization relationship must not be copied into actual");
        let surfaces = error_surfaces(&error);

        assert!(surfaces.contains("fixture.normalization.linked_ids"));
        assert!(!surfaces.contains("chatcmpl-mapping"));
        assert!(!surfaces.contains("chatcmpl-recorded-9999"));
    }

    #[tokio::test]
    async fn replay_transport_error_never_exposes_request_query() {
        let port = free_port_guard().release();
        let client = reqwest::Client::new();
        let request = RecordedRequest {
            method: "GET".to_owned(),
            path: "/?credential=transport-secret-never-log".to_owned(),
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        };

        let error = send_recorded_request(&client, &format!("127.0.0.1:{port}"), &request)
            .await
            .expect_err("released loopback port should reject the request");
        let surfaces = error_surfaces(&error);

        assert!(!surfaces.contains("transport-secret-never-log"));
        assert!(
            error.source().is_none(),
            "opaque replay transport errors have no source chain"
        );
    }

    #[tokio::test]
    async fn materialize_rejects_an_authority_absent_from_the_named_config() {
        let mut scenario = scenario_with_turns(vec![scenario_turn("initial", "hello", false)]);
        scenario.upstream_authority = "127.0.0.1:65530".to_owned();
        let imported = imported_turn("fixture-model", "hello", chat_response("world", "chatcmpl-offline"));

        let error = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect_err("an absent authority must fail before proxy startup");

        assert_eq!(
            error.to_string(),
            "scenario upstream authority was not found in example config"
        );
    }

    #[test]
    fn scenario_config_resolution_requires_normal_relative_components() {
        let directory = tempfile::tempdir().expect("temporary config root should exist");
        let root = directory.path().join("configs");
        let nested = root.join("nested");
        std::fs::create_dir_all(&nested).expect("nested config directory should exist");
        let config = nested.join("safe.yaml");
        std::fs::write(&config, "safe").expect("temporary config should be written");

        let resolved = resolve_example_config_path(&root, Path::new("nested/safe.yaml"))
            .expect("normal relative components beneath the root should resolve");
        assert_eq!(resolved, config.canonicalize().unwrap());

        for invalid in [
            "./nested/safe.yaml",
            "nested/./safe.yaml",
            "nested/safe.yaml/.",
            "nested/../nested/safe.yaml",
            "../safe.yaml",
        ] {
            let error = resolve_example_config_path(&root, Path::new(invalid))
                .expect_err("dot and parent components must be rejected before reading");
            assert_eq!(error.to_string(), "scenario example config path is not contained");
        }

        let error = resolve_example_config_path(&root, &config)
            .expect_err("absolute config paths must be rejected before reading");
        assert_eq!(error.to_string(), "scenario example config path is not contained");

        for windows_drive_path in [
            "C:",
            "C:/configs/safe.yaml",
            "z:relative.yaml",
            "nested/C:/safe.yaml",
            "nested/z:relative.yaml",
        ] {
            assert!(
                !has_only_normal_relative_components(Path::new(windows_drive_path)),
                "Windows drive-looking components must be rejected on every host"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn scenario_config_resolution_rejects_symlink_escape_before_reading() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().expect("temporary config root should exist");
        let root = directory.path().join("configs");
        let outside = directory.path().join("outside");
        std::fs::create_dir_all(&root).expect("config root should exist");
        std::fs::create_dir_all(&outside).expect("outside directory should exist");
        std::fs::write(outside.join("escaped.yaml"), "outside").expect("outside file should exist");
        symlink(&outside, root.join("linked")).expect("test symlink should be created");

        let error = resolve_example_config_path(&root, Path::new("linked/escaped.yaml"))
            .expect_err("a symlinked component must be rejected before reading");

        assert_eq!(error.to_string(), "scenario example config path is not contained");
    }

    #[test]
    fn scenario_config_loader_rejects_traversal_even_when_it_returns_inside_the_root() {
        let mut scenario = scenario_with_turns(vec![scenario_turn("initial", "hello", false)]);
        scenario.example_config = "../configs/anthropic/messages-to-openai.yaml".to_owned();

        let error = load_and_validate_config_source(&scenario)
            .expect_err("a parent component must be rejected even when normalization lands inside the root");

        assert_eq!(error.to_string(), "scenario example config path is not contained");
    }

    #[tokio::test]
    async fn materialize_returns_redirect_without_contacting_location_target() {
        let sentinel = ScriptedHttpServer::start(Vec::new()).expect("loopback sentinel should start");
        let scenario = responses_scenario();
        let imported = ImportedUpstream {
            source_id: Some("redirect-test".to_owned()),
            provider: Some("test-provider".to_owned()),
            model: Some("fixture-model".to_owned()),
            exchange: RecordedExchange {
                request: scenario.turns[0].request.clone(),
                response: RecordedResponse {
                    status: 302,
                    headers: BTreeMap::from([(
                        "location".to_owned(),
                        vec![format!("http://{}/redirect-sentinel", sentinel.addr())],
                    )]),
                    body: RecordedBody::Empty,
                },
            },
        };

        let fixture = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect("redirect response must be returned without following Location");

        assert_eq!(fixture.turns[0].client.response.status, 302);
        let contacted = sentinel.wait_for_exchanges(1, Duration::from_millis(20)).await;
        assert!(contacted.is_err(), "redirect target must receive no requests");
        assert!(sentinel.take_exchanges().is_empty());
    }

    #[test]
    fn replay_config_rejects_any_unpatched_nonloopback_provider_endpoint() {
        let source = std::fs::read_to_string(example_config_path("openai/responses/responses-proxy.yaml"))
            .expect("example config should load")
            .replace(
                "              - \"127.0.0.1:3001\"",
                "              - \"127.0.0.1:3001\"\n              - \"192.0.2.1:443\"",
            );

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("every provider endpoint must select the replay-owned backend");

        assert_eq!(error.to_string(), "scenario outbound target is not replay-owned");
    }

    #[test]
    fn replay_config_rejects_a_nonloopback_secondary_listener() {
        let source = replay_config_source("openai/responses/responses-proxy.yaml").replace(
            "    filter_chains: [responses-proxy]\n",
            "    filter_chains: [responses-proxy]\n  - name: secondary\n    address: \"192.0.2.10:18081\"\n    filter_chains: [responses-proxy]\n",
        );

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("every listener bind must be loopback after patching");

        assert_eq!(error.to_string(), "scenario replay requires exactly one listener");
    }

    #[test]
    fn replay_config_rejects_a_nonloopback_admin_bind() {
        let source = format!(
            "{}\nadmin:\n  address: \"192.0.2.10:19090\"\ninsecure_options:\n  allow_public_admin: true\n",
            replay_config_source("openai/responses/responses-proxy.yaml")
        );

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("the admin bind must be loopback after patching");

        assert_eq!(error.to_string(), "scenario admin bind is not replay-owned");
    }

    #[test]
    fn replay_config_rejects_nonloopback_singular_endpoint_url() {
        let source = replay_config_source("openai/responses/responses-proxy.yaml").replace(
            "        name: inference\n",
            "        name: inference\n        endpoint: \"http://192.0.2.10:3001/check\"\n",
        );

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("a singular outbound endpoint must be loopback");

        assert_eq!(error.to_string(), "scenario outbound URL is not loopback");
    }

    #[test]
    fn replay_config_rejects_nonloopback_inference_url() {
        let source = replay_config_source("openai/responses/responses-proxy.yaml").replace(
            "        name: inference\n",
            "        name: inference\n        inference_url: \"https://example.test/v1/chat/completions\"\n",
        );

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("the compaction inference URL must be loopback");

        assert_eq!(error.to_string(), "scenario outbound URL is not loopback");
    }

    #[test]
    fn replay_config_rejects_a_nonloopback_network_database_url() {
        let source = replay_config_source("openai/responses/compact.yaml")
            .replace("sqlite://responses.db?mode=rwc", "postgresql://example.test/responses");

        let error = build_replay_config(&source, 19_001, "127.0.0.1:11434", 19_002)
            .expect_err("network database URLs must inherit replay containment");

        assert_eq!(error.to_string(), "scenario database backend is not replay-contained");
    }

    #[test]
    fn replay_config_rewrites_file_backed_sqlite_to_an_in_memory_database() {
        let source = replay_config_source("openai/responses/response-store.yaml").replace(
            "sqlite://responses.db?mode=rwc",
            "sqlite:///tmp/replay-owned.db?mode=rwc",
        );

        let config = build_replay_config(&source, 19_001, "127.0.0.1:8000", 19_002)
            .expect("a SQLite-backed example should remain replayable");
        let value = serde_json::to_value(config).unwrap();
        let mut database_urls = Vec::new();
        collect_values_for_key(&value, "database_url", &mut database_urls);

        assert_eq!(database_urls, ["sqlite::memory:"]);
    }

    #[test]
    fn replay_config_rejects_even_local_postgres_and_backend_tls_files() {
        for database_url in [
            "postgres://localhost/responses",
            "postgres://%2Fvar%2Frun%2Fpostgresql/responses",
            "postgres://localhost/responses?sslrootcert=/etc/hosts",
        ] {
            let source = replay_config_source_with_database_url("postgres", database_url, true);
            let error = build_replay_config(&source, 19_001, "127.0.0.1:8000", 19_002)
                .expect_err("replay must not contact a PostgreSQL service or read its TLS files");
            assert_eq!(error.to_string(), "scenario database backend is not replay-contained");
        }
    }

    #[test]
    fn replay_config_matches_response_store_scheme_less_sqlite_paths() {
        let source = replay_config_source("openai/responses/response-store.yaml");

        for database_url in [
            "test.db",
            "/tmp/test.db",
            ":memory:",
            "?mode=memory",
            "?mode=memory&cache=shared",
            "test.db?mode=rwc",
            "/tmp/test.db?mode=ro",
            "data/conversations.db?cache=shared",
        ] {
            let filter_config = serde_yaml::from_str(&format!(
                "backend: sqlite\ndatabase_url: \"{database_url}\"\nresponses_table: responses\nconversations_table: conversations\n"
            ))
            .expect("SQLite parity config should parse as YAML");
            ResponseStoreFilter::from_config(&filter_config)
                .expect("the ResponseStore API should accept the scheme-less SQLite target");

            let candidate = source.replace("sqlite://responses.db?mode=rwc", database_url);
            build_replay_config(&candidate, 19_001, "127.0.0.1:8000", 19_002)
                .expect("replay should preserve every API-accepted local SQLite target");
        }
    }

    #[test]
    fn replay_config_rejects_listener_tls_file_paths_before_config_construction() {
        let source = replay_config_source("openai/responses/responses-proxy.yaml").replace(
            "    filter_chains: [responses-proxy]\n",
            "    filter_chains: [responses-proxy]\n    tls:\n      certificates:\n        - cert_path: /dev/zero\n          key_path: /etc/hosts\n",
        );

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("replay must not read listener TLS files from the scenario config");

        assert_eq!(error.to_string(), "scenario local resource is not replay-contained");
    }

    #[test]
    fn replay_config_rejects_runtime_ca_files_before_config_construction() {
        let mut source = replay_config_source("openai/responses/responses-proxy.yaml");
        source.push_str("\nruntime:\n  upstream_ca_file: /dev/zero\n");

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("replay must not read a runtime CA file");

        assert_eq!(error.to_string(), "scenario local resource is not replay-contained");
    }

    #[test]
    fn replay_config_rejects_filters_with_independent_live_callouts() {
        let parse_config = |filters: &str| {
            serde_yaml::from_str(&format!(
                "listeners:\n  - name: replay\n    address: 127.0.0.1:19001\n    filter_chains: [replay]\nfilter_chains:\n  - name: replay\n    filters:\n{filters}"
            ))
            .expect("test filter config should parse")
        };
        let safe_config: Config = parse_config(
            "      - filter: openai_responses_format\n      - filter: openai_responses_validate\n      - filter: openai_response_store\n      - filter: openai_responses_rehydrate\n      - filter: responses_to_chat_completions\n      - filter: path_rewrite\n      - filter: router\n      - filter: load_balancer\n",
        );
        validate_replay_filters(&safe_config).expect("known safe filters must remain replayable");

        for filters in [
            "      - filter: external_callout\n",
            "      - filter: router\n        branch_chains:\n          - name: nested\n            chains:\n              - name: inline\n                filters:\n                  - filter: external_callout\n",
        ] {
            let config: Config = parse_config(filters);
            let error = validate_replay_filters(&config)
                .expect_err("unreviewed filters must not be replayable, including nested inline branches");

            assert_eq!(error.to_string(), "scenario filter is not replay-contained");
        }

        for path in [
            "openai/responses/agentic-loop.yaml",
            "openai/responses/mcp-tool-resolve.yaml",
            "openai/responses/mcp-dispatch.yaml",
            "openai/responses/file-resolve.yaml",
            "openai/responses/compact.yaml",
            "openai/responses/web-search.yaml",
            "nemo-guardrails.yaml",
        ] {
            let source = replay_config_source(path);
            let config = Config::from_yaml(&source).expect("example config should parse");
            let error = validate_replay_filters(&config)
                .expect_err("independent callout filters must not run during offline replay");

            assert_eq!(error.to_string(), "scenario filter is not replay-contained", "{path}");
        }
    }

    #[test]
    fn replay_config_rejects_endpoint_selection() {
        let config: Config = serde_yaml::from_str(
            "listeners:\n  - name: replay\n    address: 127.0.0.1:19001\n    filter_chains: [replay]\nfilter_chains:\n  - name: replay\n    filters:\n      - filter: endpoint_selector\n"
        )
        .unwrap();

        let error = validate_replay_filters(&config)
            .expect_err("endpoint selection must not escape replay-owned request accounting");

        assert_eq!(error.to_string(), "scenario filter is not replay-contained");
    }

    #[test]
    fn replay_config_allows_agentic_loop_fixture() {
        let source = replay_config_source("openai/responses/agentic-loop-fixture.yaml");
        let config = Config::from_yaml(&source).expect("agentic-loop-fixture config should parse");
        validate_replay_filters(&config).expect("agentic-loop-fixture contains only replay-safe filters");
    }

    #[test]
    fn replay_config_rejects_background_cluster_health_checks() {
        let mut source = replay_config_source("openai/responses/responses-proxy.yaml");
        source.push_str(
            "\nclusters:\n  - name: probe\n    endpoints: [\"127.0.0.1:3001\"]\n    health_check:\n      type: http\ninsecure_options:\n  allow_private_health_checks: true\n",
        );

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("background probes must not consume scripted exchanges");

        assert_eq!(error.to_string(), "scenario health check is not replay-contained");
    }

    #[test]
    fn replay_config_rejects_nonloopback_files_api_url() {
        let source = replay_config_source("openai/responses/responses-proxy.yaml").replace(
            "        name: inference\n",
            "        name: inference\n        files_api_url: \"https://example.test/v1/files\"\n",
        );

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("the Files API URL must be loopback");

        assert_eq!(error.to_string(), "scenario outbound URL is not loopback");
    }

    #[test]
    fn replay_config_fails_closed_for_a_new_http_url_setting() {
        let source = replay_config_source("openai/responses/responses-proxy.yaml").replace(
            "        name: inference\n",
            "        name: inference\n        future_callback_url: \"https://example.test/callback\"\n",
        );

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("new HTTP URL settings must inherit replay containment");

        assert_eq!(error.to_string(), "scenario outbound URL is not loopback");
    }

    #[test]
    fn replay_config_rejects_unowned_secondary_and_admin_binds() {
        let mut source = replay_config_source("openai/responses/responses-proxy.yaml").replace(
            "    filter_chains: [responses-proxy]\n",
            "    filter_chains: [responses-proxy]\n  - name: secondary\n    address: \"[::1]:18081\"\n    filter_chains: [responses-proxy]\n",
        );
        source.push_str("\nadmin:\n  address: \"127.0.0.1:19090\"\n");

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("replay must reject every bind it did not allocate");

        assert_eq!(error.to_string(), "scenario replay requires exactly one listener");
    }

    #[test]
    fn replay_config_rejects_tcp_forwarding_on_the_owned_listener() {
        let source = replay_config_source("openai/responses/responses-proxy.yaml").replace(
            "    filter_chains: [responses-proxy]\n",
            "    protocol: tcp\n    upstream: \"203.0.113.10:443\"\n",
        );

        let error = build_replay_config(&source, 19_001, "127.0.0.1:3001", 19_002)
            .expect_err("an owned bind must not become an external TCP forwarder");

        assert_eq!(error.to_string(), "scenario listener is not replay-owned");
    }

    #[test]
    fn replay_network_validation_contains_authority_and_uri_forms() {
        for local in [
            "127.0.0.1:9000",
            "[::1]:9000",
            "localhost:9000",
            "user@localhost:9000",
            "grpc://127.0.0.1:9000/service",
            "grpc://localhost:9000/service",
        ] {
            validate_generic_network_string(local).expect("literal loopback network targets should be allowed");
        }

        for nonlocal in [
            "example.test:9000",
            "user@example.test:9000",
            "[2001:db8::1]:9000",
            "grpc://example.test:9000/service",
        ] {
            let error = validate_generic_network_string(nonlocal)
                .expect_err("scheme-less and non-HTTP network targets must require loopback");
            assert!(error.to_string().contains("not loopback"));
        }

        for malformed in [
            "//example.test/service",
            "grpc://[::1",
            "grpc:/example.test/service",
            "grpc:example.test",
        ] {
            validate_generic_network_string(malformed).expect_err("network-looking malformed values must fail closed");
        }
    }

    #[test]
    fn replay_network_validation_uses_key_context_for_bare_hosts() {
        let safe = json!({
            "metadata": {
                "backup": "responses.db:backup",
                "file_url": "resolve",
                "model": "llama3.2:1b",
                "service": "service.name:api"
            }
        });
        let mut found_expected_backend = false;
        validate_replay_network_value(&safe, "127.0.0.1:19002", &mut found_expected_backend)
            .expect("ordinary colon-bearing config values are not network destinations");

        for (key, value) in [("callback_url", "localhost/callback"), ("database_host", "127.0.0.1")] {
            assert!(is_network_value_key(key));
            validate_network_string_with_context(value, true)
                .expect("bare loopback destinations beneath network-valued keys should be allowed");
        }

        for (key, value) in [
            ("callback_url", "example.test"),
            ("database_host", "user@example.test"),
            ("service_endpoint", "example.test/path"),
        ] {
            assert!(is_network_value_key(key));
            let error = validate_network_string_with_context(value, true)
                .expect_err("bare hosts beneath network-valued keys must require loopback");
            assert!(error.to_string().contains("not loopback"));
        }
    }

    #[test]
    fn replay_network_validation_allows_numeric_ports_on_contextual_bare_hosts() {
        for (key, value) in [
            ("callback_url", "localhost:9000/path"),
            ("database_host", "[::1]:9000/path"),
            ("service_endpoint", "user@localhost:9000/path"),
            ("service_url", "localhost:9000/path@opaque"),
        ] {
            assert!(is_network_value_key(key));
            validate_network_string_with_context(value, true)
                .expect("contextual bare loopback hosts may carry a numeric port and path");
        }

        for (key, value) in [
            ("callback_url", "example.test:9000/path"),
            ("database_host", "[2001:db8::1]:9000/path"),
            ("service_endpoint", "user@example.test:9000/path"),
        ] {
            assert!(is_network_value_key(key));
            let error = validate_network_string_with_context(value, true)
                .expect_err("contextual bare hosts with ports must still require loopback");
            assert!(error.to_string().contains("not loopback"));
        }

        for malformed in ["localhost:not-a-port/path", "[::1]:not-a-port/path"] {
            validate_network_string_with_context(malformed, true)
                .expect_err("contextual bare host ports must be numeric");
        }
    }

    #[test]
    fn replay_contextual_port_detection_ignores_path_query_and_fragment_colons() {
        fn validate(value: &str) -> Result<(), FixtureError> {
            validate_network_string_with_context(value, true).map(|_network| ())
        }

        for local in [
            "localhost:9000/path:123",
            "localhost:9000/resource?cursor=v:2",
            "localhost:9000/resource#fragment:3",
            "localhost/path:123",
            "localhost/resource?cursor=v:2",
            "localhost/resource#fragment:3",
            "[::1]:9000/path:123",
            "[::1]:9000/resource?cursor=v:2",
            "[::1]:9000/resource#fragment:3",
            "user@localhost:9000/path:123",
            "user@[::1]:9000/resource?cursor=v:2",
        ] {
            validate(local).expect("colons outside the authority must remain opaque suffix data");
        }

        for remote in [
            "example.test:9000/path:123",
            "example.test:9000/resource?cursor=v:2",
            "example.test:9000/resource#fragment:3",
            "example.test/path:123",
            "[2001:db8::1]:9000/resource?cursor=v:2",
            "user@example.test:9000/path:123",
        ] {
            let error = validate(remote).expect_err("the contextual authority must still require a loopback host");
            assert_eq!(error.to_string(), "scenario outbound URL is not loopback");
        }

        for malformed in [
            "localhost:not-a-port/path:123",
            "localhost:65536/resource?cursor=v:2",
            "[::1]:not-a-port/resource#fragment:3",
            "[::1]:65536/path:123",
        ] {
            let error = validate(malformed).expect_err("an invalid authority port must fail closed");
            assert_eq!(error.to_string(), "scenario outbound URL is malformed");
        }
    }

    #[test]
    fn replay_bracketed_ipv6_authority_requires_an_exact_optional_port_suffix() {
        fn validate(value: &str) -> Result<(), FixtureError> {
            validate_network_string_with_context(value, true).map(|_network| ())
        }

        for local in [
            "[::1]",
            "[::1]:0",
            "[::1]:65535/path",
            "user@[::1]/resource",
            "user:password@[::1]:9000/resource?cursor=v:2#fragment",
        ] {
            validate(local).expect("a bracketed loopback authority may have no port or one valid u16 port");
        }

        for remote in ["[2001:db8::1]", "[2001:db8::1]:9000/resource"] {
            let error = validate(remote).expect_err("a well-formed bracketed authority must still be loopback");
            assert_eq!(error.to_string(), "scenario outbound URL is not loopback");
        }

        for malformed in [
            "[::1]garbage",
            "[::1]garbage:9000",
            "[::1]garbage:not-a-port",
            "[::1]garbage/path",
            "[::1]garbage?cursor=1",
            "[::1]garbage#fragment",
            "user@[::1]garbage:9000/path",
            "[::1",
            "::1]",
            "[::1]:",
            "[::1]:65536",
            "[::1]:not-a-port",
        ] {
            let error = validate(malformed).expect_err("bracketed IPv6 authority junk must fail closed");
            assert_eq!(error.to_string(), "scenario outbound URL is malformed");
        }
    }

    #[test]
    fn replay_network_validation_treats_strong_syntax_as_network_everywhere() {
        for local in [
            "127.0.0.1",
            "::1",
            "postgresql://localhost/responses",
            "postgresql://user:pass@localhost/responses",
            "custom+transport://127.0.0.1/service",
        ] {
            validate_generic_network_string(local).expect("strong loopback syntax should remain local");
        }

        for nonlocal in [
            "192.0.2.1",
            "postgresql://example.test/responses",
            "postgresql://user:pass@example.test/responses",
            "custom+transport://example.test/service",
        ] {
            let error = validate_generic_network_string(nonlocal)
                .expect_err("strong network syntax must require a literal loopback destination");
            assert!(error.to_string().contains("not loopback"));
        }
    }

    #[test]
    fn replay_postgres_url_validation_requires_every_declared_host_to_be_loopback() {
        let local = [
            "postgres://localhost/responses",
            "postgres://127.0.0.1/responses?host=localhost",
            "postgres://[::1]/responses?hostaddr=127.0.0.1",
            "postgres://localhost/responses?host=%3A%3A1&port=5432",
        ];
        for database_url in local {
            validate_generic_network_string(database_url).unwrap_or_else(|error| {
                panic!("every declared PostgreSQL host in {database_url} is loopback: {error}")
            });
        }

        let remote = [
            "postgres://example.test/responses?host=localhost",
            "postgres://localhost/responses?host=example.test",
            "postgres://[::1]/responses?hostaddr=192.0.2.1",
            "postgres://localhost/responses?host=127.0.0.1&host=example.test",
        ];
        for database_url in remote {
            let error = validate_generic_network_string(database_url)
                .expect_err("every declared PostgreSQL target must be contained");
            assert_eq!(error.to_string(), "scenario outbound URL is not loopback");
        }
    }

    #[test]
    fn replay_postgres_url_accepts_only_explicit_loopback_hosts() {
        for database_url in [
            "postgres://localhost/responses",
            "postgres://LOCALHOST/responses",
            "postgres://127.0.0.2/responses",
            "postgres://[::1]/responses",
            "postgres://[::ffff:127.0.0.1]/responses",
            "postgres://localhost/responses?host=127.0.0.1",
            "postgres://localhost/responses?host=%3A%3A1",
            "postgres://localhost/responses?hostaddr=127.0.0.1",
            "postgres://localhost/responses?hostaddr=%3A%3A1",
        ] {
            validate_generic_network_string(database_url)
                .unwrap_or_else(|error| panic!("explicit local PostgreSQL target must pass: {error}"));
        }

        for database_url in [
            "postgres://example.test/responses",
            "postgres://192.0.2.1/responses",
            "postgres://[2001:db8::1]/responses",
            "postgres://localhost/responses?host=example.test",
            "postgres://localhost/responses?host=192.0.2.1",
            "postgres://localhost/responses?host=127.1",
            "postgres://localhost/responses?host=relative/socket",
            "postgres://localhost/responses?hostaddr=192.0.2.1",
            "postgres://%2Fvar%2Frun%2Fpostgresql/responses",
            "postgres://localhost/responses?host=%2Fvar%2Frun%2Fpostgresql",
        ] {
            let error = validate_generic_network_string(database_url)
                .expect_err("remote DNS/IP PostgreSQL targets must be rejected");
            assert_eq!(error.to_string(), "scenario outbound URL is not loopback");
        }

        for database_url in [
            "postgres://localhost/responses?hostaddr=localhost",
            "postgres://localhost/responses?hostaddr=127.1",
            "postgres://localhost/responses?hostaddr=not-an-ip",
        ] {
            let error =
                validate_generic_network_string(database_url).expect_err("hostaddr must be a literal IP address");
            assert_eq!(error.to_string(), "scenario outbound URL is malformed");
        }
    }

    #[test]
    fn replay_postgres_target_inspection_ignores_backend_owned_identity_and_database() {
        for database_url in [
            "postgres://user%FF:password@localhost/responses",
            "postgres://user:password@localhost/%FF",
        ] {
            validate_generic_network_string(database_url).unwrap_or_else(|error| {
                panic!("backend-owned identity and database text must not affect containment: {error}")
            });
        }
    }

    #[test]
    fn replay_postgres_url_requires_a_loopback_authority() {
        for database_url in [
            "postgres:///responses?host=localhost",
            "postgres:///?hostaddr=127.0.0.1",
            "postgres:///responses?host=/var/run/postgresql",
            "postgresql:///responses?host=%2Fvar%2Frun%2Fpostgresql&port=6543",
            "postgres:///responses",
            "postgres:///?port=5432",
            "postgres:///responses?host=example.test",
            "postgres:///responses?hostaddr=192.0.2.1",
            "postgres:///responses?HOST=localhost",
        ] {
            validate_generic_network_string(database_url)
                .expect_err("PostgreSQL URLs without an authority must fail closed");
        }
    }

    #[test]
    fn replay_postgres_query_port_is_u16_and_does_not_retarget() {
        for database_url in [
            "postgres://localhost/responses?port=0",
            "postgres://[::1]/responses?port=65535",
            "postgres://localhost/responses?host=127.0.0.1&port=%36%35%34%33",
            "postgres://localhost/responses?port=5432&sslmode=backend-owned&statement-cache-capacity=also-owned",
        ] {
            validate_generic_network_string(database_url)
                .unwrap_or_else(|error| panic!("valid port must preserve the local target: {error}"));
        }

        for database_url in [
            "postgres://localhost/responses?port=",
            "postgres://localhost/responses?port=-1",
            "postgres://localhost/responses?port=65536",
            "postgres://localhost/responses?port=not-a-port",
            "postgres://localhost/responses?port=not-a-port&port=5432",
            "postgres://localhost/responses?port=5432&port=65536",
        ] {
            let error = validate_generic_network_string(database_url)
                .expect_err("every PostgreSQL port override must parse as u16");
            assert_eq!(error.to_string(), "scenario outbound URL is malformed");
        }

        let error = validate_generic_network_string("postgres://example.test/responses?port=5432")
            .expect_err("a port-only override cannot make a remote host local");
        assert_eq!(error.to_string(), "scenario outbound URL is not loopback");
    }

    #[test]
    fn replay_config_rejects_all_response_store_postgres_targets() {
        let cases = [
            (
                "remote authority replaced by local host",
                "postgres://user:p%40ss@example.test:5432/responses?host=localhost",
            ),
            (
                "remote authority replaced by local hostaddr",
                "postgresql://example.test/responses?hostaddr=%31%32%37.0.0.1",
            ),
            ("hostless local host", "postgres:///responses?host=localhost"),
            (
                "hostless local socket",
                "postgres:///responses?host=%2Fvar%2Frun%2Fpostgresql",
            ),
            (
                "percent-encoded authority socket",
                "postgres://%2Fvar%2Frun%2Fpostgresql/responses",
            ),
            (
                "local authority replaced by remote host",
                "postgres://localhost/responses?host=example.test",
            ),
            (
                "local authority replaced by remote hostaddr",
                "postgres://127.0.0.1/responses?hostaddr=192.0.2.1",
            ),
            ("hostless remote host", "postgres:///responses?host=example.test"),
        ];

        for (case, database_url) in cases {
            response_store_filter_accepts_database_url("postgres", database_url, true);
            let source = replay_config_source_with_database_url("postgres", database_url, true);
            let error = build_replay_config(&source, 19_001, "127.0.0.1:8000", 19_002)
                .expect_err("replay must not contact any PostgreSQL target");
            assert_eq!(
                error.to_string(),
                "scenario database backend is not replay-contained",
                "{case}"
            );
        }
    }

    #[test]
    fn replay_network_validation_allows_sqlite_but_rejects_file_resources() {
        for local in [
            "sqlite:///tmp/responses.db",
            "sqlite::memory:",
            "sqlite://responses.db?mode=rwc",
        ] {
            validate_generic_network_string(local).expect("SQLite resources are rewritten before replay");
        }

        for external_resource in [
            "file:///tmp/replay.yaml",
            "unix:///tmp/replay.sock",
            "file://example.test/replay.yaml",
            "file://localhost/replay.yaml",
            "unix://example.test/replay.sock",
            "unix://localhost/replay.sock",
        ] {
            let error = validate_generic_network_string(external_resource)
                .expect_err("replay must reject every external file or socket resource");
            assert_eq!(error.to_string(), "scenario local resource is not replay-contained");
        }
    }

    #[test]
    fn replay_network_validation_matches_accepted_sqlite_api_forms() {
        for local in [
            "sqlite:",
            "sqlite://",
            "sqlite:?",
            "sqlite://?",
            "sqlite::memory:",
            "sqlite://:memory:",
            "sqlite:///tmp/responses.db",
            "sqlite://responses.db?mode=rwc",
            "sqlite:test.db?mode=rwc",
            "sqlite://file?mode=memory",
            "sqlite://data/conversations.db",
            "sqlite://example.test",
            "sqlite://example.test/responses.db",
            "sqlite://localhost/responses.db",
        ] {
            validate_generic_network_string(local).expect("SQLite API URL forms are local resources, not authorities");
        }
    }

    #[test]
    fn replay_config_matches_response_store_empty_sqlite_temporary_databases() {
        for database_url in ["sqlite:", "sqlite://", "sqlite:?", "sqlite://?"] {
            response_store_filter_accepts_database_url("sqlite", database_url, false);
            let source = replay_config_source_with_database_url("sqlite", database_url, false);
            build_replay_config(&source, 19_001, "127.0.0.1:8000", 19_002)
                .unwrap_or_else(|error| panic!("API-accepted temporary SQLite target must replay: {error}"));
        }
    }

    #[test]
    fn replay_singular_endpoint_requires_the_exact_replay_backend() {
        for endpoint in ["grpc://127.0.0.1:9000/service", "127.0.0.1:9000"] {
            validate_singular_endpoint(endpoint, "127.0.0.1:9000")
                .expect("the exact replay backend should remain allowed");
        }

        for endpoint in [
            "grpc://localhost:9000/service",
            "127.0.0.1:9001",
            "grpc://example.test:9000/service",
        ] {
            validate_singular_endpoint(endpoint, "127.0.0.1:9000")
                .expect_err("every ambient endpoint must be rejected");
        }
    }

    #[test]
    fn replay_endpoint_collection_recurses_into_weighted_members() {
        let endpoints = json!([{
            "address": "127.0.0.1:19002",
            "metadata": {
                "callback_url": "example.test/callback"
            },
            "weight": 1
        }]);
        let mut found_expected_backend = false;

        let error = validate_endpoint_collection(&endpoints, "127.0.0.1:19002", &mut found_expected_backend)
            .expect_err("weighted endpoint members must inherit recursive network containment");

        assert_eq!(error.to_string(), "scenario outbound URL is not loopback");
    }

    #[tokio::test]
    async fn scenario_request_path_rejects_non_origin_forms_before_networking() {
        let client = reqwest::Client::new();
        let sentinel = ScriptedHttpServer::start(Vec::new()).expect("loopback sentinel should start");
        let invalid_paths = [
            "https://example.test/v1/messages",
            "//example.test/v1/messages",
            "/a/../b",
            "/a/./b",
            "/%2e%2e/b",
            "/a/%2E/b",
            "/a/.%2e/b",
            "/a/%2e./b",
            "/v1/messages?mode=exact#hidden",
            "/v1/messages?author=O'Reilly",
            "/v1/messages\r\nx-injected: value",
            "/v1/messages?bad=%zz",
        ];

        for path in invalid_paths {
            let request = RecordedRequest {
                method: "GET".to_owned(),
                path: path.to_owned(),
                headers: BTreeMap::new(),
                body: RecordedBody::Empty,
            };
            let error = send_recorded_request(&client, &sentinel.addr().to_string(), &request)
                .await
                .expect_err("invalid origin-form must fail before reqwest constructs a URL");
            assert_eq!(
                error.to_string(),
                "replay mismatch at client.request.path: request path must be origin-form"
            );
        }

        assert!(
            sentinel.take_exchanges().is_empty(),
            "invalid paths must not reach the wire"
        );
    }

    #[tokio::test]
    async fn scenario_request_preflight_rejects_any_later_turn_before_first_wire() {
        for (invalid_method, invalid_path, expected_error) in [
            (
                "GET bad",
                "/v1/messages",
                "replay mismatch at client.request.method: invalid HTTP method",
            ),
            (
                "GET",
                "/v1/messages?author=O'Reilly",
                "replay mismatch at client.request.path: request path must be origin-form",
            ),
        ] {
            let sentinel = ScriptedHttpServer::start(Vec::new()).expect("loopback sentinel should start");
            let client = reqwest::Client::new();
            let mut scenario = scenario_with_turns(vec![
                scenario_turn("first", "must not reach the wire", false),
                scenario_turn("invalid", "must fail during preflight", false),
            ]);
            scenario.turns[1].request.method = invalid_method.to_owned();
            scenario.turns[1].request.path = invalid_path.to_owned();

            let preflight = validate_scenario_requests(&scenario);
            if preflight.is_ok() {
                for turn in &scenario.turns {
                    let _response = send_recorded_request(&client, &sentinel.addr().to_string(), &turn.request).await;
                }
            }

            let error = preflight.expect_err("every turn must be validated before the first request is sent");
            assert_eq!(error.to_string(), expected_error);
            assert!(
                sentinel.take_exchanges().is_empty(),
                "an invalid later turn must prevent the first turn from reaching the wire"
            );
        }
    }

    #[tokio::test]
    async fn materialize_preflights_every_turn_before_backend_start() {
        let mut scenario = scenario_with_turns(vec![
            scenario_turn("first", "must not start", false),
            scenario_turn("invalid", "must fail during preflight", false),
        ]);
        scenario.turns[1].request.method = "GET bad".to_owned();
        let mut invalid_script = chat_response("unused", "chatcmpl-unused");
        invalid_script.status = 101;
        let upstream = vec![
            imported_turn("fixture-model", "must not start", invalid_script.clone()),
            imported_turn("fixture-model", "must fail during preflight", invalid_script),
        ];

        let error = ScenarioRunner::materialize(&scenario, provenance(), upstream)
            .await
            .expect_err("scenario preflight must win before scripted backend validation or startup");

        assert_eq!(
            error.to_string(),
            "replay mismatch at client.request.method: invalid HTTP method"
        );
    }

    #[tokio::test]
    async fn scenario_request_path_preserves_the_exact_query_on_the_wire() {
        let server = ScriptedHttpServer::start(vec![RecordedResponse {
            status: 204,
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        }])
        .expect("loopback capture server should start");
        let client = reqwest::Client::new();
        let path = "/v1/messages?mode=a%2Fb&author=O%27Reilly&dot=%2E&empty=";
        let request = RecordedRequest {
            method: "GET".to_owned(),
            path: path.to_owned(),
            headers: BTreeMap::new(),
            body: RecordedBody::Empty,
        };

        let response = send_recorded_request(&client, &server.addr().to_string(), &request)
            .await
            .expect("a legal origin-form path should be sent");
        let captured = server.finish(1).expect("capture should stop with exact accounting");

        assert_eq!(response.status, 204);
        assert_eq!(captured[0].path, path);
    }

    #[tokio::test]
    async fn materialize_rejects_an_oversized_scenario_request_before_networking() {
        let mut scenario = scenario_with_turns(vec![scenario_turn("oversized", "small", false)]);
        scenario.turns[0].request.body = RecordedBody::Json {
            value: Value::String("x".repeat(MAX_SCENARIO_REQUEST_BODY_BYTES)),
        };
        let imported = imported_turn("fixture-model", "small", chat_response("world", "chatcmpl-oversized"));

        let error = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect_err("oversized scenario input must fail before networking");

        assert_eq!(error.to_string(), "scenario request body exceeded replay limit");
    }

    #[tokio::test]
    async fn materialize_rejects_body_bearing_head_response_before_networking() {
        let mut scenario = responses_scenario();
        scenario.turns[0].request.method = "HEAD".to_owned();
        scenario.turns[0].request.headers.clear();
        scenario.turns[0].request.body = RecordedBody::Empty;
        scenario.turns[0].expect.client_status = 200;
        scenario.turns[0].expect.client_body_kind = BodyKind::Empty;
        scenario.turns[0].expect.upstream_body_kind = BodyKind::Empty;
        let imported = ImportedUpstream {
            source_id: Some("head-content".to_owned()),
            provider: Some("test-provider".to_owned()),
            model: Some("fixture-model".to_owned()),
            exchange: RecordedExchange {
                request: scenario.turns[0].request.clone(),
                response: RecordedResponse {
                    status: 200,
                    headers: BTreeMap::from([("content-type".to_owned(), vec!["application/json".to_owned()])]),
                    body: RecordedBody::Json {
                        value: json!({"hyper_would_suppress": true}),
                    },
                },
            },
        };

        let error = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect_err("HEAD response content must fail before the backend binds");
        let rendered = format!("{error}\n{error:?}");

        assert_eq!(error.to_string(), "scripted HEAD response forbids content");
        assert!(!rendered.contains("hyper_would_suppress"));
    }

    #[test]
    fn upstream_validation_accepts_an_empty_head_response() {
        let mut scenario = responses_scenario();
        scenario.turns[0].request.method = "HEAD".to_owned();
        let imported = ImportedUpstream {
            source_id: Some("head-empty".to_owned()),
            provider: Some("test-provider".to_owned()),
            model: Some("fixture-model".to_owned()),
            exchange: RecordedExchange {
                request: scenario.turns[0].request.clone(),
                response: RecordedResponse {
                    status: 200,
                    headers: BTreeMap::new(),
                    body: RecordedBody::Empty,
                },
            },
        };

        validate_upstream_inputs(&scenario, &provenance(), &[imported])
            .expect("an empty HEAD response is wire-compatible");
    }

    #[test]
    fn backend_capture_requires_a_joined_proxy_and_releases_backend_on_timeout() {
        let backend = ScriptedHttpServer::start(Vec::new()).expect("scripted backend should bind");
        let backend_addr = backend.addr();
        let (mut proxy, release_proxy) = blocked_proxy_guard_for_test(Duration::from_millis(20));

        let error = finish_backend_after_proxy_shutdown(&mut proxy, backend, 0)
            .expect_err("backend capture must reject an unjoined proxy producer");
        let rendered = format!("{error}\n{error:?}");

        assert_eq!(error.to_string(), "scenario proxy shutdown did not complete");
        assert!(!rendered.contains(proxy.addr()));
        let rebound =
            TcpListener::bind(backend_addr).expect("early proxy shutdown failure must still drop the backend listener");
        assert_eq!(rebound.local_addr().unwrap(), backend_addr);

        release_proxy.send(()).unwrap();
        proxy.shutdown().expect("released producer should join on retry");
    }

    #[tokio::test]
    async fn materialize_rejects_malformed_upstream_request_and_unused_script() {
        let mut scenario = responses_scenario();
        let malformed = RecordedBody::Base64 {
            data: STANDARD.encode(b"{not-json"),
        };
        scenario.turns[0].request.body = malformed.clone();
        scenario.turns[0].expect.client_status = 400;
        scenario.turns[0].expect.upstream_body_kind = BodyKind::Base64;
        let imported = ImportedUpstream {
            source_id: Some("malformed-test".to_owned()),
            provider: Some("test-provider".to_owned()),
            model: Some("fixture-model".to_owned()),
            exchange: RecordedExchange {
                request: scenario.turns[0].request.clone(),
                response: RecordedResponse {
                    status: 200,
                    headers: BTreeMap::new(),
                    body: RecordedBody::Empty,
                },
            },
        };

        let error = ScenarioRunner::materialize(&scenario, provenance(), vec![imported])
            .await
            .expect_err("malformed backend request must not leave a successful unnoticed script");

        assert_eq!(error.to_string(), "scripted backend observed a malformed request");
    }

    fn error_surfaces(error: &FixtureError) -> String {
        let mut rendered = format!("{error}\n{error:?}");
        let mut source = error.source();
        while let Some(error) = source {
            write!(rendered, "\n{error}\n{error:?}").expect("writing to a String cannot fail");
            source = error.source();
        }
        rendered
    }

    fn replay_config_source(relative: &str) -> String {
        std::fs::read_to_string(example_config_path(relative)).expect("example config should load")
    }

    fn response_store_filter_accepts_database_url(backend: &str, database_url: &str, allow_private: bool) {
        let database_url = serde_json::to_string(database_url).expect("database URL should serialize");
        let private_option = if allow_private {
            "allow_private_database_url: true\n"
        } else {
            ""
        };
        let config = serde_yaml::from_str(&format!(
            "backend: {backend}\ndatabase_url: {database_url}\nresponses_table: responses\nconversations_table: conversations\n{private_option}"
        ))
        .expect("ResponseStore parity config should parse as YAML");
        ResponseStoreFilter::from_config(&config)
            .unwrap_or_else(|error| panic!("public ResponseStore API must accept parity target: {error}"));
    }

    fn replay_config_source_with_database_url(backend: &str, database_url: &str, allow_private: bool) -> String {
        let source = replay_config_source("openai/responses/response-store.yaml");
        let database_url = serde_json::to_string(database_url).expect("database URL should serialize");
        let private_option = if allow_private {
            "        allow_private_database_url: true\n"
        } else {
            ""
        };
        let patched = source
            .replacen("        backend: sqlite\n", &format!("        backend: {backend}\n"), 1)
            .replacen(
                "        database_url: \"sqlite://responses.db?mode=rwc\"\n",
                &format!("        database_url: {database_url}\n{private_option}"),
                1,
            );
        assert_ne!(patched, source, "ResponseStore example config should be patched once");
        patched
    }

    fn collect_values_for_key(value: &Value, expected_key: &str, values: &mut Vec<String>) {
        match value {
            Value::Object(object) => {
                for (key, value) in object {
                    if key == expected_key
                        && let Value::String(value) = value
                    {
                        values.push(value.clone());
                    }
                    collect_values_for_key(value, expected_key, values);
                }
            },
            Value::Array(items) => {
                for item in items {
                    collect_values_for_key(item, expected_key, values);
                }
            },
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {},
        }
    }

    fn scenario_with_turns(turns: Vec<ScenarioTurn>) -> InferenceScenario {
        InferenceScenario {
            version: 1,
            id: "messages/runner-test".to_owned(),
            description: "runner test".to_owned(),
            protocol: InferenceProtocol::AnthropicMessages,
            example_config: "anthropic/messages-to-openai.yaml".to_owned(),
            upstream_authority: "127.0.0.1:8000".to_owned(),
            features: vec!["messages.request.minimal".to_owned()],
            turns,
        }
    }

    fn responses_scenario() -> InferenceScenario {
        InferenceScenario {
            version: 1,
            id: "responses/redirect".to_owned(),
            description: "redirect containment".to_owned(),
            protocol: InferenceProtocol::OpenaiResponses,
            example_config: "openai/responses/responses-proxy.yaml".to_owned(),
            upstream_authority: "127.0.0.1:3001".to_owned(),
            features: vec!["responses.request.minimal".to_owned()],
            turns: vec![ScenarioTurn {
                name: "redirect".to_owned(),
                request: RecordedRequest {
                    method: "POST".to_owned(),
                    path: "/v1/responses".to_owned(),
                    headers: BTreeMap::from([("content-type".to_owned(), vec!["application/json".to_owned()])]),
                    body: RecordedBody::Json {
                        value: json!({"model": "fixture-model", "input": "hello"}),
                    },
                },
                expect: ScenarioExpectation {
                    client_status: 302,
                    client_body_kind: BodyKind::Empty,
                    upstream_path: "/v1/responses".to_owned(),
                    upstream_body_kind: BodyKind::Json,
                    client_sse_events: Vec::new(),
                    client_sse_repeatable_events: Vec::new(),
                    client_sse_interleaved_events: Vec::new(),
                    upstream_sse_events: Vec::new(),
                    upstream_sse_repeatable_events: Vec::new(),
                    upstream_sse_interleaved_events: Vec::new(),
                },
            }],
        }
    }

    fn root_get_scenario() -> InferenceScenario {
        InferenceScenario {
            version: 1,
            id: "messages/root-readiness".to_owned(),
            description: "real readiness followed by root request".to_owned(),
            protocol: InferenceProtocol::AnthropicMessages,
            example_config: "anthropic/messages-to-openai.yaml".to_owned(),
            upstream_authority: "127.0.0.1:8000".to_owned(),
            features: vec!["messages.request.minimal".to_owned()],
            turns: vec![ScenarioTurn {
                name: "root".to_owned(),
                request: RecordedRequest {
                    method: "GET".to_owned(),
                    path: "/".to_owned(),
                    headers: BTreeMap::new(),
                    body: RecordedBody::Empty,
                },
                expect: ScenarioExpectation {
                    client_status: 200,
                    client_body_kind: BodyKind::Json,
                    upstream_path: "/".to_owned(),
                    upstream_body_kind: BodyKind::Empty,
                    client_sse_events: Vec::new(),
                    client_sse_repeatable_events: Vec::new(),
                    client_sse_interleaved_events: Vec::new(),
                    upstream_sse_events: Vec::new(),
                    upstream_sse_repeatable_events: Vec::new(),
                    upstream_sse_interleaved_events: Vec::new(),
                },
            }],
        }
    }

    fn streaming_responses_scenario() -> InferenceScenario {
        let events = vec![
            "response.created".to_owned(),
            "response.output_text.delta".to_owned(),
            "response.completed".to_owned(),
        ];
        InferenceScenario {
            version: 1,
            id: "responses/named-stream".to_owned(),
            description: "named upstream SSE order".to_owned(),
            protocol: InferenceProtocol::OpenaiResponses,
            example_config: "openai/responses/responses-proxy.yaml".to_owned(),
            upstream_authority: "127.0.0.1:3001".to_owned(),
            features: vec!["responses.streaming".to_owned()],
            turns: vec![ScenarioTurn {
                name: "stream".to_owned(),
                request: RecordedRequest {
                    method: "POST".to_owned(),
                    path: "/v1/responses".to_owned(),
                    headers: BTreeMap::from([("content-type".to_owned(), vec!["application/json".to_owned()])]),
                    body: RecordedBody::Json {
                        value: json!({"model": "fixture-model", "input": "hello", "stream": true}),
                    },
                },
                expect: ScenarioExpectation {
                    client_status: 200,
                    client_body_kind: BodyKind::Sse,
                    upstream_path: "/v1/responses".to_owned(),
                    upstream_body_kind: BodyKind::Json,
                    client_sse_events: events.clone(),
                    client_sse_repeatable_events: Vec::new(),
                    client_sse_interleaved_events: Vec::new(),
                    upstream_sse_events: events,
                    upstream_sse_repeatable_events: Vec::new(),
                    upstream_sse_interleaved_events: Vec::new(),
                },
            }],
        }
    }

    fn imported_named_responses_stream(scenario: &InferenceScenario) -> ImportedUpstream {
        ImportedUpstream {
            source_id: Some("named-stream".to_owned()),
            provider: Some("test-provider".to_owned()),
            model: Some("fixture-model".to_owned()),
            exchange: RecordedExchange {
                request: scenario.turns[0].request.clone(),
                response: RecordedResponse {
                    status: 200,
                    headers: BTreeMap::from([("content-type".to_owned(), vec!["text/event-stream".to_owned()])]),
                    body: RecordedBody::Sse {
                        frames: vec![
                            named_frame(
                                "response.created",
                                &json!({"type": "response.created", "response": {"id": "resp_stream"}}),
                            ),
                            named_frame(
                                "response.output_text.delta",
                                &json!({"type": "response.output_text.delta", "delta": "hello"}),
                            ),
                            named_frame(
                                "response.completed",
                                &json!({"type": "response.completed", "response": {"id": "resp_stream"}}),
                            ),
                        ],
                        done: true,
                    },
                },
            },
        }
    }

    fn named_frame(event: &str, data: &Value) -> SseFrame {
        SseFrame {
            event: Some(event.to_owned()),
            data: data.to_string(),
            id: None,
            retry: None,
        }
    }

    fn scenario_turn(name: &str, prompt: &str, stream: bool) -> ScenarioTurn {
        ScenarioTurn {
            name: name.to_owned(),
            request: RecordedRequest {
                method: "POST".to_owned(),
                path: "/v1/messages".to_owned(),
                headers: BTreeMap::from([("content-type".to_owned(), vec!["application/json".to_owned()])]),
                body: RecordedBody::Json {
                    value: json!({
                        "model": "${MODEL}",
                        "max_tokens": 64,
                        "stream": stream,
                        "messages": [{"role": "user", "content": prompt}],
                    }),
                },
            },
            expect: ScenarioExpectation {
                client_status: 200,
                client_body_kind: BodyKind::Json,
                upstream_path: "/v1/chat/completions".to_owned(),
                upstream_body_kind: BodyKind::Json,
                client_sse_events: Vec::new(),
                client_sse_repeatable_events: Vec::new(),
                client_sse_interleaved_events: Vec::new(),
                upstream_sse_events: Vec::new(),
                upstream_sse_repeatable_events: Vec::new(),
                upstream_sse_interleaved_events: Vec::new(),
            },
        }
    }

    fn provenance() -> FixtureProvenance {
        FixtureProvenance {
            kind: ProvenanceKind::Imported,
            provider: "test-provider".to_owned(),
            model: "fixture-model".to_owned(),
            source_id: Some("runner-test".to_owned()),
        }
    }

    fn imported_turn(model: &str, prompt: &str, response: RecordedResponse) -> ImportedUpstream {
        ImportedUpstream {
            source_id: Some("imported-test".to_owned()),
            provider: Some("test-provider".to_owned()),
            model: Some(model.to_owned()),
            exchange: RecordedExchange {
                request: transformed_request(model, prompt, false),
                response,
            },
        }
    }

    fn imported_streaming_turn(model: &str, prompt: &str) -> ImportedUpstream {
        ImportedUpstream {
            source_id: Some("imported-stream-test".to_owned()),
            provider: Some("test-provider".to_owned()),
            model: Some(model.to_owned()),
            exchange: RecordedExchange {
                request: transformed_request(model, prompt, true),
                response: RecordedResponse {
                    status: 200,
                    headers: BTreeMap::from([
                        ("content-type".to_owned(), vec!["text/event-stream".to_owned()]),
                        ("cache-control".to_owned(), vec!["no-cache".to_owned()]),
                    ]),
                    body: RecordedBody::Sse {
                        frames: vec![
                            chat_chunk(&json!({"role": "assistant"}), None, None),
                            chat_chunk(&json!({"content": "Hello"}), None, None),
                            chat_chunk(&json!({}), Some("stop"), Some(&json!({"completion_tokens": 1}))),
                        ],
                        done: true,
                    },
                },
            },
        }
    }

    fn transformed_request(model: &str, prompt: &str, stream: bool) -> RecordedRequest {
        RecordedRequest {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: BTreeMap::from([("content-type".to_owned(), vec!["application/json".to_owned()])]),
            body: RecordedBody::Json {
                value: json!({
                    "model": model,
                    "max_completion_tokens": 64,
                    "stream": stream,
                    "messages": [{"role": "user", "content": prompt}],
                }),
            },
        }
    }

    fn chat_response(text: &str, id: &str) -> RecordedResponse {
        RecordedResponse {
            status: 200,
            headers: BTreeMap::from([("content-type".to_owned(), vec!["application/json".to_owned()])]),
            body: RecordedBody::Json {
                value: json!({
                    "id": id,
                    "model": "fixture-model",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1}
                }),
            },
        }
    }

    fn chat_chunk(delta: &Value, finish_reason: Option<&str>, usage: Option<&Value>) -> SseFrame {
        SseFrame {
            event: None,
            data: json!({
                "id": "chatcmpl-stream",
                "model": "fixture-model",
                "choices": [{"delta": delta, "index": 0, "finish_reason": finish_reason}],
                "usage": usage,
            })
            .to_string(),
            id: None,
            retry: None,
        }
    }

    trait RecordedBodyTestExt {
        fn json_value(&self) -> &Value;
        fn json_value_mut(&mut self) -> &mut Value;
        fn sse_event_names(&self) -> Vec<&str>;
    }

    impl RecordedBodyTestExt for RecordedBody {
        fn json_value(&self) -> &Value {
            let RecordedBody::Json { value } = self else {
                panic!("test expected JSON body");
            };
            value
        }

        fn json_value_mut(&mut self) -> &mut Value {
            let RecordedBody::Json { value } = self else {
                panic!("test expected JSON body");
            };
            value
        }

        fn sse_event_names(&self) -> Vec<&str> {
            let RecordedBody::Sse { frames, .. } = self else {
                panic!("test expected SSE body");
            };
            frames.iter().filter_map(|frame| frame.event.as_deref()).collect()
        }
    }
}
