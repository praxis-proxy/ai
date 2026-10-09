// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! Hot config reload: validate, build, and atomically swap filter pipelines.

use std::sync::{Arc, Mutex};

use praxis_core::{
    config::Config,
    health::{HealthRegistry, build_health_registry},
};
use praxis_filter::FilterRegistry;
use praxis_protocol::ListenerPipelines;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

#[cfg(feature = "_store-backend")]
use crate::pipelines::resolve_pipelines_with_stores;

// -----------------------------------------------------------------------------
// Reload
// -----------------------------------------------------------------------------

/// Validate a new config, rebuild pipelines, and atomically swap them
/// into the running server.
///
/// On success, cancels old health check tasks and spawns replacements.
/// On failure, logs the error and returns `Err` without modifying any
/// live state.
///
/// # Errors
///
/// Returns an error if the new config fails validation or pipeline
/// construction. The running server is unaffected.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "orchestration function"
)]
#[expect(
    clippy::allow_attributes,
    reason = "the lint only fires with some tracing feature sets"
)]
#[allow(
    clippy::cognitive_complexity,
    reason = "orchestration function; tracing macro expansion varies with enabled features"
)]
pub(crate) fn reload_pipelines(
    new_config: &Config,
    old_config: &Config,
    registry: &FilterRegistry,
    live: &ListenerPipelines,
    health_shutdown: &Arc<Mutex<CancellationToken>>,
    kv_stores: &praxis_core::kv::KvStoreRegistry,
    subrequest_client: &praxis_core::subrequest::SubRequestClient,
    store_reload: &crate::StoreReloadHandle,
    health_slot: &crate::SharedHealthRegistry,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    #[cfg(not(feature = "_store-backend"))]
    let _ = store_reload;
    info!("building new pipelines from reloaded config");

    if let Err(e) = praxis_core::logging::validate_log_overrides(new_config) {
        error!(error = %e, "config reload failed: invalid log_overrides");
        return Err(e.into());
    }

    let health_registry = build_health_registry(&new_config.clusters);

    let new_ceiling = new_config.body_limits.max_response_bytes.unwrap_or(usize::MAX);
    let updated_client = praxis_core::subrequest::SubRequestClient::with_max_response_bytes(
        subrequest_client.connector().clone(),
        new_ceiling,
    );

    // Validate the complete candidate pipeline before provisioning touches a
    // database. Factory validation alone cannot cover cross-filter contracts or
    // every filter-owned security check, such as SQLite path traversal.
    #[cfg(feature = "_store-backend")]
    let prepared_stores = if crate::store_provision::config_uses_store(new_config)
        || crate::store_provision::config_uses_store(old_config)
    {
        let (validation_registries, _service, _reload, _readiness) =
            crate::store_provision::build_store_wiring(new_config)?;
        resolve_pipelines_with_stores(
            new_config,
            registry,
            &health_registry,
            kv_stores,
            &updated_client,
            &validation_registries,
        )?;
        Some(store_reload.prepare(new_config)?)
    } else {
        None
    };

    #[cfg(feature = "_store-backend")]
    let empty_store_registries = crate::StoreRegistries::default();
    #[cfg(feature = "_store-backend")]
    let build = resolve_pipelines_with_stores(
        new_config,
        registry,
        &health_registry,
        kv_stores,
        &updated_client,
        prepared_stores
            .as_ref()
            .map_or(&empty_store_registries, |prepared| &prepared.registries),
    );
    #[cfg(all(feature = "store", not(feature = "_store-backend")))]
    let build = crate::pipelines::resolve_pipelines_with_stores(
        new_config,
        registry,
        &health_registry,
        kv_stores,
        &updated_client,
        &crate::StoreRegistries::default(),
    );
    #[cfg(not(feature = "store"))]
    let build = crate::pipelines::resolve_pipelines(new_config, registry, &health_registry, kv_stores, &updated_client);
    let new_pipelines = match build {
        Ok(p) => p,
        Err(e) => {
            #[cfg(feature = "_store-backend")]
            if let Some(prepared) = prepared_stores
                && let Err(abort_error) = store_reload.abort(prepared)
            {
                error!(error = %abort_error, "failed to release rejected store reload generation");
            }
            error!(error = %e, "config reload failed: pipeline build error");
            return Err(e);
        },
    };

    log_restart_required_changes(old_config, new_config);
    warn_stateful_filter_reset(new_config);

    #[cfg(feature = "_store-backend")]
    let old_pipelines = crate::store_provision::store_listener_names(old_config)
        .into_iter()
        .filter_map(|name| live.get(&name).map(|slot| Arc::downgrade(&slot.load_full())))
        .collect();

    // Promotion cannot fail after this acknowledgement, and the ArcSwap stores
    // below are infallible. Commit before publication so shutdown can never
    // classify an already-published generation as unattached pending state.
    #[cfg(feature = "_store-backend")]
    if let Some(prepared) = prepared_stores {
        store_reload.commit(prepared, old_pipelines)?;
    }

    let mut swapped = Vec::new();
    let mut skipped = Vec::new();

    for name in new_pipelines.listener_names() {
        if let Some(new_slot) = new_pipelines.get(name) {
            let new_arc = new_slot.load_full();
            if live.get(name).is_some() {
                live.swap(name, new_arc);
                swapped.push(name.to_owned());
            } else {
                skipped.push(name.to_owned());
            }
        }
    }

    respawn_health_checks(new_config, &health_registry, health_shutdown);
    // Publish the freshly built registry so the readiness endpoint reflects the
    // reloaded cluster health instead of the startup snapshot.
    *health_slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::clone(&health_registry);

    info!(
        swapped = ?swapped,
        skipped = ?skipped,
        "config reload complete"
    );

    Ok(())
}

// -----------------------------------------------------------------------------
// Health Check Lifecycle
// -----------------------------------------------------------------------------

/// Cancel old health check tasks and spawn new ones from the
/// updated config.
#[expect(clippy::expect_used, reason = "poisoned mutex is unrecoverable")]
fn respawn_health_checks(
    config: &Config,
    health_registry: &HealthRegistry,
    health_shutdown: &Arc<Mutex<CancellationToken>>,
) {
    let old_token = {
        let mut guard = health_shutdown.lock().expect("health shutdown lock poisoned");
        let old = guard.clone();
        *guard = CancellationToken::new();
        old
    };
    old_token.cancel();

    if health_registry.is_empty() {
        return;
    }

    let clusters = config.clusters.clone();
    let registry = Arc::clone(health_registry);
    let new_token = health_shutdown.lock().expect("health shutdown lock poisoned").clone();

    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("health check runtime");
        rt.block_on(async {
            praxis_protocol::http::pingora::health::runner::spawn_health_checks(&clusters, &registry, &new_token);
            new_token.cancelled().await;
        });
    });
}

// -----------------------------------------------------------------------------
// Restart-Required Detection
// -----------------------------------------------------------------------------

/// Compare old and new configs, logging warnings for changes that
/// require a process restart to take effect.
fn log_restart_required_changes(old: &Config, new: &Config) {
    detect_listener_topology_changes(old, new);
    detect_protocol_changes(old, new);
    detect_compression_additions(old, new);
    detect_tls_toggles(old, new);
    detect_listener_setting_changes(old, new);
    detect_subrequest_connector_changes(old, new);
    detect_process_limit_changes(old, new);
}

/// Detect listener additions, removals, and address rebinds.
#[expect(
    clippy::allow_attributes,
    reason = "the lint only fires with some tracing feature sets"
)]
#[allow(
    clippy::cognitive_complexity,
    reason = "pre-existing complexity above threshold; tracing macro expansion varies with enabled features"
)]
fn detect_listener_topology_changes(old: &Config, new: &Config) {
    let old_names: std::collections::HashSet<&str> = old.listeners.iter().map(|l| l.name.as_str()).collect();
    let new_names: std::collections::HashSet<&str> = new.listeners.iter().map(|l| l.name.as_str()).collect();

    for name in new_names.difference(&old_names) {
        warn!(
            listener = %name,
            "listener added in config; requires restart to bind"
        );
    }
    for name in old_names.difference(&new_names) {
        warn!(
            listener = %name,
            "listener removed in config; requires restart to unbind"
        );
    }

    for new_l in &new.listeners {
        if let Some(old_l) = old.listeners.iter().find(|l| l.name == new_l.name)
            && old_l.address != new_l.address
        {
            warn!(
                listener = %new_l.name,
                old_address = %old_l.address,
                new_address = %new_l.address,
                "listener address changed; requires restart to rebind"
            );
        }
    }
}

/// Detect protocol changes (e.g. HTTP to TCP).
fn detect_protocol_changes(old: &Config, new: &Config) {
    for new_l in &new.listeners {
        if let Some(old_l) = old.listeners.iter().find(|l| l.name == new_l.name)
            && old_l.protocol != new_l.protocol
        {
            warn!(
                listener = %new_l.name,
                old_protocol = ?old_l.protocol,
                new_protocol = ?new_l.protocol,
                "protocol changed; requires restart"
            );
        }
    }
}

/// Detect compression being added to a previously uncompressed listener.
fn detect_compression_additions(old: &Config, new: &Config) {
    let old_chains_with_compression = find_chains_with_compression(old);
    let new_chains_with_compression = find_chains_with_compression(new);

    for new_l in &new.listeners {
        if let Some(old_l) = old.listeners.iter().find(|l| l.name == new_l.name) {
            let old_had_compression = old_l
                .filter_chains
                .iter()
                .any(|c| old_chains_with_compression.contains(c.as_str()));

            let new_has_compression = new_l
                .filter_chains
                .iter()
                .any(|c| new_chains_with_compression.contains(c.as_str()));

            if !old_had_compression && new_has_compression {
                warn!(
                    listener = %new_l.name,
                    "compression added; requires restart (module registration is one-shot)"
                );
            }
        }
    }
}

/// Collect chain names that contain a compression filter.
fn find_chains_with_compression(config: &Config) -> std::collections::HashSet<&str> {
    config
        .filter_chains
        .iter()
        .filter(|c| c.filters.iter().any(|f| f.filter_type == "compression"))
        .map(|c| c.name.as_str())
        .collect()
}

/// Detect TLS enable/disable toggles.
fn detect_tls_toggles(old: &Config, new: &Config) {
    for new_l in &new.listeners {
        if let Some(old_l) = old.listeners.iter().find(|l| l.name == new_l.name) {
            match (&old_l.tls, &new_l.tls) {
                (None, Some(_)) => {
                    warn!(
                        listener = %new_l.name,
                        "TLS enabled; requires restart"
                    );
                },
                (Some(_), None) => {
                    warn!(
                        listener = %new_l.name,
                        "TLS disabled; requires restart"
                    );
                },
                _ => {},
            }
        }
    }
}

/// Detect changes to listener settings the HTTP handler captures once at
/// startup: connection limits and downstream timeouts.
fn detect_listener_setting_changes(old: &Config, new: &Config) {
    for new_l in &new.listeners {
        let Some(old_l) = old.listeners.iter().find(|l| l.name == new_l.name) else {
            continue;
        };
        for (field, changed) in [
            ("max_connections", old_l.max_connections != new_l.max_connections),
            (
                "downstream_keepalive_timeout_ms",
                old_l.downstream_keepalive_timeout_ms != new_l.downstream_keepalive_timeout_ms,
            ),
            (
                "downstream_read_timeout_ms",
                old_l.downstream_read_timeout_ms != new_l.downstream_read_timeout_ms,
            ),
        ] {
            if changed {
                warn!(
                    listener = %new_l.name,
                    field,
                    "listener setting changed; requires restart (applied when the listener starts)"
                );
            }
        }
    }
}

/// Detect changes to sub-request connector parameters.
fn detect_subrequest_connector_changes(old: &Config, new: &Config) {
    if old.runtime.subrequest_pool_size != new.runtime.subrequest_pool_size {
        warn!(
            old_pool_size = ?old.runtime.subrequest_pool_size,
            new_pool_size = ?new.runtime.subrequest_pool_size,
            "subrequest pool size changed; requires restart"
        );
    }
    if old.runtime.subrequest_max_connections != new.runtime.subrequest_max_connections {
        warn!(
            old_max = ?old.runtime.subrequest_max_connections,
            new_max = ?new.runtime.subrequest_max_connections,
            "subrequest max connections changed; requires restart"
        );
    }
    detect_subrequest_circuit_breaker_change(old, new);
}

/// Detect changes to process limits applied once at startup.
fn detect_process_limit_changes(old: &Config, new: &Config) {
    if old.runtime.max_open_files != new.runtime.max_open_files {
        warn!(
            old = ?old.runtime.max_open_files,
            new = ?new.runtime.max_open_files,
            "runtime.max_open_files changed; requires restart (the open file limit is set once at startup)"
        );
    }
    if old.runtime.shed_on_fd_pressure != new.runtime.shed_on_fd_pressure {
        warn!(
            old = old.runtime.shed_on_fd_pressure,
            new = new.runtime.shed_on_fd_pressure,
            "runtime.shed_on_fd_pressure changed; requires restart (the descriptor monitor starts once)"
        );
    }
}

/// Detect `runtime.subrequest_circuit_breaker` changes that require a restart.
fn detect_subrequest_circuit_breaker_change(old: &Config, new: &Config) {
    let old_cb = &old.runtime.subrequest_circuit_breaker;
    let new_cb = &new.runtime.subrequest_circuit_breaker;
    let changed = match (old_cb, new_cb) {
        (None, None) => false,
        (None, Some(_)) | (Some(_), None) => true,
        (Some(a), Some(b)) => {
            a.consecutive_failures != b.consecutive_failures
                || a.recovery_window_secs != b.recovery_window_secs
                || a.half_open_timeout_secs != b.half_open_timeout_secs
        },
    };
    if changed {
        warn!(
            old = ?old_cb.as_ref().map(|c| format!(
                "failures={}, recovery={}s, half_open={}s",
                c.consecutive_failures, c.recovery_window_secs, c.half_open_timeout_secs
            )),
            new = ?new_cb.as_ref().map(|c| format!(
                "failures={}, recovery={}s, half_open={}s",
                c.consecutive_failures, c.recovery_window_secs, c.half_open_timeout_secs
            )),
            "runtime.subrequest_circuit_breaker changed; requires restart \
             (circuit breaker registry is bound to the connector)"
        );
    }
}

// -----------------------------------------------------------------------------
// Stateful Filter Warnings
// -----------------------------------------------------------------------------

/// Log a warning when the new config contains stateful filters
/// whose state will reset on reload (e.g. rate limiters).
fn warn_stateful_filter_reset(config: &Config) {
    let has_rate_limiter = config.filter_chains.iter().any(|c| {
        c.filters
            .iter()
            .any(|f| f.filter_type == "rate_limit" || f.filter_type == "circuit_breaker")
    });

    if has_rate_limiter {
        warn!(
            "stateful filters (rate_limit, circuit_breaker) have been \
             reset; in-flight requests retain old state via Arc guard"
        );
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

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
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    use praxis_core::{config::Config, health::HealthRegistry};
    use praxis_filter::FilterRegistry;
    use tokio_util::sync::CancellationToken;
    use tracing_subscriber::layer::SubscriberExt as _;

    use super::*;
    use crate::pipelines::resolve_pipelines;

    #[test]
    fn valid_reload_swaps_pipeline() {
        let (live, old_config, registry, shutdown) = setup_live_pipelines();
        let old_ptr = Arc::as_ptr(&live.get("web").unwrap().load());

        let new_config = valid_config();
        let result = reload_pipelines(
            &new_config,
            &old_config,
            &registry,
            &live,
            &shutdown,
            &empty_kv_stores(),
            &test_client(),
            &crate::StoreReloadHandle::default(),
            &crate::SharedHealthRegistry::default(),
        );

        assert!(result.is_ok(), "valid reload should succeed");
        let new_ptr = Arc::as_ptr(&live.get("web").unwrap().load());
        assert_ne!(old_ptr, new_ptr, "pipeline pointer should change after reload");
    }

    #[test]
    fn invalid_filter_returns_err_old_pipeline_untouched() {
        let (live, old_config, registry, shutdown) = setup_live_pipelines();
        let old_ptr = Arc::as_ptr(&live.get("web").unwrap().load());

        let bad_config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: nonexistent_filter_xyz
"#,
        )
        .unwrap();

        let result = reload_pipelines(
            &bad_config,
            &old_config,
            &registry,
            &live,
            &shutdown,
            &empty_kv_stores(),
            &test_client(),
            &crate::StoreReloadHandle::default(),
            &crate::SharedHealthRegistry::default(),
        );
        assert!(result.is_err(), "invalid filter should return Err");

        let current_ptr = Arc::as_ptr(&live.get("web").unwrap().load());
        assert_eq!(old_ptr, current_ptr, "pipeline should be untouched after failure");
    }

    #[cfg(feature = "store-sqlite")]
    #[test]
    fn invalid_store_candidate_is_rejected_before_provisioning() {
        let old_config = valid_config();
        let client = test_client();
        let registry = crate::build_full_registry(&client);
        let health_registry: HealthRegistry = Arc::new(HashMap::new());
        let kv_stores = empty_kv_stores();
        let live = resolve_pipelines(&old_config, &registry, &health_registry, &kv_stores, &client)
            .expect("initial pipelines");
        let temp = tempfile::tempdir().expect("tempdir");
        let outside = temp.path().join("outside.db");
        let database_url = format!("sqlite://{}/allowed/../outside.db?mode=rwc", temp.path().display());
        let new_config = Config::from_yaml(&format!(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: openai_response_store
        backend: sqlite
        database_url: "{database_url}"
        responses_table: responses
        conversations_table: conversations
"#,
        ))
        .expect("candidate config");

        let result = reload_pipelines(
            &new_config,
            &old_config,
            &registry,
            &live,
            &Arc::new(Mutex::new(CancellationToken::new())),
            &kv_stores,
            &client,
            &crate::StoreReloadHandle::default(),
            &crate::SharedHealthRegistry::default(),
        );

        let error = result
            .expect_err("path traversal must reject the candidate")
            .to_string();
        assert!(
            error.contains("must not contain '..' path traversal"),
            "unexpected error: {error}"
        );
        assert!(!outside.exists(), "rejected reload must not create its SQLite database");
    }

    #[test]
    fn old_cancellation_token_cancelled_on_success() {
        let (live, old_config, registry, shutdown) = setup_live_pipelines();
        let old_token = shutdown.lock().unwrap().clone();

        let new_config = valid_config();
        reload_pipelines(
            &new_config,
            &old_config,
            &registry,
            &live,
            &shutdown,
            &empty_kv_stores(),
            &test_client(),
            &crate::StoreReloadHandle::default(),
            &crate::SharedHealthRegistry::default(),
        )
        .unwrap();

        assert!(
            old_token.is_cancelled(),
            "old token should be cancelled after successful reload"
        );
    }

    #[test]
    fn new_cancellation_token_created_on_success() {
        let (live, old_config, registry, shutdown) = setup_live_pipelines();
        let old_token = shutdown.lock().unwrap().clone();

        let new_config = valid_config();
        reload_pipelines(
            &new_config,
            &old_config,
            &registry,
            &live,
            &shutdown,
            &empty_kv_stores(),
            &test_client(),
            &crate::StoreReloadHandle::default(),
            &crate::SharedHealthRegistry::default(),
        )
        .unwrap();

        let new_token = shutdown.lock().unwrap().clone();
        assert!(
            !new_token.is_cancelled(),
            "new token should not be cancelled after successful reload"
        );
        assert!(old_token.is_cancelled(), "old token should be cancelled");
    }

    #[test]
    fn health_checks_not_cancelled_on_failure() {
        let (live, old_config, registry, shutdown) = setup_live_pipelines();
        let old_token = shutdown.lock().unwrap().clone();

        let bad_config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: nonexistent_filter_xyz
"#,
        )
        .unwrap();

        let _err = reload_pipelines(
            &bad_config,
            &old_config,
            &registry,
            &live,
            &shutdown,
            &empty_kv_stores(),
            &test_client(),
            &crate::StoreReloadHandle::default(),
            &crate::SharedHealthRegistry::default(),
        );
        assert!(
            !old_token.is_cancelled(),
            "health check token should not be cancelled on validation failure"
        );
    }

    #[test]
    fn new_listener_in_config_is_skipped() {
        let (live, old_config, registry, shutdown) = setup_live_pipelines();

        let new_config = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
  - name: new_listener
    address: "127.0.0.1:9090"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: static_response
        status: 200
"#,
        )
        .unwrap();

        let result = reload_pipelines(
            &new_config,
            &old_config,
            &registry,
            &live,
            &shutdown,
            &empty_kv_stores(),
            &test_client(),
            &crate::StoreReloadHandle::default(),
            &crate::SharedHealthRegistry::default(),
        );
        assert!(result.is_ok(), "reload with new listener should succeed");
        assert!(
            live.get("new_listener").is_none(),
            "new listener should not appear in live pipelines"
        );
    }

    #[test]
    fn listener_added_detected() {
        let old = valid_config();
        let new = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
  - name: api
    address: "127.0.0.1:9090"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: static_response
        status: 200
"#,
        )
        .unwrap();

        log_restart_required_changes(&old, &new);
    }

    #[test]
    fn listener_removed_detected() {
        let old = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
  - name: api
    address: "127.0.0.1:9090"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: static_response
        status: 200
"#,
        )
        .unwrap();
        let new = valid_config();

        log_restart_required_changes(&old, &new);
    }

    #[test]
    fn listener_address_changed_detected() {
        let old = valid_config();
        let new = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:9999"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: static_response
        status: 200
"#,
        )
        .unwrap();

        log_restart_required_changes(&old, &new);
    }

    #[test]
    fn protocol_changed_detected() {
        let old = valid_config();
        let new = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    protocol: tcp
    upstream: "10.0.0.1:80"
filter_chains:
  - name: main
    filters:
      - filter: static_response
        status: 200
"#,
        )
        .unwrap();

        log_restart_required_changes(&old, &new);
    }

    #[test]
    fn tls_toggle_detected() {
        let old = valid_config();
        let new = Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
    tls:
      certificates:
        - cert_path: "/tmp/cert.pem"
          key_path: "/tmp/key.pem"
filter_chains:
  - name: main
    filters:
      - filter: static_response
        status: 200
"#,
        )
        .unwrap();

        log_restart_required_changes(&old, &new);
    }

    #[test]
    fn no_restart_required_no_warnings() {
        let old = valid_config();
        let new = valid_config();
        log_restart_required_changes(&old, &new);
    }

    #[test]
    fn subrequest_connector_change_detected() {
        let old = valid_config();
        let mut new = valid_config();
        new.runtime.subrequest_pool_size = Some(32);
        new.runtime.subrequest_max_connections = Some(256);

        log_restart_required_changes(&old, &new);
    }

    #[test]
    fn circuit_breaker_added_warns() {
        let old = valid_config();
        let new = config_with_circuit_breaker(Some(5));
        let warnings = capture_warnings(|| detect_subrequest_circuit_breaker_change(&old, &new));
        assert_eq!(warnings.len(), 1, "adding breaker should produce one warning");
        assert!(
            warnings[0].contains("subrequest_circuit_breaker"),
            "warning should mention circuit breaker: {:?}",
            warnings[0]
        );
    }

    #[test]
    fn circuit_breaker_removed_warns() {
        let old = config_with_circuit_breaker(Some(5));
        let new = valid_config();
        let warnings = capture_warnings(|| detect_subrequest_circuit_breaker_change(&old, &new));
        assert_eq!(warnings.len(), 1, "removing breaker should produce one warning");
    }

    #[test]
    fn circuit_breaker_threshold_changed_warns() {
        let old = config_with_circuit_breaker(Some(3));
        let new = config_with_circuit_breaker(Some(5));
        let warnings = capture_warnings(|| detect_subrequest_circuit_breaker_change(&old, &new));
        assert_eq!(warnings.len(), 1, "changed threshold should produce one warning");
    }

    #[test]
    fn listener_keepalive_timeout_change_warns() {
        let old = config_with_listener_line("");
        let new = config_with_listener_line("    downstream_keepalive_timeout_ms: 5000\n");
        let warnings = capture_warnings(|| detect_listener_setting_changes(&old, &new));
        assert_eq!(
            warnings.len(),
            1,
            "one changed listener setting, one warning: {warnings:?}"
        );
        assert!(
            warnings[0].contains("requires restart"),
            "the timeout is applied when the listener starts: {:?}",
            warnings[0]
        );
    }

    #[test]
    fn listener_limit_and_read_timeout_changes_warn() {
        let old = config_with_listener_line("");
        let new = config_with_listener_line("    max_connections: 10\n    downstream_read_timeout_ms: 5000\n");
        let warnings = capture_warnings(|| detect_listener_setting_changes(&old, &new));
        assert_eq!(warnings.len(), 2, "each changed setting warns: {warnings:?}");
    }

    #[test]
    fn unchanged_listener_settings_do_not_warn() {
        let config = config_with_listener_line("    downstream_keepalive_timeout_ms: 5000\n");
        let warnings = capture_warnings(|| detect_listener_setting_changes(&config, &config));
        assert!(warnings.is_empty(), "nothing changed: {warnings:?}");
    }

    #[test]
    fn max_open_files_change_warns() {
        let old = config_with_runtime_line("threads: 1");
        let new = config_with_runtime_line("max_open_files: 4096");
        let warnings = capture_warnings(|| detect_process_limit_changes(&old, &new));
        assert_eq!(warnings.len(), 1, "one changed limit, one warning: {warnings:?}");
        assert!(
            warnings[0].contains("requires restart"),
            "the open file limit is set once at startup: {:?}",
            warnings[0]
        );
    }

    #[test]
    fn shed_on_fd_pressure_change_warns() {
        let old = config_with_runtime_line("threads: 1");
        let new = config_with_runtime_line("shed_on_fd_pressure: false");
        let warnings = capture_warnings(|| detect_process_limit_changes(&old, &new));
        assert_eq!(warnings.len(), 1, "one changed setting, one warning: {warnings:?}");
        assert!(
            warnings[0].contains("requires restart"),
            "the descriptor monitor starts once: {:?}",
            warnings[0]
        );
    }

    #[test]
    fn unchanged_process_limits_do_not_warn() {
        let config = config_with_runtime_line("max_open_files: 4096");
        let warnings = capture_warnings(|| detect_process_limit_changes(&config, &config));
        assert!(warnings.is_empty(), "nothing changed: {warnings:?}");
    }

    #[test]
    fn circuit_breaker_unchanged_no_warning() {
        let config = config_with_circuit_breaker(Some(5));
        let warnings = capture_warnings(|| detect_subrequest_circuit_breaker_change(&config, &config));
        assert!(warnings.is_empty(), "unchanged config should produce no warnings");
    }

    #[test]
    fn circuit_breaker_both_none_no_warning() {
        let config = valid_config();
        let warnings = capture_warnings(|| detect_subrequest_circuit_breaker_change(&config, &config));
        assert!(warnings.is_empty(), "both-absent should produce no warnings");
    }

    // -------------------------------------------------------------------------
    // Test Utilities
    // -------------------------------------------------------------------------

    /// Minimal valid config for reload tests.
    fn valid_config() -> Config {
        Config::from_yaml(
            r#"
listeners:
  - name: web
    address: "127.0.0.1:8080"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: static_response
        status: 200
"#,
        )
        .unwrap()
    }

    /// Set up live pipelines, registry, and shutdown token for reload tests.
    fn setup_live_pipelines() -> (ListenerPipelines, Config, FilterRegistry, Arc<Mutex<CancellationToken>>) {
        let config = valid_config();
        let registry = FilterRegistry::with_builtins();
        let health_registry: HealthRegistry = Arc::new(HashMap::new());
        let pipelines =
            resolve_pipelines(&config, &registry, &health_registry, &empty_kv_stores(), &test_client()).unwrap();
        let shutdown = Arc::new(Mutex::new(CancellationToken::new()));
        (pipelines, config, registry, shutdown)
    }

    /// Empty KV store registry for tests without KV stores.
    fn empty_kv_stores() -> praxis_core::kv::KvStoreRegistry {
        praxis_core::kv::KvStoreRegistry::new()
    }

    /// Minimal sub-request client for tests. The connector builds a rustls
    /// config, which needs the process-wide crypto provider first (the binary
    /// installs it at startup; a no-op after the first call).
    fn test_client() -> praxis_core::subrequest::SubRequestClient {
        praxis_tls::provider::install();
        praxis_core::subrequest::SubRequestClient::new(praxis_core::subrequest::SubRequestConnector::new(8, None))
    }

    fn config_with_listener_line(line: &str) -> Config {
        Config::from_yaml(&format!(
            "listeners:\n  - name: web\n    address: \"127.0.0.1:8080\"\n{line}    \
             filter_chains: [main]\nfilter_chains:\n  - name: main\n    \
             filters:\n      - filter: static_response\n        status: 200\n"
        ))
        .unwrap()
    }

    fn config_with_runtime_line(line: &str) -> Config {
        Config::from_yaml(&format!(
            "listeners:\n  - name: web\n    address: \"127.0.0.1:8080\"\n    \
             filter_chains: [main]\nruntime:\n  {line}\nfilter_chains:\n  - name: main\n    \
             filters:\n      - filter: static_response\n        status: 200\n"
        ))
        .unwrap()
    }

    fn config_with_circuit_breaker(failures: Option<u32>) -> Config {
        let cb = failures.map_or_else(String::new, |n| {
            format!(
                "runtime:\n  subrequest_circuit_breaker:\n    \
                 consecutive_failures: {n}\n    recovery_window_secs: 30\n"
            )
        });
        Config::from_yaml(&format!(
            "listeners:\n  - name: web\n    address: \"127.0.0.1:8080\"\n    \
             filter_chains: [main]\n{cb}filter_chains:\n  - name: main\n    \
             filters:\n      - filter: static_response\n        status: 200\n"
        ))
        .unwrap()
    }

    fn capture_warnings<F: FnOnce()>(f: F) -> Vec<String> {
        let messages = Arc::new(Mutex::new(Vec::<String>::new()));
        let capture = WarningCapture(Arc::clone(&messages));
        let subscriber = tracing_subscriber::registry().with(capture);
        tracing::subscriber::with_default(subscriber, f);
        std::mem::take(&mut *messages.lock().unwrap())
    }

    struct WarningCapture(Arc<Mutex<Vec<String>>>);

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for WarningCapture {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
            if *event.metadata().level() == tracing::Level::WARN {
                let mut visitor = MessageVisitor(String::new());
                event.record(&mut visitor);
                self.0.lock().unwrap().push(visitor.0);
            }
        }
    }

    struct MessageVisitor(String);

    impl tracing::field::Visit for MessageVisitor {
        fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
            if field.name() == "message" {
                self.0 = format!("{value:?}");
            }
        }
    }
}
