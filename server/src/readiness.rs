// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Store-provisioning readiness endpoint.
//!
//! A standalone HTTP service on its own port so an orchestrator readiness probe
//! gates traffic on store provisioning. It reads the same readiness watch the
//! provisioning service owns and composes it with cluster readiness: ready only
//! when both are ready, and 503 otherwise so a provisioning failure stops
//! reporting healthy. It cannot be a route on the protocol admin service, which
//! owns its own route set, so it binds its own listener.

#![cfg(any(feature = "store-postgres", feature = "store-sqlite"))]

use async_trait::async_trait;
use http::{Response, StatusCode, header};
use pingora_core::{apps::http_app::ServeHttp, protocols::http::ServerSession};
use praxis_protocol::http::pingora::health::PingoraHealthService;

use crate::store_provision::{StoreReadiness, StoreReadinessHandle};

/// Default listen address for the readiness endpoint.
pub const DEFAULT_READINESS_ADDR: &str = "0.0.0.0:9200";

/// Environment variable that overrides the readiness listen address.
pub const READINESS_ADDR_ENV: &str = "PRAXIS_STORE_READINESS_ADDR";

/// Request path the readiness probe targets.
pub const READINESS_PATH: &str = "/ready";

/// The readiness listen address: the override when set, else the default.
#[must_use]
pub fn readiness_addr() -> String {
    std::env::var(READINESS_ADDR_ENV).unwrap_or_else(|_| DEFAULT_READINESS_ADDR.to_owned())
}

/// Compose the readiness verdict from store and cluster state.
///
/// Ready (200) only when store provisioning is `Ready` and cluster health is
/// ready. Any other store state, or a degraded cluster, is 503.
fn compose_readiness(store: StoreReadiness, cluster_ready: bool) -> (u16, String) {
    let store_label = match store {
        StoreReadiness::Ready => "ready",
        StoreReadiness::Pending => "pending",
        StoreReadiness::Failed => "failed",
    };
    let ready = store == StoreReadiness::Ready && cluster_ready;
    let code = if ready { 200 } else { 503 };
    let status = if ready { "ready" } else { "not_ready" };
    let body = format!(r#"{{"status":"{status}","store":"{store_label}","clusters_ready":{cluster_ready}}}"#);
    (code, body)
}

/// Readiness HTTP service composing store provisioning and cluster health.
pub struct StoreReadinessService {
    /// Store-provisioning readiness, shared with the provisioning service.
    readiness: StoreReadinessHandle,
    /// Shared cluster-health registry, read live so a reload's replacement is
    /// reflected instead of a startup snapshot.
    health: crate::SharedHealthRegistry,
}

impl StoreReadinessService {
    /// Build the service over the store handle and the cluster health registry.
    #[must_use]
    pub fn new(readiness: StoreReadinessHandle, health: crate::SharedHealthRegistry) -> Self {
        Self { readiness, health }
    }

    /// The current readiness verdict.
    fn verdict(&self) -> (u16, String) {
        let current = std::sync::Arc::clone(&self.health.lock().unwrap_or_else(std::sync::PoisonError::into_inner));
        let cluster_ready = PingoraHealthService::new(Some(current), false).ready_response().0 == 200;
        compose_readiness(self.readiness.current(), cluster_ready)
    }
}

#[async_trait]
impl ServeHttp for StoreReadinessService {
    async fn response(&self, http_session: &mut ServerSession) -> Response<Vec<u8>> {
        let (code, body) = if http_session.req_header().uri.path() == READINESS_PATH {
            self.verdict()
        } else {
            (404, r#"{"error":"not found"}"#.to_owned())
        };
        let mut response = Response::new(body.into_bytes());
        *response.status_mut() = StatusCode::from_u16(code).unwrap_or(StatusCode::SERVICE_UNAVAILABLE);
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        response
    }
}

#[cfg(test)]
mod tests {
    use super::{StoreReadiness, compose_readiness};

    #[test]
    fn ready_only_when_store_and_cluster_ready() {
        assert_eq!(compose_readiness(StoreReadiness::Ready, true).0, 200);
    }

    #[test]
    fn not_ready_when_store_not_ready() {
        assert_eq!(compose_readiness(StoreReadiness::Pending, true).0, 503);
        assert_eq!(compose_readiness(StoreReadiness::Failed, true).0, 503);
    }

    #[test]
    fn not_ready_when_cluster_degraded_even_if_store_ready() {
        assert_eq!(compose_readiness(StoreReadiness::Ready, false).0, 503);
    }

    #[test]
    fn body_reports_store_and_cluster_state() {
        let (_, body) = compose_readiness(StoreReadiness::Failed, false);
        assert!(body.contains(r#""store":"failed""#));
        assert!(body.contains(r#""clusters_ready":false"#));
    }
}
