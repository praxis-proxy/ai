// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! In-memory, per-model in-flight tracking store.
//!
//! This is the "datastore" the issue refers to: a pipeline extension that
//! holds live counts and summed `max_tokens` per model. The request-metadata
//! filter increments it on request start and decrements it on completion;
//! a future in-flight-requests scorer reads [`InFlightRegistry::snapshot`] to
//! bias candidate selection.
//!
//! # Scope
//!
//! State is **single-process** and lost on restart, matching the existing
//! `SessionAffinity` map in `routing::intelligent_route`. Cross-replica
//! sharing is explicitly out of scope for this first cut; if it is ever
//! required, swap the backing map for a shared backend behind the same API.

use std::sync::Arc;

use dashmap::DashMap;
use tracing::trace;

/// Live counters for a single model.
///
/// `Copy` so a snapshot can be handed to a scorer without touching the map.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModelMetrics {
    /// Requests currently in flight for this model.
    pub in_flight: u64,

    /// Sum of `max_tokens` (or `max_output_tokens`) across in-flight requests.
    pub reserved_tokens: u64,
}

/// Per-model in-flight tracking, shared across every request in the pipeline.
///
/// `Clone` is cheap and shares the backing map: the pipeline inserts a clone
/// into each request's extensions via [`praxis_filter::PipelineExtension`], and
/// every clone reads and mutates the *same* counters through the inner `Arc`.
#[derive(Clone, Default)]
pub struct InFlightRegistry {
    /// Model name -> live counters. `DashMap` gives per-key locking so
    /// concurrent requests for different models never contend.
    inner: Arc<DashMap<String, ModelMetrics>>,
}

impl InFlightRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
        }
    }

    /// Record a request entering the pipeline: `in_flight += 1` and
    /// `reserved_tokens += max_tokens`.
    ///
    /// Returns the `reserved_tokens` delta that was actually added, so the
    /// caller can stash it and subtract exactly that much on completion
    /// (the request body might not have carried a `max_tokens`, in which
    /// case the token delta is zero but the count still moved).
    pub fn on_request_start(&self, model: &str, max_tokens: u64) -> u64 {
        let mut entry = self.inner.entry(model.to_owned()).or_default();

        entry.in_flight = entry.in_flight.saturating_add(1);
        entry.reserved_tokens = entry.reserved_tokens.saturating_add(max_tokens);

        trace!(
            model,
            in_flight = entry.in_flight,
            reserved_tokens = entry.reserved_tokens,
            "inflight: request start"
        );

        max_tokens
    }

    /// Record a request leaving the pipeline: `in_flight -= 1` and
    /// `reserved_tokens -= reserved` — both floored at zero.
    ///
    /// `reserved` must be the value returned by [`on_request_start`] for the
    /// same request so the reservation nets out exactly.
    ///
    /// [`on_request_start`]: Self::on_request_start
    pub fn on_request_complete(&self, model: &str, reserved: u64) {
        let drained = {
            let Some(mut entry) = self.inner.get_mut(model) else {
                trace!(model, "inflight: stray completion ignored");
                return;
            };
            entry.in_flight = entry.in_flight.saturating_sub(1);
            entry.reserved_tokens = entry.reserved_tokens.saturating_sub(reserved);
            trace!(
                model,
                in_flight = entry.in_flight,
                reserved_tokens = entry.reserved_tokens,
                "inflight: request complete"
            );
            entry.in_flight == 0 && entry.reserved_tokens == 0
        };

        if drained {
            self.inner
                .remove_if(model, |_key, m| m.in_flight == 0 && m.reserved_tokens == 0);
        }
    }

    /// Read-only snapshot of one model's counters, for a scorer to consult.
    #[must_use]
    pub fn get(&self, model: &str) -> ModelMetrics {
        self.inner.get(model).map(|e| *e.value()).unwrap_or_default()
    }

    /// Full snapshot of every tracked model. Cloned out so the caller holds
    /// no locks while scoring.
    #[must_use]
    pub fn snapshot(&self) -> Vec<(String, ModelMetrics)> {
        self.inner
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect()
    }
}

impl praxis_filter::PipelineExtension for InFlightRegistry {
    fn prepare(&self, extensions: &mut praxis_filter::RequestExtensions) {
        // Hand each request a clone that shares the same inner Arc<DashMap>.
        extensions.insert(self.clone());
    }
}
