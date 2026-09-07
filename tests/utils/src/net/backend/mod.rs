// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2024 Praxis Contributors

//! HTTP backends for integration testing.

mod echo;
mod simple;
mod specialized;
mod websocket;

pub use echo::{
    CapturingBackendGuard, start_capturing_backend, start_echo_backend, start_header_echo_backend,
    start_uri_echo_backend,
};
pub use simple::{
    Backend, CapturedRequest, ChunkedBackend, RoutedBackend, StatefulBackend, StatefulCapturingBackend,
    StatefulCapturingGuard, start_backend, start_backend_v6, start_backend_with_shutdown,
};
pub use specialized::{BackendGuard, StatefulBackendGuard, start_stateful_backend};
pub use websocket::{
    CapturedWsMessage, WsBackendEvent, WsBackendGuard, WsServerAction, start_scripted_websocket_backend,
    start_scripted_websocket_backend_turns,
};
