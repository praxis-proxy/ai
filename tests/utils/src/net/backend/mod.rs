// SPDX-License-Identifier: MIT
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
    Backend, ChunkedBackend, RoutedBackend, start_backend, start_backend_v6, start_backend_with_shutdown,
};
pub use specialized::BackendGuard;
pub use websocket::{
    CapturedWsMessage, WsBackendEvent, WsBackendGuard, WsServerAction, start_scripted_websocket_backend,
    start_scripted_websocket_backend_turns,
};
