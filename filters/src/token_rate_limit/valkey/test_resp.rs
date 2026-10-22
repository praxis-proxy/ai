// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Strict, hermetic RESP exchanges for backend unit tests.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

use tokio::{
    io::AsyncWriteExt as _,
    task::{JoinHandle, JoinSet},
};

use super::connection::tests::read_resp_command;

/// A controlled datastore reply or transport fault.
pub(super) enum Reply {
    /// A complete RESP response.
    Bytes(Vec<u8>),
    /// Drop this socket without acknowledging the command.
    Disconnect,
    /// Leave the command pending until the caller's deadline expires.
    Stall,
}

/// One exact command and its independently specified response.
pub(super) struct Step {
    expected: Vec<u8>,
    reply: Reply,
}

impl Step {
    /// Own the expected wire command at the fixture boundary.
    pub(super) fn new(command: &redis::Cmd, reply: Reply) -> Self {
        Self {
            expected: command.get_packed_command(),
            reply,
        }
    }

    /// Reply with a complete RESP frame.
    pub(super) fn reply(command: &redis::Cmd, reply: impl Into<Vec<u8>>) -> Self {
        Self::new(command, Reply::Bytes(reply.into()))
    }
}

type Commands = Arc<Mutex<Vec<Vec<Vec<u8>>>>>;
type Steps = Arc<Mutex<VecDeque<Step>>>;

/// Loopback-only server whose entire command transcript must be consumed.
pub(super) struct ScriptedRespServer {
    port: u16,
    steps: Steps,
    commands: Commands,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl ScriptedRespServer {
    /// Start without a real Redis/Valkey process, container, or environment variable.
    pub(super) async fn start(steps: Vec<Step>) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind scripted RESP server");
        let port = listener.local_addr().expect("scripted RESP address").port();
        let steps = Arc::new(Mutex::new(steps.into()));
        let commands = Arc::new(Mutex::new(Vec::new()));
        let (shutdown, shutdown_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(serve(listener, Arc::clone(&steps), Arc::clone(&commands), shutdown_rx));
        Self {
            port,
            steps,
            commands,
            shutdown: Some(shutdown),
            task: Some(task),
        }
    }

    /// Standalone native-client address for this fixture only.
    pub(super) fn url(&self) -> String {
        format!("redis://127.0.0.1:{}", self.port)
    }

    /// Count received commands without copying their captured arguments.
    pub(super) fn count(&self, name: &str) -> usize {
        self.commands
            .lock()
            .expect("captured RESP commands")
            .iter()
            .filter(|command| command.first().is_some_and(|value| value == name.as_bytes()))
            .count()
    }

    /// Join all server tasks and surface transcript failures before returning.
    pub(super) async fn finish(mut self) {
        let _shutdown_result = self.shutdown.take().expect("server shutdown sender").send(());
        self.task
            .take()
            .expect("server task")
            .await
            .expect("RESP transcript must match");
        assert!(
            self.steps.lock().expect("RESP script").is_empty(),
            "every expected RESP exchange must run"
        );
    }
}

impl Drop for ScriptedRespServer {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _shutdown_result = shutdown.send(());
        }
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

async fn serve(
    listener: tokio::net::TcpListener,
    steps: Steps,
    commands: Commands,
    mut shutdown: tokio::sync::oneshot::Receiver<()>,
) {
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => {
                let (stream, _) = accepted.expect("accept scripted RESP connection");
                connections.spawn(serve_connection(stream, Arc::clone(&steps), Arc::clone(&commands)));
            },
            completed = connections.join_next(), if !connections.is_empty() => {
                completed.expect("RESP connection task").expect("RESP transcript must match");
            },
        }
    }
    connections.abort_all();
    while let Some(completed) = connections.join_next().await {
        if let Err(error) = completed {
            assert!(error.is_cancelled(), "RESP connection failed: {error}");
        }
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "strict transcript validation and fault dispatch share one loop"
)]
async fn serve_connection(mut stream: tokio::net::TcpStream, steps: Steps, commands: Commands) {
    while let Some(command) = read_resp_command(&mut stream).await {
        if command.first().is_some_and(|value| value == b"CLIENT") {
            assert_eq!(
                command.get(1).map(Vec::as_slice),
                Some(b"SETINFO".as_slice()),
                "only native client metadata is implicit"
            );
            stream
                .write_all(b"+OK\r\n")
                .await
                .expect("reply to native client metadata");
            continue;
        }
        let step = steps
            .lock()
            .expect("RESP script")
            .pop_front()
            .expect("unexpected extra datastore command");
        assert_eq!(
            redis::Cmd::new().arg(&command).get_packed_command(),
            step.expected,
            "backend command must match the independent transcript"
        );
        commands.lock().expect("captured RESP commands").push(command);
        match step.reply {
            Reply::Bytes(bytes) => stream.write_all(&bytes).await.expect("write scripted RESP response"),
            Reply::Disconnect => return,
            Reply::Stall => std::future::pending::<()>().await,
        }
    }
}

/// Encode a controlled bulk string (or missing value) independently of backend code.
pub(super) fn bulk(value: Option<&str>) -> Vec<u8> {
    value.map_or_else(
        || b"$-1\r\n".to_vec(),
        |value| format!("${}\r\n{value}\r\n", value.len()).into_bytes(),
    )
}

/// Encode a controlled array reply.
pub(super) fn array(values: &[&str]) -> Vec<u8> {
    let mut reply = format!("*{}\r\n", values.len()).into_bytes();
    for value in values {
        reply.extend(bulk(Some(value)));
    }
    reply
}
