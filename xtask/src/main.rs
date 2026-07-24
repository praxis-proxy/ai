// SPDX-License-Identifier: MIT
// Copyright (c) 2024 Praxis Contributors

//! Development tasks for Praxis AI.

#![allow(
    clippy::exit,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unused_result_ok,
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "development tooling"
)]
#![allow(let_underscore_drop, reason = "development tooling")]

mod debug;
mod echo;
mod filter_docs;
mod lint_deps;
mod lint_example_tests;
mod lint_separators;
mod make_replay_fixture;
mod openai_conformance;
mod port;
mod sync_example_readme;

use clap::{Parser, Subcommand};

// -----------------------------------------------------------------------------
// CLI Definition
// -----------------------------------------------------------------------------

/// Top-level CLI for xtask development commands.
#[derive(Parser)]
#[command(name = "xtask", about = "Praxis AI development tasks")]
struct Cli {
    /// The subcommand to run.
    #[command(subcommand)]
    command: Command,
}

/// Available xtask subcommands.
#[derive(Subcommand)]
enum Command {
    /// Start a quick HTTP test server returning a static
    /// response to every request.
    Echo(echo::Args),

    /// Run praxis-ai with development settings.
    /// Runs single-threaded by default.
    Debug(debug::Args),

    /// Check that workspace dependency versions use
    /// three-component semver.
    LintDeps(lint_deps::Args),

    /// Check that every example config has a corresponding
    /// integration test.
    LintExampleTests(lint_example_tests::Args),

    /// Check that separator comments total exactly 80 columns.
    LintSeparators(lint_separators::Args),

    /// Convert a Claude Code or Codex session log into a replay fixture.
    MakeReplayFixture(make_replay_fixture::Args),

    /// Verify or regenerate the `examples/README.md` table
    /// from YAML config header comments.
    SyncExampleReadme(sync_example_readme::Args),

    /// Generate per-filter documentation under `docs/filters/`.
    GenerateFilterDocs(filter_docs::GenerateArgs),

    /// Check that filter doc files are up to date.
    LintFilterDocs(filter_docs::LintArgs),

    /// Compare Conversations coverage with OpenAI's `OpenAPI` spec.
    OpenaiConformance(openai_conformance::Args),

    /// Refresh or verify the pinned complete OpenAI reference.
    OpenaiConformanceReference(openai_conformance::ReferenceArgs),
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

/// Dispatch the CLI subcommand to its handler.
fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Echo(args) => echo::run(args),
        Command::Debug(args) => debug::run(&args),
        Command::LintDeps(args) => lint_deps::run(args),
        Command::LintExampleTests(args) => lint_example_tests::run(args),
        Command::LintSeparators(args) => lint_separators::run(args),
        Command::MakeReplayFixture(args) => make_replay_fixture::run(args),
        Command::SyncExampleReadme(args) => sync_example_readme::run(&args),
        Command::GenerateFilterDocs(args) => filter_docs::generate(args),
        Command::LintFilterDocs(args) => filter_docs::lint(args),
        Command::OpenaiConformance(args) => openai_conformance::run(&args),
        Command::OpenaiConformanceReference(args) => openai_conformance::run_reference(&args),
    }
}

// -----------------------------------------------------------------------------
// Tracing Setup
// -----------------------------------------------------------------------------

/// Initialize tracing with the given default level.
///
/// Respects `RUST_LOG` if set, otherwise falls back to
/// `default_level`. Set `PRAXIS_LOG_FORMAT=json` for
/// structured JSON output.
pub(crate) fn init_tracing(default_level: &str) {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_level));

    let json = std::env::var("PRAXIS_LOG_FORMAT").is_ok_and(|v| v.eq_ignore_ascii_case("json"));

    if json {
        tracing_subscriber::fmt().json().with_env_filter(env_filter).init();
    } else {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    }
}
