// SPDX-License-Identifier: Apache-2.0
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

#[cfg(feature = "dev")]
mod debug;
#[cfg(feature = "dev")]
mod echo;
#[cfg(feature = "dev")]
mod filter_docs;
mod fips;
#[cfg(feature = "dev")]
mod inference_fixtures;
#[cfg(feature = "dev")]
mod lint_deps;
#[cfg(feature = "dev")]
mod lint_example_tests;
#[cfg(feature = "dev")]
mod lint_markdown_links;
#[cfg(feature = "dev")]
mod lint_separators;
#[cfg(feature = "dev")]
mod make_replay_fixture;
#[cfg(feature = "dev")]
mod openai_conformance;
#[cfg(feature = "dev")]
mod openai_conformance_gate;
#[cfg(feature = "dev")]
mod openresponses_coverage;
#[cfg(feature = "dev")]
mod port;
#[cfg(feature = "dev")]
mod sync_example_readme;
#[cfg(feature = "dev")]
mod sync_inference_readme;
#[cfg(feature = "dev")]
mod sync_responses_readme;

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
    /// Validate inference fixture coverage.
    #[cfg(feature = "dev")]
    CheckInference(inference_fixtures::CheckArgs),

    /// Check the runtime Responses operation registry against
    /// the pinned OpenAI specification.
    #[cfg(feature = "dev")]
    CheckResponsesRegistry,

    /// Check the runtime Chat Completions operation registry against
    /// the pinned OpenAI specification.
    #[cfg(feature = "dev")]
    CheckChatCompletionsRegistry,

    /// Start a quick HTTP test server returning a static
    /// response to every request.
    #[cfg(feature = "dev")]
    Echo(echo::Args),

    /// Run praxis-ai with development settings.
    /// Runs single-threaded by default.
    #[cfg(feature = "dev")]
    Debug(debug::Args),

    /// Check that workspace dependency versions use
    /// three-component semver.
    #[cfg(feature = "dev")]
    LintDeps(lint_deps::Args),

    /// Check that every example config has a corresponding
    /// integration test.
    #[cfg(feature = "dev")]
    LintExampleTests(lint_example_tests::Args),

    /// Check that local Markdown link targets exist.
    #[cfg(feature = "dev")]
    LintMarkdownLinks(lint_markdown_links::Args),

    /// Check that separator comments total exactly 80 columns.
    #[cfg(feature = "dev")]
    LintSeparators(lint_separators::Args),

    /// Import an external provider recording into a two-sided fixture.
    #[cfg(feature = "dev")]
    ImportInference(inference_fixtures::ImportArgs),

    /// Convert a Claude Code or Codex session log into a replay fixture.
    #[cfg(feature = "dev")]
    MakeReplayFixture(make_replay_fixture::Args),

    /// Verify or regenerate the `examples/README.md` table
    /// from YAML config header comments.
    #[cfg(feature = "dev")]
    SyncExampleReadme(sync_example_readme::Args),

    /// Verify or regenerate the inference fixture coverage inventory.
    #[cfg(feature = "dev")]
    SyncInferenceReadme(sync_inference_readme::Args),

    /// Generate per-filter documentation under `docs/filters/`.
    #[cfg(feature = "dev")]
    GenerateFilterDocs(filter_docs::GenerateArgs),

    /// Check that filter doc files are up to date.
    #[cfg(feature = "dev")]
    LintFilterDocs(filter_docs::LintArgs),

    /// Compare registered API areas with OpenAI's `OpenAPI` spec.
    #[cfg(feature = "dev")]
    OpenaiConformance(openai_conformance::Args),

    /// Refresh or verify the pinned complete OpenAI reference.
    #[cfg(feature = "dev")]
    OpenaiConformanceReference(openai_conformance::ReferenceArgs),

    /// Regenerate or verify official Conversation item schemas.
    #[cfg(feature = "dev")]
    OpenaiConversationItemContracts(openai_conformance::ItemContractsArgs),

    /// Enforce or acknowledge failures in a generated conformance report.
    #[cfg(feature = "dev")]
    OpenaiConformanceGate(openai_conformance_gate::Args),

    /// Regenerate or verify the `OpenResponses` translation coverage report
    /// from the triage manifest.
    #[cfg(feature = "dev")]
    OpenresponsesCoverage(openresponses_coverage::Args),

    /// Record a two-sided fixture against a live provider.
    #[cfg(feature = "dev")]
    RecordInference(inference_fixtures::RecordArgs),

    /// Generate the pipeline-overview table in
    /// `apis/src/openai/responses/README.md`.
    #[cfg(feature = "dev")]
    SyncResponsesReadme(sync_responses_readme::Args),

    /// FIPS build tooling: the compliance report, Red Hat image verification
    /// and podman's signature store.
    Fips(fips::Args),
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

/// Dispatch the CLI subcommand to its handler.
fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Fips(args) => fips::run(args),
        #[cfg(feature = "dev")]
        command => run_dev(command),
    }
}

/// Dispatch a development subcommand to its handler.
#[cfg(feature = "dev")]
fn run_dev(command: Command) {
    match command {
        Command::CheckInference(args) => inference_fixtures::run_check(&args),
        Command::CheckResponsesRegistry => openai_conformance::run_responses_registry_check(),
        Command::CheckChatCompletionsRegistry => openai_conformance::run_chat_completions_registry_check(),
        Command::Echo(args) => echo::run(args),
        Command::Debug(args) => debug::run(&args),
        Command::LintDeps(args) => lint_deps::run(args),
        Command::LintExampleTests(args) => lint_example_tests::run(args),
        Command::LintMarkdownLinks(args) => lint_markdown_links::run(args),
        Command::LintSeparators(args) => lint_separators::run(args),
        Command::ImportInference(args) => inference_fixtures::run_import(args),
        Command::MakeReplayFixture(args) => make_replay_fixture::run(args),
        Command::SyncExampleReadme(args) => sync_example_readme::run(&args),
        Command::GenerateFilterDocs(args) => filter_docs::generate(args),
        Command::LintFilterDocs(args) => filter_docs::lint(args),
        Command::OpenaiConformance(args) => openai_conformance::run(&args),
        Command::OpenaiConformanceReference(args) => openai_conformance::run_reference(&args),
        Command::OpenaiConversationItemContracts(args) => openai_conformance::run_item_contracts(&args),
        Command::OpenaiConformanceGate(args) => openai_conformance_gate::run(&args),
        Command::OpenresponsesCoverage(args) => openresponses_coverage::run(&args),
        Command::RecordInference(args) => inference_fixtures::run_record(args),
        Command::SyncInferenceReadme(args) => sync_inference_readme::run(&args),
        Command::SyncResponsesReadme(args) => sync_responses_readme::run(&args),
        Command::Fips(args) => fips::run(args),
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
#[cfg(feature = "dev")]
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
