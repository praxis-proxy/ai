// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask fips runtime-probe`: drive Praxis AI's listener probes
//! against the shipped image on a FIPS host.

mod runtime_probe;

use clap::{Parser, Subcommand};

/// The canonical FIPS feature set, built `--no-default-features`.
///
/// This is the single Rust-side source of truth for the FIPS profile; it must
/// stay in sync with `FIPS_FEATURES` in the `Makefile`. The runtime probe
/// references this constant rather than re-spelling the literal.
pub(crate) const FIPS_FEATURES: &str = "aws-sigv4-filter";

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask fips`.
#[derive(Parser)]
pub(crate) struct Args {
    /// The FIPS task to run.
    #[command(subcommand)]
    command: Command,
}

/// FIPS tasks.
#[derive(Subcommand)]
enum Command {
    /// Run the shipped FIPS image on this FIPS host and drive the listener
    /// probes against it.
    RuntimeProbe(runtime_probe::Args),
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Dispatch a FIPS task.
pub(crate) fn run(args: Args) {
    match args.command {
        Command::RuntimeProbe(args) => runtime_probe::run(&args),
    }
}
