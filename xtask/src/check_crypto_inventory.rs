// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask check-crypto-inventory` — verify the cryptographic inventory
//! manifest against the runtime dependency graphs of the shipped profiles, and
//! generate the prose companion from it.
//!
//! The manifest (`docs/architecture/cryptographic-inventory.yaml`) is the
//! authoritative inventory; the prose companion
//! (`docs/architecture/cryptographic-inventory.md`) is generated from it by this
//! task (`--fix`). The AI proxy ships two build profiles and both are tracked:
//!
//! * `full` — the published container image (`PRAXIS_AI_FEATURES ?= full`);
//! * `fips` — the reduced FIPS runtime, built `--no-default-features` with the feature set shared from
//!   [`crate::fips::FIPS_FEATURES`].
//!
//! This check is a **maintained map of the production cryptographic operations
//! and a thin drift alarm** — deliberately *not* a second implementation of
//! Cargo's dependency resolver. For each profile it resolves
//! `cargo tree -p praxis-ai-proxy --edges normal --no-default-features
//! --features <profile>` once, on the host target, and cross-checks the manifest
//! against the resolved graphs. It fails when:
//!
//! * a crypto-family-named crate (matched by the manifest's `watch_tokens`, or named exactly on the FIPS denylist
//!   [`crate::fips::graph::DENIED`]) enters a profile's runtime graph without being referenced by an operation or
//!   listed in `allow` — the "undeclared crypto crate fails CI" property, enforced on *both* profiles so a denylisted
//!   cipher whose concatenated name the token split cannot reduce (e.g. `chacha20poly1305`, `ctr`, `cbc`) is still
//!   caught in the published `full` image, not only in `fips`;
//! * an `allow` ("reviewed non-primitive") entry names a FIPS-denylisted crate (a denied name is a real primitive by
//!   definition, so it must be an operation, not silenced), duplicates another classification, or no longer resolves
//!   (dead documentation);
//! * a pinned provider drifts its version, gains a forbidden feature, or loses a required one while the crate name
//!   stays put (e.g. `aws-lc-rs` gaining `fips`, `rustls` regaining `ring`, or `openssl` gaining `vendored`); unlisted
//!   features are permitted by design, since feature sets grow across patch releases and an exact-set pin would churn
//!   on benign additions;
//! * an operation is incompletely specified, references a crate absent from a profile it declares, or carries a
//!   non-compliant disposition without a remediation reference;
//! * a FIPS-required operation (the FIPS runtime's load-bearing crypto — [`FIPS_REQUIRED_OPERATIONS`]) is missing or no
//!   longer declares the `fips` profile;
//! * a crate on Red Hat's FIPS denylist enters the `fips` runtime graph.
//!
//! # Known limitations
//!
//! * **Host-target resolution.** The graphs come from the host target, not all three tier-1 targets. The published
//!   image (`full`) is a superset of the FIPS runtime (`fips ⊆ full`), and the crypto-relevant crate set is
//!   host-independent *except* the OS trust-store backends ([`PLATFORM_ONLY`]) — those are exempt from the
//!   operation-resolution and drift checks so a Linux CI host validates the macOS/Windows declarations without
//!   re-resolving those targets.
//! * **Name tripwire, not a totalizing gate.** A genuinely novel crypto crate whose name matches neither a watch token
//!   nor the denylist is caught instead at Cargo.lock / `cargo deny` review, where new dependencies land (widen
//!   `watch_tokens` when one appears).
//! * **Resolver-trusted features.** Provider features are validated against the set `cargo tree` reports — Cargo's own
//!   already-unified resolution — rather than independently re-derived across every activation path.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    path::{Path, PathBuf},
    process::Command,
};

use clap::Parser;
use serde::Deserialize;

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask check-crypto-inventory`.
#[derive(Parser)]
pub(crate) struct Args {
    /// Regenerate the prose companion from the manifest instead of only checking
    /// that it is in sync.
    #[arg(long)]
    fix: bool,
}

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Relative path from the workspace root to the manifest.
const MANIFEST_REL: &str = "docs/architecture/cryptographic-inventory.yaml";

/// Relative path to the generated prose companion.
const DOC_REL: &str = "docs/architecture/cryptographic-inventory.md";

/// Basename of the manifest, used for the same-directory link in the prose.
const MANIFEST_BASENAME: &str = "cryptographic-inventory.yaml";

/// OS trust-store backend crates that resolve on only one target: `schannel`
/// (Windows), `security-framework`/`security-framework-sys` (macOS). They are
/// referenced by operations (cert verification, native-tls store) but the host
/// target may not resolve them, so they are exempt from the
/// operation-resolution and drift checks — keeping the host-target check
/// host-independent.
const PLATFORM_ONLY: &[&str] = &["schannel", "security-framework", "security-framework-sys"];

/// The FIPS runtime's load-bearing cryptographic operations: they MUST exist and
/// MUST declare the `fips` profile. This is the real FIPS-capability assertion,
/// grounded in the FIPS feature set ([`crate::fips::FIPS_FEATURES`] =
/// `openai-responses,openai-file-resolve-filter,aws-sigv4-filter,store-postgres-cert-auth`):
/// `T1` is the process TLS provider,
/// `A3`/`A4` are the AWS `SigV4` HMAC/SHA-256 operations enabled by
/// `aws-sigv4-filter`, and `A12` is the OpenSSL-backed HTTP Basic Auth digest
/// compiled into both profiles. If any loses its `fips` declaration the inventory
/// no longer describes the shipped FIPS build.
const FIPS_REQUIRED_OPERATIONS: &[&str] = &["T1", "A3", "A4", "A12"];

/// The shipped build profiles the inventory tracks, in display order.
const PROFILES: [Profile; 2] = [
    Profile {
        label: "full",
        features: "full",
        description: "Published container image (`standard` + `openai-all` + `store-postgres`).",
    },
    Profile {
        label: "fips",
        features: crate::fips::FIPS_FEATURES,
        description: "Reduced FIPS runtime, built `--no-default-features`.",
    },
];

/// Valid profile labels (the `label` of every [`PROFILES`] entry). Kept as a
/// standalone array so error messages can print the expected set; a unit test
/// asserts it stays in step with [`PROFILES`].
const PROFILE_LABELS: [&str; 2] = ["full", "fips"];

/// Dispositions every operation must declare (kept in sync with the generated
/// prose legend, which is rendered from this list).
const VALID_DISPOSITIONS: [&str; 5] = [
    "validated",
    "non-validated",
    "needs-remediation",
    "upstream-owned",
    "n-a",
];

/// The dispositions that are non-compliant in a FIPS inventory: a real primitive
/// whose validated execution is not proven (`non-validated`), or one that should
/// change (`needs-remediation`). Operations carrying either MUST cite a
/// remediation reference. `validated`/`n-a`/`upstream-owned` are not required to
/// (compliant, non-primitive, or owned and tracked elsewhere).
const NON_COMPLIANT_DISPOSITIONS: [&str; 2] = ["non-validated", "needs-remediation"];

/// A shipped build profile: a label, its `--features` argument (always built
/// `--no-default-features`), and a one-line description rendered into the prose.
struct Profile {
    /// Manifest/label token (`full` / `fips`).
    label: &'static str,
    /// The `--features` argument value.
    features: &'static str,
    /// One-line description for the generated companion document.
    description: &'static str,
}

/// Human-readable meaning of each disposition, rendered into the prose legend.
fn disposition_doc(disposition: &str) -> &'static str {
    match disposition {
        "validated" => "Executes the intended FIPS-validated module with FIPS mode proven in effect.",
        "non-validated" => "A real cryptographic primitive whose validated execution is not proven in this build.",
        "needs-remediation" => "A primitive that should change (e.g. move onto the OpenSSL-backed provider).",
        "upstream-owned" => "A primitive owned and executed entirely within an upstream dependency.",
        "n-a" => "Not a security-function primitive (identifier hashing, key zeroization, or a support layer).",
        _ => "",
    }
}

// -----------------------------------------------------------------------------
// Manifest
// -----------------------------------------------------------------------------

/// The parsed cryptographic inventory manifest.
///
/// `deny_unknown_fields` makes a mistyped key a hard parse error rather than a
/// silently dropped section (e.g. `provders:` losing the entire drift guard).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    /// Name tokens that mark a crate as crypto-relevant.
    watch_tokens: Vec<String>,
    /// Crypto-relevant crates that are reviewed non-primitives: present in the
    /// runtime graph and crypto-family-named, but not standalone operations
    /// (support layers, code generators, FFI companions). Listing them here keeps
    /// the drift alarm quiet without hiding a real primitive.
    allow: Vec<AllowEntry>,
    /// Load-bearing providers whose version and features are pinned.
    #[serde(default)]
    providers: Vec<Provider>,
    /// Production-reachable cryptographic operations — the heart of the inventory
    /// (#1220): where cryptography is used, for what algorithm, under whose
    /// ownership, and what remediation is required.
    #[serde(default)]
    operations: Vec<Operation>,
}

/// One reviewed non-primitive crate.
///
/// `deny_unknown_fields` rejects mistyped keys instead of silently defaulting.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AllowEntry {
    /// The crate name as it appears in `cargo tree`.
    #[serde(rename = "crate")]
    crate_name: String,
    /// Human-readable rationale. Validated non-empty so no allow-listed crate is
    /// left undocumented.
    #[serde(default)]
    note: String,
}

/// A provider crate whose version and feature set are pinned so the
/// provider-selection findings cannot rot while the crate name stays put.
///
/// `deny_unknown_fields` is load-bearing here: a mistyped `forbid_feature:`
/// would otherwise be dropped, leaving an empty constraint that silently accepts
/// a forbidden feature (e.g. `fips`) and defeats the drift guard.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Provider {
    /// The provider crate name.
    #[serde(rename = "crate")]
    crate_name: String,
    /// The exact version expected in the lockfile.
    version: String,
    /// The build profiles this provider is pinned in (`full`/`fips`).
    #[serde(default)]
    profiles: Vec<String>,
    /// Features that must all be enabled.
    #[serde(default)]
    require_features: Vec<String>,
    /// Features that must all be absent.
    #[serde(default)]
    forbid_features: Vec<String>,
    /// Human-readable rationale for the pin. Validated non-empty.
    #[serde(default)]
    note: String,
}

/// One production-reachable cryptographic operation: a caller invoking a
/// primitive through a specific implementation provider.
///
/// `deny_unknown_fields` makes a mistyped key a hard parse error rather than a
/// silently dropped field.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Operation {
    /// Stable operation identifier (e.g. `A3`, `H1`, `T1`). Must be unique.
    id: String,
    /// Short human-readable operation name.
    name: String,
    /// The component/repository responsible for the operation and its remediation
    /// (e.g. `praxis-ai (this repo)`, `praxis-core tls`). Distinct from `caller`
    /// (where it is invoked) and `provider` (what executes the primitive).
    owner: String,
    /// Who invokes the operation, ideally with a `file:line` anchor.
    caller: String,
    /// The algorithm(s) the operation executes.
    algorithm: String,
    /// Human-readable label for the implementation provider (e.g.
    /// `OpenSSL EVP via apis::hash`, `aws-lc-rs via jsonwebtoken`).
    provider: String,
    /// The crate names that implement or support this operation. Each is
    /// validated to resolve in every profile the operation declares (except the
    /// [`PLATFORM_ONLY`] OS trust-store backends, which the host may not resolve).
    crates: Vec<String>,
    /// The build profiles the operation executes in (`full`/`fips`).
    profiles: Vec<String>,
    /// The operation's disposition ([`VALID_DISPOSITIONS`]).
    disposition: String,
    /// Remediation reference. Required (non-empty and pointing at an issue, docs
    /// path, or upstream repo — see [`looks_like_reference`]) when the disposition
    /// is non-compliant ([`NON_COMPLIANT_DISPOSITIONS`]).
    #[serde(default)]
    remediation: String,
    /// Human-readable rationale/context. Validated non-empty.
    #[serde(default)]
    note: String,
}

// -----------------------------------------------------------------------------
// Resolved graph
// -----------------------------------------------------------------------------

/// Version and enabled features of one resolved crate.
struct CrateFacts {
    /// The resolved crate version (without the leading `v`).
    version: String,
    /// The set of features cargo reports enabled for the crate.
    features: BTreeSet<String>,
}

/// One resolved profile runtime graph: crate name -> every resolved version.
///
/// The value is a `Vec` because Cargo can co-resolve several versions of the
/// same crate; keeping them all is what lets the provider-drift check see a
/// second, upgraded copy of a pinned provider instead of silently accepting the
/// first occurrence.
struct ProfileTree {
    /// Profile label this graph was resolved for (`full`/`fips`).
    profile: &'static str,
    /// Resolved crate facts keyed by crate name (one entry per resolved version).
    facts: BTreeMap<String, Vec<CrateFacts>>,
}

impl ProfileTree {
    /// Whether the crate appears anywhere in this profile's resolved tree.
    fn contains(&self, name: &str) -> bool {
        self.facts.contains_key(name)
    }
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Verify the manifest against the resolved runtime graphs, then check (or, with
/// `--fix`, regenerate) the prose companion.
#[expect(
    clippy::too_many_lines,
    reason = "one linear load -> resolve -> evaluate -> report pass"
)]
pub(crate) fn run(args: &Args) {
    let root = workspace_root();
    let manifest_path = root.join(MANIFEST_REL);
    let doc_path = root.join(DOC_REL);

    let raw = std::fs::read_to_string(&manifest_path).unwrap_or_else(|err| {
        eprintln!("failed to read {}: {err}", manifest_path.display());
        std::process::exit(1);
    });
    let manifest: Manifest = serde_yaml::from_str(&raw).unwrap_or_else(|err| {
        eprintln!("failed to parse {MANIFEST_REL}: {err}");
        std::process::exit(1);
    });

    let trees = resolve_trees(&root);

    let violations = evaluate(&manifest, &trees);
    if !violations.is_empty() {
        eprintln!("crypto inventory check failed ({} violation(s)):", violations.len());
        for violation in &violations {
            eprintln!("  - {violation}");
        }
        eprintln!("\nReview {DOC_REL} and update {MANIFEST_REL} to match the change.");
        std::process::exit(1);
    }

    // The manifest is authoritative and internally consistent; the prose is
    // generated from it, never edited by hand.
    let content = render_doc(&manifest);
    if args.fix {
        std::fs::write(&doc_path, &content).unwrap_or_else(|err| {
            eprintln!("failed to write {}: {err}", doc_path.display());
            std::process::exit(1);
        });
        println!("wrote {DOC_REL}");
    } else {
        let current = std::fs::read_to_string(&doc_path).unwrap_or_default();
        if current == content {
            print_summary(&manifest, &trees);
        } else {
            eprintln!("{DOC_REL} is stale");
            eprintln!("\nrun: cargo xtask check-crypto-inventory --fix");
            std::process::exit(1);
        }
    }
}

/// Resolve every shipped profile into a [`ProfileTree`] on the host target.
fn resolve_trees(root: &Path) -> Vec<ProfileTree> {
    PROFILES.iter().map(|p| resolve_profile(root, p)).collect()
}

/// Print the success summary: manifest counts plus the resolved profile set.
fn print_summary(manifest: &Manifest, trees: &[ProfileTree]) {
    let union: BTreeSet<&str> = trees.iter().flat_map(|t| t.facts.keys().map(String::as_str)).collect();
    let profiles: Vec<&str> = PROFILES.iter().map(|p| p.label).collect();
    println!(
        "crypto inventory in sync ({} operations, {} providers, {} allow; profiles: {}; {} crates across graphs)",
        manifest.operations.len(),
        manifest.providers.len(),
        manifest.allow.len(),
        profiles.join("+"),
        union.len(),
    );
}

// -----------------------------------------------------------------------------
// Evaluation
// -----------------------------------------------------------------------------

/// Compare the manifest against the resolved per-profile `trees`, returning a
/// human-readable violation for each mismatch. Pure: all IO happens in [`run`].
fn evaluate(manifest: &Manifest, trees: &[ProfileTree]) -> Vec<String> {
    let mut out = Vec::new();
    check_notes(manifest, &mut out);
    check_allow(manifest, trees, &mut out);
    check_providers(&manifest.providers, trees, &mut out);
    check_operations(manifest, trees, &mut out);
    check_required_operations(manifest, &mut out);
    check_drift(manifest, trees, &mut out);
    check_denied_fips(trees, &mut out);
    out
}

/// Documentation check: every allow entry and pinned provider must carry a
/// non-empty `note` (operation notes are validated in [`check_operations`]).
/// Reading `note` here also anchors the field that `deny_unknown_fields` requires
/// be declared to accept the manifest's `note:`.
fn check_notes(manifest: &Manifest, out: &mut Vec<String>) {
    for entry in &manifest.allow {
        if entry.note.trim().is_empty() {
            out.push(format!(
                "`allow` entry `{}` has an empty `note` (document why it is a reviewed non-primitive)",
                entry.crate_name
            ));
        }
    }
    for provider in &manifest.providers {
        if provider.note.trim().is_empty() {
            out.push(format!("pinned provider `{}` has an empty `note`", provider.crate_name));
        }
    }
}

/// `allow` bucket checks: an allow entry must be a unique, non-denylisted,
/// still-resolving reviewed non-primitive, and must not double as an operation
/// crate.
///
/// * **denylist** — a crate on [`crate::fips::graph::DENIED`] is a real primitive by definition, so it can never be a
///   "reviewed non-primitive"; it must be an operation carrying a disposition. Without this guard `allow` would be an
///   escape hatch that silences [`check_drift`] for a denied cipher shipping in `full`.
/// * **double classification** — a crate both allow-listed and referenced by an operation has two classifications, one
///   masking the other.
/// * **stale** — an allow entry absent from every resolved graph is dead documentation. [`PLATFORM_ONLY`] backends are
///   exempt (the host may not resolve them).
fn check_allow(manifest: &Manifest, trees: &[ProfileTree], out: &mut Vec<String>) {
    let denied: BTreeSet<&str> = crate::fips::graph::DENIED.iter().copied().collect();
    let op_crates: BTreeSet<&str> = operation_crates(manifest);
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for entry in &manifest.allow {
        let name = entry.crate_name.as_str();
        if !seen.insert(name) {
            out.push(format!("duplicate `allow` entry: `{name}`"));
        }
        if denied.contains(name) {
            out.push(format!(
                "FIPS denylist crate `{name}` is declared in `allow` — a denylisted primitive is never a reviewed non-primitive; track it as an operation with a disposition"
            ));
        }
        if op_crates.contains(name) {
            out.push(format!(
                "`{name}` is in `allow` and also referenced by an operation — a crate has exactly one classification"
            ));
        }
        if !PLATFORM_ONLY.contains(&name) && !trees.iter().any(|t| t.contains(name)) {
            out.push(format!(
                "stale `allow` entry: `{name}` is absent from every resolved profile graph"
            ));
        }
    }
}

/// The union of every crate referenced by any operation.
fn operation_crates(manifest: &Manifest) -> BTreeSet<&str> {
    manifest
        .operations
        .iter()
        .flat_map(|op| op.crates.iter().map(String::as_str))
        .collect()
}

/// Provider-drift check across all pinned providers.
fn check_providers(providers: &[Provider], trees: &[ProfileTree], out: &mut Vec<String>) {
    let valid: BTreeSet<&str> = PROFILE_LABELS.iter().copied().collect();
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for provider in providers {
        if !seen.insert(provider.crate_name.as_str()) {
            out.push(format!("duplicate pinned provider: `{}`", provider.crate_name));
        }
        if provider.profiles.is_empty() {
            out.push(format!(
                "pinned provider `{}` declares no `profiles` (expected a non-empty subset of {PROFILE_LABELS:?})",
                provider.crate_name
            ));
        }
        for profile in &provider.profiles {
            if !valid.contains(profile.as_str()) {
                out.push(format!(
                    "pinned provider `{}` has unknown profile `{profile}` (expected one of {PROFILE_LABELS:?})",
                    provider.crate_name
                ));
                continue;
            }
            check_provider_in_profile(provider, profile, trees, out);
        }
    }
}

/// Verify one pinned provider in one declared profile: it must be present, and
/// every resolved copy must match the pinned version (and, on the pinned copy,
/// the required/forbidden features).
fn check_provider_in_profile(provider: &Provider, profile: &str, trees: &[ProfileTree], out: &mut Vec<String>) {
    let Some(tree) = trees.iter().find(|t| t.profile == profile) else {
        return;
    };
    let Some(instances) = tree.facts.get(&provider.crate_name) else {
        out.push(format!(
            "pinned provider `{}` is absent from the {profile} profile runtime graph",
            provider.crate_name
        ));
        return;
    };
    for facts in instances {
        if facts.version == provider.version {
            check_provider_features(provider, facts, profile, out);
        } else {
            out.push(format!(
                "provider drift: `{}` is `{}` in the {profile} graph but the manifest pins `{}`",
                provider.crate_name, facts.version, provider.version
            ));
        }
    }
}

/// Verify a provider's required features are present and forbidden ones absent in
/// one graph's resolved feature set. Unlisted (extra) features are permitted by
/// design: feature sets legitimately grow across patch releases, so exact-set
/// pinning would churn on benign additions. A provider with no constraints (e.g.
/// `native-tls`) pins only its version.
fn check_provider_features(provider: &Provider, facts: &CrateFacts, profile: &str, out: &mut Vec<String>) {
    for want in &provider.require_features {
        if !facts.features.contains(want) {
            out.push(format!(
                "provider drift: `{}` is missing required feature `{want}` in the {profile} graph",
                provider.crate_name
            ));
        }
    }
    for deny in &provider.forbid_features {
        if facts.features.contains(deny) {
            out.push(format!(
                "provider drift: `{}` has forbidden feature `{deny}` enabled in the {profile} graph",
                provider.crate_name
            ));
        }
    }
}

/// Operations check: validate the operation-level inventory (#1220).
///
/// Enforces:
/// 0. **non-empty inventory** — at least one operation must be declared (the inventory is a maintained map and drift
///    alarm, not an opt-out);
/// 1. **unique ids and complete fields** — every operation has a unique `id` and non-empty
///    `name`/`owner`/`caller`/`algorithm`/`provider`/`note`, a non-empty `crates` list (unless `n-a`), a non-empty
///    `profiles` list over `{full, fips}`, and a `disposition` in [`VALID_DISPOSITIONS`];
/// 2. **crate resolution** — every crate an operation references (except the [`PLATFORM_ONLY`] OS trust-store backends)
///    resolves in every profile the operation declares;
/// 3. **remediation links** — an operation whose disposition is non-compliant ([`NON_COMPLIANT_DISPOSITIONS`]) cites an
///    actionable remediation reference ([`looks_like_reference`]).
#[expect(
    clippy::too_many_lines,
    reason = "one linear validation pass per operation invariant"
)]
fn check_operations(manifest: &Manifest, trees: &[ProfileTree], out: &mut Vec<String>) {
    if manifest.operations.is_empty() {
        out.push(
            "operations inventory is empty: at least one cryptographic operation must be declared (the inventory is a maintained map and drift alarm, not an opt-out)".to_owned(),
        );
        return;
    }

    let valid_disp: BTreeSet<&str> = VALID_DISPOSITIONS.iter().copied().collect();
    let valid_profiles: BTreeSet<&str> = PROFILE_LABELS.iter().copied().collect();
    let mut seen_ids: BTreeSet<&str> = BTreeSet::new();

    for op in &manifest.operations {
        let id = op.id.trim();
        let label = if id.is_empty() { "<empty id>" } else { id };
        if id.is_empty() {
            out.push("operation has an empty `id`".to_owned());
        } else if !seen_ids.insert(id) {
            out.push(format!("operation id `{id}` is declared more than once"));
        }
        for (field, value) in [
            ("name", &op.name),
            ("owner", &op.owner),
            ("caller", &op.caller),
            ("algorithm", &op.algorithm),
            ("provider", &op.provider),
            ("note", &op.note),
        ] {
            if value.trim().is_empty() {
                out.push(format!("operation `{label}` has an empty `{field}`"));
            }
        }
        if !valid_disp.contains(op.disposition.as_str()) {
            out.push(format!(
                "operation `{label}` has missing/invalid disposition `{}` (expected one of {VALID_DISPOSITIONS:?})",
                op.disposition
            ));
        }
        if op.profiles.is_empty() {
            out.push(format!("operation `{label}` declares no `profiles`"));
        }
        for p in &op.profiles {
            if !valid_profiles.contains(p.as_str()) {
                out.push(format!(
                    "operation `{label}` declares invalid profile `{p}` (expected one of {PROFILE_LABELS:?})"
                ));
            }
        }
        // A crate-backed operation must name its crates. An `n-a` operation may
        // reference none: it documents non-cryptographic hashing on the standard
        // library (reviewed and dispositioned out of scope), which has no crate.
        if op.crates.is_empty() && op.disposition != "n-a" {
            out.push(format!("operation `{label}` references no `crates`"));
        }
        check_operation_crates_resolve(op, label, trees, out);
        if NON_COMPLIANT_DISPOSITIONS.contains(&op.disposition.as_str()) && !looks_like_reference(&op.remediation) {
            out.push(format!(
                "operation `{label}` has disposition `{}` but no remediation reference (cite an issue #NNN, a docs path, or an upstream repo)",
                op.disposition
            ));
        }
    }
}

/// Verify every crate an operation references resolves in every profile it
/// declares. The [`PLATFORM_ONLY`] OS trust-store backends are exempt: they are
/// legitimately referenced (cert verification, native-tls store) but resolve on
/// only one target, which the host may not be.
fn check_operation_crates_resolve(op: &Operation, label: &str, trees: &[ProfileTree], out: &mut Vec<String>) {
    let valid_profiles: BTreeSet<&str> = PROFILE_LABELS.iter().copied().collect();
    for crate_name in &op.crates {
        if PLATFORM_ONLY.contains(&crate_name.as_str()) {
            continue;
        }
        for p in &op.profiles {
            if !valid_profiles.contains(p.as_str()) {
                continue;
            }
            let Some(tree) = trees.iter().find(|t| t.profile == p) else {
                continue;
            };
            if !tree.contains(crate_name) {
                out.push(format!(
                    "operation `{label}` declares profile `{p}` but its crate `{crate_name}` is not resolved in that profile graph"
                ));
            }
        }
    }
}

/// FIPS-capability check: every [`FIPS_REQUIRED_OPERATIONS`] entry must exist and
/// declare the `fips` profile, so the FIPS runtime's load-bearing cryptographic
/// operations stay documented and tracked in the profile they execute in.
fn check_required_operations(manifest: &Manifest, out: &mut Vec<String>) {
    let by_id: BTreeMap<&str, &Operation> = manifest.operations.iter().map(|op| (op.id.trim(), op)).collect();
    for &required in FIPS_REQUIRED_OPERATIONS {
        match by_id.get(required) {
            None => out.push(format!(
                "FIPS-required operation `{required}` is missing from the inventory (the FIPS runtime's load-bearing crypto operations must be documented and declare the `fips` profile)"
            )),
            Some(op) => {
                if !op.profiles.iter().any(|p| p == "fips") {
                    out.push(format!(
                        "FIPS-required operation `{required}` does not declare the `fips` profile (it executes in the FIPS runtime and must be tracked there)"
                    ));
                }
            },
        }
    }
}

/// Drift alarm: a crypto-relevant crate present in a profile's runtime graph that
/// is neither referenced by an operation nor listed in `allow`. A crate is
/// crypto-relevant when it matches a `watch_tokens` entry OR is named exactly on
/// the FIPS denylist [`crate::fips::graph::DENIED`]. The denylist arm closes a
/// hole in the token tripwire: the split-on-`-`/`_`-then-strip-trailing-digits
/// tokenizer cannot reduce a separator-less cipher name (e.g. `chacha20poly1305`)
/// to its `chacha`/`poly1305` tokens, and mode crates like `ctr`/`cbc` have no
/// token at all, so without an exact-name check such a crate could link into the
/// published `full` image undetected ([`check_denied_fips`] only guards `fips`).
///
/// Enforced on both profiles. [`PLATFORM_ONLY`] backends are treated as declared
/// so the host-target check stays host-independent.
fn check_drift(manifest: &Manifest, trees: &[ProfileTree], out: &mut Vec<String>) {
    let watch: BTreeSet<String> = manifest.watch_tokens.iter().map(|t| t.to_ascii_lowercase()).collect();
    let denied: BTreeSet<&str> = crate::fips::graph::DENIED.iter().copied().collect();
    let mut declared: BTreeSet<&str> = operation_crates(manifest);
    declared.extend(manifest.allow.iter().map(|e| e.crate_name.as_str()));
    declared.extend(PLATFORM_ONLY.iter().copied());

    for tree in trees {
        for name in tree.facts.keys() {
            let crypto_relevant = is_watch_match(name, &watch) || denied.contains(name.as_str());
            if crypto_relevant && !declared.contains(name.as_str()) {
                out.push(format!(
                    "undeclared crypto crate in the {} runtime graph: `{name}` is crypto-relevant (matches a watch token or the FIPS denylist) but is referenced by no operation and is not in `allow`",
                    tree.profile
                ));
            }
        }
    }
}

/// FIPS denylist check: no crate on Red Hat's denylist (shared with the FIPS
/// report through [`crate::fips::graph::DENIED`]) may enter the `fips` runtime
/// graph. Defense-in-depth over the operation profile assignments.
fn check_denied_fips(trees: &[ProfileTree], out: &mut Vec<String>) {
    let Some(fips) = trees.iter().find(|t| t.profile == "fips") else {
        return;
    };
    for &denied in crate::fips::graph::DENIED {
        if fips.contains(denied) {
            out.push(format!(
                "FIPS denylist crate `{denied}` is present in the fips runtime graph (must never enter the FIPS build)"
            ));
        }
    }
}

/// Whether `remediation` cites something actionable: an issue (`#<digit>`), a URL
/// (`http`), a docs path (`.md`), or an upstream repo (`praxis-proxy/`). Used to
/// reject a bare non-empty string (e.g. `"TODO"`) as a remediation "link".
fn looks_like_reference(remediation: &str) -> bool {
    remediation.contains("http")
        || remediation.contains(".md")
        || remediation.contains("praxis-proxy/")
        || remediation
            .as_bytes()
            .windows(2)
            .any(|w| matches!(w, [b'#', d] if d.is_ascii_digit()))
}

/// Whether `name` matches any watch token, using the same tokenization the
/// manifest documents: split on `-`/`_`, then also compare each part with its
/// trailing digits stripped (so `sha2` matches `sha`, `md-5` matches `md`).
fn is_watch_match(name: &str, watch: &BTreeSet<String>) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.split(['-', '_']).any(|part| {
        if watch.contains(part) {
            return true;
        }
        let base = part.trim_end_matches(|c: char| c.is_ascii_digit());
        !base.is_empty() && watch.contains(base)
    })
}

// -----------------------------------------------------------------------------
// Prose generation
// -----------------------------------------------------------------------------

/// Render the full generated companion document from the manifest. Pure and
/// deterministic: iterates the manifest in declaration order, so the same
/// manifest always renders byte-for-byte the same document.
#[expect(clippy::too_many_lines, reason = "one linear section-by-section document render")]
fn render_doc(manifest: &Manifest) -> String {
    let mut out = String::new();

    writeln!(out, "<!-- SPDX-License-Identifier: Apache-2.0 -->").unwrap();
    writeln!(out, "<!-- Copyright (c) 2026 Praxis Contributors -->").unwrap();
    writeln!(
        out,
        "<!-- Generated by `cargo xtask check-crypto-inventory --fix`. Do not edit"
    )
    .unwrap();
    writeln!(out, "     by hand — edit {MANIFEST_BASENAME} and regenerate. -->").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "# Cryptographic operations inventory").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "This page is generated from the machine-readable manifest").unwrap();
    writeln!(
        out,
        "[`{MANIFEST_BASENAME}`]({MANIFEST_BASENAME}), which is the authoritative source of"
    )
    .unwrap();
    writeln!(
        out,
        "record for the Praxis AI proxy's production cryptography ([#1220][i1220])."
    )
    .unwrap();
    writeln!(
        out,
        "`cargo xtask check-crypto-inventory` resolves the runtime dependency graph of"
    )
    .unwrap();
    writeln!(
        out,
        "each shipped profile and fails if the manifest and the resolved graphs disagree;"
    )
    .unwrap();
    writeln!(out, "run it with `--fix` to regenerate this page.").unwrap();
    writeln!(out).unwrap();

    render_profiles(&mut out);
    render_dispositions(&mut out);
    render_operations(manifest, &mut out);
    render_providers(manifest, &mut out);
    render_allow(manifest, &mut out);
    render_watch_tokens(manifest, &mut out);
    render_maintenance(&mut out);

    writeln!(out).unwrap();
    writeln!(out, "[i1220]: https://github.com/praxis-proxy/ai/issues/1220").unwrap();

    out
}

/// Render the build-profiles table.
fn render_profiles(out: &mut String) {
    writeln!(out, "## Build profiles").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "The inventory tracks two shipped profiles; every operation and pinned provider is"
    )
    .unwrap();
    writeln!(out, "assigned to the profiles it is resolved in. Both are built").unwrap();
    writeln!(out, "`--no-default-features`.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Profile | Cargo features | Description |").unwrap();
    writeln!(out, "| --- | --- | --- |").unwrap();
    for profile in &PROFILES {
        writeln!(
            out,
            "| `{}` | `{}` | {} |",
            profile.label,
            profile.features,
            cell(profile.description)
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the disposition legend.
fn render_dispositions(out: &mut String) {
    writeln!(out, "## Dispositions").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "Every operation carries one disposition:").unwrap();
    writeln!(out).unwrap();
    for disposition in VALID_DISPOSITIONS {
        writeln!(out, "- `{disposition}` — {}", disposition_doc(disposition)).unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the operations inventory: one row per production-reachable
/// cryptographic operation.
#[expect(
    clippy::too_many_lines,
    reason = "one linear table render with a fixed wide column set"
)]
fn render_operations(manifest: &Manifest, out: &mut String) {
    writeln!(out, "## Cryptographic operations").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Every production-reachable cryptographic operation: where it is used, for what"
    )
    .unwrap();
    writeln!(
        out,
        "algorithm, under whose ownership, by which crates, and what remediation it"
    )
    .unwrap();
    writeln!(
        out,
        "requires. Each operation's crates are cross-checked against the resolved graphs."
    )
    .unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| ID | Operation | Owner | Caller | Algorithm | Provider | Crates | Profiles | Disposition | Remediation | Notes |"
    )
    .unwrap();
    writeln!(
        out,
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    .unwrap();
    for op in &manifest.operations {
        // `n-a` std operations reference no crypto crate; show an em dash there.
        let crates = if op.crates.is_empty() {
            "—".to_owned()
        } else {
            op.crates
                .iter()
                .map(|c| format!("`{c}`"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            cell(&op.id),
            cell(&op.name),
            cell(&op.owner),
            cell(&op.caller),
            cell(&op.algorithm),
            cell(&op.provider),
            cell(&crates),
            op.profiles.join(", "),
            cell(&op.disposition),
            cell(&op.remediation),
            cell(&op.note),
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the pinned-providers table.
fn render_providers(manifest: &Manifest, out: &mut String) {
    writeln!(out, "## Pinned providers").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Providers whose selection is load-bearing: the manifest pins the version and the"
    )
    .unwrap();
    writeln!(out, "security-relevant features so a drift cannot pass unnoticed.").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "| Crate | Profiles | Required features | Forbidden features | Notes |"
    )
    .unwrap();
    writeln!(out, "| --- | --- | --- | --- | --- |").unwrap();
    for provider in &manifest.providers {
        writeln!(
            out,
            "| `{}` | {} | {} | {} | {} |",
            provider.crate_name,
            provider.profiles.join(", "),
            features_cell(&provider.require_features),
            features_cell(&provider.forbid_features),
            cell(&provider.note),
        )
        .unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the reviewed non-primitives (`allow`) table.
fn render_allow(manifest: &Manifest, out: &mut String) {
    writeln!(out, "## Reviewed non-primitives (`allow`)").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Crypto-family-named crates that resolve into a runtime graph but are not"
    )
    .unwrap();
    writeln!(
        out,
        "standalone operations — support layers, code generators, and FFI companions,"
    )
    .unwrap();
    writeln!(out, "each reviewed and documented so the drift alarm stays quiet.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "| Crate | Notes |").unwrap();
    writeln!(out, "| --- | --- |").unwrap();
    for entry in &manifest.allow {
        writeln!(out, "| `{}` | {} |", entry.crate_name, cell(&entry.note)).unwrap();
    }
    writeln!(out).unwrap();
}

/// Render the watch-tokens note.
fn render_watch_tokens(manifest: &Manifest, out: &mut String) {
    writeln!(out, "## Watch tokens").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "The check treats a crate as crypto-relevant when its name matches one of these"
    )
    .unwrap();
    writeln!(out, "tokens (split on `-`/`_`, trailing digits stripped):").unwrap();
    writeln!(out).unwrap();
    let tokens: Vec<String> = manifest.watch_tokens.iter().map(|t| format!("`{t}`")).collect();
    writeln!(out, "{}", tokens.join(", ")).unwrap();
    writeln!(out).unwrap();
}

/// Render the maintenance note.
fn render_maintenance(out: &mut String) {
    writeln!(out, "## Maintenance").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "Edit [`{MANIFEST_BASENAME}`]({MANIFEST_BASENAME}) and run:").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "```console").unwrap();
    writeln!(out, "cargo xtask check-crypto-inventory --fix").unwrap();
    writeln!(out, "```").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "The same check without `--fix` runs in `make lint` and fails on any drift"
    )
    .unwrap();
    writeln!(
        out,
        "between the manifest, this page, and the resolved dependency graphs."
    )
    .unwrap();
}

/// Escape one cell of a Markdown table: collapse newlines to spaces and escape
/// pipes so a `note` cannot break the table layout.
fn cell(text: &str) -> String {
    text.replace('\n', " ").replace('|', "\\|").trim().to_owned()
}

/// Render a feature list into one table cell, or an em dash when empty.
fn features_cell(features: &[String]) -> String {
    if features.is_empty() {
        "—".to_owned()
    } else {
        features.iter().map(|f| format!("`{f}`")).collect::<Vec<_>>().join(", ")
    }
}

// -----------------------------------------------------------------------------
// Cargo Graph
// -----------------------------------------------------------------------------

/// Resolve the runtime (`--edges normal`) dependency graph of one profile on the
/// host target and return its crate facts.
#[expect(clippy::too_many_lines, reason = "one linear cargo-tree invocation and parse")]
fn resolve_profile(root: &Path, profile: &Profile) -> ProfileTree {
    let output = Command::new("cargo")
        .current_dir(root)
        .args([
            "tree",
            "-p",
            "praxis-ai-proxy",
            "--edges",
            "normal",
            "--no-default-features",
            "--features",
            profile.features,
            "--prefix",
            "none",
            "--format",
            "{p}|{f}",
        ])
        .output()
        .expect("failed to run cargo tree");

    if !output.status.success() {
        eprintln!(
            "cargo tree (profile {}) failed:\n{}",
            profile.label,
            String::from_utf8_lossy(&output.stderr)
        );
        std::process::exit(1);
    }

    let text = String::from_utf8(output.stdout).expect("cargo tree output is not UTF-8");
    ProfileTree {
        profile: profile.label,
        facts: parse_facts(&text),
    }
}

/// Parse `cargo tree --prefix none --format "{p}|{f}"` output into crate facts.
///
/// Each line is `name vX.Y.Z [(source)] [(proc-macro)]|feat1,feat2 [(*)]`; the
/// trailing `(*)` marks a subtree cargo already printed. Every distinct resolved
/// version of a crate is retained (the `(*)` repeats of the same version are
/// de-duplicated), so a crate co-resolved at two versions keeps both.
fn parse_facts(text: &str) -> BTreeMap<String, Vec<CrateFacts>> {
    let mut facts: BTreeMap<String, Vec<CrateFacts>> = BTreeMap::new();
    for raw in text.lines() {
        let trimmed = raw.trim().trim_end_matches("(*)").trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let (left, right) = trimmed.split_once('|').unwrap_or((trimmed, ""));
        let mut fields = left.split_whitespace();
        let Some(name) = fields.next() else {
            continue;
        };
        let version = fields.next().unwrap_or("").trim_start_matches('v').to_owned();
        let features: BTreeSet<String> = right
            .split(',')
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .map(str::to_owned)
            .collect();
        let versions = facts.entry(name.to_owned()).or_default();
        if !versions.iter().any(|f| f.version == version) {
            versions.push(CrateFacts { version, features });
        }
    }
    facts
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

/// Find the workspace root by locating the top-level `Cargo.toml`.
fn workspace_root() -> PathBuf {
    let output = Command::new("cargo")
        .args(["locate-project", "--workspace", "--message-format=plain"])
        .output()
        .expect("failed to run cargo locate-project");
    let path = String::from_utf8(output.stdout).expect("non-utf8 path");
    PathBuf::from(path.trim())
        .parent()
        .expect("Cargo.toml has no parent")
        .to_owned()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing, reason = "tests")]
mod tests {
    use super::*;

    /// A profile graph from bare crate names (version `0.0.0`, no features).
    fn names_tree(profile: &'static str, names: &[&str]) -> ProfileTree {
        ProfileTree {
            profile,
            facts: names
                .iter()
                .map(|n| {
                    (
                        (*n).to_owned(),
                        vec![CrateFacts {
                            version: "0.0.0".to_owned(),
                            features: BTreeSet::new(),
                        }],
                    )
                })
                .collect(),
        }
    }

    /// Insert or replace a crate in `tree` with a specific version and features.
    fn set_facts(tree: &mut ProfileTree, name: &str, version: &str, features: &[&str]) {
        tree.facts.insert(
            name.to_owned(),
            vec![CrateFacts {
                version: version.to_owned(),
                features: features.iter().map(|f| (*f).to_owned()).collect(),
            }],
        );
    }

    /// The clean two-profile graph set matching [`sample_manifest`]: both profiles
    /// resolve the same crates, with the pinned `openssl` at its manifest version.
    /// The [`PLATFORM_ONLY`] backends are absent (host may not resolve them).
    fn sample_trees() -> Vec<ProfileTree> {
        let build = |profile| {
            let mut tree = names_tree(
                profile,
                &[
                    "rustls",
                    "rustls-webpki",
                    "subtle",
                    "aws-smithy-types",
                    "openssl-macros",
                    "serde",
                ],
            );
            set_facts(&mut tree, "openssl", "0.10.81", &[]);
            tree
        };
        vec![build("full"), build("fips")]
    }

    /// A sample manifest covering every FIPS-required operation plus a
    /// platform-only cert-verification operation and one pinned provider.
    #[expect(clippy::too_many_lines, reason = "inline YAML fixture is one long string literal")]
    fn sample_manifest() -> Manifest {
        let yaml = r#"
watch_tokens: [sha, md, hmac, rustls, ring, openssl, aws, rand]
allow:
  - crate: aws-smithy-types
    note: AWS SDK support types; no primitive.
  - crate: openssl-macros
    note: OpenSSL derive macros; build-host code generator.
providers:
  - crate: openssl
    version: "0.10.81"
    profiles: [full, fips]
    forbid_features: [vendored]
    note: libcrypto binding; must not vendor.
operations:
  - id: T1
    name: Process rustls crypto-provider install
    owner: praxis core
    caller: server.rs:98
    algorithm: rustls_openssl install_default
    provider: OpenSSL via rustls_openssl
    crates: [rustls, openssl]
    profiles: [full, fips]
    disposition: non-validated
    remediation: docs/fips.md
    note: TLS provider install.
  - id: A3
    name: AWS SigV4 signature
    owner: praxis-ai
    caller: sigv4.rs:293
    algorithm: HMAC-SHA256
    provider: OpenSSL EVP
    crates: [openssl]
    profiles: [full, fips]
    disposition: non-validated
    remediation: docs/fips.md
    note: SigV4 signing key + signature.
  - id: A4
    name: AWS SigV4 payload SHA-256
    owner: praxis-ai
    caller: signing.rs:216
    algorithm: SHA-256
    provider: OpenSSL EVP
    crates: [openssl]
    profiles: [full, fips]
    disposition: non-validated
    remediation: docs/fips.md
    note: SigV4 payload + canonical-request hash.
  - id: A12
    name: HTTP Basic Auth credential verification
    owner: praxis core
    caller: basic_auth/filter.rs:253
    algorithm: unsalted SHA-256 + constant-time compare
    provider: OpenSSL EVP + subtle
    crates: [openssl, subtle]
    profiles: [full, fips]
    disposition: upstream-owned
    note: Basic Auth digest compare.
  - id: C1
    name: Server-cert chain and hostname verification
    owner: praxis core
    caller: pingora connector
    algorithm: X.509 path validation + signature check
    provider: rustls-webpki + platform verifier
    crates: [rustls-webpki, openssl, security-framework, schannel]
    profiles: [full, fips]
    disposition: non-validated
    remediation: docs/fips.md
    note: Cert chain and hostname verification.
"#;
        serde_yaml::from_str(yaml).expect("sample manifest parses")
    }

    #[test]
    fn profile_labels_match_profiles() {
        let labels: Vec<&str> = PROFILES.iter().map(|p| p.label).collect();
        assert_eq!(labels, PROFILE_LABELS, "PROFILE_LABELS must mirror PROFILES");
    }

    #[test]
    fn fips_required_operations_are_valid_ids() {
        // Guards against a typo in the const: every required id must look like an
        // operation id (non-empty, no whitespace).
        for &id in FIPS_REQUIRED_OPERATIONS {
            assert!(
                !id.is_empty() && !id.contains(char::is_whitespace),
                "bad required id `{id}`"
            );
        }
    }

    #[test]
    fn clean_manifest_has_no_violations() {
        let violations = evaluate(&sample_manifest(), &sample_trees());
        assert!(violations.is_empty(), "expected no violations, got: {violations:?}");
    }

    #[test]
    fn tokenizer_matches_digit_suffixes_and_parts() {
        let watch: BTreeSet<String> = ["sha", "md", "rustls", "ring", "aws"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        assert!(is_watch_match("sha2", &watch), "sha2 -> sha");
        assert!(is_watch_match("md-5", &watch), "md-5 -> md");
        assert!(is_watch_match("tokio-rustls", &watch), "part rustls");
        assert!(is_watch_match("ring", &watch), "exact ring");
        assert!(is_watch_match("aws-smithy-types", &watch), "part aws");
        assert!(!is_watch_match("serde_json", &watch), "no crypto token");
        assert!(
            !is_watch_match("string_cache", &watch),
            "ring is not a token of string_cache"
        );
    }

    #[test]
    fn looks_like_reference_accepts_actionable_forms() {
        assert!(looks_like_reference("#1220"), "issue ref");
        assert!(looks_like_reference("docs/fips.md"), "docs path");
        assert!(looks_like_reference("https://example.com"), "url");
        assert!(looks_like_reference("praxis-proxy/ai"), "upstream repo");
        assert!(!looks_like_reference("TODO"), "bare string is not a reference");
        assert!(!looks_like_reference(""), "empty is not a reference");
    }

    #[test]
    fn undeclared_crypto_crate_is_flagged() {
        let mut trees = sample_trees();
        set_facts(&mut trees[0], "hmac", "0.12.0", &[]);
        let violations = evaluate(&sample_manifest(), &trees);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("hmac") && v.contains("undeclared")),
            "hmac is an undeclared crypto crate: {violations:?}"
        );
    }

    #[test]
    fn allow_list_suppresses_undeclared() {
        // openssl-macros matches `openssl` but is on the allow list -> quiet.
        let violations = evaluate(&sample_manifest(), &sample_trees());
        assert!(
            !violations.iter().any(|v| v.contains("openssl-macros")),
            "allow-listed crate must not be flagged: {violations:?}"
        );
    }

    #[test]
    fn denylisted_cipher_without_watch_token_is_flagged() {
        // The token split cannot reduce `chacha20poly1305`/`ctr`/`cbc` to a token,
        // but the exact-name denylist arm of check_drift still catches them in the
        // published full image.
        let watch: BTreeSet<String> = sample_manifest()
            .watch_tokens
            .iter()
            .map(|t| t.to_ascii_lowercase())
            .collect();
        assert!(!is_watch_match("chacha20poly1305", &watch), "precondition: no token");
        assert!(!is_watch_match("ctr", &watch), "precondition: no token");

        let mut trees = sample_trees();
        for name in ["chacha20poly1305", "ctr", "cbc"] {
            set_facts(&mut trees[0], name, "0.1.0", &[]);
        }
        let violations = evaluate(&sample_manifest(), &trees);
        for name in ["chacha20poly1305", "ctr", "cbc"] {
            assert!(
                violations.iter().any(|v| v.contains(name) && v.contains("undeclared")),
                "denylisted cipher `{name}` must be flagged undeclared: {violations:?}"
            );
        }
    }

    #[test]
    fn denylisted_crate_in_allow_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.allow.push(AllowEntry {
            crate_name: "aes-gcm".to_owned(),
            note: "should not be allowed".to_owned(),
        });
        let mut trees = sample_trees();
        set_facts(&mut trees[0], "aes-gcm", "0.10.0", &[]);
        let violations = evaluate(&manifest, &trees);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("aes-gcm") && v.contains("allow") && v.contains("denylist")),
            "a denylisted crate in `allow` must be flagged: {violations:?}"
        );
    }

    #[test]
    fn allow_crate_also_referenced_by_operation_is_flagged() {
        let mut manifest = sample_manifest();
        // `openssl` is an operation crate; also listing it in allow is a double
        // classification.
        manifest.allow.push(AllowEntry {
            crate_name: "openssl".to_owned(),
            note: "double classified".to_owned(),
        });
        let violations = evaluate(&manifest, &sample_trees());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("openssl") && v.contains("exactly one classification")),
            "a double-classified crate must be flagged: {violations:?}"
        );
    }

    #[test]
    fn stale_allow_entry_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.allow.push(AllowEntry {
            crate_name: "ghost-crate".to_owned(),
            note: "no longer resolves".to_owned(),
        });
        let violations = evaluate(&manifest, &sample_trees());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("ghost-crate") && v.contains("stale")),
            "an allow entry absent from every graph must be flagged: {violations:?}"
        );
    }

    #[test]
    fn empty_allow_note_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.allow.push(AllowEntry {
            crate_name: "aws-smithy-async".to_owned(),
            note: "  ".to_owned(),
        });
        let mut trees = sample_trees();
        set_facts(&mut trees[0], "aws-smithy-async", "1.0.0", &[]);
        let violations = evaluate(&manifest, &trees);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("aws-smithy-async") && v.contains("empty `note`")),
            "an empty allow note must be flagged: {violations:?}"
        );
    }

    #[test]
    fn provider_version_drift_is_flagged() {
        let mut trees = sample_trees();
        set_facts(&mut trees[0], "openssl", "0.10.99", &[]);
        let violations = evaluate(&sample_manifest(), &trees);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("openssl") && v.contains("0.10.99") && v.contains("drift")),
            "a provider version drift must be flagged: {violations:?}"
        );
    }

    #[test]
    fn provider_forbidden_feature_is_flagged() {
        let mut trees = sample_trees();
        set_facts(&mut trees[0], "openssl", "0.10.81", &["vendored"]);
        let violations = evaluate(&sample_manifest(), &trees);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("openssl") && v.contains("vendored") && v.contains("forbidden")),
            "a forbidden provider feature must be flagged: {violations:?}"
        );
    }

    #[test]
    fn provider_missing_required_feature_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.providers[0].require_features = vec!["tls12".to_owned()];
        // sample_trees openssl has no features -> missing tls12.
        let violations = evaluate(&manifest, &sample_trees());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("openssl") && v.contains("tls12") && v.contains("missing")),
            "a missing required feature must be flagged: {violations:?}"
        );
    }

    #[test]
    fn provider_absent_from_declared_profile_is_flagged() {
        let mut trees = sample_trees();
        trees[1].facts.remove("openssl"); // drop from fips
        let violations = evaluate(&sample_manifest(), &trees);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("openssl") && v.contains("absent") && v.contains("fips")),
            "a provider absent from a declared profile must be flagged: {violations:?}"
        );
    }

    #[test]
    fn operation_crate_not_resolved_in_profile_is_flagged() {
        let mut trees = sample_trees();
        trees[1].facts.remove("rustls"); // rustls gone from fips, T1 declares fips
        let violations = evaluate(&sample_manifest(), &trees);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("T1") && v.contains("rustls") && v.contains("not resolved")),
            "an operation crate absent from a declared profile must be flagged: {violations:?}"
        );
    }

    #[test]
    fn platform_only_operation_crate_is_exempt() {
        // C1 references security-framework/schannel, absent from the host graphs;
        // they must not be flagged as unresolved.
        let violations = evaluate(&sample_manifest(), &sample_trees());
        assert!(
            !violations
                .iter()
                .any(|v| v.contains("security-framework") || v.contains("schannel")),
            "platform-only crates must be exempt from the resolution check: {violations:?}"
        );
    }

    #[test]
    fn non_compliant_operation_without_remediation_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.operations[1].remediation = String::new(); // A3 non-validated
        let violations = evaluate(&manifest, &sample_trees());
        assert!(
            violations.iter().any(|v| v.contains("A3") && v.contains("remediation")),
            "a non-compliant op without a remediation reference must be flagged: {violations:?}"
        );
    }

    #[test]
    fn duplicate_operation_id_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.operations[1].id = "T1".to_owned(); // clash with operations[0]
        let violations = evaluate(&manifest, &sample_trees());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("T1") && v.contains("more than once")),
            "a duplicate operation id must be flagged: {violations:?}"
        );
    }

    #[test]
    fn empty_operation_field_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.operations[0].caller = "  ".to_owned();
        let violations = evaluate(&manifest, &sample_trees());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("T1") && v.contains("empty `caller`")),
            "an empty operation field must be flagged: {violations:?}"
        );
    }

    #[test]
    fn invalid_operation_disposition_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.operations[0].disposition = "totally-fine".to_owned();
        let violations = evaluate(&manifest, &sample_trees());
        assert!(
            violations.iter().any(|v| v.contains("T1") && v.contains("disposition")),
            "an invalid disposition must be flagged: {violations:?}"
        );
    }

    #[test]
    fn empty_operations_inventory_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.operations.clear();
        let violations = evaluate(&manifest, &sample_trees());
        assert!(
            violations.iter().any(|v| v.contains("operations inventory is empty")),
            "an empty operations inventory must be flagged: {violations:?}"
        );
    }

    #[test]
    fn missing_fips_required_operation_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.operations.retain(|op| op.id != "A12");
        let violations = evaluate(&manifest, &sample_trees());
        assert!(
            violations.iter().any(|v| v.contains("A12") && v.contains("missing")),
            "a missing FIPS-required operation must be flagged: {violations:?}"
        );
    }

    #[test]
    fn fips_required_operation_without_fips_profile_is_flagged() {
        let mut manifest = sample_manifest();
        for op in &mut manifest.operations {
            if op.id == "A3" {
                op.profiles = vec!["full".to_owned()];
            }
        }
        // Keep the fips graph consistent so only the required-op check fires.
        let violations = evaluate(&manifest, &sample_trees());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("A3") && v.contains("does not declare the `fips` profile")),
            "a FIPS-required op dropping the fips profile must be flagged: {violations:?}"
        );
    }

    #[test]
    fn denied_crate_in_fips_graph_is_flagged() {
        let mut trees = sample_trees();
        set_facts(&mut trees[1], "sha2", "0.10.0", &[]); // sha2 into fips
        // sha2 is declared (add an operation) so drift stays quiet and only the
        // fips-denylist backstop fires.
        let mut manifest = sample_manifest();
        manifest.operations.push(Operation {
            id: "X1".to_owned(),
            name: "n".to_owned(),
            owner: "o".to_owned(),
            caller: "c".to_owned(),
            algorithm: "a".to_owned(),
            provider: "p".to_owned(),
            crates: vec!["sha2".to_owned()],
            profiles: vec!["fips".to_owned()],
            disposition: "non-validated".to_owned(),
            remediation: "#1220".to_owned(),
            note: "n".to_owned(),
        });
        let violations = evaluate(&manifest, &trees);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("sha2") && v.contains("fips") && v.contains("denylist")),
            "a denylisted crate in the fips graph must be flagged: {violations:?}"
        );
    }

    #[test]
    fn parse_facts_reads_name_version_features() {
        let text =
            "praxis-ai-proxy v0.3.0|default\nopenssl v0.10.81|bindgen (*)\nrustls v0.23.45|custom-provider,std,tls12\n";
        let facts = parse_facts(text);
        assert_eq!(facts["openssl"][0].version, "0.10.81");
        assert!(facts["rustls"][0].features.contains("tls12"));
        assert!(facts["rustls"][0].features.contains("custom-provider"));
        assert_eq!(facts["praxis-ai-proxy"][0].version, "0.3.0");
    }

    #[test]
    fn parse_facts_dedupes_repeated_versions() {
        let text = "openssl v0.10.81|a\nopenssl v0.10.81|a (*)\n";
        let facts = parse_facts(text);
        assert_eq!(facts["openssl"].len(), 1, "same version must be deduped");
    }

    #[test]
    fn render_doc_is_deterministic() {
        let manifest = sample_manifest();
        assert_eq!(render_doc(&manifest), render_doc(&manifest), "render must be pure");
    }

    #[test]
    fn render_doc_contains_key_sections() {
        let doc = render_doc(&sample_manifest());
        for heading in [
            "## Build profiles",
            "## Dispositions",
            "## Cryptographic operations",
            "## Pinned providers",
            "## Reviewed non-primitives (`allow`)",
            "## Watch tokens",
            "## Maintenance",
        ] {
            assert!(doc.contains(heading), "generated doc is missing `{heading}`");
        }
    }
}
