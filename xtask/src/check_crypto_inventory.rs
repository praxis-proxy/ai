// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask check-crypto-inventory` — verify the cryptographic inventory
//! manifest against the actual runtime dependency graph of the production
//! profile.
//!
//! The manifest (`docs/architecture/cryptographic-inventory.yaml`) is the stable
//! inventory backing `docs/architecture/cryptographic-inventory.md`. This check
//! resolves the runtime graph of the profile of record (the shipped production
//! artifact, `PRAXIS_AI_FEATURES ?= full`) —
//! `cargo tree -p praxis-ai-proxy --edges normal --no-default-features
//! --features full --target <triple>` — on all three tier-1 targets,
//! so the result is host-independent and CI validates the macOS/Windows
//! declarations even though it runs on Linux. It fails when the manifest and the
//! resolved graphs disagree, so:
//!
//! * a crypto-family-named crate (matched by the manifest's `watch_tokens`) cannot enter a target's runtime graph
//!   undeclared — the guard is a name tripwire, not a totalizing gate: a genuinely novel crypto crate whose name
//!   matches no token is caught instead at Cargo.lock / `cargo deny` review, where new dependencies land (widen
//!   `watch_tokens` when one appears);
//! * a `production` entry cannot go unclassified (missing/unknown `disposition`);
//! * a pinned provider cannot drift its version, gain a forbidden feature, or lose a required one while the crate name
//!   stays put (e.g. `aws-lc-rs` gaining `fips`, `rustls` regaining `ring`, or `openssl` gaining `vendored`); unlisted
//!   features are permitted by design, since feature sets grow across patch releases and an exact-set pin would churn
//!   on benign additions;
//! * a proc-macro crate (build-host code generator, present under `--edges normal` but not linked into the shipped
//!   binary — e.g. `zeroize_derive`) cannot be misfiled as runtime-binary content, and a runtime crate cannot hide in
//!   the build-host `proc_macro` bucket;
//! * a crate reachable only *beneath* a proc-macro subtree (a code generator's own dependency — e.g. `tiny-keccak`
//!   under `const-random-macro`) executes on the build host too, so it cannot be filed as runtime content: runtime
//!   linkage is computed by walking the tree from the root and refusing to cross into proc-macro nodes, not by mere
//!   presence in `--edges normal` output;
//! * test/build-only cryptography cannot silently link into the runtime binary;
//! * a crate cannot carry two classifications at once (a duplicate across `production`/`allow`/`proc_macro`/
//!   `test_only`/`build_only` would let one bucket's guard mask another's).

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
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
pub(crate) struct Args;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Relative path from the workspace root to the manifest.
const MANIFEST_REL: &str = "docs/architecture/cryptographic-inventory.yaml";

/// Relative path to the prose companion (referenced in failure output).
const DOC_REL: &str = "docs/architecture/cryptographic-inventory.md";

/// Tier-1 targets resolved so the check is host-independent: the graphs come
/// from the lockfile, not the host, so CI on Linux still validates the
/// macOS/Windows declarations. `(platform label, target triple)`.
const TARGETS: [(&str, &str); 3] = [
    ("linux", "x86_64-unknown-linux-gnu"),
    ("macos", "aarch64-apple-darwin"),
    ("windows", "x86_64-pc-windows-msvc"),
];

/// Dispositions every `production` entry must declare (kept in sync with the
/// prose legend in the companion document).
const VALID_DISPOSITIONS: [&str; 5] = [
    "validated",
    "non-validated",
    "needs-remediation",
    "upstream-owned",
    "n-a",
];

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
    /// Crypto/crypto-support crates expected in the runtime binary.
    production: Vec<Entry>,
    /// Watch-matched crates that are not tracked primitives (reviewed).
    allow: Vec<Entry>,
    /// Crypto that must never appear in the runtime graph (dev-only).
    test_only: Vec<Entry>,
    /// Crypto that runs only on the build host (build-dependencies).
    build_only: Vec<Entry>,
    /// Crypto-adjacent proc-macro crates: present under `--edges normal` but
    /// executing on the build host as code generators, not linked into the
    /// shipped binary. Tracked separately so they are not counted as runtime
    /// contents (e.g. `zeroize_derive`, `asn1-rs-derive`).
    #[serde(default)]
    proc_macro: Vec<Entry>,
    /// Load-bearing providers whose version and features are pinned.
    #[serde(default)]
    providers: Vec<Provider>,
}

/// One crate entry in the manifest.
///
/// `deny_unknown_fields` rejects mistyped keys (e.g. `platfrom:`) instead of
/// silently defaulting them, which would quietly change the entry's scope.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    /// The crate name as it appears in `cargo tree`.
    #[serde(rename = "crate")]
    crate_name: String,
    /// The host OS this entry is present on (`linux`/`macos`/`windows`).
    ///
    /// When set, the stale check only applies on that target; an unset platform
    /// means the crate is expected on every target.
    #[serde(default)]
    platform: Option<String>,
    /// The classification (`validated`/`non-validated`/…). Required and
    /// validated for `production` entries; advisory elsewhere.
    #[serde(default)]
    disposition: Option<String>,
    /// Human-readable rationale. Validated non-empty for `production` entries so
    /// no tracked crypto path is left undocumented.
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

// -----------------------------------------------------------------------------
// Resolved graph
// -----------------------------------------------------------------------------

/// Version and enabled features of one crate in a target's runtime graph.
#[derive(Debug)]
struct CrateFacts {
    /// The resolved crate version (without the leading `v`).
    version: String,
    /// The set of features cargo reports enabled for the crate.
    features: BTreeSet<String>,
    /// Whether cargo marks this crate `(proc-macro)` — a build-host code
    /// generator that is not linked into the runtime binary.
    is_proc_macro: bool,
}

/// One resolved target's runtime graph: crate name -> every resolved version.
///
/// The value is a `Vec` because Cargo can co-resolve several versions of the
/// same crate; keeping them all is what lets the provider-drift check see a
/// second, upgraded copy of a pinned provider instead of silently accepting the
/// first occurrence.
struct TargetGraph {
    /// Platform label (`linux`/`macos`/`windows`).
    platform: String,
    /// Resolved crate facts keyed by crate name (one entry per resolved version).
    facts: BTreeMap<String, Vec<CrateFacts>>,
    /// Crate names actually linked into the runtime binary: reachable from the
    /// root crate over `--edges normal` edges *without crossing a proc-macro
    /// node*. Proc-macros and any crate reachable only beneath one (a code
    /// generator's own dependency, e.g. `tiny-keccak` under `const-random-macro`)
    /// run on the build host and are excluded even though `cargo tree` lists them.
    runtime_linked: BTreeSet<String>,
}

impl TargetGraph {
    /// Whether the crate appears anywhere in this target's resolved tree
    /// (including build-host-only subtrees). Presence, not runtime linkage — use
    /// [`Self::is_runtime_linked`] to ask what actually ships in the binary.
    fn contains(&self, name: &str) -> bool {
        self.facts.contains_key(name)
    }

    /// Whether any resolved copy of the crate is a build-host proc-macro.
    fn is_proc_macro(&self, name: &str) -> bool {
        self.facts.get(name).is_some_and(|v| v.iter().any(|f| f.is_proc_macro))
    }

    /// Whether the crate is linked into the runtime binary — reachable from the
    /// root over normal edges without passing through a proc-macro node.
    fn is_runtime_linked(&self, name: &str) -> bool {
        self.runtime_linked.contains(name)
    }
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Verify the manifest against the resolved runtime graphs.
pub(crate) fn run(_args: Args) {
    let root = workspace_root();
    let manifest_path = root.join(MANIFEST_REL);

    let raw = std::fs::read_to_string(&manifest_path).unwrap_or_else(|err| {
        eprintln!("failed to read {}: {err}", manifest_path.display());
        std::process::exit(1);
    });
    let manifest: Manifest = serde_yaml::from_str(&raw).unwrap_or_else(|err| {
        eprintln!("failed to parse {MANIFEST_REL}: {err}");
        std::process::exit(1);
    });

    let graphs: Vec<TargetGraph> = TARGETS
        .iter()
        .map(|(platform, triple)| resolve_target(&root, platform, triple))
        .collect();

    let violations = evaluate(&manifest, &graphs);

    if violations.is_empty() {
        print_summary(&manifest, &graphs);
    } else {
        eprintln!("crypto inventory check failed ({} violation(s)):", violations.len());
        for violation in &violations {
            eprintln!("  - {violation}");
        }
        eprintln!("\nReview {DOC_REL} and update {MANIFEST_REL} to match the change.");
        std::process::exit(1);
    }
}

/// Print the success summary: manifest counts plus the resolved target set.
fn print_summary(manifest: &Manifest, graphs: &[TargetGraph]) {
    let union: BTreeSet<&str> = graphs.iter().flat_map(|g| g.facts.keys().map(String::as_str)).collect();
    let targets: Vec<&str> = graphs.iter().map(|g| g.platform.as_str()).collect();
    println!(
        "crypto inventory in sync ({} production, {} allow, {} providers, {} proc-macro, {} test-only, {} build-only; targets: {}; {} crates across targets)",
        manifest.production.len(),
        manifest.allow.len(),
        manifest.providers.len(),
        manifest.proc_macro.len(),
        manifest.test_only.len(),
        manifest.build_only.len(),
        targets.join("/"),
        union.len(),
    );
}

// -----------------------------------------------------------------------------
// Evaluation
// -----------------------------------------------------------------------------

/// Compare the manifest against the resolved per-target `graphs`, returning a
/// human-readable violation for each mismatch. Pure: all IO happens in [`run`].
fn evaluate(manifest: &Manifest, graphs: &[TargetGraph]) -> Vec<String> {
    let platforms: Vec<&str> = graphs.iter().map(|g| g.platform.as_str()).collect();
    let known = platforms.join("/");

    let mut out = Vec::new();
    check_duplicates(manifest, &mut out);
    check_dispositions(&manifest.production, &mut out);
    check_notes(manifest, &mut out);
    check_undeclared(manifest, graphs, &mut out);
    check_stale_production(&manifest.production, graphs, &known, &mut out);
    check_stale_allow(&manifest.allow, graphs, &known, &mut out);
    check_proc_macro(manifest, graphs, &mut out);
    check_runtime_linkage(manifest, graphs, &mut out);
    check_containment(manifest, graphs, &mut out);
    check_providers(&manifest.providers, graphs, &mut out);
    out
}

/// Uniqueness check: every crate must carry exactly one classification. A crate
/// declared in two lists lets one bucket's guard mask another's (e.g. a crate in
/// both `build_only` and `production` is exempt from the containment check while
/// still counted as runtime content). Providers are excluded from the mutual-
/// exclusion union because they intentionally overlay `production`; they are
/// instead checked for self-duplication and for spilling into any non-`production`
/// list.
fn check_duplicates(manifest: &Manifest, out: &mut Vec<String>) {
    let mut buckets: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let labelled: [(&str, &[Entry]); 5] = [
        ("production", &manifest.production),
        ("allow", &manifest.allow),
        ("proc_macro", &manifest.proc_macro),
        ("test_only", &manifest.test_only),
        ("build_only", &manifest.build_only),
    ];
    for (label, entries) in labelled {
        for entry in entries {
            buckets.entry(entry.crate_name.as_str()).or_default().push(label);
        }
    }
    for (name, labels) in &buckets {
        if labels.len() > 1 {
            out.push(format!(
                "duplicate manifest entry: `{name}` is classified in {} lists ({}) but must have exactly one classification",
                labels.len(),
                labels.join(", ")
            ));
        }
    }
    check_provider_overlap(&manifest.providers, &buckets, out);
}

/// Duplicate sub-check for the pinned providers: a provider may be listed only
/// once and may overlap only the `production` classification (never `allow`,
/// `proc_macro`, `test_only`, or `build_only`).
fn check_provider_overlap(providers: &[Provider], buckets: &BTreeMap<&str, Vec<&str>>, out: &mut Vec<String>) {
    let mut seen_providers: BTreeSet<&str> = BTreeSet::new();
    for provider in providers {
        let name = provider.crate_name.as_str();
        if !seen_providers.insert(name) {
            out.push(format!("duplicate pinned provider: `{name}` is listed more than once"));
        }
        if let Some(labels) = buckets
            .get(name)
            .filter(|labels| labels.iter().any(|l| *l != "production"))
        {
            out.push(format!(
                "pinned provider `{name}` also appears in a non-`production` list ({}) — a provider may only overlap `production`",
                labels.join(", ")
            ));
        }
    }
}

/// Classification check: every `production` entry must carry a known
/// `disposition`, so a missing or misspelled classification cannot pass.
fn check_dispositions(production: &[Entry], out: &mut Vec<String>) {
    let valid: BTreeSet<&str> = VALID_DISPOSITIONS.iter().copied().collect();
    for entry in production {
        match entry.disposition.as_deref() {
            None => out.push(format!(
                "`production` entry `{}` has no `disposition` (expected one of {VALID_DISPOSITIONS:?})",
                entry.crate_name
            )),
            Some(d) if !valid.contains(d) => out.push(format!(
                "`production` entry `{}` has unknown disposition `{d}` (expected one of {VALID_DISPOSITIONS:?})",
                entry.crate_name
            )),
            Some(_) => {},
        }
    }
}

/// Documentation check: every load-bearing entry (production primitives, tracked
/// proc-macros, and pinned providers) must carry a non-empty `note`, so no
/// tracked crypto path is left undocumented. Reading `note` here also anchors the
/// field that `deny_unknown_fields` requires be declared to accept the manifest's
/// `note:`.
fn check_notes(manifest: &Manifest, out: &mut Vec<String>) {
    for entry in manifest.production.iter().chain(&manifest.proc_macro) {
        if entry.note.trim().is_empty() {
            out.push(format!(
                "`{}` has an empty `note` (document its rationale)",
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

/// Undeclared-crypto check: a watch-matched crate present in a target's graph
/// that is not declared applicable to THAT platform in either `production` or
/// `allow`. Each target is checked against the entries that apply to it, so a
/// `platform:`-tagged entry (e.g. Linux-only `openssl`) does not silently
/// suppress the same crate appearing unexpectedly on another target. Resolving
/// every target from the lockfile keeps the check host-independent.
fn check_undeclared(manifest: &Manifest, graphs: &[TargetGraph], out: &mut Vec<String>) {
    let watch: BTreeSet<String> = manifest.watch_tokens.iter().map(|t| t.to_ascii_lowercase()).collect();
    for g in graphs {
        for name in g.facts.keys() {
            if is_watch_match(name, &watch)
                && !declared_for(&manifest.production, name, &g.platform)
                && !declared_for(&manifest.allow, name, &g.platform)
                && !declared_for(&manifest.proc_macro, name, &g.platform)
                && !declared_for(&manifest.build_only, name, &g.platform)
            {
                out.push(format!(
                    "undeclared crypto crate in the {} resolved tree: `{name}` matches a watch token but is declared in none of `production`, `allow`, `proc_macro`, or `build_only` for that platform",
                    g.platform
                ));
            }
        }
    }
}

/// Whether `entries` declares `name` as applicable to `platform`: an untagged
/// entry applies to every target, a `platform:`-tagged one only to its target.
fn declared_for(entries: &[Entry], name: &str, platform: &str) -> bool {
    entries
        .iter()
        .any(|e| e.crate_name == name && e.platform.as_deref().is_none_or(|p| p == platform))
}

/// Evaluate one `platform:`-tagged entry against the single target it names,
/// returning a violation if the platform is unknown or the crate is absent from
/// that target's graph. Shared by the `production` and `allow` stale checks.
fn stale_tagged(kind: &str, name: &str, platform: &str, graphs: &[TargetGraph], known: &str) -> Option<String> {
    match graphs.iter().find(|g| g.platform == platform) {
        None => Some(format!(
            "`{kind}` entry `{name}` has unknown platform `{platform}` (expected one of {known})"
        )),
        Some(g) if !g.contains(name) => Some(format!(
            "stale `{kind}` entry: `{name}` (platform: {platform}) is declared but absent from the {platform} runtime graph"
        )),
        Some(_) => None,
    }
}

/// Stale-declaration check: a `production` entry missing from a target it applies
/// to. A `platform:`-tagged entry is checked on its own target; an untagged entry
/// must appear on every target.
fn check_stale_production(entries: &[Entry], graphs: &[TargetGraph], known: &str, out: &mut Vec<String>) {
    for entry in entries {
        match entry.platform.as_deref() {
            Some(p) => out.extend(stale_tagged("production", &entry.crate_name, p, graphs, known)),
            None => {
                for g in graphs {
                    if !g.contains(&entry.crate_name) {
                        out.push(format!(
                            "stale `production` entry: `{}` (platform: any) is declared but absent from the {} runtime graph",
                            entry.crate_name, g.platform
                        ));
                    }
                }
            },
        }
    }
}

/// Stale-exception check: an `allow` entry that no longer resolves anywhere (it
/// stops suppressing a real crate and becomes dead documentation).
fn check_stale_allow(entries: &[Entry], graphs: &[TargetGraph], known: &str, out: &mut Vec<String>) {
    for entry in entries {
        match entry.platform.as_deref() {
            Some(p) => out.extend(stale_tagged("allow", &entry.crate_name, p, graphs, known)),
            None => {
                if !graphs.iter().any(|g| g.contains(&entry.crate_name)) {
                    out.push(format!(
                        "stale `allow` entry: `{}` is declared but absent from every target's runtime graph",
                        entry.crate_name
                    ));
                }
            },
        }
    }
}

/// Proc-macro classification check. Proc-macros are build-host code generators:
/// `cargo tree --edges normal` lists them, but they are not linked into the
/// shipped binary, so they must not be counted as runtime-binary content. Both
/// directions are enforced:
///
/// * every `proc_macro` entry must be present AND actually marked `(proc-macro)` in some target's graph — otherwise a
///   normal (linked) crate is hiding in the build-host bucket, dodging runtime scrutiny;
/// * no `production` or `allow` entry may be a proc-macro — that would overstate the runtime-binary contents (the
///   inaccuracy this check exists to prevent).
fn check_proc_macro(manifest: &Manifest, graphs: &[TargetGraph], out: &mut Vec<String>) {
    for entry in &manifest.proc_macro {
        if !graphs.iter().any(|g| g.contains(&entry.crate_name)) {
            out.push(format!(
                "stale `proc_macro` entry: `{}` is absent from every target's runtime graph",
                entry.crate_name
            ));
        } else if !graphs.iter().any(|g| g.is_proc_macro(&entry.crate_name)) {
            out.push(format!(
                "misclassified `proc_macro` entry: `{}` is a normal linked crate, not a proc-macro — move it to `production`/`allow`",
                entry.crate_name
            ));
        }
    }
    for entry in manifest.production.iter().chain(&manifest.allow) {
        if graphs.iter().any(|g| g.is_proc_macro(&entry.crate_name)) {
            out.push(format!(
                "`{}` is a build-host proc-macro (not linked into the runtime binary) but is declared in a runtime list — move it to `proc_macro`",
                entry.crate_name
            ));
        }
    }
}

/// Runtime-linkage check: a `production`/`allow` entry that resolves into a
/// target's tree but is NOT linked into the runtime binary there — reachable
/// only beneath a proc-macro subtree (a build-host code generator's own
/// dependency, e.g. `tiny-keccak` under `const-random-macro`) — overstates the
/// runtime contents and must move to `build_only`. Proc-macro entries are handled
/// by [`check_proc_macro`] and skipped here, so this fires only on ordinary
/// crates buried under a code generator.
fn check_runtime_linkage(manifest: &Manifest, graphs: &[TargetGraph], out: &mut Vec<String>) {
    for entry in manifest.production.iter().chain(&manifest.allow) {
        for g in graphs {
            if g.contains(&entry.crate_name)
                && !g.is_runtime_linked(&entry.crate_name)
                && !g.is_proc_macro(&entry.crate_name)
            {
                out.push(format!(
                    "`{}` resolves in the {} tree but is reachable only via a proc-macro subtree (build-host only, not linked into the runtime binary) — move it to `build_only`",
                    entry.crate_name, g.platform
                ));
            }
        }
    }
}

/// Containment check: dev-only or build-only crypto actually linked into any
/// target's runtime binary. Keyed on runtime linkage, not mere presence, so a
/// `build_only` crate that only appears beneath a proc-macro subtree (its
/// legitimate build-host home) is allowed while a genuine leak into the shipped
/// binary is still caught.
fn check_containment(manifest: &Manifest, graphs: &[TargetGraph], out: &mut Vec<String>) {
    for entry in manifest.test_only.iter().chain(&manifest.build_only) {
        for g in graphs {
            if g.is_runtime_linked(&entry.crate_name) {
                out.push(format!(
                    "containment regression: `{}` is linked into the {} runtime binary",
                    entry.crate_name, g.platform
                ));
            }
        }
    }
}

/// Provider-drift check across all pinned providers.
fn check_providers(providers: &[Provider], graphs: &[TargetGraph], out: &mut Vec<String>) {
    for provider in providers {
        check_provider(provider, graphs, out);
    }
}

/// Verify one pinned provider's version and feature set on every target it
/// appears in, and that it appears at all — the load-bearing provider-selection
/// facts cannot rot while the crate name stays put.
fn check_provider(provider: &Provider, graphs: &[TargetGraph], out: &mut Vec<String>) {
    let mut seen = false;
    for g in graphs {
        let Some(instances) = g.facts.get(&provider.crate_name) else {
            continue;
        };
        seen = true;
        // Every resolved copy must match the pin. An off-version copy is drift on
        // its own; features are only meaningful on the pinned-version copy.
        for facts in instances {
            if facts.version == provider.version {
                check_provider_features(provider, facts, &g.platform, out);
            } else {
                out.push(format!(
                    "provider drift: `{}` is `{}` in the {} graph but the manifest pins `{}`",
                    provider.crate_name, facts.version, g.platform, provider.version
                ));
            }
        }
    }
    if !seen {
        out.push(format!(
            "pinned provider `{}` is absent from every target's runtime graph",
            provider.crate_name
        ));
    }
}

/// Verify a provider's required features are present and forbidden ones absent
/// in one target's resolved feature set. Unlisted (extra) features are permitted
/// by design: feature sets legitimately grow across patch releases, so exact-set
/// pinning would churn on benign additions. Only the security-relevant invariants
/// are expressed — as `require_features` (must stay) and `forbid_features` (must
/// not appear, e.g. `fips`). A provider with no constraints (e.g. `ring`) pins
/// only its version, because none of its features affect provider selection.
fn check_provider_features(provider: &Provider, facts: &CrateFacts, platform: &str, out: &mut Vec<String>) {
    for want in &provider.require_features {
        if !facts.features.contains(want) {
            out.push(format!(
                "provider drift: `{}` is missing required feature `{want}` in the {platform} graph",
                provider.crate_name
            ));
        }
    }
    for deny in &provider.forbid_features {
        if facts.features.contains(deny) {
            out.push(format!(
                "provider drift: `{}` has forbidden feature `{deny}` enabled in the {platform} graph",
                provider.crate_name
            ));
        }
    }
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
// Cargo Graph
// -----------------------------------------------------------------------------

/// The fixed `cargo tree` arguments describing the profile of record; the
/// per-target `--target <triple>` pair is appended in [`resolve_target`].
const TREE_ARGS: [&str; 12] = [
    "tree",
    "-p",
    "praxis-ai-proxy",
    "--edges",
    "normal",
    "--no-default-features",
    "--features",
    "full",
    "--prefix",
    "depth",
    "--format",
    "{p}|{f}",
];

/// Resolve the runtime (`--edges normal`) dependency graph of the profile of
/// record for one target triple and return its crate facts.
fn resolve_target(root: &Path, platform: &str, triple: &str) -> TargetGraph {
    let output = Command::new("cargo")
        .current_dir(root)
        .args(TREE_ARGS)
        .args(["--target", triple])
        .output()
        .expect("failed to run cargo tree");

    if !output.status.success() {
        eprintln!(
            "cargo tree (--target {triple}) failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::process::exit(1);
    }

    let text = String::from_utf8(output.stdout).expect("cargo tree output is not UTF-8");
    TargetGraph {
        platform: platform.to_owned(),
        runtime_linked: compute_runtime_linked(&text),
        facts: parse_facts(&text),
    }
}

/// Split a `--prefix depth` line into its integer depth and the remaining text.
///
/// `cargo tree --prefix depth` glues the depth digits directly onto the crate
/// name (`0praxis-ai-proxy v0.3.0 …`, `6tiny-keccak v2.0.2 …`). Lines with no
/// leading digit (the synthetic test fixtures, which omit the prefix) parse as
/// depth 0 with the whole line returned — no crate name begins with a digit, so
/// this is unambiguous.
fn split_depth(line: &str) -> (usize, &str) {
    let end = line
        .char_indices()
        .find(|(_, c)| !c.is_ascii_digit())
        .map_or(line.len(), |(i, _)| i);
    let (digits, rest) = line.split_at(end);
    (digits.parse::<usize>().unwrap_or(0), rest)
}

/// Parse `cargo tree --prefix depth --format "{p}|{f}"` output into crate facts.
///
/// Each line is `<depth>name vX.Y.Z [(source)] [(proc-macro)]|feat1,feat2 [(*)]`;
/// the leading digits are the tree depth (stripped here — linkage lives in
/// [`compute_runtime_linked`]), the trailing `(*)` marks a subtree cargo already
/// printed, and a `(proc-macro)` marker flags a build-host code generator that is
/// not linked into the runtime binary. Every distinct resolved version of a crate
/// is retained (the `(*)` repeats of the same version are de-duplicated), so a
/// crate co-resolved at two versions keeps both.
fn parse_facts(text: &str) -> BTreeMap<String, Vec<CrateFacts>> {
    let mut facts: BTreeMap<String, Vec<CrateFacts>> = BTreeMap::new();
    for raw in text.lines() {
        let trimmed = raw.trim().trim_end_matches("(*)").trim_end();
        if trimmed.is_empty() {
            continue;
        }
        let (_, line) = split_depth(trimmed);
        let (left, right) = line.split_once('|').unwrap_or((line, ""));
        let is_proc_macro = left.contains("(proc-macro)");
        let mut fields = left.split_whitespace();
        let Some(name) = fields.next() else { continue };
        let version = fields.next().unwrap_or("").trim_start_matches('v').to_owned();
        let features: BTreeSet<String> = right
            .split(',')
            .map(str::trim)
            .filter(|f| !f.is_empty())
            .map(str::to_owned)
            .collect();
        let versions = facts.entry(name.to_owned()).or_default();
        if !versions.iter().any(|f| f.version == version) {
            versions.push(CrateFacts {
                version,
                features,
                is_proc_macro,
            });
        }
    }
    facts
}

/// The name-keyed edge set of a `cargo tree --prefix depth` graph, plus the set
/// of proc-macro nodes and the root crate.
struct TreeEdges {
    /// For each crate, the set of crates it depends on (by name).
    edges: BTreeMap<String, BTreeSet<String>>,
    /// Crates marked `(proc-macro)` in the graph (build-host code generators).
    proc_macros: BTreeSet<String>,
    /// The root crate (`praxis-ai-proxy`), if any line was at depth 0.
    root: Option<String>,
}

/// Reconstruct the dependency edges from a `cargo tree --prefix depth` walk.
///
/// `cargo tree --prefix depth` prints a depth-first walk with an integer depth on
/// every line, so the parent of a line at depth `d` is the most recent line at
/// depth `d - 1`. That reconstructs a global, name-keyed edge set (a `(*)` repeat
/// still contributes its parent edge; its children were already printed at the
/// first, fully expanded occurrence).
fn parse_tree_edges(text: &str) -> TreeEdges {
    let mut edges: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut proc_macros: BTreeSet<String> = BTreeSet::new();
    let mut root: Option<String> = None;
    let mut stack: Vec<String> = Vec::new();
    for raw in text.lines() {
        let Some((depth, name, is_proc_macro)) = parse_tree_line(raw) else {
            continue;
        };
        if is_proc_macro {
            proc_macros.insert(name.to_owned());
        }
        if depth == 0 {
            root.get_or_insert_with(|| name.to_owned());
        } else if let Some(parent) = stack.get(depth - 1) {
            edges.entry(parent.clone()).or_default().insert(name.to_owned());
        }
        stack.truncate(depth);
        stack.push(name.to_owned());
    }
    TreeEdges {
        edges,
        proc_macros,
        root,
    }
}

/// Parse one `cargo tree --prefix depth` line into `(depth, crate name,
/// is-proc-macro)`, or `None` for a blank line. The trailing `(*)` (a subtree
/// cargo already printed) is stripped before parsing; it still contributes a
/// parent edge but its children were printed at the first occurrence.
fn parse_tree_line(raw: &str) -> Option<(usize, &str, bool)> {
    let trimmed = raw.trim().trim_end_matches("(*)").trim_end();
    if trimmed.is_empty() {
        return None;
    }
    let (depth, line) = split_depth(trimmed);
    let left = line.split_once('|').map_or(line, |(l, _)| l);
    let is_proc_macro = left.contains("(proc-macro)");
    let name = left.split_whitespace().next()?;
    Some((depth, name, is_proc_macro))
}

/// Compute the set of crates linked into the runtime binary: reachable from the
/// root crate over `--edges normal` edges without passing through a proc-macro
/// node.
///
/// A breadth-first walk from the root collects every reachable crate while
/// refusing to enter proc-macro nodes — so a proc-macro (a build-host code
/// generator) and anything reachable *only* beneath one (e.g. `tiny-keccak` under
/// `const-random-macro`) is excluded, while a crate that is *also* reachable by a
/// normal path stays linked.
fn compute_runtime_linked(text: &str) -> BTreeSet<String> {
    let TreeEdges {
        edges,
        proc_macros,
        root,
    } = parse_tree_edges(text);
    let mut linked: BTreeSet<String> = BTreeSet::new();
    let Some(root) = root else { return linked };
    if proc_macros.contains(&root) {
        return linked;
    }
    let mut queue: VecDeque<String> = VecDeque::new();
    linked.insert(root.clone());
    queue.push_back(root);
    while let Some(node) = queue.pop_front() {
        let Some(children) = edges.get(&node) else { continue };
        for child in children {
            if proc_macros.contains(child) || linked.contains(child) {
                continue;
            }
            linked.insert(child.clone());
            queue.push_back(child.clone());
        }
    }
    linked
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

    /// Assert no crate appears in more than one classification list. Providers
    /// intentionally overlap `production`, so they are excluded.
    fn assert_no_duplicate_entries(manifest: &Manifest) {
        let mut seen = BTreeSet::new();
        for entry in manifest
            .production
            .iter()
            .chain(&manifest.allow)
            .chain(&manifest.test_only)
            .chain(&manifest.build_only)
            .chain(&manifest.proc_macro)
        {
            assert!(
                seen.insert(entry.crate_name.clone()),
                "duplicate manifest entry: {}",
                entry.crate_name
            );
        }
    }

    fn sample_manifest() -> Manifest {
        // No `providers:` block here — provider drift has its own focused tests
        // so the name-based tests can build graphs from plain crate names.
        let yaml = "
watch_tokens: [sha, md, hmac, rustls, ring, openssl, aws, rand]
production:
  - crate: sha2
    disposition: non-validated
    note: test
  - crate: rustls
    disposition: non-validated
    note: test
  - crate: openssl
    disposition: non-validated
    platform: linux
    note: test
  - crate: security-framework
    disposition: non-validated
    platform: macos
    note: test
allow:
  - crate: aws-smithy-types
  - crate: openssl-macros
test_only:
  - crate: rcgen
  - crate: sha1
build_only:
  - crate: sha3
";
        serde_yaml::from_str(yaml).expect("sample manifest parses")
    }

    /// Build a target graph from bare crate names (version/features irrelevant).
    /// Every name is treated as a normal, runtime-linked crate; proc-macro /
    /// build-host-only crates are layered on via [`insert_proc_macro`] /
    /// [`insert_build_host_only`].
    fn names_tg(platform: &str, names: &[&str]) -> TargetGraph {
        TargetGraph {
            platform: platform.to_owned(),
            facts: names
                .iter()
                .map(|n| {
                    (
                        (*n).to_owned(),
                        vec![CrateFacts {
                            version: "0.0.0".to_owned(),
                            features: BTreeSet::new(),
                            is_proc_macro: false,
                        }],
                    )
                })
                .collect(),
            runtime_linked: names.iter().map(|n| (*n).to_owned()).collect(),
        }
    }

    /// Insert `name` into `g` marked as a build-host proc-macro (present in the
    /// tree but never linked into the runtime binary).
    fn insert_proc_macro(g: &mut TargetGraph, name: &str) {
        g.facts.insert(
            name.to_owned(),
            vec![CrateFacts {
                version: "0.0.0".to_owned(),
                features: BTreeSet::new(),
                is_proc_macro: true,
            }],
        );
        g.runtime_linked.remove(name);
    }

    /// Insert `name` into `g` as an ordinary crate that is present in the tree but
    /// NOT runtime-linked — simulating a crate reachable only beneath a proc-macro
    /// subtree (e.g. `tiny-keccak` under `const-random-macro`).
    fn insert_build_host_only(g: &mut TargetGraph, name: &str) {
        g.facts.insert(
            name.to_owned(),
            vec![CrateFacts {
                version: "0.0.0".to_owned(),
                features: BTreeSet::new(),
                is_proc_macro: false,
            }],
        );
        g.runtime_linked.remove(name);
    }

    /// Build a target graph with `normal` crates plus `proc_macros` marked as
    /// build-host proc-macros.
    fn tg_with(platform: &str, normal: &[&str], proc_macros: &[&str]) -> TargetGraph {
        let mut g = names_tg(platform, normal);
        for name in proc_macros {
            insert_proc_macro(&mut g, name);
        }
        g
    }

    /// The three-target graph set for a clean sample tree.
    fn clean_graphs() -> Vec<TargetGraph> {
        vec![
            names_tg(
                "linux",
                &[
                    "sha2",
                    "rustls",
                    "openssl",
                    "openssl-macros",
                    "aws-smithy-types",
                    "serde",
                ],
            ),
            names_tg("macos", &["sha2", "rustls", "security-framework", "serde"]),
            names_tg("windows", &["sha2", "rustls", "serde"]),
        ]
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
    fn clean_graphs_have_no_violations() {
        let violations = evaluate(&sample_manifest(), &clean_graphs());
        assert!(violations.is_empty(), "expected no violations, got: {violations:?}");
    }

    #[test]
    fn undeclared_crypto_crate_is_flagged_from_any_target() {
        // `hmac` present only on Linux is still caught: every target graph is
        // resolved from the lockfile and checked, so the host does not matter.
        let mut graphs = clean_graphs();
        graphs[0].facts.insert(
            "hmac".to_owned(),
            vec![CrateFacts {
                version: "0.0.0".to_owned(),
                features: BTreeSet::new(),
                is_proc_macro: false,
            }],
        );
        let violations = evaluate(&sample_manifest(), &graphs);
        assert_eq!(violations.len(), 1, "hmac is undeclared: {violations:?}");
        let msg = violations.first().expect("exactly one violation");
        assert!(msg.contains("hmac"), "{msg}");
        assert!(msg.contains("undeclared"), "{msg}");
    }

    #[test]
    fn allow_list_suppresses_undeclared() {
        // openssl-macros matches `openssl` but is on the allow list -> quiet.
        let violations = evaluate(&sample_manifest(), &clean_graphs());
        assert!(
            !violations.iter().any(|v| v.contains("openssl-macros")),
            "allow-listed crate must not be flagged: {violations:?}"
        );
    }

    #[test]
    fn untagged_production_entry_missing_everywhere_is_flagged() {
        // rustls (platform any) removed from all three graphs -> stale on each.
        let graphs = vec![
            names_tg("linux", &["sha2", "openssl", "openssl-macros", "aws-smithy-types"]),
            names_tg("macos", &["sha2", "security-framework"]),
            names_tg("windows", &["sha2"]),
        ];
        let violations = evaluate(&sample_manifest(), &graphs);
        let stale: Vec<&String> = violations
            .iter()
            .filter(|v| v.contains("stale") && v.contains("rustls"))
            .collect();
        assert_eq!(stale.len(), 3, "one stale per target: {violations:?}");
    }

    #[test]
    fn off_target_platform_entry_is_not_flagged_stale() {
        // openssl (linux) is absent from macOS/Windows and must NOT be flagged.
        let violations = evaluate(&sample_manifest(), &clean_graphs());
        assert!(
            !violations.iter().any(|v| v.contains("openssl") && v.contains("stale")),
            "{violations:?}"
        );
    }

    #[test]
    fn platform_entry_missing_on_its_own_target_is_flagged() {
        // security-framework (macos) is absent from the macOS graph -> stale.
        let mut graphs = clean_graphs();
        graphs[1] = names_tg("macos", &["sha2", "rustls", "serde"]);
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("stale") && v.contains("security-framework")),
            "{violations:?}"
        );
    }

    #[test]
    fn platform_tagged_entry_does_not_suppress_other_platforms() {
        // `openssl` is declared only for linux; an unexpected copy on the windows
        // graph must still be flagged undeclared (platform-scoped suppression).
        let mut graphs = clean_graphs();
        graphs[2] = names_tg("windows", &["sha2", "rustls", "serde", "openssl"]);
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("undeclared") && v.contains("openssl") && v.contains("windows")),
            "{violations:?}"
        );
    }

    #[test]
    fn unknown_platform_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "made-up".to_owned(),
            platform: Some("solaris".to_owned()),
            disposition: Some("n-a".to_owned()),
            note: "test".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("unknown platform") && v.contains("solaris")),
            "{violations:?}"
        );
    }

    #[test]
    fn missing_disposition_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "sha2".to_owned(),
            platform: None,
            disposition: None,
            note: "test".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations.iter().any(|v| v.contains("no `disposition`")),
            "{violations:?}"
        );
    }

    #[test]
    fn unknown_disposition_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "sha2".to_owned(),
            platform: None,
            disposition: Some("bogus".to_owned()),
            note: "test".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("unknown disposition") && v.contains("bogus")),
            "{violations:?}"
        );
    }

    #[test]
    fn empty_production_note_is_flagged() {
        let mut manifest = sample_manifest();
        manifest.production.push(Entry {
            crate_name: "sha2".to_owned(),
            platform: None,
            disposition: Some("n-a".to_owned()),
            note: "  ".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(violations.iter().any(|v| v.contains("empty `note`")), "{violations:?}");
    }

    #[test]
    fn misspelled_provider_constraint_key_is_rejected() {
        // `forbid_feature` (typo) must be a hard parse error, not a silently
        // dropped constraint that would defeat the drift guard.
        let yaml = "
watch_tokens: [aws]
production:
  - crate: aws-lc-rs
    disposition: non-validated
    note: test
allow: []
test_only: []
build_only: []
providers:
  - crate: aws-lc-rs
    version: \"1.18.1\"
    forbid_feature: [fips]
    note: test
";
        let parsed: Result<Manifest, _> = serde_yaml::from_str(yaml);
        assert!(parsed.is_err(), "unknown provider field must be rejected");
    }

    #[test]
    fn test_only_crate_in_runtime_is_containment_regression() {
        // rcgen linked into the runtime binary (not merely present) is a leak.
        let mut graphs = clean_graphs();
        graphs[0].facts.insert(
            "rcgen".to_owned(),
            vec![CrateFacts {
                version: "0.0.0".to_owned(),
                features: BTreeSet::new(),
                is_proc_macro: false,
            }],
        );
        graphs[0].runtime_linked.insert("rcgen".to_owned());
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("containment regression") && v.contains("rcgen")),
            "{violations:?}"
        );
    }

    #[test]
    fn build_only_crate_in_runtime_is_containment_regression() {
        // sha3 linked into the runtime binary (not merely present) is a leak.
        let mut graphs = clean_graphs();
        graphs[2].facts.insert(
            "sha3".to_owned(),
            vec![CrateFacts {
                version: "0.0.0".to_owned(),
                features: BTreeSet::new(),
                is_proc_macro: false,
            }],
        );
        graphs[2].runtime_linked.insert("sha3".to_owned());
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("containment regression") && v.contains("sha3")),
            "{violations:?}"
        );
    }

    #[test]
    fn build_only_crate_present_but_not_runtime_linked_is_quiet() {
        // A `build_only` crate that appears in the tree only beneath a proc-macro
        // subtree (present but not runtime-linked) is its legitimate build-host
        // home — no containment regression. Modelled on `tiny-keccak`.
        let mut manifest = sample_manifest();
        manifest.build_only.push(Entry {
            crate_name: "tiny-keccak".to_owned(),
            platform: None,
            disposition: None,
            note: "build-host only".to_owned(),
        });
        let mut graphs = clean_graphs();
        insert_build_host_only(&mut graphs[0], "tiny-keccak");
        let violations = evaluate(&manifest, &graphs);
        assert!(
            !violations.iter().any(|v| v.contains("tiny-keccak")),
            "build-host-only crate must be quiet: {violations:?}"
        );
    }

    #[test]
    fn runtime_list_entry_not_linked_is_flagged() {
        // A crate declared in `allow` (a runtime list) but reachable only beneath
        // a proc-macro subtree overstates the runtime contents -> flagged with a
        // pointer to `build_only`. This is the tiny-keccak-in-allow regression.
        let mut manifest = sample_manifest();
        manifest.allow.push(Entry {
            crate_name: "tiny-keccak".to_owned(),
            platform: None,
            disposition: None,
            note: String::new(),
        });
        let mut graphs = clean_graphs();
        insert_build_host_only(&mut graphs[0], "tiny-keccak");
        let violations = evaluate(&manifest, &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("tiny-keccak") && v.contains("build_only")),
            "{violations:?}"
        );
    }

    #[test]
    fn duplicate_entry_across_lists_is_flagged() {
        // A crate classified in two lists lets one bucket's guard mask another's.
        let mut manifest = sample_manifest();
        manifest.build_only.push(Entry {
            crate_name: "sha2".to_owned(), // already a `production` entry
            platform: None,
            disposition: None,
            note: "dup".to_owned(),
        });
        let violations = evaluate(&manifest, &clean_graphs());
        assert!(
            violations
                .iter()
                .any(|v| v.contains("duplicate manifest entry") && v.contains("sha2")),
            "{violations:?}"
        );
    }

    #[test]
    fn compute_runtime_linked_excludes_proc_macro_subtree() {
        // `under-pm` is reachable only beneath a proc-macro node -> build-host
        // only. `shared` sits under a normal path -> runtime-linked.
        let text = "0root v1.0.0|\n\
                    1normal-a v1.0.0|\n\
                    2shared v1.0.0|\n\
                    1pm-crate v1.0.0 (proc-macro)|\n\
                    2under-pm v1.0.0|\n";
        let linked = compute_runtime_linked(text);
        assert!(linked.contains("root"), "{linked:?}");
        assert!(linked.contains("normal-a"), "{linked:?}");
        assert!(linked.contains("shared"), "{linked:?}");
        assert!(!linked.contains("pm-crate"), "proc-macro excluded: {linked:?}");
        assert!(
            !linked.contains("under-pm"),
            "crate only under a proc-macro excluded: {linked:?}"
        );
    }

    #[test]
    fn compute_runtime_linked_keeps_crate_with_a_normal_path() {
        // `dual` is reachable via a proc-macro AND via a normal crate; the normal
        // path keeps it linked.
        let text = "0root v1.0.0|\n\
                    1pm-crate v1.0.0 (proc-macro)|\n\
                    2dual v1.0.0|\n\
                    1normal-b v1.0.0|\n\
                    2dual v1.0.0|\n";
        let linked = compute_runtime_linked(text);
        assert!(linked.contains("normal-b"), "{linked:?}");
        assert!(
            linked.contains("dual"),
            "reachable via a normal path stays linked: {linked:?}"
        );
        assert!(!linked.contains("pm-crate"), "{linked:?}");
    }

    fn proc_macro_manifest() -> Manifest {
        // `zeroize_derive` matches the `zeroize` token and is declared as a
        // build-host proc-macro; `sha2`/`rustls` are ordinary runtime primitives.
        let yaml = "
watch_tokens: [sha, rustls, zeroize]
production:
  - crate: sha2
    disposition: non-validated
    note: test
  - crate: rustls
    disposition: non-validated
    note: test
allow: []
test_only: []
build_only: []
proc_macro:
  - crate: zeroize_derive
    note: test
";
        serde_yaml::from_str(yaml).expect("proc_macro manifest parses")
    }

    #[test]
    fn parse_facts_detects_proc_macro_marker() {
        let text = "zeroize_derive v1.5.0 (proc-macro)|\n\
                    async-trait v0.1.92 (proc-macro)|feat (*)\n\
                    sha2 v0.10.0|default\n";
        let facts = parse_facts(text);
        assert!(
            facts
                .get("zeroize_derive")
                .and_then(|v| v.first())
                .expect("present")
                .is_proc_macro,
            "marker detected"
        );
        assert!(
            facts
                .get("async-trait")
                .and_then(|v| v.first())
                .expect("present")
                .is_proc_macro,
            "marker detected before trailing (*)"
        );
        assert!(
            !facts
                .get("sha2")
                .and_then(|v| v.first())
                .expect("present")
                .is_proc_macro,
            "normal crate not flagged"
        );
    }

    #[test]
    fn well_classified_proc_macro_has_no_violations() {
        // zeroize_derive is a build-host proc-macro declared in `proc_macro`: it
        // must be suppressed for undeclared, never flagged stale/misclassified,
        // and never counted as runtime content.
        let graphs = vec![
            tg_with("linux", &["sha2", "rustls"], &["zeroize_derive"]),
            tg_with("macos", &["sha2", "rustls"], &["zeroize_derive"]),
            tg_with("windows", &["sha2", "rustls"], &["zeroize_derive"]),
        ];
        let violations = evaluate(&proc_macro_manifest(), &graphs);
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn proc_macro_in_runtime_list_is_flagged() {
        // A production entry reported as a proc-macro in the graph overstates the
        // runtime-binary contents and must be flagged.
        let mut graphs = clean_graphs();
        insert_proc_macro(&mut graphs[0], "sha2");
        let violations = evaluate(&sample_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("sha2") && v.contains("runtime list")),
            "{violations:?}"
        );
    }

    #[test]
    fn misclassified_proc_macro_entry_is_flagged() {
        // zeroize_derive declared as a proc-macro but resolving as a normal
        // (linked) crate means a runtime crate is hiding in the build-host bucket.
        let graphs = vec![
            tg_with("linux", &["sha2", "rustls", "zeroize_derive"], &[]),
            tg_with("macos", &["sha2", "rustls", "zeroize_derive"], &[]),
            tg_with("windows", &["sha2", "rustls", "zeroize_derive"], &[]),
        ];
        let violations = evaluate(&proc_macro_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("misclassified") && v.contains("zeroize_derive")),
            "{violations:?}"
        );
    }

    #[test]
    fn stale_proc_macro_entry_is_flagged() {
        // zeroize_derive declared but absent from every graph -> dead entry.
        let graphs = vec![
            tg_with("linux", &["sha2", "rustls"], &[]),
            tg_with("macos", &["sha2", "rustls"], &[]),
            tg_with("windows", &["sha2", "rustls"], &[]),
        ];
        let violations = evaluate(&proc_macro_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("stale `proc_macro`") && v.contains("zeroize_derive")),
            "{violations:?}"
        );
    }

    fn provider_manifest() -> Manifest {
        // aws-lc-rs is both a production primitive and a pinned provider (as in
        // the real manifest), so the undeclared check stays quiet and only the
        // provider-drift path is exercised.
        let yaml = "
watch_tokens: [aws]
production:
  - crate: aws-lc-rs
    disposition: non-validated
    note: test
allow: []
test_only: []
build_only: []
providers:
  - crate: aws-lc-rs
    version: \"1.18.1\"
    require_features: [aws-lc-sys]
    forbid_features: [fips]
    note: test
";
        serde_yaml::from_str(yaml).expect("provider manifest parses")
    }

    /// Three-target graphs where `aws-lc-rs` resolves at the given
    /// `(version, &[features])` instances on every target.
    fn provider_graphs(instances: &[(&str, &[&str])]) -> Vec<TargetGraph> {
        let build = || {
            instances
                .iter()
                .map(|(v, f)| CrateFacts {
                    version: (*v).to_owned(),
                    features: f.iter().map(|s| (*s).to_owned()).collect(),
                    is_proc_macro: false,
                })
                .collect::<Vec<_>>()
        };
        TARGETS
            .iter()
            .map(|(platform, _)| {
                let facts = [("aws-lc-rs".to_owned(), build())].into_iter().collect();
                TargetGraph {
                    platform: (*platform).to_owned(),
                    facts,
                    runtime_linked: ["aws-lc-rs".to_owned()].into_iter().collect(),
                }
            })
            .collect()
    }

    #[test]
    fn provider_clean_has_no_violations() {
        let graphs = provider_graphs(&[("1.18.1", &["aws-lc-sys", "alloc", "default"])]);
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn provider_version_drift_is_flagged() {
        let graphs = provider_graphs(&[("1.19.0", &["aws-lc-sys"])]);
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("provider drift") && v.contains("1.19.0")),
            "{violations:?}"
        );
    }

    #[test]
    fn provider_forbidden_feature_is_flagged() {
        let graphs = provider_graphs(&[("1.18.1", &["aws-lc-sys", "fips"])]);
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("forbidden feature") && v.contains("fips")),
            "{violations:?}"
        );
    }

    #[test]
    fn provider_missing_required_feature_is_flagged() {
        let graphs = provider_graphs(&[("1.18.1", &["alloc"])]);
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("missing required feature") && v.contains("aws-lc-sys")),
            "{violations:?}"
        );
    }

    #[test]
    fn provider_absent_everywhere_is_flagged() {
        let graphs = vec![
            names_tg("linux", &["serde"]),
            names_tg("macos", &["serde"]),
            names_tg("windows", &["serde"]),
        ];
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("pinned provider") && v.contains("aws-lc-rs")),
            "{violations:?}"
        );
    }

    #[test]
    fn parse_facts_extracts_name_version_and_features() {
        let text = "praxis-ai-proxy v0.3.0 (/path/to/server)|store-postgres\n\
                    ring v0.17.14|alloc,default,dev_urandom_fallback\n\
                    aws-lc-rs v1.18.1|alloc,aws-lc-sys,default (*)\n\
                    serde v1.0.0|\n\n";
        let facts = parse_facts(text);
        assert_eq!(
            facts.len(),
            4,
            "blank lines ignored: {:?}",
            facts.keys().collect::<Vec<_>>()
        );
        let ring = facts.get("ring").and_then(|v| v.first()).expect("ring present");
        assert_eq!(ring.version, "0.17.14");
        assert!(ring.features.contains("dev_urandom_fallback"));
        let aws = facts
            .get("aws-lc-rs")
            .and_then(|v| v.first())
            .expect("aws-lc-rs present");
        assert_eq!(aws.version, "1.18.1", "(*) marker stripped");
        assert!(aws.features.contains("aws-lc-sys"));
        assert!(!aws.features.contains("fips"));
        let serde = facts.get("serde").and_then(|v| v.first()).expect("serde present");
        assert!(serde.features.is_empty(), "no features");
    }

    #[test]
    fn parse_facts_keeps_multiple_resolved_versions() {
        // A crate co-resolved at two versions must retain both (provider-drift
        // relies on seeing the second, upgraded copy).
        let text = "aws-lc-rs v1.18.1|aws-lc-sys\n\
                    aws-lc-rs v2.0.0|aws-lc-sys,fips\n\
                    aws-lc-rs v1.18.1|aws-lc-sys (*)\n";
        let facts = parse_facts(text);
        let versions = facts.get("aws-lc-rs").expect("aws-lc-rs present");
        assert_eq!(
            versions.len(),
            2,
            "both versions kept, (*) repeat de-duplicated: {versions:?}"
        );
        assert!(versions.iter().any(|f| f.version == "1.18.1"));
        assert!(
            versions
                .iter()
                .any(|f| f.version == "2.0.0" && f.features.contains("fips"))
        );
    }

    #[test]
    fn provider_second_coresolved_version_is_flagged() {
        // The pinned 1.18.1 copy is fine, but a co-resolved 2.0.0 copy must not
        // be silently accepted — this is the P2 false-negative guard.
        let graphs = provider_graphs(&[("1.18.1", &["aws-lc-sys"]), ("2.0.0", &["aws-lc-sys"])]);
        let violations = evaluate(&provider_manifest(), &graphs);
        assert!(
            violations
                .iter()
                .any(|v| v.contains("provider drift") && v.contains("2.0.0")),
            "{violations:?}"
        );
    }

    #[test]
    fn real_manifest_parses_and_is_internally_consistent() {
        let root = workspace_root();
        let raw = std::fs::read_to_string(root.join(MANIFEST_REL)).expect("manifest readable");
        let manifest: Manifest = serde_yaml::from_str(&raw).expect("real manifest parses");
        assert!(!manifest.watch_tokens.is_empty());
        assert!(!manifest.production.is_empty());
        assert!(!manifest.providers.is_empty(), "providers pinned");

        // No crate may appear in more than one classification list (providers
        // intentionally overlap `production`, so they are excluded here).
        assert_no_duplicate_entries(&manifest);

        // Every production entry carries a valid disposition.
        let valid: BTreeSet<&str> = VALID_DISPOSITIONS.iter().copied().collect();
        for entry in &manifest.production {
            let disp = entry.disposition.as_deref();
            assert!(
                disp.is_some_and(|d| valid.contains(d)),
                "{} has missing/invalid disposition {disp:?}",
                entry.crate_name
            );
        }

        // Every pinned provider is also tracked as a production primitive.
        let production: BTreeSet<&str> = manifest.production.iter().map(|e| e.crate_name.as_str()).collect();
        for provider in &manifest.providers {
            assert!(
                production.contains(provider.crate_name.as_str()),
                "pinned provider {} must also be a production entry",
                provider.crate_name
            );
        }

        // The canonical SHA token guards against an empty/malformed token list.
        let watch: BTreeSet<String> = manifest.watch_tokens.iter().map(|t| t.to_ascii_lowercase()).collect();
        assert!(watch.contains("sha"), "expected canonical token present");
    }
}
