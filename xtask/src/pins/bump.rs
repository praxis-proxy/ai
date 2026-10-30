// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! `cargo xtask bump-pins` — move the pinned coding clients and the vLLM image
//! to their latest upstream releases.
//!
//! This automates the mechanical half of a bump: discover the newest release
//! within each manifest's `major_series`, fetch its checksum, rewrite every
//! site in the shared table, swap the checksum fixture, and run `check-pins`
//! on the staged tree before writing anything.
//!
//! The behavioral half stays human. Tool lists, exit-code semantics,
//! version-conditioned test expectations, replay recordings, and "verified
//! against X" claims are never touched; the report lists them as a
//! re-validation checklist instead.

use std::{fmt::Write as _, fs, path::PathBuf};

use clap::Parser;

use super::{
    Repo, check,
    manifest::{self, Manifests, OpenCode},
    sites::{self, CLAUDE_SHA256_SITE, Pin},
    upstream::{NPM_REGISTRY, Upstream},
    workspace_root,
};

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask bump-pins`.
#[derive(Parser)]
pub(crate) struct BumpArgs {
    /// Pin to bump; repeat for several. Every pin when omitted.
    #[arg(long = "pin", value_enum)]
    pins: Vec<Pin>,

    /// Discover and validate the bumps, but write nothing.
    #[arg(long)]
    dry_run: bool,

    /// Also write the markdown report (the bump PR body) to this file.
    #[arg(long)]
    report: Option<PathBuf>,
}

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Discover, stage, validate, and (unless `--dry-run`) write every bump.
pub(crate) fn run(args: &BumpArgs) {
    if let Err(e) = bump(args) {
        eprintln!("bump-pins failed: {e}");
        std::process::exit(1);
    }
}

/// [`run`], returning the first fatal error.
fn bump(args: &BumpArgs) -> Result<(), String> {
    let mut repo = Repo::new(workspace_root());
    let outcomes = stage_all(&mut repo, &args.pins)?;
    let after = Manifests::load(&repo)?;
    let errors = check::check(&repo, &after);
    if !errors.is_empty() {
        return Err(format!(
            "the bumped tree fails check-pins; nothing was written:\n  - {}",
            errors.join("\n  - ")
        ));
    }

    let report = render_report(&repo, &outcomes);
    println!("{report}");
    if let Some(path) = &args.report {
        fs::write(path, &report).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    if args.dry_run {
        eprintln!("dry run: nothing written");
        return Ok(());
    }
    repo.commit()?;
    for path in repo.staged_paths() {
        eprintln!("updated {path}");
    }
    Ok(())
}

/// Check every selected pin (all when `pins` is empty) and stage each bump.
fn stage_all(repo: &mut Repo, pins: &[Pin]) -> Result<Vec<Outcome>, String> {
    let before = Manifests::load(repo)?;
    let upstream = Upstream::new()?;
    let mut outcomes = Vec::new();
    for pin in Pin::ALL.into_iter().filter(|pin| pins.is_empty() || pins.contains(pin)) {
        eprintln!("checking {}...", pin.name());
        let outcome = discover(pin, &before, &upstream).unwrap_or_else(|e| Outcome::Skipped(pin, e));
        if let Outcome::Bump(update) = &outcome {
            stage(repo, update)?;
        }
        outcomes.push(outcome);
    }
    Ok(outcomes)
}

// -----------------------------------------------------------------------------
// Discovery
// -----------------------------------------------------------------------------

/// The result of checking one pin against upstream.
enum Outcome {
    /// Already at the newest eligible release.
    Current(Pin, String),

    /// A newer release is available.
    Bump(Update),

    /// Could not be checked, or the newest release is not eligible.
    Skipped(Pin, String),
}

/// A pin moving from `old` to `new`.
struct Update {
    /// Which pin.
    pin: Pin,

    /// Current value.
    old: String,

    /// New value.
    new: String,

    /// sha256 of the new artifact (the Claude Code manifest's inline checksum
    /// or the checksum fixture's digest); `None` for the vLLM image, whose
    /// digest is part of the value.
    sha256: Option<String>,
}

/// A discovered release: the new value and its artifact's sha256.
type Discovered = (String, Option<String>);

/// Find the newest eligible upstream release for `pin`.
fn discover(pin: Pin, manifests: &Manifests, upstream: &Upstream) -> Result<Outcome, String> {
    let old = pin.current(manifests)?.to_owned();
    let discovered = match pin {
        Pin::Claude => discover_claude(manifests, &old, upstream),
        Pin::Opencode => discover_opencode(&manifests.opencode.opencode, &old, upstream),
        Pin::OpencodeV2 => discover_opencode(&manifests.opencode_v2.opencode, &old, upstream),
        Pin::Pi => discover_pi(manifests, upstream),
        Pin::Codex => discover_codex(manifests, upstream),
        Pin::Vllm => discover_vllm(&old, upstream),
    };
    Ok(match discovered? {
        Err(reason) => Outcome::Skipped(pin, reason),
        Ok((new, _)) if new == old => Outcome::Current(pin, old),
        Ok((new, sha256)) => Outcome::Bump(Update { pin, old, new, sha256 }),
    })
}

/// Claude Code: npm `latest` within `major_series`; the checksum comes from
/// the release's own manifest.
fn discover_claude(
    manifests: &Manifests,
    old: &str,
    upstream: &Upstream,
) -> Result<Result<Discovered, String>, String> {
    let version = upstream.npm_latest("@anthropic-ai/claude-code", manifests.claude.claude_code.major_series)?;
    let sha256 = (version != old)
        .then(|| upstream.claude_checksum(&version))
        .transpose()?;
    Ok(Ok((version, sha256)))
}

/// `OpenCode`: npm `latest` within `major_series`. npm publishes only sha512
/// integrity, so the platform tarball is streamed through sha256.
fn discover_opencode(client: &OpenCode, old: &str, upstream: &Upstream) -> Result<Result<Discovered, String>, String> {
    let version = upstream.npm_latest(&client.npm_package, client.major_series)?;
    if version == old {
        return Ok(Ok((version, None)));
    }
    let package = client
        .platform_package
        .clone()
        .unwrap_or_else(|| format!("opencode-{}", client.platform));
    let tarball = package.rsplit('/').next().unwrap_or(&package);
    let sha256 = upstream.sha256(&format!("{NPM_REGISTRY}/{package}/-/{tarball}-{version}.tgz"))?;
    Ok(Ok((version, Some(sha256))))
}

/// pi: the newest GitHub release within `major_series`.
fn discover_pi(manifests: &Manifests, upstream: &Upstream) -> Result<Result<Discovered, String>, String> {
    let pi = &manifests.pi.pi;
    let Some(release) = upstream.github_newest("earendil-works/pi", "v", Some(pi.major_series))? else {
        return Ok(Err(format!(
            "no stable {}.x release among the recent ones; re-validate the manifest by hand",
            pi.major_series
        )));
    };
    let sha256 = release.asset_sha256(&pi.archive)?.to_owned();
    Ok(Ok((release.version, Some(sha256))))
}

/// Codex: the newest GitHub release under `release_tag_prefix`.
fn discover_codex(manifests: &Manifests, upstream: &Upstream) -> Result<Result<Discovered, String>, String> {
    let codex = &manifests.codex.codex;
    let Some(release) = upstream.github_newest("openai/codex", &codex.release_tag_prefix, None)? else {
        return Ok(Err(format!(
            "no stable `{}` release among the recent ones",
            codex.release_tag_prefix
        )));
    };
    let sha256 = release.asset_sha256(&codex.archive)?.to_owned();
    Ok(Ok((release.version, Some(sha256))))
}

/// The newest vLLM release's image reference, or a reason to skip it.
///
/// The repository part of the current reference is kept; only the tag and
/// digest move. A release whose image is not on Docker Hub yet is skipped, not
/// failed: the next scheduled run picks it up.
fn discover_vllm(old: &str, upstream: &Upstream) -> Result<Result<Discovered, String>, String> {
    let (name, _) = old
        .split_once('@')
        .and_then(|(name_tag, _)| name_tag.rsplit_once(':'))
        .ok_or_else(|| format!("vLLM image `{old}` is not `<repo>:<tag>@sha256:<digest>`"))?;
    let Some(hub_repository) = name.strip_prefix("docker.io/") else {
        return Ok(Err(format!("{name} is not a Docker Hub image")));
    };
    let Some(release) = upstream.github_newest("vllm-project/vllm", "v", None)? else {
        return Ok(Err("no stable release among the recent ones".to_owned()));
    };
    Ok(match upstream.docker_hub_digest(hub_repository, &release.tag)? {
        Some(digest) => Ok((format!("{name}:{}@{digest}", release.tag), None)),
        None => Err(format!("{name}:{} is not published on Docker Hub yet", release.tag)),
    })
}

// -----------------------------------------------------------------------------
// Staging
// -----------------------------------------------------------------------------

/// Stage every file change one update needs.
fn stage(repo: &mut Repo, update: &Update) -> Result<(), String> {
    let old_fixture = fixture_of(repo, update.pin)?;
    for site in update.pin.sites() {
        let contents = repo.read(site.path)?;
        let rewritten = site.rewrite(&contents, &update.old, &update.new)?;
        repo.stage(site.path, rewritten);
    }
    let Some(sha256) = &update.sha256 else {
        return Ok(());
    };
    if update.pin == Pin::Claude {
        let manifests = Manifests::load(repo)?;
        let contents = repo.read(manifest::CLAUDE)?;
        let old_sha256 = &manifests.claude.claude_code.sha256;
        repo.stage(
            manifest::CLAUDE,
            CLAUDE_SHA256_SITE.rewrite(&contents, old_sha256, sha256)?,
        );
        return Ok(());
    }
    // The fixture name and artifact derive from the rewritten manifest.
    let (Some(old_fixture), Some(new_fixture)) = (old_fixture, fixture_of(repo, update.pin)?) else {
        return Err(format!("{} has no checksum fixture", update.pin.name()));
    };
    repo.stage_removal(&old_fixture.path());
    repo.stage(&new_fixture.path(), format!("{sha256}  {}\n", new_fixture.artifact));
    Ok(())
}

/// The checksum fixture the staged manifests require for `pin`, if any.
fn fixture_of(repo: &Repo, pin: Pin) -> Result<Option<sites::Fixture>, String> {
    let manifests = Manifests::load(repo)?;
    Ok(sites::fixtures(&manifests)
        .into_iter()
        .find(|fixture| fixture.pin == pin))
}

// -----------------------------------------------------------------------------
// Report
// -----------------------------------------------------------------------------

/// Directories scanned for leftover mentions of an old version.
const CLAIM_ROOTS: &[&str] = &[
    "docs",
    "tests/integration/fixtures",
    "tests/integration/tests/suite",
    ".github/workflows",
    "vllm",
];

/// Render the markdown report: what moved, what to re-validate, and every
/// remaining line that still names an old version.
fn render_report(repo: &Repo, outcomes: &[Outcome]) -> String {
    let updates: Vec<&Update> = outcomes
        .iter()
        .filter_map(|outcome| match outcome {
            Outcome::Bump(update) => Some(update),
            _ => None,
        })
        .collect();
    let mut out = String::from("## Pinned client bumps\n\n");
    if updates.is_empty() {
        out.push_str("All checked pins are current.\n");
    } else {
        render_table(&mut out, &updates);
    }
    render_status(&mut out, outcomes);
    for update in &updates {
        render_checklist(&mut out, repo, update);
    }
    out
}

/// The old → new table and the review instructions.
fn render_table(out: &mut String, updates: &[&Update]) {
    out.push_str("| Pin | From | To |\n|---|---|---|\n");
    for update in updates {
        writeln!(
            out,
            "| {} | `{}` | [`{}`]({}) |",
            update.pin.name(),
            update.old,
            update.new,
            release_link(update.pin, &update.new)
        )
        .unwrap();
    }
    out.push_str(
        "\nGenerated by `cargo xtask bump-pins`. `cargo xtask check-pins` passes on the bumped tree. Nothing \
         behavioral was changed: work through the checklist below, run the GPU acceptance lanes \
         (`vllm-full-suite` label), and only then mark this ready for review.\n",
    );
}

/// List the pins that were current or skipped.
fn render_status(out: &mut String, outcomes: &[Outcome]) {
    let mut current = Vec::new();
    let mut skipped = Vec::new();
    for outcome in outcomes {
        match outcome {
            Outcome::Current(pin, value) => current.push(format!("{} `{value}`", pin.name())),
            Outcome::Skipped(pin, reason) => skipped.push(format!("- {}: {reason}", pin.name())),
            Outcome::Bump(_) => {},
        }
    }
    if !current.is_empty() {
        writeln!(out, "\nAlready current: {}.", current.join(", ")).unwrap();
    }
    if !skipped.is_empty() {
        writeln!(out, "\nNot bumped:\n\n{}", skipped.join("\n")).unwrap();
    }
}

/// The re-validation checklist and leftover old-version mentions for one bump.
fn render_checklist(out: &mut String, repo: &Repo, update: &Update) {
    let (old, new) = (
        display_version(update.pin, &update.old),
        display_version(update.pin, &update.new),
    );
    writeln!(out, "\n### {} `{old}` → `{new}`\n", update.pin.name()).unwrap();
    for item in checklist(update.pin, &new) {
        writeln!(out, "- [ ] {item}").unwrap();
    }
    let mentions = mentions_of(repo, &old);
    if !mentions.is_empty() {
        writeln!(
            out,
            "\nLines still naming `{old}`. Most are behavioral claims; update each only after re-validating it \
             against `{new}`:\n"
        )
        .unwrap();
        for mention in mentions {
            writeln!(out, "- {mention}").unwrap();
        }
    }
}

/// The bare version a pin's value names (the vLLM image's tag without `v`).
pub(crate) fn display_version(pin: Pin, value: &str) -> String {
    if pin != Pin::Vllm {
        return value.to_owned();
    }
    value
        .split_once('@')
        .and_then(|(name_tag, _)| name_tag.rsplit_once(':'))
        .map_or(value, |(_, tag)| tag.trim_start_matches('v'))
        .to_owned()
}

/// Where a reviewer reads what changed in `version`.
fn release_link(pin: Pin, value: &str) -> String {
    let version = display_version(pin, value);
    match pin {
        Pin::Claude => format!("https://www.npmjs.com/package/@anthropic-ai/claude-code/v/{version}"),
        Pin::Opencode => format!("https://www.npmjs.com/package/opencode-ai/v/{version}"),
        Pin::OpencodeV2 => format!("https://www.npmjs.com/package/@opencode/cli/v/{version}"),
        Pin::Pi => format!("https://github.com/earendil-works/pi/releases/tag/v{version}"),
        Pin::Codex => format!("https://github.com/openai/codex/releases/tag/rust-v{version}"),
        Pin::Vllm => format!("https://github.com/vllm-project/vllm/releases/tag/v{version}"),
    }
}

/// Behavioral pins to re-validate by hand after bumping `pin` to `new`.
pub(crate) fn checklist(pin: Pin, new: &str) -> Vec<String> {
    let mut items: Vec<String> = checklist_items(pin).iter().map(|item| (*item).to_owned()).collect();
    if pin == Pin::Codex {
        items.push(format!(
            "Record `tests/integration/fixtures/openai/responses/websocket-codex-{new}.json` against the new build; \
             `codex_pin::websocket_recording_matches_the_pinned_version` stays red until then."
        ));
    }
    items
}

/// The fixed checklist items for each pin.
const fn checklist_items(pin: Pin) -> &'static [&'static str] {
    match pin {
        Pin::Claude => &[
            "Re-validate `[claude_code.launch]` (exposed tools, permission modes, flags) against the new build.",
            "Re-measure the 15,048-token compaction reserve (`CLAUDE_COMPACTION_RESERVE` in `xtask/src/pins/check.rs` \
             and the manifest comment).",
            "Check the CPU lane: the npm build must report the same version.",
        ],
        Pin::Opencode => &[
            "Re-capture `[opencode.tools] registered` from the new build.",
            "Re-check exit-code semantics (`exit_code_meaningful`) and stdin behavior.",
            "Re-check `options.headers` credential resolution, and the schema-key allowlists.",
        ],
        Pin::OpencodeV2 => &[
            "Re-capture `[opencode.tools] registered` from the new build.",
            "Re-check `rejects_unknown_keys`, the auth plugin hook, and `--` prompt handling.",
            "Re-check the schema-key allowlists (`PROVIDER_KEYS`, `MODEL_KEYS`).",
        ],
        Pin::Pi => &[
            "Re-capture `[pi.tools] registered` from the new build.",
            "Re-check JSON-mode exit codes, stdin behavior, and `unresolved_header_error`.",
            "Update \"verified against pi\" in the client guide; `pi::documented_models_json_matches_fixture` stays \
             red until then.",
        ],
        Pin::Codex => &["Re-check the observed behaviors cited in `codex_http.rs` against the new build."],
        Pin::Vllm => &[
            "Review version-conditioned xfails and skips in the live SDK and suite tests.",
            "Re-check the client guide's vLLM claims and the CUDA image notes in `vllm-integration.yaml`.",
            "Rebuild the baked-model image (`vllm-gpu-container.yaml`).",
        ],
    }
}

/// Every `path:line — text` under [`CLAIM_ROOTS`] that names `version`.
fn mentions_of(repo: &Repo, version: &str) -> Vec<String> {
    let root = workspace_root();
    let mut files = Vec::new();
    for dir in CLAIM_ROOTS {
        collect_text_files(&root, &root.join(dir), &mut files);
    }
    files.sort();
    let mut mentions = Vec::new();
    for path in files {
        let Ok(contents) = repo.read(&path) else {
            continue;
        };
        for (index, line) in contents.lines().enumerate() {
            if names_version(line, version) {
                let text: String = line.trim().chars().take(160).collect();
                mentions.push(format!("`{path}:{}` — {text}", index + 1));
            }
        }
    }
    mentions
}

/// Recursively collect repository-relative text files under `dir`.
fn collect_text_files(root: &std::path::Path, dir: &std::path::Path, files: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_text_files(root, &path, files);
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let text = ["md", "toml", "rs", "yaml", "yml"]
            .iter()
            .any(|ext| name.ends_with(&format!(".{ext}")))
            || name == "Containerfile";
        if let (true, Ok(relative)) = (text, path.strip_prefix(root)) {
            files.push(relative.to_string_lossy().into_owned());
        }
    }
}

/// Whether `line` names `version` as a whole version, not as part of a longer
/// one (`1.4.0` must not match `11.4.0` or `1.4.0.1`).
pub(crate) fn names_version(line: &str, version: &str) -> bool {
    line.match_indices(version).any(|(start, _)| {
        let before = line.get(..start).and_then(|s| s.chars().next_back());
        let after = line.get(start + version.len()..).unwrap_or_default();
        let mut rest = after.chars();
        let next = rest.next();
        let continues = match next {
            Some(c) if c.is_ascii_digit() => true,
            Some('.') => rest.next().is_some_and(|c| c.is_ascii_digit()),
            _ => false,
        };
        !before.is_some_and(|c| c.is_ascii_digit() || c == '.') && !continues
    })
}
