// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Cross-file pin invariants.
//!
//! Every check pushes a message instead of returning early, so one run reports
//! every broken invariant at once.

use super::{
    Repo,
    manifest::{self, Compaction, Manifests, OpenCodeManifest, Pi, Vllm},
    sites::{self, Pin},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Tokens Claude Code holds back from its window: 2,048 output plus 13,000
/// compaction (measured on 2.1.267; re-validate on a bump).
const CLAUDE_COMPACTION_RESERVE: u64 = 15_048;

/// Approximate bytes per token, for sizing the compaction ballast.
const BYTES_PER_TOKEN: f64 = 4.0;

// -----------------------------------------------------------------------------
// Entry Point
// -----------------------------------------------------------------------------

/// Run every invariant and return the violations; empty means all hold.
pub(crate) fn check(repo: &Repo, manifests: &Manifests) -> Vec<String> {
    let mut errors = Vec::new();
    check_sites(repo, manifests, &mut errors);
    check_fixtures(repo, manifests, &mut errors);
    check_vllm_image(manifests, &mut errors);
    check_claude(manifests, &mut errors);
    check_opencode(manifests, &mut errors);
    check_pi(repo, manifests, &mut errors);
    check_codex(manifests, &mut errors);
    for (path, gateway, matrix) in [
        (
            manifest::OPENCODE,
            &manifests.opencode.gateway,
            &manifests.opencode.startup_matrix,
        ),
        (
            manifest::OPENCODE_V2,
            &manifests.opencode_v2.gateway,
            &manifests.opencode_v2.startup_matrix,
        ),
        (manifest::PI, &manifests.pi.gateway, &manifests.pi.startup_matrix),
    ] {
        check_routes(
            repo,
            path,
            &gateway.example_config,
            &matrix.allowed_prefixes,
            &mut errors,
        );
    }
    errors
}

// -----------------------------------------------------------------------------
// Sites and Fixtures
// -----------------------------------------------------------------------------

/// Every site in the table carries its manifest's current value.
fn check_sites(repo: &Repo, manifests: &Manifests, errors: &mut Vec<String>) {
    for pin in Pin::ALL {
        let value = match pin.current(manifests) {
            Ok(value) => value,
            Err(e) => {
                errors.push(e);
                continue;
            },
        };
        for site in pin.sites() {
            let result = repo.read(site.path).and_then(|contents| site.expect(&contents, value));
            if let Err(e) = result {
                errors.push(format!("{} pin {value}: {e}", pin.name()));
            }
        }
    }
}

/// Each client's checksum fixture exists for the pinned version, is in
/// `sha256sum` format, names the artifact the lanes download, and has no
/// stale sibling left behind by an earlier version.
fn check_fixtures(repo: &Repo, manifests: &Manifests, errors: &mut Vec<String>) {
    for fixture in sites::fixtures(manifests) {
        let path = fixture.path();
        match repo.read(&path) {
            Ok(line) => {
                if let Err(e) = check_checksum_line(&line, &fixture.artifact) {
                    errors.push(format!("{path}: {e}"));
                }
            },
            Err(_) => errors.push(format!("{} needs a checksum fixture at {path}", fixture.pin.name())),
        }
        match repo.list(fixture.dir) {
            Ok(files) => {
                for file in files {
                    if file.ends_with(".sha256") && file != fixture.file {
                        errors.push(format!(
                            "{}/{file} is stale: the pinned {} fixture is {}",
                            fixture.dir,
                            fixture.pin.name(),
                            fixture.file
                        ));
                    }
                }
            },
            Err(e) => errors.push(e),
        }
    }
}

/// Validate one `<hex>  <artifact>` checksum line.
pub(crate) fn check_checksum_line(line: &str, artifact: &str) -> Result<(), String> {
    let (digest, file) = line
        .trim_end_matches('\n')
        .split_once("  ")
        .ok_or("must be in `sha256sum` format: `<sha256>  <file>`")?;
    if !is_sha256(digest) {
        return Err(format!("`{digest}` is not a sha256 hex digest"));
    }
    if file != artifact {
        return Err(format!("names `{file}`, but the lanes download `{artifact}`"));
    }
    Ok(())
}

/// Whether `value` is 64 lowercase hex characters.
pub(crate) fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Both `[vllm]` blocks pin the same `tag@sha256:<digest>` image; the site
/// table then holds every workflow and the Containerfile to it.
fn check_vllm_image(manifests: &Manifests, errors: &mut Vec<String>) {
    let Some(opencode) = manifests.opencode.vllm.as_ref() else {
        return;
    };
    let image = &opencode.image;
    if image != &manifests.claude.vllm.image {
        errors.push(format!(
            "{} and {} must pin the same vLLM image: `{image}` vs `{}`",
            manifest::OPENCODE,
            manifest::CLAUDE,
            manifests.claude.vllm.image
        ));
    }
    let digest_form = image
        .split_once("@sha256:")
        .is_some_and(|(name, digest)| name.contains(':') && is_sha256(digest));
    if !digest_form {
        errors.push(format!(
            "vLLM image `{image}` must have the form `<repo>:<tag>@sha256:<digest>`"
        ));
    }
}

// -----------------------------------------------------------------------------
// Claude Code
// -----------------------------------------------------------------------------

/// Claude Code manifest invariants: release pins, context agreement, and the
/// compaction scenario's arithmetic.
fn check_claude(manifests: &Manifests, errors: &mut Vec<String>) {
    let path = manifest::CLAUDE;
    let claude = &manifests.claude.claude_code;
    let vllm = &manifests.claude.vllm;
    let launch = &claude.launch;

    check_major(path, "claude_code", &claude.version, claude.major_series, errors);
    if !is_sha256(&claude.sha256) {
        errors.push(format!("{path}: claude_code.sha256 must be a sha256 hex digest"));
    }

    let context = vllm.context_length;
    match vllm.flag_context() {
        Ok(flag) if flag != context => errors.push(format!(
            "{path}: vllm.context_length must match --max-model-len in vllm.launch.flags"
        )),
        Ok(_) => {},
        Err(e) => errors.push(format!("{path}: {e}")),
    }
    if context < launch.auto_minimum_backend_context_tokens {
        errors.push(format!(
            "{path}: vLLM context {context} is below Claude auto-mode minimum {}",
            launch.auto_minimum_backend_context_tokens
        ));
    }
    if context < launch.max_context_tokens {
        errors.push(format!(
            "{path}: vLLM context {context} is below Claude client context {}",
            launch.max_context_tokens
        ));
    }
    check_compaction(path, manifests, errors);
}

/// The compaction scenario runs the client with a LOWERED window so it
/// auto-compacts mid-task.
///
/// The window must stay positive and strictly below the real backend window,
/// or the client never crosses its threshold and the scenario proves nothing.
fn check_compaction(path: &str, manifests: &Manifests, errors: &mut Vec<String>) {
    let context = manifests.claude.vllm.context_length;
    let compaction = &manifests.claude.claude_code.launch.compaction;
    let window = compaction.max_context_tokens;
    if !(0 < window && window < context) {
        errors.push(format!("{path}: compaction window {window} must be in (0, {context})"));
    }
    match window.checked_sub(CLAUDE_COMPACTION_RESERVE).filter(|h| *h > 0) {
        Some(headroom) => check_ballast(path, compaction, headroom, errors),
        None => errors.push(format!(
            "{path}: compaction window {window} must exceed the {CLAUDE_COMPACTION_RESERVE}-token Claude Code \
             reserve so the client has usable headroom"
        )),
    }
}

/// A single ballast read must fit well under the post-reserve headroom, or one
/// post-compaction read immediately re-crosses the threshold and the client's
/// rapid-refill breaker aborts the run. Total ballast must still exceed the
/// threshold so compaction fires mid-task.
fn check_ballast(path: &str, compaction: &Compaction, headroom: u64, errors: &mut Vec<String>) {
    #[expect(clippy::cast_precision_loss, reason = "token estimates are approximate")]
    let (per_chapter, chapters, headroom_tokens) = (
        compaction.ballast_bytes as f64 / BYTES_PER_TOKEN,
        compaction.ballast_chapters as f64,
        headroom as f64,
    );
    if per_chapter * 2.0 >= headroom_tokens {
        errors.push(format!(
            "{path}: a single ballast chapter (~{per_chapter:.0} tokens) must be well under the ~{headroom}-token \
             post-compaction headroom to avoid the rapid-refill breaker"
        ));
    }
    if per_chapter * chapters <= headroom_tokens {
        errors.push(format!(
            "{path}: total ballast (~{:.0} tokens) must exceed the ~{headroom}-token threshold so compaction fires \
             mid-task",
            per_chapter * chapters
        ));
    }
}

// -----------------------------------------------------------------------------
// OpenCode
// -----------------------------------------------------------------------------

/// `OpenCode` invariants for both majors.
///
/// Each major has its own manifest and credential recipe. Every manifest is
/// validated against the major it declares AND the major its path is reserved
/// for, so a 2.x version dropped into the 1.x manifest (or the reverse) still
/// fails: crossing the boundary needs a rewritten client configuration and a
/// re-validated manifest, not a bump.
fn check_opencode(manifests: &Manifests, errors: &mut Vec<String>) {
    for (lane, path, data) in [
        (1, manifest::OPENCODE, &manifests.opencode),
        (2, manifest::OPENCODE_V2, &manifests.opencode_v2),
    ] {
        let client = &data.opencode;
        check_major(path, "opencode", &client.version, client.major_series, errors);
        if major(&client.version) != Some(lane) {
            errors.push(format!(
                "{path}: the OpenCode acceptance lane is pinned to the {lane}.x client configuration; version {} \
                 needs its own manifest and a rewritten client configuration",
                client.version
            ));
        }
        // The prompt is positional on both majors; the flag that would swallow
        // it differs (-p was the 1.x server password).
        if !client.launch.prompt_is_positional {
            errors.push(format!("{path}: opencode.launch.prompt_is_positional must stay true"));
        }
    }
    check_opencode_v1(&manifests.opencode, errors);
    check_opencode_v2(manifests, errors);
}

/// The 1.x manifest owns the shared Chat Completions backend pins.
fn check_opencode_v1(data: &OpenCodeManifest, errors: &mut Vec<String>) {
    let path = manifest::OPENCODE;
    let client = &data.opencode;
    let Some(vllm) = data.vllm.as_ref() else {
        errors.push(format!(
            "{path}: must declare [vllm], the backend pins every Chat Completions lane shares"
        ));
        return;
    };

    // The client window, the manifest, and the server flag must agree, or the
    // client advertises a context the backend will not honor.
    let context = client.provider.context;
    match vllm.flag_context() {
        Ok(flag) if !(context == vllm.context_length && context == flag) => errors.push(format!(
            "{path}: OpenCode context {context}, vllm.context_length {}, and --max-model-len {flag} must all agree",
            vllm.context_length
        )),
        Ok(_) => {},
        Err(e) => errors.push(format!("{path}: {e}")),
    }
    check_client_window(
        &ClientWindow {
            path,
            table: "opencode.provider",
            model_id: &client.provider.model_id,
            context: None,
            output: client.provider.output,
        },
        data,
        errors,
    );
    check_opencode_v1_backend(path, data, vllm, errors);
}

/// The shared backend serves tool calls and, for a release install, is
/// reproducible.
fn check_opencode_v1_backend(path: &str, data: &OpenCodeManifest, vllm: &Vllm, errors: &mut Vec<String>) {
    // Tool-calling must be served, or the tool-call scenario silently degrades
    // to a plain text turn and proves nothing.
    if vllm.tool_call_parser.is_empty() || !vllm.launch.flags.iter().any(|f| f == "--enable-auto-tool-choice") {
        errors.push(format!(
            "{path}: the OpenCode backend must serve tool calls: set vllm.tool_call_parser and pass \
             --enable-auto-tool-choice"
        ));
    }

    // A github-release install is only reproducible with both pins.
    let client = &data.opencode;
    if client.install == "github-release" {
        for (key, value) in [("url", &client.url), ("sha256", &client.sha256)] {
            if value.is_empty() || value.contains("TBD") {
                errors.push(format!(
                    "{path}: install=github-release requires a concrete opencode.{key}"
                ));
            }
        }
    }
}

/// The 2.x scenarios run against the 1.x lane's vLLM container, so its backend
/// pins live once, in the 1.x manifest.
fn check_opencode_v2(manifests: &Manifests, errors: &mut Vec<String>) {
    let path = manifest::OPENCODE_V2;
    let v2 = &manifests.opencode_v2;
    let provider = &v2.opencode.provider;
    if v2.vllm.is_some() {
        errors.push(format!(
            "{path} must not declare [vllm]: the 2.x scenarios share the 1.x lane's vLLM container, whose pins \
             live in the 1.x manifest"
        ));
    }
    check_client_window(
        &ClientWindow {
            path,
            table: "opencode.provider",
            model_id: &provider.model_id,
            context: Some(provider.context),
            output: provider.output,
        },
        &manifests.opencode,
        errors,
    );
    // Both 2.x lanes stage the registry tarball verified against the committed
    // checksum; that is what makes them byte-identical.
    if v2.opencode.install != "registry-tarball" {
        errors.push(format!(
            "{path}: OpenCode 2.x must install from the verified registry tarball"
        ));
    }
    let expected = format!("opencode v{}", v2.opencode.version);
    if v2.opencode.version_output.as_deref() != Some(expected.as_str()) {
        errors.push(format!("{path}: opencode.version_output must be `{expected}`"));
    }
}

/// The model and window a Chat Completions client advertises.
#[derive(Clone, Copy)]
struct ClientWindow<'a> {
    /// Manifest path, for messages.
    path: &'a str,

    /// Provider table name, for messages.
    table: &'a str,

    /// Model the client addresses.
    model_id: &'a str,

    /// Context window the client advertises, when it declares one.
    context: Option<u64>,

    /// Output budget the client advertises.
    output: u64,
}

/// A Chat Completions client addresses the model the shared backend serves and
/// does not advertise a window it will not honor.
///
/// `context` is `None` for the 1.x client, whose window is checked against the
/// server flag separately.
fn check_client_window(client: &ClientWindow<'_>, backend: &OpenCodeManifest, errors: &mut Vec<String>) {
    let ClientWindow {
        path,
        table,
        model_id,
        context,
        output,
    } = *client;
    let Some(vllm) = backend.vllm.as_ref() else {
        return;
    };
    if model_id != vllm.served_model_name {
        errors.push(format!(
            "{path}: {table} model {model_id} must match the served model {}",
            vllm.served_model_name
        ));
    }
    if let Some(context) = context.filter(|c| *c != vllm.context_length) {
        errors.push(format!(
            "{path}: {table} context {context} must equal vllm.context_length {}",
            vllm.context_length
        ));
    }
    if output >= vllm.context_length {
        errors.push(format!(
            "{path}: {table} output budget must be below the context window"
        ));
    }
}

// -----------------------------------------------------------------------------
// pi
// -----------------------------------------------------------------------------

/// pi runs against the same Chat Completions backend as `OpenCode`, so its
/// backend pins live once, in the 1.x `OpenCode` manifest.
fn check_pi(repo: &Repo, manifests: &Manifests, errors: &mut Vec<String>) {
    let path = manifest::PI;
    let data = &manifests.pi;
    let pi = &data.pi;
    let provider = &pi.provider;
    if data.vllm.is_some() || data.gateway.backend_pins.as_deref() != Some("opencode-cli/pin.toml") {
        errors.push(format!(
            "{path} must not declare [vllm]: pi shares the backend pins in opencode-cli/pin.toml, and \
             gateway.backend_pins must say so"
        ));
    }
    check_major(path, "pi", &pi.version, pi.major_series, errors);
    check_client_window(
        &ClientWindow {
            path,
            table: "pi.provider",
            model_id: &provider.model_id,
            context: Some(provider.context_window),
            output: provider.max_tokens,
        },
        &manifests.opencode,
        errors,
    );
    check_pi_release(path, pi, errors);
    check_pi_models(repo, manifests, errors);
}

/// pi is supported over Chat Completions only, and both pi lanes stage the
/// release archive verified against the committed checksum; that is what
/// makes them byte-identical.
fn check_pi_release(path: &str, pi: &Pi, errors: &mut Vec<String>) {
    if pi.provider.api != "openai-completions" {
        errors.push(format!("{path}: pi is supported over openai-completions only"));
    }
    let archive = format!("pi-{}.tar.gz", pi.platform);
    let url = format!(
        "https://github.com/earendil-works/pi/releases/download/v{}/{archive}",
        pi.version
    );
    if pi.install != "github-release" || pi.url != url || pi.archive != archive {
        errors.push(format!(
            "{path}: pi must install from the verified release archive {url}"
        ));
    }
}

/// The documented `models.json` is what both pi lanes run; it must agree with
/// the manifest.
fn check_pi_models(repo: &Repo, manifests: &Manifests, errors: &mut Vec<String>) {
    let pi = &manifests.pi.pi;
    let provider = &pi.provider;
    let dir = manifest::PI.rsplit_once('/').map_or("", |(dir, _)| dir);
    let models_path = format!("{dir}/{}", pi.auth.fixture);
    let models: serde_json::Value = match repo
        .read(&models_path)
        .and_then(|text| serde_json::from_str(&text).map_err(|e| format!("{models_path}: {e}")))
    {
        Ok(models) => models,
        Err(e) => {
            errors.push(e);
            return;
        },
    };
    let model = models.pointer(&format!("/providers/{}/models/0", provider.id));
    let field = |key: &str| model.and_then(|m| m.get(key));
    let agrees = field("id").and_then(serde_json::Value::as_str) == Some(provider.model_id.as_str())
        && field("contextWindow").and_then(serde_json::Value::as_u64) == Some(provider.context_window)
        && field("maxTokens").and_then(serde_json::Value::as_u64) == Some(provider.max_tokens);
    if !agrees {
        errors.push(format!("{models_path} disagrees with {}", manifest::PI));
    }
}

// -----------------------------------------------------------------------------
// Codex
// -----------------------------------------------------------------------------

/// Codex manifest invariants.
fn check_codex(manifests: &Manifests, errors: &mut Vec<String>) {
    let path = manifest::CODEX;
    let codex = &manifests.codex.codex;
    if codex.install != "github-release" || codex.release_tag_prefix.is_empty() {
        errors.push(format!(
            "{path}: Codex must install from a GitHub release with a non-empty release_tag_prefix"
        ));
    }
    if codex.archive != format!("codex-{}.tar.gz", codex.platform) {
        errors.push(format!("{path}: codex.archive must be the platform release archive"));
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// The leading numeric component of a dotted version.
pub(crate) fn major(version: &str) -> Option<u64> {
    version.split('.').next()?.parse().ok()
}

/// A manifest's `major_series` tripwire agrees with its version.
fn check_major(path: &str, table: &str, version: &str, major_series: u64, errors: &mut Vec<String>) {
    if major(version) != Some(major_series) {
        errors.push(format!(
            "{path}: {table}.major_series {major_series} disagrees with version {version}; crossing a major must \
             re-validate the manifest"
        ));
    }
}

/// Every prefix a client may reach has an explicit route in its example config,
/// so the client path can never silently regress onto the catch-all that exists
/// for health checks and startup probes.
fn check_routes(repo: &Repo, path: &str, example: &str, prefixes: &[String], errors: &mut Vec<String>) {
    let example_path = format!("examples/configs/{example}");
    let text = match repo.read(&example_path) {
        Ok(text) => text,
        Err(e) => {
            errors.push(format!("{path}: {e}"));
            return;
        },
    };
    for prefix in prefixes {
        if !text.contains(&format!("- path: \"{prefix}\"")) {
            errors.push(format!(
                "{example_path} must declare an explicit route for {prefix}; relying on the catch-all is not a \
                 supported client path"
            ));
        }
    }
}
