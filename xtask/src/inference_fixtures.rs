// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Praxis Contributors

//! Record, import, and validate two-sided inference fixtures.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    future::Future,
    io::{Read as _, Write as _},
    path::{Path, PathBuf},
};

use clap::Parser;
use praxis_test_utils::inference_fixture::{
    CoverageReport, CoverageStatus, FixtureError, FixtureProvenance, InferenceScenario, ProvenanceKind, ProviderTarget,
    RedactionRules, ScenarioRunner, ScenarioSnapshot, WireFixture, check_coverage, discover_scenario_snapshots,
    import_external_recording, validate_commit_safe_with_rules,
};
use reqwest::{Url, header::HeaderValue};
use serde::de::{self, MapAccess, Visitor};
use tempfile::NamedTempFile;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Default root containing inference scenario declarations and recordings.
const DEFAULT_ROOT: &str = "tests/integration/fixtures/inference";

/// Default provider base URL for OpenAI recording.
const OPENAI_BASE_URL: &str = "https://api.openai.com";

/// Exact first-party base URL authorized to receive an Anthropic credential.
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";

/// Default provider base URL for local OpenAI-compatible recording.
const COMPATIBLE_BASE_URL: &str = "http://127.0.0.1:8000";

/// Environment variable containing the required OpenAI credential.
const OPENAI_API_KEY: &str = "OPENAI_API_KEY";

/// Environment variable containing the required Anthropic credential.
const ANTHROPIC_API_KEY: &str = "ANTHROPIC_API_KEY";

/// Anthropic API version sent by native recording requests.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Environment variable optionally authorizing non-OpenAI providers.
const INFERENCE_PROVIDER_API_KEY: &str = "INFERENCE_PROVIDER_API_KEY";

/// Maximum external recording size accepted by the importer.
const MAX_EXTERNAL_RECORDING_BYTES: usize = 16_777_216; // 16 MiB

/// Maximum custom literal-redaction document size.
const MAX_REDACTIONS_FILE_BYTES: usize = 1_048_576; // 1 MiB

// -----------------------------------------------------------------------------
// CLI Arguments
// -----------------------------------------------------------------------------

/// CLI arguments for `cargo xtask record-inference`.
#[derive(Parser)]
pub(crate) struct RecordArgs {
    /// Stable scenario ID declared beneath the selected root.
    #[arg(long)]
    scenario: String,

    /// Provider provenance and credential policy name.
    #[arg(long, default_value = "vllm")]
    provider: String,

    /// HTTP(S) provider base URL; defaults according to the provider name.
    #[arg(long)]
    provider_base_url: Option<String>,

    /// Provider model bound into exact `${MODEL}` scenario values.
    #[arg(long)]
    model: String,

    /// Inference fixture root containing `scenarios/`.
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,

    /// Explicit new output path instead of the provider/scenario default.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Bounded JSON object containing literal source/replacement strings.
    #[arg(long)]
    redactions_file: Option<PathBuf>,
}

/// CLI arguments for `cargo xtask import-inference`.
#[derive(Parser)]
pub(crate) struct ImportArgs {
    /// External request-hashed provider recording to import.
    #[arg(long)]
    recording: PathBuf,

    /// Stable scenario ID declared beneath the selected root.
    #[arg(long)]
    scenario: String,

    /// Provider provenance recorded in the final fixture.
    #[arg(long, default_value = "openai")]
    provider: String,

    /// Mark this import as originating from a controlled synthetic backend.
    #[arg(long)]
    controlled_synthetic: bool,

    /// Inference fixture root containing `scenarios/`.
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,

    /// Explicit new output path instead of the provider/scenario default.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Bounded JSON object containing literal source/replacement strings.
    #[arg(long)]
    redactions_file: Option<PathBuf>,
}

/// CLI arguments for `cargo xtask check-inference`.
#[derive(Parser)]
pub(crate) struct CheckArgs {
    /// Inference fixture root containing coverage, scenarios, and recordings.
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
}

// -----------------------------------------------------------------------------
// Entry Points
// -----------------------------------------------------------------------------

/// Record one scenario against a configured live provider.
pub(crate) fn run_record(args: RecordArgs) {
    let mut stdout = std::io::stdout().lock();
    exit_on_error(run_record_with(args, &ProcessEnv, &mut stdout));
}

/// Import one external provider recording through its declared scenario.
pub(crate) fn run_import(args: ImportArgs) {
    let mut stdout = std::io::stdout().lock();
    exit_on_error(run_import_with(args, &mut stdout));
}

/// Validate the complete inference fixture coverage tree.
pub(crate) fn run_check(args: &CheckArgs) {
    let mut stdout = std::io::stdout().lock();
    exit_on_error(run_check_with(args, &mut stdout));
}

/// Prints one opaque command error and exits nonzero.
fn exit_on_error(result: Result<(), String>) {
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

/// Fallible live-recording handler with injectable environment and output.
fn run_record_with(args: RecordArgs, env: &dyn EnvReader, stdout: &mut dyn std::io::Write) -> Result<(), String> {
    validate_provider(&args.provider)?;
    let scenario = load_declared_scenario(&args.root, &args.scenario)?;
    let rules = load_optional_redaction_rules(args.redactions_file.as_deref())?;
    let output = OutputDestination::resolve(&args.root, &args.provider, &args.scenario, args.out)?;
    output.preflight()?;
    let base_url = args
        .provider_base_url
        .as_deref()
        .unwrap_or_else(|| default_provider_base_url(&args.provider));
    let target = provider_target(&args.provider, base_url, &args.model, env)?;
    let fixture = run_fixture_future(ScenarioRunner::record_live_with_rules(&scenario, target, &rules))?;
    validate_commit_safe_with_rules(&fixture, &rules).map_err(|error| error.to_string())?;

    persist_fixture(&output, &fixture)?;
    writeln!(stdout, "wrote inference fixture {}", output.path().display())
        .map_err(|_error| "inference command output could not be written".to_owned())
}

/// Fallible external-import handler with injectable output.
#[expect(
    clippy::too_many_lines,
    reason = "validation, import, provenance, materialization, and persistence remain one ordered transaction"
)]
fn run_import_with(args: ImportArgs, stdout: &mut dyn std::io::Write) -> Result<(), String> {
    validate_provider(&args.provider)?;
    validate_controlled_synthetic(&args.provider, args.controlled_synthetic)?;
    let scenario = load_declared_scenario(&args.root, &args.scenario)?;
    let rules = load_optional_redaction_rules(args.redactions_file.as_deref())?;
    let output = OutputDestination::resolve(&args.root, &args.provider, &args.scenario, args.out)?;
    output.preflight()?;
    let content = read_bounded(&args.recording, MAX_EXTERNAL_RECORDING_BYTES, "external recording")?;
    let content = std::str::from_utf8(&content).map_err(|_error| "external recording is not valid UTF-8".to_owned())?;
    let mut imported = import_external_recording(content).map_err(|error| error.to_string())?;
    // Materialization retains the imported model for coherence checks while
    // fixture provenance needs its own owned copy.
    let model = imported
        .model
        .as_ref()
        .filter(|model| !model.trim().is_empty())
        .cloned()
        .ok_or_else(|| "external recording model is invalid".to_owned())?;
    let provenance = import_provenance(
        args.provider,
        model,
        imported.source_id.take(),
        args.controlled_synthetic,
    );
    let fixture = run_fixture_future(ScenarioRunner::materialize_with_rules(
        &scenario,
        provenance,
        vec![imported],
        &rules,
    ))?;
    validate_commit_safe_with_rules(&fixture, &rules).map_err(|error| error.to_string())?;

    persist_fixture(&output, &fixture)?;
    writeln!(stdout, "wrote inference fixture {}", output.path().display())
        .map_err(|_error| "inference command output could not be written".to_owned())
}

/// Builds provenance from the explicit import origin selected by the caller.
fn import_provenance(
    provider: String,
    model: String,
    source_id: Option<String>,
    controlled_synthetic: bool,
) -> FixtureProvenance {
    let kind = if controlled_synthetic {
        ProvenanceKind::Synthetic
    } else {
        ProvenanceKind::Imported
    };
    FixtureProvenance {
        kind,
        provider,
        model,
        source_id,
    }
}

/// Fallible coverage-check handler with injectable output.
fn run_check_with(args: &CheckArgs, stdout: &mut dyn std::io::Write) -> Result<(), String> {
    let report = check_coverage(&args.root).map_err(|error| opaque_check_error(&error).to_owned())?;
    writeln!(stdout, "{}", format_coverage_report(&report))
        .map_err(|_error| "inference command output could not be written".to_owned())
}

// -----------------------------------------------------------------------------
// Command Boundaries
// -----------------------------------------------------------------------------

/// Builds one current-thread Tokio runtime for a fixture operation.
fn run_fixture_future<T>(future: impl Future<Output = Result<T, FixtureError>>) -> Result<T, String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|_error| "inference command runtime could not start".to_owned())?;
    runtime.block_on(future).map_err(|error| error.to_string())
}

/// Reads process environment values without formatting their contents.
trait EnvReader {
    /// Returns the encoded bytes for `name`, if it is present.
    fn read(&self, name: &'static str) -> Option<Vec<u8>>;
}

/// Production environment reader.
struct ProcessEnv;

impl EnvReader for ProcessEnv {
    fn read(&self, name: &'static str) -> Option<Vec<u8>> {
        std::env::var_os(name).map(|value| value.as_encoded_bytes().to_vec())
    }
}

/// Credential and native-header policy selected by a canonical provider slug.
enum ProviderPolicy {
    /// OpenAI's dedicated required bearer credential.
    OpenAi,
    /// Anthropic's origin-contained native credential and version headers.
    Anthropic,
    /// Credentialless local or private-network vLLM provider.
    Vllm,
    /// Optional bearer credential for all other canonical provider slugs.
    Compatible,
}

impl ProviderPolicy {
    /// Selects the typed policy for a previously validated provider slug.
    fn for_provider(provider: &str) -> Self {
        match provider {
            "openai" => Self::OpenAi,
            "anthropic" => Self::Anthropic,
            "vllm" => Self::Vllm,
            _ => Self::Compatible,
        }
    }
}

/// Builds a validated provider target with an in-memory credential header.
#[expect(
    clippy::too_many_lines,
    reason = "provider-specific credential policies remain explicit and colocated"
)]
fn provider_target(provider: &str, base_url: &str, model: &str, env: &dyn EnvReader) -> Result<ProviderTarget, String> {
    validate_provider(provider)?;
    let base_url = Url::parse(base_url).map_err(|_error| "provider base URL is invalid".to_owned())?;
    let mut outbound_headers = reqwest::header::HeaderMap::new();
    match ProviderPolicy::for_provider(provider) {
        ProviderPolicy::OpenAi => {
            validate_openai_origin(&base_url)?;
            let credential =
                required_credential(env, OPENAI_API_KEY, "OPENAI_API_KEY is required for provider openai")?;
            insert_bearer_credential(&mut outbound_headers, &credential)?;
        },
        ProviderPolicy::Anthropic => {
            validate_anthropic_origin(&base_url)?;
            let credential = required_credential(
                env,
                ANTHROPIC_API_KEY,
                "ANTHROPIC_API_KEY is required for provider anthropic",
            )?;
            insert_anthropic_credential(&mut outbound_headers, &credential)?;
        },
        ProviderPolicy::Vllm => {},
        ProviderPolicy::Compatible => {
            if let Some(credential) = env.read(INFERENCE_PROVIDER_API_KEY).filter(|value| !value.is_empty()) {
                insert_bearer_credential(&mut outbound_headers, &credential)?;
            }
        },
    }
    Ok(ProviderTarget {
        provider: provider.to_owned(),
        model: model.to_owned(),
        base_url,
        outbound_headers,
    })
}

/// Reads one required, nonempty credential without formatting its bytes.
fn required_credential(
    env: &dyn EnvReader,
    name: &'static str,
    missing_error: &'static str,
) -> Result<Vec<u8>, String> {
    env.read(name)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| missing_error.to_owned())
}

/// Rejects every target except the root of OpenAI's exact first-party origin.
fn validate_openai_origin(base_url: &Url) -> Result<(), String> {
    let is_first_party_root = base_url.scheme() == "https"
        && base_url.host_str() == Some("api.openai.com")
        && base_url.port_or_known_default() == Some(443)
        && base_url.username().is_empty()
        && base_url.password().is_none()
        && base_url.path() == "/"
        && base_url.query().is_none()
        && base_url.fragment().is_none();
    if !is_first_party_root {
        return Err("OpenAI provider base URL must be https://api.openai.com/".to_owned());
    }
    Ok(())
}

/// Rejects every target except the root of Anthropic's exact first-party origin.
fn validate_anthropic_origin(base_url: &Url) -> Result<(), String> {
    let is_first_party_root = base_url.scheme() == "https"
        && base_url.host_str() == Some("api.anthropic.com")
        && base_url.port_or_known_default() == Some(443)
        && base_url.username().is_empty()
        && base_url.password().is_none()
        && base_url.path() == "/"
        && base_url.query().is_none()
        && base_url.fragment().is_none();
    if !is_first_party_root {
        return Err("Anthropic provider base URL must be https://api.anthropic.com/".to_owned());
    }
    Ok(())
}

/// Adds Anthropic's sensitive native key and static API-version headers.
fn insert_anthropic_credential(
    outbound_headers: &mut reqwest::header::HeaderMap,
    credential: &[u8],
) -> Result<(), String> {
    let mut value =
        HeaderValue::from_bytes(credential).map_err(|_error| "provider credential could not be used".to_owned())?;
    value.set_sensitive(true);
    outbound_headers.insert("x-api-key", value);
    outbound_headers.insert("anthropic-version", HeaderValue::from_static(ANTHROPIC_VERSION));
    Ok(())
}

/// Adds one sensitive bearer credential without formatting its bytes.
fn insert_bearer_credential(
    outbound_headers: &mut reqwest::header::HeaderMap,
    credential: &[u8],
) -> Result<(), String> {
    let mut bearer = Vec::with_capacity(7_usize.saturating_add(credential.len()));
    bearer.extend_from_slice(b"Bearer ");
    bearer.extend_from_slice(credential);
    let mut value =
        HeaderValue::from_bytes(&bearer).map_err(|_error| "provider credential could not be used".to_owned())?;
    value.set_sensitive(true);
    outbound_headers.insert(reqwest::header::AUTHORIZATION, value);
    Ok(())
}

/// Selects the provider default without reading environment state.
fn default_provider_base_url(provider: &str) -> &'static str {
    match ProviderPolicy::for_provider(provider) {
        ProviderPolicy::OpenAi => OPENAI_BASE_URL,
        ProviderPolicy::Anthropic => ANTHROPIC_BASE_URL,
        ProviderPolicy::Vllm | ProviderPolicy::Compatible => COMPATIBLE_BASE_URL,
    }
}

/// Loads a scenario only after deterministic declaration discovery.
fn load_declared_scenario(root: &Path, requested_id: &str) -> Result<InferenceScenario, String> {
    load_declared_scenario_with_discovery(root, requested_id, discover_scenario_snapshots)
}

/// Selects one owned scenario from a coherent discovered snapshot.
fn load_declared_scenario_with_discovery(
    root: &Path,
    requested_id: &str,
    discover: impl FnOnce(&Path) -> Result<BTreeMap<String, ScenarioSnapshot>, FixtureError>,
) -> Result<InferenceScenario, String> {
    validate_scenario_id(requested_id)?;
    let mut scenarios = discover(root).map_err(|error| error.to_string())?;
    let snapshot = scenarios.remove(requested_id).ok_or_else(|| {
        format!(
            "inference scenario `{requested_id}` is not declared below `{}`",
            root.display()
        )
    })?;
    let scenario = snapshot.scenario;
    if scenario.id != requested_id {
        return Err(format!(
            "inference scenario `{requested_id}` declaration identity changed"
        ));
    }
    Ok(scenario)
}

/// Validates a slash-separated stable scenario ID for safe default output use.
fn validate_scenario_id(id: &str) -> Result<(), String> {
    if id.is_empty() || id.split('/').any(|component| !is_canonical_component(component)) {
        return Err("inference scenario ID is not a safe relative path".to_owned());
    }
    Ok(())
}

/// Validates one provider identity and output-path component.
fn validate_provider(provider: &str) -> Result<(), String> {
    if !is_canonical_component(provider) {
        return Err("inference provider is not a safe path component".to_owned());
    }
    Ok(())
}

/// Requires provider `synthetic` when controlled synthetic provenance is requested.
fn validate_controlled_synthetic(provider: &str, controlled_synthetic: bool) -> Result<(), String> {
    if controlled_synthetic && provider != "synthetic" {
        return Err("controlled synthetic imports require provider `synthetic`".to_owned());
    }
    Ok(())
}

/// Returns whether one stable-ID component is a portable lowercase ASCII slug.
///
/// Components begin with `[a-z0-9]`, continue with `[a-z0-9._-]`, do not end
/// in a dot, and exclude Windows device basenames even when an extension is
/// present. This single grammar is applied before both path and env policies.
fn is_canonical_component(component: &str) -> bool {
    let bytes = component.as_bytes();
    let Some(first) = bytes.first() else {
        return false;
    };
    if !(first.is_ascii_lowercase() || first.is_ascii_digit())
        || bytes
            .iter()
            .any(|byte| !(byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'_' | b'-')))
        || bytes.last() == Some(&b'.')
    {
        return false;
    }
    !is_windows_device_basename(component.split('.').next().unwrap_or(component))
}

/// Rejects case-insensitive Windows device basenames and extension aliases.
fn is_windows_device_basename(basename: &str) -> bool {
    if ["con", "prn", "aux", "nul", "clock$"]
        .into_iter()
        .any(|reserved| basename.eq_ignore_ascii_case(reserved))
    {
        return true;
    }
    let bytes = basename.as_bytes();
    let Some(suffix) = bytes.get(3) else {
        return false;
    };
    bytes.len() == 4
        && bytes
            .get(..3)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(b"com") || prefix.eq_ignore_ascii_case(b"lpt"))
        && suffix.is_ascii_digit()
        && suffix != &b'0'
}

/// An explicit user-authorized output or a contained default destination.
enum OutputDestination {
    /// Arbitrary caller-selected path; parent symlinks remain user-authorized.
    Explicit(PathBuf),
    /// Canonical root and stable relative output path governed by no-follow preflight.
    Default {
        /// Resolved root under which the generated hierarchy must remain.
        canonical_root: PathBuf,
        /// Final canonical output path derived from validated components.
        path: PathBuf,
    },
}

impl OutputDestination {
    /// Resolves an explicit path or canonicalizes the selected default root.
    fn resolve(root: &Path, provider: &str, scenario: &str, explicit: Option<PathBuf>) -> Result<Self, String> {
        validate_provider(provider)?;
        validate_scenario_id(scenario)?;
        if let Some(path) = explicit {
            return Ok(Self::Explicit(path));
        }
        // Resolving the selected root once intentionally permits the root itself
        // to be a symlink while keeping all generated child paths canonical.
        let canonical_root =
            fs::canonicalize(root).map_err(|_error| "inference fixture root could not be resolved".to_owned())?;
        if !canonical_root.is_dir() {
            return Err("inference fixture root is not a directory".to_owned());
        }
        let path = canonical_root
            .join("recordings")
            .join(provider)
            .join(format!("{scenario}.json"));
        Ok(Self::Default { canonical_root, path })
    }

    /// Returns the final path used for output and reporting.
    fn path(&self) -> &Path {
        match self {
            Self::Explicit(path) | Self::Default { path, .. } => path,
        }
    }

    /// Refuses replacement and unsafe existing default parents before execution.
    fn preflight(&self) -> Result<(), String> {
        match self {
            Self::Explicit(path) => reject_existing_output(path, false),
            Self::Default { canonical_root, path } => {
                preflight_default_parents(canonical_root, path)?;
                reject_existing_output(path, true)
            },
        }
    }
}

/// Refuses any existing final entry, including a dangling symlink.
fn reject_existing_output(path: &Path, opaque: bool) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(_) if opaque => Err("inference fixture output already exists".to_owned()),
        Ok(_) => Err(format!("inference fixture output `{}` already exists", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_error) if opaque => Err("default inference fixture output path could not be inspected".to_owned()),
        Err(_error) => Err(format!(
            "inference fixture output `{}` could not be inspected",
            path.display()
        )),
    }
}

/// Persists pretty JSON with one newline through a same-directory no-clobber rename.
fn persist_fixture(output: &OutputDestination, fixture: &WireFixture) -> Result<(), String> {
    let document = fixture.to_pretty_json_document(output.path()).map_err(|_error| {
        format!(
            "inference fixture `{}` could not be serialized",
            output.path().display()
        )
    })?;
    match output {
        OutputDestination::Explicit(path) => {
            let parent = prepare_explicit_output_parent(path)?;
            persist_document(path, parent, &document, None)
        },
        OutputDestination::Default { canonical_root, path } => {
            let parent = prepare_default_output_parent(canonical_root, path)?;
            persist_document(path, &parent, &document, Some(canonical_root))
        },
    }
}

/// Creates an explicit output parent after fixture validation succeeds.
fn prepare_explicit_output_parent(path: &Path) -> Result<&Path, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if parent != Path::new(".") {
        fs::create_dir_all(parent)
            .map_err(|_error| format!("inference fixture parent `{}` could not be created", parent.display()))?;
    }
    Ok(parent)
}

/// Writes and no-clobber persists one complete fixture document.
fn persist_document(path: &Path, parent: &Path, document: &[u8], contained_root: Option<&Path>) -> Result<(), String> {
    if let Some(root) = contained_root {
        verify_default_parent(root, parent)?;
        reject_existing_output(path, true)?;
    }
    let mut temporary = NamedTempFile::new_in(parent).map_err(|_error| {
        format!(
            "temporary inference fixture in `{}` could not be created",
            parent.display()
        )
    })?;
    temporary
        .write_all(document)
        .and_then(|()| temporary.flush())
        .map_err(|_error| {
            format!(
                "temporary inference fixture in `{}` could not be written",
                parent.display()
            )
        })?;
    if let Some(root) = contained_root {
        verify_default_parent(root, parent)?;
        reject_existing_output(path, true)?;
    }
    temporary.persist_noclobber(path).map_err(|_error| {
        format!(
            "inference fixture output `{}` already exists or could not be persisted",
            path.display()
        )
    })?;
    Ok(())
}

/// Checks all existing default-output parents without creating any component.
fn preflight_default_parents(root: &Path, path: &Path) -> Result<(), String> {
    let parent = default_relative_parent(root, path)?;
    let mut current = root.to_path_buf();
    for component in parent.components() {
        let std::path::Component::Normal(component) = component else {
            return Err("default inference fixture output path is unsafe".to_owned());
        };
        current.push(component);
        match fs::symlink_metadata(&current) {
            Ok(metadata) => verify_default_directory(root, &current, &metadata)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(_error) => return Err("default inference fixture output path could not be inspected".to_owned()),
        }
    }
    Ok(())
}

/// Creates default-output parents one component at a time without following symlinks.
///
/// This is a local-tool no-follow preflight, not a race-proof `openat` walk. Each
/// component and canonical containment are therefore rechecked at persistence
/// boundaries, which protects ordinary worktrees from existing symlink escapes.
fn prepare_default_output_parent(root: &Path, path: &Path) -> Result<PathBuf, String> {
    let parent = default_relative_parent(root, path)?;
    let mut current = root.to_path_buf();
    for component in parent.components() {
        let std::path::Component::Normal(component) = component else {
            return Err("default inference fixture output path is unsafe".to_owned());
        };
        current.push(component);
        let metadata = match fs::symlink_metadata(&current) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&current)
                    .map_err(|_error| "default inference fixture output parent could not be created".to_owned())?;
                fs::symlink_metadata(&current)
                    .map_err(|_error| "default inference fixture output parent could not be inspected".to_owned())?
            },
            Err(_error) => return Err("default inference fixture output parent could not be inspected".to_owned()),
        };
        verify_default_directory(root, &current, &metadata)?;
    }
    verify_default_parent(root, &current)?;
    Ok(current)
}

/// Returns the safe relative parent below a canonical default root.
fn default_relative_parent<'a>(root: &'a Path, path: &'a Path) -> Result<&'a Path, String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_error| "default inference fixture output path is unsafe".to_owned())?;
    relative
        .parent()
        .ok_or_else(|| "default inference fixture output path is unsafe".to_owned())
}

/// Rejects one existing symlink/non-directory and verifies canonical containment.
fn verify_default_directory(root: &Path, path: &Path, metadata: &fs::Metadata) -> Result<(), String> {
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("default inference fixture output path is unsafe".to_owned());
    }
    verify_default_parent(root, path)
}

/// Rechecks that one non-symlink directory canonically remains below the root.
fn verify_default_parent(root: &Path, parent: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(parent)
        .map_err(|_error| "default inference fixture output parent could not be inspected".to_owned())?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("default inference fixture output path is unsafe".to_owned());
    }
    let canonical = fs::canonicalize(parent)
        .map_err(|_error| "default inference fixture output parent could not be resolved".to_owned())?;
    if !canonical.starts_with(root) {
        return Err("default inference fixture output path is unsafe".to_owned());
    }
    Ok(())
}

/// Reads at most `limit` bytes without exposing file content in errors.
fn read_bounded(path: &Path, limit: usize, label: &'static str) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|_error| format!("{label} `{}` could not be opened", path.display()))?;
    let maximum = u64::try_from(limit)
        .ok()
        .and_then(|limit| limit.checked_add(1))
        .ok_or_else(|| format!("{label} size limit is invalid"))?;
    let mut bytes = Vec::with_capacity(limit.min(8192));
    file.take(maximum)
        .read_to_end(&mut bytes)
        .map_err(|_error| format!("{label} `{}` could not be read", path.display()))?;
    if bytes.len() > limit {
        return Err(format!("{label} exceeds its size limit"));
    }
    Ok(bytes)
}

/// Loads optional strict literal-redaction rules.
fn load_optional_redaction_rules(path: Option<&Path>) -> Result<RedactionRules, String> {
    path.map_or_else(|| Ok(RedactionRules::default()), load_redaction_rules)
}

/// Loads a bounded JSON object while rejecting duplicate and empty sources.
fn load_redaction_rules(path: &Path) -> Result<RedactionRules, String> {
    let document = read_bounded(path, MAX_REDACTIONS_FILE_BYTES, "redactions file")?;
    let mut deserializer = serde_json::Deserializer::from_slice(&document);
    let RedactionDocument(literals) = <RedactionDocument as serde::Deserialize>::deserialize(&mut deserializer)
        .map_err(|_error| "redactions file is invalid".to_owned())?;
    deserializer
        .end()
        .map_err(|_error| "redactions file must contain exactly one JSON object".to_owned())?;
    if literals.keys().any(String::is_empty) {
        return Err("redactions file contains an empty literal source".to_owned());
    }
    Ok(RedactionRules { literals })
}

/// Strict duplicate-detecting redaction document.
struct RedactionDocument(BTreeMap<String, String>);

impl<'de> serde::Deserialize<'de> for RedactionDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(RedactionVisitor)
    }
}

/// Serde visitor for a strict string-to-string JSON object.
struct RedactionVisitor;

impl<'de> Visitor<'de> for RedactionVisitor {
    type Value = RedactionDocument;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON object of unique string replacements")
    }

    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let mut literals = BTreeMap::new();
        while let Some((source, replacement)) = map.next_entry::<String, String>()? {
            if literals.insert(source, replacement).is_some() {
                return Err(de::Error::custom("duplicate literal source"));
            }
        }
        Ok(RedactionDocument(literals))
    }
}

/// Formats only deterministic aggregate counts and stable status identifiers.
fn format_coverage_report(report: &CoverageReport) -> String {
    let statuses = report
        .counts_by_status
        .iter()
        .map(|(status, count)| format!("{}:{count}", coverage_status_name(status)))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "inference coverage ok: features={} scenarios={} recordings={} statuses={statuses}",
        report.features_total, report.scenarios_total, report.recordings_total
    )
}

/// Maps rich library diagnostics to static command-boundary failure categories.
fn opaque_check_error(error: &FixtureError) -> &'static str {
    match error {
        FixtureError::JsonFixture { .. }
        | FixtureError::CommitSafety { .. }
        | FixtureError::UnsupportedWireFixtureVersion { .. }
        | FixtureError::PersistedDocumentTooLarge { .. }
        | FixtureError::PersistedDocumentAllocation
        | FixtureError::FixtureDocumentChanged => "inference recording validation failed",
        FixtureError::Read { .. } | FixtureError::ReadDirectory { .. } | FixtureError::FixtureSymlink { .. } => {
            "inference fixture tree validation failed"
        },
        FixtureError::UnsupportedCoverageManifestVersion { .. }
        | FixtureError::DuplicateScenarioId { .. }
        | FixtureError::DuplicateCoverageFeatureId { .. }
        | FixtureError::CoverageUnknownFeature { .. }
        | FixtureError::CoverageMissingScenario { .. }
        | FixtureError::CoverageOneSidedLink { .. }
        | FixtureError::CoverageScenarioWithoutFeatures { .. }
        | FixtureError::CoverageReasonRequired { .. }
        | FixtureError::CoverageProviderReasonRequired { .. }
        | FixtureError::CoverageProviderRequired { .. }
        | FixtureError::CoverageLiveRecordingRequired { .. }
        | FixtureError::CoverageSyntheticRecordingRequired { .. }
        | FixtureError::CoverageUnknownRecordingScenario { .. }
        | FixtureError::CoverageInvariant { .. }
        | FixtureError::DuplicateRecordingIdentity { .. }
        | FixtureError::UnsupportedInferenceScenarioVersion { .. }
        | FixtureError::InvalidScenarioExpectation { .. }
        | FixtureError::YamlFixture { .. } => "inference coverage validation failed",
        _ => "inference fixture validation failed",
    }
}

/// Returns the stable serialized name of one coverage status.
const fn coverage_status_name(status: &CoverageStatus) -> &'static str {
    match status {
        CoverageStatus::Covered => "covered",
        CoverageStatus::LiveCovered => "live_covered",
        CoverageStatus::SyntheticOnly => "synthetic_only",
        CoverageStatus::Unsupported => "unsupported",
        CoverageStatus::ProviderUnsupported => "provider_unsupported",
    }
}

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests;
