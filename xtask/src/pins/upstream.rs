// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Upstream release discovery for `bump-pins`.
//!
//! Network access is confined to [`Upstream`]; every response is handed to a
//! pure parsing function so selection rules are unit-tested on canned JSON.

use std::{io::Read as _, time::Duration};

use serde_json::Value;
use sha2::{Digest as _, Sha256};

use super::check::{is_sha256, major};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// User agent for every request; the GitHub API rejects requests without one.
const USER_AGENT: &str = "praxis-ai-bump-pins";

/// Ceiling for one request, sized for the ~180 MB `OpenCode` tarball.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// npm registry base URL.
pub(crate) const NPM_REGISTRY: &str = "https://registry.npmjs.org";

/// Read buffer for streamed downloads.
const CHUNK: usize = 64 * 1024;

/// GitHub releases per listing page. Codex attaches hundreds of assets to every
/// release (~650 KB of JSON each), and GitHub times out building larger pages.
const RELEASES_PER_PAGE: usize = 30;

/// How many listing pages are searched before giving up on a repository.
const RELEASE_PAGES: usize = 10;

// -----------------------------------------------------------------------------
// Client
// -----------------------------------------------------------------------------

/// Blocking HTTP access to the upstream registries.
pub(crate) struct Upstream {
    /// Shared HTTP client.
    client: reqwest::blocking::Client,

    /// Optional `GITHUB_TOKEN`, which lifts the anonymous API rate limit.
    github_token: Option<String>,
}

impl Upstream {
    /// Build the client.
    pub(crate) fn new() -> Result<Self, String> {
        // reqwest resolves TLS through the process rustls provider
        // (`rustls-no-provider`, see the workspace Cargo.toml) and building its
        // client aborts if none is installed yet.
        praxis_ai::install_crypto_provider();
        let client = reqwest::blocking::Client::builder()
            .user_agent(USER_AGENT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| format!("failed to build HTTP client: {e}"))?;
        let github_token = std::env::var("GITHUB_TOKEN").ok().filter(|t| !t.is_empty());
        Ok(Self { client, github_token })
    }

    /// GET `url`; `Ok(None)` on HTTP 404, an error on any other failure.
    fn fetch(&self, url: &str, accept: Option<&str>) -> Result<Option<reqwest::blocking::Response>, String> {
        let mut request = self.client.get(url);
        if let Some(accept) = accept {
            request = request.header(reqwest::header::ACCEPT, accept);
        }
        if let Some(token) = self
            .github_token
            .as_deref()
            .filter(|_| url.starts_with("https://api.github.com/"))
        {
            request = request.bearer_auth(token);
        }
        let response = request.send().map_err(|e| format!("failed to fetch {url}: {e}"))?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(format!("failed to fetch {url}: HTTP {status}"));
        }
        Ok(Some(response))
    }

    /// GET `url`, treating 404 as an error.
    fn get(&self, url: &str, accept: Option<&str>) -> Result<reqwest::blocking::Response, String> {
        self.fetch(url, accept)?
            .ok_or_else(|| format!("failed to fetch {url}: HTTP 404 Not Found"))
    }

    /// GET `url` as JSON; `Ok(None)` on HTTP 404.
    fn json(&self, url: &str, accept: Option<&str>) -> Result<Option<Value>, String> {
        self.fetch(url, accept)?
            .map(|response| response.json().map_err(|e| format!("failed to parse {url}: {e}")))
            .transpose()
    }

    /// GET `url` as JSON, treating 404 as an error.
    fn require_json(&self, url: &str, accept: Option<&str>) -> Result<Value, String> {
        self.get(url, accept)?
            .json()
            .map_err(|e| format!("failed to parse {url}: {e}"))
    }

    /// The newest stable npm release of `package` in `major_series`.
    pub(crate) fn npm_latest(&self, package: &str, major_series: u64) -> Result<String, String> {
        let encoded = package.replace('/', "%2f");
        let tags = self.require_json(&format!("{NPM_REGISTRY}/-/package/{encoded}/dist-tags"), None)?;
        if let Some(latest) = tags_latest_in_series(&tags, major_series) {
            return Ok(latest);
        }
        // `latest` crossed (or never reached) the series; pick from every
        // published version instead. The abbreviated packument omits READMEs.
        let packument = self.require_json(
            &format!("{NPM_REGISTRY}/{encoded}"),
            Some("application/vnd.npm.install-v1+json"),
        )?;
        newest_in_series(&packument, major_series)
            .ok_or_else(|| format!("npm {package} has no stable {major_series}.x release"))
    }

    /// The newest stable GitHub release of `repo` whose tag starts with
    /// `tag_prefix` and, when given, whose version is in `major_series`.
    ///
    /// `/releases/latest` is not used: it names a single release regardless of
    /// prefix or series, so a new major (or another tag family sharing the
    /// repository) would hide an eligible release behind it. The listing is
    /// newest first, so the search stops at the first page with an eligible
    /// release.
    pub(crate) fn github_newest(
        &self,
        repo: &str,
        tag_prefix: &str,
        major_series: Option<u64>,
    ) -> Result<Option<Release>, String> {
        for page in 1..=RELEASE_PAGES {
            let releases = self.require_json(
                &format!("https://api.github.com/repos/{repo}/releases?per_page={RELEASES_PER_PAGE}&page={page}"),
                None,
            )?;
            if let Some(release) =
                newest_release(&releases, tag_prefix, major_series).map_err(|e| format!("{repo}: {e}"))?
            {
                return Ok(Some(release));
            }
            if releases.as_array().is_none_or(|page| page.len() < RELEASES_PER_PAGE) {
                break;
            }
        }
        Ok(None)
    }

    /// The linux-x64 checksum from a Claude Code release manifest.
    pub(crate) fn claude_checksum(&self, version: &str) -> Result<String, String> {
        let url = format!("https://downloads.claude.ai/claude-code-releases/{version}/manifest.json");
        let manifest = self.require_json(&url, None)?;
        parse_claude_checksum(&manifest, "linux-x64").map_err(|e| format!("{url}: {e}"))
    }

    /// The manifest-list digest of a Docker Hub tag; `None` while the tag is
    /// not published yet.
    pub(crate) fn docker_hub_digest(&self, repository: &str, tag: &str) -> Result<Option<String>, String> {
        let url = format!("https://hub.docker.com/v2/repositories/{repository}/tags/{tag}");
        match self.json(&url, None)? {
            Some(body) => parse_hub_digest(&body).map(Some).map_err(|e| format!("{url}: {e}")),
            None => Ok(None),
        }
    }

    /// Stream `url` through SHA-256 without buffering the body.
    pub(crate) fn sha256(&self, url: &str) -> Result<String, String> {
        let mut response = self.get(url, None)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; CHUNK];
        loop {
            let read = response
                .read(&mut buffer)
                .map_err(|e| format!("failed to read {url}: {e}"))?;
            if read == 0 {
                break;
            }
            hasher.update(buffer.get(..read).unwrap_or_default());
        }
        let mut hex = String::with_capacity(64);
        for byte in hasher.finalize() {
            use std::fmt::Write as _;
            write!(hex, "{byte:02x}").unwrap();
        }
        Ok(hex)
    }
}

// -----------------------------------------------------------------------------
// Parsing
// -----------------------------------------------------------------------------

/// A GitHub release.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Release {
    /// Full tag name, e.g. `rust-v0.144.1`.
    pub(crate) tag: String,

    /// The tag without its prefix.
    pub(crate) version: String,

    /// `(asset name, sha256 hex)` for every asset GitHub published a digest for.
    pub(crate) assets: Vec<(String, String)>,
}

impl Release {
    /// The sha256 of the named asset.
    pub(crate) fn asset_sha256(&self, name: &str) -> Result<&str, String> {
        self.assets
            .iter()
            .find(|(asset, _)| asset == name)
            .map(|(_, digest)| digest.as_str())
            .ok_or_else(|| format!("release {} has no sha256 digest for asset {name}", self.tag))
    }
}

/// `dist-tags.latest`, when it is a stable release in `major_series`.
pub(crate) fn tags_latest_in_series(tags: &Value, major_series: u64) -> Option<String> {
    let latest = tags.get("latest")?.as_str()?;
    (is_stable(latest) && major(latest) == Some(major_series)).then(|| latest.to_owned())
}

/// The highest stable version in `major_series` among a packument's versions.
pub(crate) fn newest_in_series(packument: &Value, major_series: u64) -> Option<String> {
    packument
        .get("versions")?
        .as_object()?
        .keys()
        .filter(|version| is_stable(version) && major(version) == Some(major_series))
        .max_by_key(|version| semver_key(version))
        .cloned()
}

/// The highest eligible release in a GitHub releases listing: stable, tagged
/// `tag_prefix<major>.<minor>.<patch>`, and in `major_series` when given.
pub(crate) fn newest_release(
    releases: &Value,
    tag_prefix: &str,
    major_series: Option<u64>,
) -> Result<Option<Release>, String> {
    let releases = releases.as_array().ok_or("releases response is not an array")?;
    Ok(releases
        .iter()
        .filter_map(|release| parse_release(release, tag_prefix).ok())
        .filter(|release| major_series.is_none_or(|series| major(&release.version) == Some(series)))
        .max_by_key(|release| semver_key(&release.version)))
}

/// Parse a GitHub release object.
pub(crate) fn parse_release(release: &Value, tag_prefix: &str) -> Result<Release, String> {
    let tag = release
        .get("tag_name")
        .and_then(Value::as_str)
        .ok_or("release has no tag_name")?;
    if release.get("prerelease").and_then(Value::as_bool) == Some(true)
        || release.get("draft").and_then(Value::as_bool) == Some(true)
    {
        return Err(format!("release {tag} is a prerelease or draft"));
    }
    let version = tag
        .strip_prefix(tag_prefix)
        .filter(|v| is_stable(v))
        .ok_or_else(|| format!("release tag {tag} is not `{tag_prefix}<major>.<minor>.<patch>`"))?;
    Ok(Release {
        tag: tag.to_owned(),
        version: version.to_owned(),
        assets: release_assets(release),
    })
}

/// `(name, sha256 hex)` for every release asset with a sha256 digest.
fn release_assets(release: &Value) -> Vec<(String, String)> {
    release
        .get("assets")
        .and_then(Value::as_array)
        .map(|assets| {
            assets
                .iter()
                .filter_map(|asset| {
                    let name = asset.get("name")?.as_str()?;
                    let digest = asset.get("digest")?.as_str()?.strip_prefix("sha256:")?;
                    is_sha256(digest).then(|| (name.to_owned(), digest.to_owned()))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `.platforms[<slug>].checksum` from a Claude Code release manifest.
pub(crate) fn parse_claude_checksum(manifest: &Value, slug: &str) -> Result<String, String> {
    manifest
        .pointer(&format!("/platforms/{slug}/checksum"))
        .and_then(Value::as_str)
        .filter(|checksum| is_sha256(checksum))
        .map(str::to_owned)
        .ok_or_else(|| format!("no sha256 checksum for platform {slug}"))
}

/// The top-level (manifest-list) digest of a Docker Hub tag.
pub(crate) fn parse_hub_digest(tag: &Value) -> Result<String, String> {
    tag.get("digest")
        .and_then(Value::as_str)
        .filter(|digest| digest.strip_prefix("sha256:").is_some_and(is_sha256))
        .map(str::to_owned)
        .ok_or_else(|| "tag has no sha256 digest".to_owned())
}

/// Whether `version` is a plain `major.minor.patch` release.
pub(crate) fn is_stable(version: &str) -> bool {
    let parts: Vec<&str> = version.split('.').collect();
    parts.len() == 3
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
}

/// Numeric sort key for a stable version.
fn semver_key(version: &str) -> Vec<u64> {
    version.split('.').filter_map(|part| part.parse().ok()).collect()
}
