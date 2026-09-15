// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Live load signals polled from the local grid operator.
//!
//! Polls the operator's federation endpoint and keeps a bounded per-series
//! window, so routing scores candidates on current load. Samples key on the
//! operator's observation time, so a republished cache value never reads as new.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use dashmap::DashMap;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use praxis_filter::FilterError;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

/// Default poll interval against the operator.
const DEFAULT_INTERVAL_MS: u64 = 1_000;

/// Default retention for each series.
const DEFAULT_WINDOW_SECS: u64 = 300;

/// Default age past which a sample no longer describes the present.
const DEFAULT_MAX_AGE_MS: i64 = 30_000;

/// Default per-request timeout. May exceed the poll interval: ticks are
/// sequential, so a slow poll delays the next tick rather than overlapping.
const DEFAULT_TIMEOUT_MS: u64 = 1_500;

/// Cap on providers retained, so a misconfigured or hostile endpoint cannot grow
/// the store without bound.
const MAX_PROVIDERS: usize = 4_096;

/// Cap on distinct metric names per provider, bounding a peer that floods unique
/// names past the provider cap.
const MAX_METRICS_PER_PROVIDER: usize = 64;

/// TLS material the collector presents and verifies against, by path.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LoadTls {
    /// CA bundle the endpoint certificate is verified against.
    pub ca_path: String,

    /// Certificate presented to the endpoint. Omitted, an access-enforcing
    /// listener refuses the unidentified collector.
    #[serde(default)]
    pub cert_path: Option<String>,

    /// Private key for `cert_path`.
    #[serde(default)]
    pub key_path: Option<String>,
}

/// Collector configuration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LoadConfig {
    /// Signals endpoint on the local operator (e.g. `.../metrics`).
    ///
    /// Unqualified it carries the local site and every collected peer. Adding
    /// `?target=<site>` narrows it to one and leaves remote candidates unscored.
    #[serde(default = "default_signals_endpoint")]
    pub endpoint: String,

    /// Poll interval, in milliseconds.
    #[serde(default = "default_interval_ms", deserialize_with = "deserialize_interval_ms")]
    pub interval_ms: u64,

    /// Retention per series, in seconds.
    #[serde(default = "default_window_secs")]
    pub window_secs: u64,

    /// Liveness bound, in milliseconds: past this a sample is ignored, so a dead
    /// operator stops pinning routing to values that no longer describe anything.
    /// Rejected at parse time when negative, which would empty the freshness
    /// window and silently mark every sample stale.
    #[serde(default = "default_max_age_ms", deserialize_with = "deserialize_max_age_ms")]
    pub max_age_ms: i64,

    /// Request timeout, in milliseconds.
    #[serde(default = "default_timeout_ms", deserialize_with = "deserialize_timeout_ms")]
    pub timeout_ms: u64,

    /// TLS material for the endpoint. Without it the collector is a plain client,
    /// with no grid CA trust and indistinguishable from any other caller.
    #[serde(default)]
    pub tls: Option<LoadTls>,

    /// Signals to score candidates on. Also the metric names polled, so an
    /// unlisted signal is neither collected nor scored.
    #[serde(default = "default_signals")]
    pub signals: Vec<SignalConfig>,
}

/// Default poll interval.
const fn default_interval_ms() -> u64 {
    DEFAULT_INTERVAL_MS
}

/// Default retention per series.
const fn default_window_secs() -> u64 {
    DEFAULT_WINDOW_SECS
}

/// Default liveness bound on a sample.
const fn default_max_age_ms() -> i64 {
    DEFAULT_MAX_AGE_MS
}

/// Reject a negative `max_age_ms`. A negative bound empties the freshness range
/// and would mark every sample stale, silently disabling load scoring.
fn deserialize_max_age_ms<'de, D>(deserializer: D) -> Result<i64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = i64::deserialize(deserializer)?;
    if value < 0 {
        return Err(serde::de::Error::custom("max_age_ms must not be negative"));
    }
    Ok(value)
}

/// Reject a zero `interval_ms`: `tokio::time::interval` panics on a zero period.
fn deserialize_interval_ms<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = u64::deserialize(deserializer)?;
    if value == 0 {
        return Err(serde::de::Error::custom("interval_ms must be greater than zero"));
    }
    Ok(value)
}

/// Reject a zero `timeout_ms`: a zero request timeout elapses immediately, so
/// every poll fails before it can read a sample.
fn deserialize_timeout_ms<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = u64::deserialize(deserializer)?;
    if value == 0 {
        return Err(serde::de::Error::custom("timeout_ms must be greater than zero"));
    }
    Ok(value)
}

/// Default request timeout.
const fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// One observation of a series.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sample {
    /// Operator observation time, in milliseconds since the epoch.
    pub at_ms: i64,
    /// Value as the provider reported it.
    pub value: f64,
}

/// A bounded window of one series, oldest first.
#[derive(Debug, Default)]
struct Series {
    /// Samples in timestamp order.
    samples: Vec<Sample>,
}

impl Series {
    /// Append `sample` if it is newer than what is held, then evict past `window`.
    fn push(&mut self, sample: Sample, window: Duration) {
        if self.samples.last().is_some_and(|last| sample.at_ms <= last.at_ms) {
            return;
        }
        self.samples.push(sample);
        let Ok(window_ms) = i64::try_from(window.as_millis()) else {
            return;
        };
        let cutoff = sample.at_ms.saturating_sub(window_ms);
        let keep_from = self.samples.partition_point(|s| s.at_ms < cutoff);
        if keep_from > 0 {
            self.samples.drain(..keep_from);
        }
    }
}

/// Series held for one provider, keyed by metric name.
#[derive(Debug, Default)]
struct Provider {
    /// Metric name to its window.
    metrics: HashMap<Box<str>, Series>,
}

/// Windowed signals per provider, keyed by `"site/cluster"` so a request-path
/// lookup matches a route candidate with no allocation.
#[derive(Debug)]
pub(crate) struct LoadStore {
    /// Provider key to its series.
    providers: DashMap<Box<str>, Provider>,
    /// Retention per series.
    window: Duration,
}

impl LoadStore {
    /// Create an empty store retaining `window` of history per series.
    pub fn new(window: Duration) -> Self {
        Self {
            providers: DashMap::new(),
            window,
        }
    }

    /// The key under which a candidate's series are held.
    pub fn key(site: &str, cluster: &str) -> Box<str> {
        format!("{site}/{cluster}").into_boxed_str()
    }

    /// Most recent sample of `metric` for `key`. Test-only since scoring moved
    /// to [`Self::window_worst`].
    #[cfg(test)]
    pub fn latest(&self, key: &str, metric: &str) -> Option<Sample> {
        let provider = self.providers.get(key)?;
        provider.metrics.get(metric)?.samples.last().copied()
    }

    /// Most recent sample of `metric` for `key` younger than `max_age_ms`.
    /// Test-only since scoring moved to [`Self::window_worst`].
    #[cfg(test)]
    pub fn fresh(&self, key: &str, metric: &str, now_ms: i64, max_age_ms: i64) -> Option<Sample> {
        // Range starts at zero: a future timestamp (publisher clock ahead) yields
        // a negative age that would otherwise read as fresh forever.
        self.latest(key, metric)
            .filter(|s| (0..=max_age_ms).contains(&now_ms.saturating_sub(s.at_ms)))
    }

    /// Worst reading of `metric` for `key` within the last `window_ms`, or `None`
    /// when the window holds no sample.
    ///
    /// Worst is the max when lower is better, so a drained burst stays penalised
    /// until it ages out rather than snapping to idle. Future-stamped samples are
    /// skipped.
    #[expect(
        clippy::too_many_arguments,
        clippy::significant_drop_tightening,
        reason = "keyed lookup with window bounds and polarity, holding the read guard for the bounded in-window scan"
    )]
    pub fn window_worst(
        &self,
        key: &str,
        metric: &str,
        now_ms: i64,
        window_ms: i64,
        lower_is_better: bool,
    ) -> Option<f64> {
        let provider = self.providers.get(key)?;
        let cutoff = now_ms.saturating_sub(window_ms);
        let mut worst: Option<f64> = None;
        for sample in &provider.metrics.get(metric)?.samples {
            if sample.at_ms < cutoff || sample.at_ms > now_ms {
                continue;
            }
            worst = Some(match worst {
                None => sample.value,
                Some(w) if lower_is_better => w.max(sample.value),
                Some(w) => w.min(sample.value),
            });
        }
        worst
    }

    /// Number of providers held.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Absorb an exposition response, skipping lines that do not parse so one
    /// bad line does not cost the rest.
    pub fn ingest(&self, text: &str) {
        for line in text.lines() {
            let Some(observation) = parse_line(line) else {
                continue;
            };
            // A non-finite value (NaN or infinity) would poison window_worst,
            // where it can survive as the sole in-window reading.
            if !observation.value.is_finite() {
                continue;
            }
            let key = Self::key(observation.site, observation.cluster);
            // Not atomic with the entry() insert below, but only the single
            // collector poll thread calls ingest, so the cap holds by construction.
            if !self.providers.contains_key(&key) && self.providers.len() >= MAX_PROVIDERS {
                continue;
            }
            let sample = Sample {
                at_ms: observation.at_ms,
                value: observation.value,
            };
            let mut provider = self.providers.entry(key).or_default();
            // Bound the distinct metric names one provider can create, so a peer
            // flooding unique names cannot grow the store past the provider cap.
            if !provider.metrics.contains_key(observation.metric) && provider.metrics.len() >= MAX_METRICS_PER_PROVIDER
            {
                continue;
            }
            provider
                .metrics
                .entry(observation.metric.into())
                .or_default()
                .push(sample, self.window);
        }
    }
}

/// One parsed sample line.
struct Observation<'a> {
    /// Metric name.
    metric: &'a str,
    /// Owning site, from the `grid_site` label.
    site: &'a str,
    /// Owning provider, from the `grid_provider` label.
    cluster: &'a str,
    /// Reported value.
    value: f64,
    /// Operator observation time.
    at_ms: i64,
}

/// Parse a `name{labels} value timestamp` line. One without a timestamp is
/// skipped: without it a republished sample cannot be told from a new one.
fn parse_line(line: &str) -> Option<Observation<'_>> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let (head, timestamp) = line.rsplit_once(' ')?;
    let (head, value) = head.rsplit_once(' ')?;
    let at_ms = timestamp.parse().ok()?;
    let value = value.parse().ok()?;
    let (metric, labels) = head.split_once('{')?;
    let labels = labels.strip_suffix('}')?;
    let mut site = None;
    let mut cluster = None;
    for pair in labels.split(',') {
        match pair.trim().split_once('=') {
            Some(("grid_site", v)) => site = Some(v.trim_matches('"')),
            Some(("grid_provider", v)) => cluster = Some(v.trim_matches('"')),
            _ => {},
        }
    }
    Some(Observation {
        metric,
        site: site?,
        cluster: cluster?,
        value,
        at_ms,
    })
}

/// Milliseconds since the epoch.
pub(crate) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// A running collector, stopped on drop.
#[derive(Debug)]
pub(crate) struct LoadCollector {
    /// Signals the poll loop to exit.
    cancel: CancellationToken,
}

impl Drop for LoadCollector {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Start polling `config.endpoint` into a new store.
///
/// Runs on its own thread and current-thread runtime, so it needs no runtime
/// current when the filter is built. A failed poll is logged and retried next
/// tick, and the store keeps what it had until `max_age_ms` expires it.
pub(crate) fn spawn(config: &LoadConfig, collect: &[String]) -> Result<(Arc<LoadStore>, LoadCollector), FilterError> {
    let store = Arc::new(LoadStore::new(Duration::from_secs(config.window_secs)));
    let cancel = CancellationToken::new();
    let polling_cfg = Polling::build(config, collect)?;

    let polling = Arc::clone(&store);
    let stopping = cancel.clone();
    std::thread::Builder::new()
        .name("load-collector".to_owned())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(rt) => rt,
                Err(error) => {
                    tracing::error!(%error, "load collector runtime unavailable; routing falls back to overlay order");
                    return;
                },
            };
            runtime.block_on(poll_loop(polling, stopping, polling_cfg));
        })
        .map_err(|e| -> FilterError { format!("intelligent_route: load collector thread: {e}").into() })?;

    Ok((store, LoadCollector { cancel }))
}

/// TLS material read from disk once at start, held as bytes so a mid-build
/// rotation cannot leave a certificate that no longer matches its key.
#[derive(Clone)]
pub(crate) struct ClientTls {
    /// CA bundle in PEM.
    ca: Vec<u8>,
    /// Certificate and key in one PEM, when the caller identifies itself.
    identity: Option<Vec<u8>>,
}

impl ClientTls {
    /// Read the configured material.
    ///
    /// # Errors
    ///
    /// Returns the IO error when a configured path cannot be read. A missing file
    /// fails here rather than starting an anonymous client a listener will refuse.
    pub(crate) fn load(cfg: &LoadTls) -> std::io::Result<Self> {
        let identity = match (&cfg.cert_path, &cfg.key_path) {
            (Some(cert), Some(key)) => {
                let mut pem = std::fs::read(cert)?;
                pem.extend_from_slice(&std::fs::read(key)?);
                Some(pem)
            },
            _ => None,
        };
        Ok(Self {
            ca: std::fs::read(&cfg.ca_path)?,
            identity,
        })
    }
}

/// Build the polling client, with TLS when the endpoint needs it.
///
/// Trusts the grid CA only and skips the hostname check: the operator serves the
/// site identity, whose SAN names the site, not this service. `tls_certs_only`
/// (reqwest 0.13 `ClientBuilder`) drops the built-in roots, which reqwest
/// requires before the hostname check can be turned off.
fn build_client(timeout: Duration, tls: Option<&ClientTls>) -> reqwest::Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder().timeout(timeout);
    if let Some(tls) = tls {
        let roots = reqwest::Certificate::from_pem_bundle(&tls.ca)?;
        builder = builder.tls_certs_only(roots).tls_danger_accept_invalid_hostnames(true);
        if let Some(identity) = &tls.identity {
            builder = builder.identity(reqwest::Identity::from_pem(identity)?);
        }
    }
    builder.build()
}

/// Where and how the collector polls.
struct Polling {
    /// Fully built request URL, `collect[]` included.
    url: String,
    /// Gap between polls.
    interval: Duration,
    /// Per-request timeout.
    timeout: Duration,
    /// Material presented and verified against, when the endpoint speaks TLS.
    tls: Option<ClientTls>,
}

impl Polling {
    /// Resolve the configuration, reading any TLS material from disk here so an
    /// unreadable file surfaces as a config error the caller sees rather than a
    /// silent per-poll refusal.
    ///
    /// # Errors
    ///
    /// Returns a [`FilterError`] when a configured path cannot be read.
    fn build(config: &LoadConfig, collect: &[String]) -> Result<Self, FilterError> {
        let tls = config
            .tls
            .as_ref()
            .map(ClientTls::load)
            .transpose()
            .map_err(|e| -> FilterError { format!("intelligent_route: load collector TLS: {e}").into() })?;
        Ok(Self {
            url: build_url(&config.endpoint, collect),
            interval: Duration::from_millis(config.interval_ms),
            timeout: Duration::from_millis(config.timeout_ms),
            tls,
        })
    }
}

/// Poll until cancelled, feeding every response into `store`.
async fn poll_loop(store: Arc<LoadStore>, cancel: CancellationToken, polling: Polling) {
    let Polling {
        url,
        interval,
        timeout,
        tls,
    } = polling;
    let client = match build_client(timeout, tls.as_ref()) {
        Ok(c) => c,
        Err(error) => {
            tracing::error!(%error, "load collector client unavailable; routing falls back to overlay order");
            return;
        },
    };
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            () = cancel.cancelled() => return,
            _ = ticker.tick() => {},
        }
        poll_once(&client, &url, &store).await;
    }
}

/// Fetch once and absorb the response, logging rather than propagating failure.
async fn poll_once(client: &reqwest::Client, url: &str, store: &LoadStore) {
    match client.get(url).send().await {
        Ok(response) => match response.text().await {
            Ok(body) => store.ingest(&body),
            Err(error) => tracing::debug!(%error, "load endpoint body unreadable"),
        },
        Err(error) => tracing::debug!(%error, "load endpoint poll failed"),
    }
}

/// Everything escaped except the RFC 3986 unreserved set.
const QUERY_VALUE: &AsciiSet = &NON_ALPHANUMERIC.remove(b'-').remove(b'_').remove(b'.').remove(b'~');

/// Append `collect[]` parameters for each metric name.
fn build_url(endpoint: &str, collect: &[String]) -> String {
    if collect.is_empty() {
        return endpoint.to_owned();
    }
    let query = collect
        .iter()
        .map(|s| format!("collect[]={}", utf8_percent_encode(s, QUERY_VALUE)))
        .collect::<Vec<_>>()
        .join("&");
    let separator = if endpoint.contains('?') { '&' } else { '?' };
    format!("{endpoint}{separator}{query}")
}

/// The default signals endpoint: the operator's cross-site mTLS rollup, which
/// needs the grid client identity via `signals_tls`. The single-site
/// `/metrics` exporter on the same host is the configurable alternative.
pub(crate) fn default_signals_endpoint() -> String {
    "https://grid-operator-signals:9091/v1/site/signals".to_owned()
}

/// The signals a grid scores on when it names none: the two the endpoint
/// picker's multicluster scorers read, under the operator's republished names.
pub(crate) fn default_signals() -> Vec<SignalConfig> {
    vec![
        SignalConfig {
            key: "llm_d_epp_average_queue_size".to_owned(),
            weight: default_signal_weight(),
            lower_is_better: true,
            scale: SignalScale::Relative,
            // Measured: across 183 ticks the inter-site spread is bimodal (30%
            // under 1.0, 64% over 2.0), so 1.0 is the conservative end of the
            // gap. Averaged over pods, so on two pods 1.0 is two queued requests.
            deadband: 1.0,
        },
        SignalConfig {
            key: "llm_d_epp_average_kv_cache_utilization".to_owned(),
            weight: default_signal_weight(),
            lower_is_better: true,
            scale: SignalScale::Ratio,
            // Measured: with no deadband, utilisation decided every queue tie on
            // spreads under 0.020, moving 28 of 183 decisions off-local for
            // nothing. 0.05 filters that noise with margin, far below real spread.
            deadband: 0.05,
        },
    ]
}

/// Default weight for a load signal.
const fn default_signal_weight() -> f64 {
    1.0
}

/// Default direction for a load signal.
const fn default_lower_is_better() -> bool {
    true
}

/// Reject a non-finite or non-positive `weight`. A non-finite weight poisons the
/// combined score with `NaN`, which then wins every pick. Zero erases the signal
/// and a negative inverts it, routing to the worst candidate. All silent.
fn deserialize_signal_weight<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = f64::deserialize(deserializer)?;
    if !value.is_finite() || value <= 0.0 {
        return Err(serde::de::Error::custom("weight must be finite and greater than zero"));
    }
    Ok(value)
}

/// Reject a non-finite or negative `deadband`. A non-finite band suppresses every
/// spread, so the signal never expresses a preference. A negative one is meaningless.
fn deserialize_deadband<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = f64::deserialize(deserializer)?;
    if !value.is_finite() || value < 0.0 {
        return Err(serde::de::Error::custom("deadband must be finite and not negative"));
    }
    Ok(value)
}

/// One signal the gateway scores candidates on.
#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SignalConfig {
    /// The signal's name in the publishing source's keyspace. A key the source
    /// does not publish is not an error: that signal says nothing, and the others
    /// decide the route.
    pub key: String,

    /// Relative weight in the combined score.
    #[serde(default = "default_signal_weight", deserialize_with = "deserialize_signal_weight")]
    pub weight: f64,

    /// Whether a lower reading is better. True for queue depth and utilisation,
    /// so it is the default.
    #[serde(default = "default_lower_is_better")]
    pub lower_is_better: bool,

    /// How the reading becomes a rating.
    #[serde(default)]
    pub scale: SignalScale,

    /// Spread below which candidates tie on this signal, in the signal's own
    /// units. Relative scaling stretches any spread to 0..1, so without a
    /// deadband half a queued request reads as decisively as two hundred.
    ///
    /// Below it the signal expresses no preference and the decision falls to the
    /// other signals, then to overlay (locality) order. The value should clear
    /// the move's network cost, `(L_remote - L_local) / T_service` in queue
    /// units, and the drift in the difference within one poll.
    #[serde(default, deserialize_with = "deserialize_deadband")]
    pub deadband: f64,
}

/// How a raw reading is turned into a 0.0 to 1.0 rating.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SignalScale {
    /// Rate against the other candidates, for an unbounded quantity such as queue
    /// depth.
    #[default]
    Relative,

    /// Take the reading as the rating, for a quantity already a ratio such as
    /// cache utilisation.
    Ratio,
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::significant_drop_tightening,
    clippy::unwrap_used,
    reason = "tests"
)]
mod tests {
    use super::*;

    /// Write PEM-ish bytes to a temp file and return its path.
    fn temp_pem(name: &str, body: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("load-tls-{name}-{}", std::process::id()));
        std::fs::write(&path, body).unwrap_or_else(|_| std::process::abort());
        path
    }

    #[test]
    fn tls_is_optional_and_absent_by_default() {
        let cfg: LoadConfig =
            serde_yaml::from_str("endpoint: http://operator:9091/metrics\n").unwrap_or_else(|_| std::process::abort());
        assert!(cfg.tls.is_none(), "a plaintext endpoint needs no material");
    }

    #[test]
    fn tls_paths_are_accepted() {
        let cfg: LoadConfig = serde_yaml::from_str(
            "endpoint: https://operator:9091/metrics\ntls:\n  ca_path: /etc/praxis/tls/ca.crt\n  cert_path: /etc/praxis/tls/tls.crt\n  key_path: /etc/praxis/tls/tls.key\n",
        )
        .unwrap_or_else(|_| std::process::abort());
        let tls = cfg.tls.unwrap_or_else(|| std::process::abort());
        assert_eq!(tls.ca_path, "/etc/praxis/tls/ca.crt");
        assert_eq!(tls.cert_path.as_deref(), Some("/etc/praxis/tls/tls.crt"));
    }

    #[test]
    fn a_certificate_without_a_key_names_nobody() {
        // Half an identity is not an identity.
        let ca = temp_pem("ca", b"-----BEGIN CERTIFICATE-----\n");
        let cfg = LoadTls {
            ca_path: ca.to_string_lossy().into_owned(),
            cert_path: Some("/nonexistent".to_owned()),
            key_path: None,
        };
        let loaded = ClientTls::load(&cfg).unwrap_or_else(|_| std::process::abort());
        assert!(loaded.identity.is_none(), "an unpaired certificate is not presented");
        drop(std::fs::remove_file(ca));
    }

    #[test]
    fn a_missing_file_is_an_error_not_an_anonymous_client() {
        let cfg = LoadTls {
            ca_path: "/nonexistent/ca.crt".to_owned(),
            cert_path: None,
            key_path: None,
        };
        assert!(
            ClientTls::load(&cfg).is_err(),
            "starting anonymous against a listener that will refuse us is worse than failing here"
        );
    }

    const QUEUE: &str = "inference_pool_average_queue_size";

    fn line(site: &str, cluster: &str, value: f64, at_ms: i64) -> String {
        format!(r#"{QUEUE}{{grid_site="{site}",grid_provider="{cluster}"}} {value} {at_ms}"#)
    }

    fn store() -> LoadStore {
        LoadStore::new(Duration::from_secs(300))
    }

    #[test]
    fn ingests_a_labelled_sample() {
        let store = store();
        store.ingest(&line("east", "pool-a", 3.0, 1_000));
        let sample = store.latest(&LoadStore::key("east", "pool-a"), QUEUE).expect("sample");
        assert_eq!(
            sample,
            Sample {
                at_ms: 1_000,
                value: 3.0
            },
            "value and time as reported"
        );
    }

    #[test]
    fn a_republished_sample_does_not_advance_the_series() {
        let store = store();
        let repeated = line("east", "pool-a", 3.0, 1_000);
        store.ingest(&repeated);
        store.ingest(&repeated);
        store.ingest(&repeated);
        let key = LoadStore::key("east", "pool-a");
        let provider = store.providers.get(&key).expect("provider");
        let held = provider.metrics.get(QUEUE).expect("series").samples.len();
        assert_eq!(held, 1, "the operator's cached republish is not a new observation");
    }

    #[test]
    fn a_newer_sample_advances_the_series() {
        let store = store();
        store.ingest(&line("east", "pool-a", 3.0, 1_000));
        store.ingest(&line("east", "pool-a", 5.0, 2_000));
        let sample = store.latest(&LoadStore::key("east", "pool-a"), QUEUE).expect("sample");
        assert_eq!(sample.value, 5.0, "the newer value wins");
    }

    #[test]
    fn samples_older_than_the_window_are_evicted() {
        let store = LoadStore::new(Duration::from_secs(10));
        for at_ms in [1_000, 5_000, 20_000] {
            store.ingest(&line("east", "pool-a", 1.0, at_ms));
        }
        let key = LoadStore::key("east", "pool-a");
        let provider = store.providers.get(&key).expect("provider");
        let samples = &provider.metrics.get(QUEUE).expect("series").samples;
        assert_eq!(samples.len(), 1, "only what falls inside the window: {samples:?}");
        assert_eq!(samples.first().map(|s| s.at_ms), Some(20_000), "the newest survives");
    }

    #[test]
    fn sites_do_not_collide_on_a_shared_cluster_name() {
        let store = store();
        store.ingest(&line("east", "pool-a", 1.0, 1_000));
        store.ingest(&line("west", "pool-a", 9.0, 1_000));
        assert_eq!(store.len(), 2, "the site is part of the key");
        let west = store.latest(&LoadStore::key("west", "pool-a"), QUEUE).expect("west");
        assert_eq!(west.value, 9.0, "each site keeps its own value");
    }

    #[test]
    fn a_line_without_a_timestamp_is_skipped() {
        let store = store();
        store.ingest(&format!(r#"{QUEUE}{{grid_site="east",grid_provider="pool-a"}} 3"#));
        assert_eq!(
            store.len(),
            0,
            "without a timestamp there is no way to order the sample"
        );
    }

    #[test]
    fn unlabelled_and_malformed_lines_are_skipped_without_losing_the_rest() {
        let store = store();
        let text = format!(
            "# HELP something\n{QUEUE} 3 1000\nnot a metric\n{}",
            line("east", "pool-a", 3.0, 1_000)
        );
        store.ingest(&text);
        assert_eq!(store.len(), 1, "the one usable line still lands");
    }

    #[test]
    fn a_provider_cannot_exceed_the_metric_name_cap() {
        let store = store();
        for i in 0..(MAX_METRICS_PER_PROVIDER + 10) {
            store.ingest(&format!(
                r#"metric_{i}{{grid_site="east",grid_provider="pool-a"}} 1 1000"#
            ));
        }
        let key = LoadStore::key("east", "pool-a");
        let provider = store.providers.get(&key).expect("provider");
        assert_eq!(
            provider.metrics.len(),
            MAX_METRICS_PER_PROVIDER,
            "a flood of unique metric names is bounded per provider"
        );
    }

    #[test]
    fn max_age_ms_rejects_a_negative_value() {
        let result: Result<LoadConfig, _> =
            serde_yaml::from_str("endpoint: http://operator:9091/metrics\nmax_age_ms: -1\n");
        assert!(result.is_err(), "a negative max_age_ms must be rejected at parse time");
    }

    #[test]
    fn interval_ms_rejects_zero() {
        let result: Result<LoadConfig, _> =
            serde_yaml::from_str("endpoint: http://operator:9091/metrics\ninterval_ms: 0\n");
        assert!(
            result.is_err(),
            "a zero interval_ms panics tokio::time::interval; reject at parse time"
        );
    }

    #[test]
    fn timeout_ms_rejects_zero() {
        let result: Result<LoadConfig, _> =
            serde_yaml::from_str("endpoint: http://operator:9091/metrics\ntimeout_ms: 0\n");
        assert!(
            result.is_err(),
            "a zero timeout_ms fails every poll immediately; reject at parse time"
        );
    }

    #[test]
    fn a_non_finite_signal_weight_is_rejected() {
        let result: Result<LoadConfig, _> = serde_yaml::from_str(
            "endpoint: http://operator:9091/metrics\nsignals:\n  - key: queue\n    weight: .inf\n",
        );
        assert!(
            result.is_err(),
            "a non-finite weight poisons the combined score with NaN"
        );
    }

    #[test]
    fn a_non_positive_signal_weight_is_rejected() {
        let result: Result<LoadConfig, _> =
            serde_yaml::from_str("endpoint: http://operator:9091/metrics\nsignals:\n  - key: queue\n    weight: 0\n");
        assert!(
            result.is_err(),
            "zero or negative weight silently erases or inverts the signal"
        );
    }

    #[test]
    fn a_non_finite_deadband_is_rejected() {
        let result: Result<LoadConfig, _> = serde_yaml::from_str(
            "endpoint: http://operator:9091/metrics\nsignals:\n  - key: queue\n    deadband: .inf\n",
        );
        assert!(result.is_err(), "a non-finite deadband suppresses every spread");
    }

    #[test]
    fn a_stale_sample_is_withheld_from_routing() {
        let store = store();
        store.ingest(&line("east", "pool-a", 3.0, 1_000));
        let key = LoadStore::key("east", "pool-a");
        assert!(store.fresh(&key, QUEUE, 10_000, 30_000).is_some(), "inside the bound");
        assert!(store.fresh(&key, QUEUE, 60_000, 30_000).is_none(), "past the bound");
    }

    #[test]
    fn windowed_worst_persists_a_drained_burst() {
        let store = store();
        let key = LoadStore::key("east", "pool-a");
        store.ingest(&line("east", "pool-a", 30.0, 1_000)); // burst
        store.ingest(&line("east", "pool-a", 1.0, 5_000)); // drained to idle
        // lower_is_better keeps the worst (max) in the window, so the burst persists.
        assert_eq!(
            store.window_worst(&key, QUEUE, 5_000, 30_000, true),
            Some(30.0),
            "a drained burst must persist as the worst reading in the window"
        );
        // The last-value view (what scoring used before) would have snapped to idle.
        assert_eq!(store.latest(&key, QUEUE).map(|s| s.value), Some(1.0));
    }

    #[test]
    fn a_sample_from_a_clock_ahead_of_ours_is_withheld() {
        let store = store();
        store.ingest(&line("east", "pool-a", 3.0, 60_000));
        let key = LoadStore::key("east", "pool-a");
        assert!(
            store.fresh(&key, QUEUE, 10_000, 30_000).is_none(),
            "a future timestamp must not read as fresh, or a dead site keeps winning"
        );
    }

    #[test]
    fn metric_names_become_collect_parameters() {
        let url = build_url("http://operator:9091/metrics", &[QUEUE.to_owned()]);
        assert_eq!(
            url,
            format!("http://operator:9091/metrics?collect[]={QUEUE}"),
            "one parameter carrying a bare metric name"
        );
    }

    #[test]
    fn an_endpoint_without_metric_names_is_left_alone() {
        let url = build_url("http://operator:9091/metrics", &[]);
        assert_eq!(url, "http://operator:9091/metrics", "nothing appended");
    }
}
