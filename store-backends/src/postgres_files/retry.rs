// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Error classification and bounded retry for the file catalog.
//!
//! `sqlx` errors are classified into [`PgFailure`] classes before they are
//! mapped onto [`CatalogError`](praxis_ai_store::CatalogError), so the retry
//! wrapper can repeat only the failures a repeat can fix. The classified error
//! itself travels inside the [`CatalogError::Backend`] variant as a
//! [`PgBackendError`], keeping the connection URL and its credentials out of
//! the message.

use std::{
    error::Error,
    fmt,
    future::Future,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

/// Why a `PostgreSQL` operation failed, in retry terms.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PgFailure {
    /// The pool is unreachable or exhausted; fail fast, the caller decides
    /// whether to back off.
    Unavailable,
    /// The transaction was rolled back (`40001` serialization failure, `40P01`
    /// deadlock, `55P03` lock timeout) or the connection was lost mid-query.
    /// A repeat is safe when the operation is idempotent by state check.
    Retryable,
    /// Constraint, syntax, decode and every other failure: repeating cannot
    /// change the outcome.
    NonRetryable,
}

impl PgFailure {
    /// Classify a `PostgreSQL` SQLSTATE code.
    pub(super) fn from_code(code: &str) -> Self {
        match code {
            // serialization failure
            "40001" |
            // deadlock detected
            "40P01" |
            // lock not available (lock timeout)
            "55P03" |
            // admin shutdown (terminated backend)
            "57P01" => Self::Retryable,
            _ => Self::NonRetryable,
        }
    }
}

/// Classify a `sqlx` error for retry decisions.
///
/// Pool exhaustion and closure are unavailable. `IO` failures are classified
/// retryable: `sqlx` does not distinguish a failed connect from a lost
/// established connection, and the plan's retryable class covers connection
/// loss, terminated backends and IO errors. The constructor classifies its
/// connect failure as unavailable separately, before any retry applies.
pub(crate) fn classify(err: &sqlx::Error) -> PgFailure {
    match err {
        sqlx::Error::PoolTimedOut | sqlx::Error::PoolClosed => PgFailure::Unavailable,
        sqlx::Error::Io(_) => PgFailure::Retryable,
        sqlx::Error::Database(db) => db
            .code()
            .as_deref()
            .map_or(PgFailure::NonRetryable, PgFailure::from_code),
        _ => PgFailure::NonRetryable,
    }
}

/// A classified, redacted backend failure carried by
/// [`CatalogError::Backend`](praxis_ai_store::CatalogError::Backend).
#[derive(Debug)]
pub(crate) struct PgBackendError {
    /// Retry classification of the original failure.
    pub class: PgFailure,
    /// Redacted human-readable description.
    message: String,
}

impl PgBackendError {
    /// Build a classified error from an already-redacted message.
    pub(crate) fn new(class: PgFailure, message: String) -> Self {
        Self { class, message }
    }

    /// A retryable failure for tests and internal reuse.
    #[cfg(test)]
    pub(crate) fn retryable(message: &str) -> Self {
        Self::new(PgFailure::Retryable, message.to_owned())
    }

    /// A non-retryable failure for tests.
    #[cfg(test)]
    pub(crate) fn non_retryable(message: &str) -> Self {
        Self::new(PgFailure::NonRetryable, message.to_owned())
    }
}

impl fmt::Display for PgBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.class, self.message)
    }
}

impl Error for PgBackendError {}

impl fmt::Display for PgFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => write!(f, "postgres unavailable"),
            Self::Retryable => write!(f, "postgres retryable failure"),
            Self::NonRetryable => write!(f, "postgres failure"),
        }
    }
}

// -----------------------------------------------------------------------------
// Bounded retry
// -----------------------------------------------------------------------------

/// Maximum attempts per retried operation, including the first.
pub(super) const MAX_ATTEMPTS: u32 = 3;

/// Backoff before the first retry.
pub(super) const BASE_BACKOFF: Duration = Duration::from_millis(50);

/// Upper bound of the backoff, before jitter.
pub(super) const MAX_BACKOFF: Duration = Duration::from_millis(500);

/// Run `op`, retrying while it fails retryably.
///
/// Each attempt runs in a fresh transaction owned by `op`; only errors whose
/// class guarantees a rollback (`40001`, `40P01`, `55P03`) or a lost
/// connection are retried, and only up to [`MAX_ATTEMPTS`] attempts. The delay
/// grows exponentially from [`BASE_BACKOFF`] to [`MAX_BACKOFF`] with
/// time-derived jitter, so concurrent workers do not retry in lockstep.
///
/// Apply this only to operations that are idempotent by state check: a retry
/// after an ambiguous commit must observe the earlier attempt's result and
/// turn into a no-op. Never apply it to `claim_cleanup` or the batch marking
/// operations, whose retry would duplicate work.
pub(crate) async fn with_retry<T, F, Fut>(mut op: F) -> Result<T, PgBackendError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, PgBackendError>>,
{
    let mut attempts = 0;
    loop {
        attempts += 1;
        match op().await {
            Ok(value) => return Ok(value),
            Err(err) if err.class == PgFailure::Retryable && attempts < MAX_ATTEMPTS => {
                tokio::time::sleep(backoff_delay(attempts)).await;
            },
            Err(err) => return Err(err),
        }
    }
}

/// Exponential backoff with jitter for the `attempts`-th failed attempt.
///
/// Jitter is derived from the wall clock rather than a random source, keeping
/// the crate free of a `rand` dependency; uniformity is not load-bearing here,
/// only decorrelation between workers.
pub(super) fn backoff_delay(failed_attempts: u32) -> Duration {
    let shift = (failed_attempts - 1).min(4);
    let exponential = BASE_BACKOFF.saturating_mul(1 << shift);
    let capped = exponential.min(MAX_BACKOFF);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or_default();
    // Jitter of at most half the capped delay, in whole milliseconds.
    let max_jitter_millis = capped.as_millis() / 2;
    let jitter_millis = if max_jitter_millis == 0 {
        0
    } else {
        u64::try_from(nanos % (max_jitter_millis + 1)).unwrap_or_default()
    };
    capped + Duration::from_millis(jitter_millis)
}
